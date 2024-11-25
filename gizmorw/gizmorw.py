import h5py
import numpy as np
from collections.abc import Iterable


# Expects that the file_name does not have extension
def read_gizmo_file(file_name, nfiles, part_types = 0, part_keys = 'ParticleIDs'):
    headers = {}
    part_dict = {}

    # Can take in a single number for the particle types to read
    if not isinstance(part_types, Iterable):
        part_types = [part_types]
    if not isinstance(part_keys, Iterable):
        part_keys = [part_keys]

    if nfiles > 1:
        # First, I will grab the total number of particles and the header information
        with h5py.File('%s.0.hdf5' % file_name, 'r') as f:
            for key in f['Header'].attrs:
                headers.update({key: f['Header'].attrs[key]})

            tot_num_part = np.array(f['Header'].attrs['NumPart_Total'], dtype = np.uint64)

            # Check if this wasn't stored for some reason
            if np.sum(tot_num_part) == 0 and np.sum(headers['NumPart_ThisFile']) > 0:
                # Start with the values in the 0.hdf5 file, then add on the rest
                tot_num_part = np.array(headers['NumPart_ThisFile'], dtype = np.uint64)
                print('start tot_num_part[1]=%d' % tot_num_part[1])
                for file_num in range(1,nfiles):
                    f2 = h5py.File('%s.%s.hdf5' % (file_name, str(file_num)), 'r')
                    tot_num_part += np.array(f2['Header'].attrs['NumPart_ThisFile'], dtype = np.uint64)
                    f2.close()

                print('Total number of particles (re-calculated): %d' % np.sum(tot_num_part))
            elif np.sum(tot_num_part) == 0:
                raise ValueError('There are no particles in NumPart_ThisFile or NumPart_Total!')
            else:
                print('Total number of particles: %d' % np.sum(tot_num_part))

            # Need to allocate the arrays beforehand, since we want all of the particles
            # properties in a single array (per property, for tot_num_part).
            for part_type in part_types:
                pt = int(part_type)  # Be careful with arrays

                if tot_num_part[pt] == 0:
                    continue

                part_key = 'PartType%d' % pt
                part_dict.update({part_key: {}})

                print('%s: %d' % (part_key, tot_num_part[pt]))

                for key in f[part_key]:
                    if key not in part_keys:
                        continue
                    full_key = '%s/%s' % (part_key, key)
                    data_shape = np.array(f[full_key]).shape
                    try:
                        if data_shape[1] > 0:
                            real_shape = (tot_num_part[pt], data_shape[1])
                    except IndexError:
                        real_shape = tot_num_part[pt]

                    part_dict[part_key].update({key: np.zeros(real_shape)})

        # Fill up the arrays of length tot_num_part in slices depending on what is read in.
        idx_pointer = [0 for i in range(0, 6)]
        for file_num in range(nfiles):
            with h5py.File('%s.%s.hdf5' % (file_name, str(file_num)), 'r') as f:
                for part_type in part_types:
                    pt = int(part_type)
                    num_part_this = int(f['Header'].attrs['NumPart_ThisFile'][pt])
                    if num_part_this == 0:
                        continue

                    part_key = 'PartType%d' % pt

                    print('N=%d %s in file=%d' % (num_part_this, part_key, file_num))
                    print('Index pointer idx_pointer[%d]=%d' % (pt, idx_pointer[pt]))

                    for key in f[part_key]:
                        if key not in part_keys:
                            continue
                        full_key = '%s/%s' % (part_key, key)
                        part_dict[part_key][key][idx_pointer[pt]:idx_pointer[pt] + num_part_this] = np.copy(np.array(f[full_key]))

                    idx_pointer[pt] += num_part_this
    else:
        with h5py.File('%s.hdf5' % file_name, 'r') as f:
            for key in f['Header'].attrs:
                headers.update({key: f['Header'].attrs[key]})

            for part_type in part_types:
                pt = int(part_type)
                part_key = 'PartType%d' % pt

                if int(f['Header'].attrs['NumPart_Total'][pt]) == 0:
                    continue

                part_dict.update({part_key: {}})

                for key in f[part_key]:
                    if key not in part_keys:
                        continue
                    full_key = '%s/%s' % (part_key, key)
                    part_dict[part_key].update({key: np.array(f[full_key])})

    return headers, part_dict


def write_gizmo_file(file_name, nfiles, headers, part_dict):
    if not isinstance(headers, dict):
        raise ValueError('Must pass a dict for the headers variable.')
    if not isinstance(part_dict, dict):
        raise ValueError('Must pass a dict for the particle data.')

    part_types = []
    for i, num_part in enumerate(headers['NumPart_Total']):
        if num_part > 0:
            part_types.append(i) 

    if nfiles > 1:
        # For nfiles, we need to chunk the particle dictionary and ensure that the
        # header makes sense. The header needs to know the number of particles in
        # TOTAL and the number of particles in EACH FILE separately.
        for part_type in part_types:
            pt = int(part_type)
            part_key = 'PartType%d' % pt

            for key in part_dict[part_key]:
                # This is now a list of length nfiles
                part_dict[part_key][key] = np.array_split(part_dict[part_key][key], nfiles)  
                
        print('Writing to nfiles=%d' % nfiles)
        for file_num in range(nfiles):
            print('Write to file %d' % file_num)
            for i in range(0, 6):
                part_key = 'PartType%d' % i
                if i in part_types:
                    headers['NumPart_ThisFile'][i] = len(part_dict[part_key]['ParticleIDs'][file_num])

            with h5py.File('%s.%s.hdf5' % (file_name, str(file_num)), 'w') as f:
                h = f.create_group('Header')
                for key in headers:
                    h.attrs[key] = headers[key]

                
                for part_type in part_types:
                    pt = int(part_type)
                    part_key = 'PartType%d' % pt

                    print('N=%d %s in file_num=%d' % (h.attrs['NumPart_ThisFile'][pt], part_key, file_num))
                    p = f.create_group(part_key)
                    for key in part_dict[part_key]:
                        p.create_dataset(key, data = part_dict[part_key][key][file_num])
    else:
        # For 1 file it is simple, just write out the dictionaries to the HDF5 file
        # as they are in the structure.
        with h5py.File('%s.hdf5' % file_name, 'w') as f:
            h = f.create_group('Header')
            for key in headers:
                h.attrs[key] = headers[key]

            for part_type in part_types:
                pt = int(part_type)
                part_key = 'PartType%d' % pt

                p = f.create_group(part_key)
                for key in part_dict[part_key]:
                    p.create_dataset(key, data = part_dict[part_key][key])



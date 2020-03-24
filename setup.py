from setuptools import setup, find_packages


setup(name = 'gizmorw',
      version = '0.1',
      description = 'gizmorw: A Python HDF5 read/write tool for GIZMO HDF5 files',
      url = 'https://github.com/rennehan/gizmorw',
      author = 'Doug Rennehan',
      author_email = 'douglas.rennehan@gmail.com',
      license = 'GPLv3',
      packages = find_packages(),
      zip_safe = False, install_requires=['numpy', 'h5py'])

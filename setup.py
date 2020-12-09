from setuptools import setup, find_packages
import os
"""
def copy_dir(dir_path):
    base_dir = os.path.join('tpstorch/', dir_path)
    for (dirpath, dirnames, files) in os.walk(base_dir):
        for f in files:
            if "git" in f:
                continue
            else:
                yield os.path.join(dirpath.split('/', 1)[1], f)
"""
setup(name='tpstorch',
      version='0.1',
      description='Some Random Package',
      url='http://github.com/muhammadhasyim/tps-torch',
      author='M.R. Hasyim, C.B. Batton',
      author_email='muhammad_hasyim@berkeley.edu, chbatton@berkeley.edu',
      license='MIT',
      packages=find_packages(),
      package_data={"":["*.so",
                        "CMake/*.cmake",
                        "include/tpstorch/*.h",
                        "include/tpstorch/fts/*.h"]},
      install_requires=['numpy','scipy','pybind11','tqdm'],
      zip_safe=False)

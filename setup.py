from setuptools import setup, find_packages

setup(name='tpstorch',
      version='0.1',
      description='Some Random Package',
      url='http://github.com/muhammadhasyim/tps-torch',
      author='M.R. Hasyim, C.B. Batton',
      author_email='muhammad_hasyim@berkeley.edu, chbatton@berkeley.edu',
      license='MIT',
      packages=find_packages(),
      package_data={"":["*.so"]},
      install_requires=['numpy','scipy'],
      zip_safe=False)
"""Setup module for the liteflow package."""

from setuptools import setup, find_packages

setup(name='dket',
      version='0.0.1',
      description='Deep Knowledge Extraction from Text',
      url='https://github.com/dkmfbk/dket',
      author='Giulio Petrucci (petrux)',
      author_email='giulio.petrucci@gmail.com',
      license='Apache License 2.0',
      packages=find_packages(exclude=["dket.tests"]),
      install_requires=[
          'appdirs==1.4.3',
          'funcsigs==1.0.2',
          'mock==2.0.0',
          'numpy==1.12.1',
          'packaging==16.8',
          'pbr==2.1.0',
          'protobuf==3.2.0',
          'pyparsing==2.2.0',
          'six==1.10.0',
          'tensorflow==1.0.1',
          'liteflow',
      ],
      dependency_links=[
          'git+https://github.com/petrux/LiTeFlow.git@master#egg=liteflow-0'
      ],
      zip_safe=False)

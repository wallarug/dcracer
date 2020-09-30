#!/usr/bin/env python

from distutils.core import setup

setup(name='dcracer',
      version='0.1',
      description='',
      author='Robotics Masters Limited',
      author_email='cian@roboticsmasters.co',
      url='https://github.com/wallarug/dcracer',
      install_requires=['donkeycar',
                        ],
      extras_require={
                      'nano': [
                              'pyserial',
                              '',
                              ],
                      },
      packages=['dcracer'],
     )

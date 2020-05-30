#!/usr/bin/env python3
from setuptools import setup, find_packages

description = 'A collection of useful signal processing and astronomy functionality'

setup(
    name='doodads',
    version='0.0.1.dev',
    url='https://github.com/joseph-long/doodads',
    description=description,
    author='Joseph D. Long',
    author_email='me@joseph-long.com',
    packages=['doodads'],
    package_data={
        'doodads.ref': ['3.9um_Clio.dat'],
    },
    install_requires=['pytest>=5.4.2', 'numpy>=1.18.4',
                      'scipy>=1.2.1', 'matplotlib>=3.1.3', 'astropy>=4.0.1', ],
    entry_points={
        'console_scripts': [
            'dd-get-reference-data=doodads.commands:get_reference_data',
            'dd-run-diagnostics=doodads.commands:run_diagnostics',
        ],
    }
)

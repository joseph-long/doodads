#!/usr/bin/env python3
from setuptools import setup, find_packages

description='The algorithmic junk drawer'

setup(
    name='doodads',
    version='0.0.1.dev',
    url='https://github.com/joseph-long/doodads',
    description=description,
    author='Joseph D. Long',
    author_email='me@joseph-long.com',
    packages=['doodads'],
    install_requires=['pytest'],
    entry_points={
        'console_scripts': [
            'download_and_convert_settl_cond=doodads.modeling.io.settl_cond:download_and_convert_settl_cond',
        ],
    }
)

#!/usr/bin/env python3
from setuptools import setup

description = "A collection of useful signal processing and astronomy functionality"

setup(
    name="doodads",
    version="0.0.1.dev",
    url="https://github.com/joseph-long/doodads",
    description=description,
    author="Joseph D. Long",
    author_email="me@joseph-long.com",
    packages=["doodads"],
    package_data={
        "doodads.ref": ["3.9um_Clio.dat"],
    },
    install_requires=[
        "pytest>=5.4.2",
        "numpy>=1.16",
        "scipy>=1.2.1",
        "matplotlib>=3.1.3",
        "astropy>=4.0.1",
        "joblib>=0.14.1",
        "scikit-image>=0.16.2",
        "requests>=2.27.1",
        "xconf>=0.0.1",
        "tqdm>=4.62.3,<5",
        "coloredlogs>=15.0.1,<16",
        "ray>=2.7.0",
        "xconf>=0.0.0",
        "projecc>=1.0,<2",
        "fsspec",
    ],
    entry_points={
        "console_scripts": [
            "ddx=doodads.commands:DISPATCHER.main",
        ],
    },
)

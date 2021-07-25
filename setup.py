#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="lviv",
    version="0.1",
    description="A collection of code for LVIV 2021 workshop on Deep Learning in Neuroscience",
    author="Edgar Y. Walker, Zhuokun Ding",
    author_email="edgar.walker@uni-tuebingen.edu",
    packages=find_packages(exclude=[]),
    install_requires=["neuralpredictors~=0.0.1", "torch", "numpy"],
)
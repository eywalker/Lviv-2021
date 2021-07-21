  
#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="iviv",
    version="0.1",
    description="",
    author="Zhuokun Ding, Edgar Walker",
    author_email="zhuokund@bcm.edu",
    packages=find_packages(exclude=[]),
    install_requires=[
        "neuralpredictors",
        "pytorch",
        "numpy"
    ],
)
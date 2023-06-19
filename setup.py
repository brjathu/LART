#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="lart",
    version="0.0.0",
    description="Action recognition with 3D poses and tracking on any videos.",
    author="Jathushan Rajasegaran",
    author_email="jathushan@berkeley.edu",
    url="https://github.com/brjathu/LART",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    packages=find_packages(),
    install_requires=[
        "pytorch-lightning", 
        "lightning",
        "submitit",
        "pyrootutils",
        "opencv-python",
        "joblib",
        "rich",
        "einops",
        "hydra-core",
        "hydra-submitit-launcher",
        "timm",
        "phalp[all,blur] @ git+https://github.com/brjathu/PHALP.git",
    ],
    extras_require={
        'demo': [
            "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git",
            "slowfast @ git+https://github.com/brjathu/SlowFast.git",
        ],
    },
)

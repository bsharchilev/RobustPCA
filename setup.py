import os
from setuptools import setup

setup(
    name = "MRobustPCA",
    version = "0.1",
    author = "Boris Sharchilev",
    author_email = "bsharchilev@gmail.com",
    description = ("A scikit-learn-compatible implementation of Robust PCA using M-estimator loss."),
    url = "https://github.com/bsharchilev/RobustPCA",
    packages = ['rpca', 'rpca.data', 'rpca.baselines'],
    install_requires = [
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas'
    ],
    package_data = {
        'rpca.data': ['sleep.txt']
    },
    classifiers = [
        "Development Status :: 3 - Alpha",
    ],
)

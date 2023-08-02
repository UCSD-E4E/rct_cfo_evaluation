"""Example setup file
"""
from setuptools import setup, find_packages

setup(
    name='cfo_estimation',
    version='0.0.0.1',
    author='UCSD Engineers for Exploration',
    author_email='e4e@ucsd.edu',
    entry_points={
        'console_scripts': [
        ]
    },
    packages=find_packages(),
    install_requires=[
        'numpy',
        'jupyter',
        'matplotlib',
        'scipy',
    ],
    extras_require={
        'dev': [
            'pytest',
            'coverage',
            'pylint',
            'wheel',
        ]
    },
)

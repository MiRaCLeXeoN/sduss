from setuptools import setup, find_packages

setup(
    name='sduss',
    version='0.0.2',
    packages=find_packages(
        where='.',
        include=['sduss*'],
    )
)
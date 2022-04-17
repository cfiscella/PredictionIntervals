from setuptools import setup, find_packages

setup(
    name='PredictionIntervals',
    version='0.1',
    packages=find_packages(where = 'src')
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
)

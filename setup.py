from setuptools import setup, find_packages

setup(
    name='PredictionIntervals',
    version='0.1',
    #packages=(find_packages()+find_packages(where="./features")+find_packages(where="./models")),
    #packages=['src', 'src.subpackages', 'src.subpackages.ts_process', 'src.subpackages.model']
    packages = find_packages()
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires = [
                        'tensorboard>=2.8.0',
                        'statsmodels>=0.13.2',
                        'numpy>=1.21.5',
                        'tensorflow>=2.8.0',
                        'oauth2client>=4.1.3',
                        'sklearn>=0.0',
                        'arch>=5.2.0',
                        'matplotlib>=3.2.2',
                        'pandas>=1.3.5',
                        'scipy>=1.4.1',
                        'keras>=2.8.0',
                        'fuzzy-c-means>=1.6.3'],
    #long_description=open('README.txt').read(),
)

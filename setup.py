from setuptools import setup, find_packages

setup(
    name='PredictionIntervals',
    version='0.1',
    packages=find_packages()
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires = ['numexpr==2.8.1',
                        'pygments==2.6.1',
                        'requests==2.23.0',
                        'idna==2.10',
                        'uritemplate==3.0.1',
                        'tensorboard==2.8.0',
                        'astor==0.8.1',
                        'statsmodels==0.13.2',
                        'psutil==5.4.8',
                        'gast==0.5.3',
                        'numpy==1.21.5',
                        'numba==0.51.2',
                        'traitlets==5.1.1',
                        'portpicker==1.3.9',
                        'tblib==1.7.0',
                        'certifi==2021.10.8',
                        'tensorflow==2.8.0',
                        'oauth2client==4.1.3',
                        'sklearn==0.0',
                        'llvmlite==0.34.0',
                        'bottleneck==1.3.4',
                        'httplib2==0.17.4',
                        'chardet==3.0.4',
                        'debugpy==1.0.0',
                        'cffi==1.15.0',
                        'tornado==5.1.1',
                        'pyasn1==0.4.8',
                        'pyparsing==3.0.8',
                        'urllib3==1.24.3',
                        'wcwidth==0.2.5',
                        'packaging==21.3',
                        'kiwisolver==1.4.2',
                        'ipykernel==4.10.1',
                        'arch==5.2.0',
                        'ptyprocess==0.7.0',
                        'pycparser==2.21',
                        'six==1.15.0',
                        'pyarrow==6.0.1',
                        'matplotlib==3.2.2',
                        'pandas==1.3.5',
                        'pexpect==4.8.0',
                        'scipy==1.4.1',
                        'cloudpickle==1.3.0',
                        'jax==0.3.4',
                        'simplegeneric==0.8.1',
                        'decorator==4.4.2',
                        'dill==0.3.4',
                        'rsa==4.8',
                        'threadpoolctl==3.1.0',
                        'joblib==1.1.0',
                        'flatbuffers==2.0',
                        'termcolor==1.1.0',
                        'jaxlib==0.3.2',
                        '+cuda11.cudnn805cycler==0.11.0',
                        'keras==2.8.0',
                        'astunparse==1.6.3',
                        'patsy==0.5.2',
                        'pathlib==1.0.1',
                        'pydantic==1.9.0',
                        'wrapt==1.14.0',
                        'pytz==2018.9',
                        'ipywidgets==7.7.0',
                        'pickleshare==0.7.5',
                        'h5py==3.1.0',
                        'fuzzy-c-means==1.6.3']
    long_description=open('README.txt').read(),
)

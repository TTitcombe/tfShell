from distutils.core import setup

setup(
    name = 'tfShell',
    version = '0.1dev',
    author = 'Tom Titcombe',
    url = 'https://github.com/TTitcombe/tfShell',
    packages = ['tfShell'],
    install_requires = ['numpy', 'tensorflow'],
    license = 'MIT',
    description = 'Shell classes to automate and scale the building, training, and testing of ML models',
)

from setuptools import find_packages, setup

setup(
    name='fscrp',
    version='0.0.3',
    description='A Python library for implementing for tree structured chinese restaurant process models.',
    author='Andrew Roth',
    author_email='andrewjlroth@gmail.com',
    url='https://bitbucket.org/aroth85/fscrp',
    package_dir={'': 'lib'},
    packages=find_packages(),
    license='GPL v3'
)

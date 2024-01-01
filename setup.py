from setuptools import find_packages, setup

setup(
    name='PhyClone',
    version='0.2.0',
    description='A method for inferring clonal phylogenies from SNV data.',
    author='Andrew Roth',
    author_email='andrewjlroth@gmail.com',
    url='https://bitbucket.org/aroth85/phyclone',
    packages=find_packages(),
    license='GPL v3',
    entry_points={
        'console_scripts': [
            'phyclone = phyclone.cli:main',
        ]
    }
)

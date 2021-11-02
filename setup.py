from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='spherical_opt',
    version='0.3.0',
    packages=find_packages(),
    license='Apache 2.0',
    author='Philipp Eller',
    author_email='peller.phys@gmail.com',
    url='https://github.com/philippeller/spherical_opt',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy>=1.11',
    ],
)

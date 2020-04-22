from distutils.core import setup

setup(
    name='spherical_opt',
    version='0.2dev',
    packages=['spherical_opt',],
    license='Apache 2.0',
    author='Philipp Eller',
    long_description=open('README.md').read(),

    setup_requires=[
        'pip>=1.8',
        'setuptools>18.5',
        'numpy>=1.11'
    ],
)

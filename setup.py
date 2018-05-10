from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='evostra',

    version='2.0',

    description='Evolution Strategy Solver in Python',
    long_description=long_description,

    url='https://github.com/alirezamika/evostra',

    author='Alireza Mika',
    author_email='alirezamika@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],

    keywords='evolution strategy in machine learning - deep reinforcement learning',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['numpy'],

)

from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='qexp',
    url='https://github.com/qexp',
    author='Vu Tuan Hai',
    author_email='haivt@uit.edu.vn',
    # Needed to actually package something
    packages=['qexp'],
    # Needed for dependencies
    install_requires=['qiskit',
                      'qiskit-aer',
                      'qiskit-ibmq-provider',
                      'qiskit-ignis',
                      'qiskit-nature',
                      'qiskit-terra ',
                      'matplotlib',
                      'tdqm',
                      'pytest'],
    version='2.0',
    license='MIT',
    description='A Python package for quantum compilation',
    long_description=open('README.md').read(),
)

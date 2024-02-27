from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='qsee',
    url='https://github.com/vutuanhai237/qsee',
    author='Vu Tuan Hai',
    author_email='haivt@uit.edu.vn',
    # Needed to actually package something
    packages=['qsee'],
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
    description='qsee: A quantum object search engine',
    long_description=open('README.md').read(),
)

from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='qoop',
    url='https://github.com/vutuanhai237/qoop',
    author='Vu Tuan Hai',
    author_email='haivt@uit.edu.vn',
    # Needed to actually package something
    packages=['qsee'],
    # Needed for dependencies
    install_requires=['qiskit',
                      'qiskit-aer',
                      'qiskit-ignis',
                      'qiskit-nature',
                      'matplotlib',
                      'tdqm',
                      'pytest'],
    version='2.0',
    license='MIT',
    description='<qo|op>: A quantum object search engine',
    long_description=open('README.md').read(),
)

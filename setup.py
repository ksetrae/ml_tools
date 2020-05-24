from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_ = f.read()

setup(
    name='sklearn_search_tools',
    version='0.0.3',
    description='Tools for tuning the hyper-parameters that are based on scikit-learn modules',
    long_description=readme,
    author='ksetrae',
    author_email='ahisaw@gmail.com',
    license=license_,
    url='https://github.com/ksetrae/sklearn_search_tools.git',
    packages=find_packages()
)

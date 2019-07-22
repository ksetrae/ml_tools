from setuptools import setup, find_packages


# with open('README.rst') as f:
#     readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ml_tools',
    version='0.0.2',
    description='',
    long_description='',
    author='ksetrae',
    author_email='ksetrae@gmail.com',
    license=license,
    url='https://github.com/ksetrae/ml_tools.git',
    packages=find_packages()
)

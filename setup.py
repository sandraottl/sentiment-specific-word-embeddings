from setuptools import setup, find_packages

setup(
   name='sentiwords',
   version='0.1',
   description='A useful module',
   packages=find_packages(),
   install_requires=['nltk', 'numpy', 'pandas'],
   zip_safe=False
)

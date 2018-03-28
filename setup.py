from setuptools import setup, find_packages

setup(
   name='sentiwords',
   version='0.1',
   description='Package for training and working with Sentiment-Specific word embeddings.',
   packages=find_packages(),
   install_requires=['nltk', 'numpy', 'pandas', 'tqdm', 'sklearn', 'scipy', 'matplotlib'],
   extras_require={'gpu': ['tensorflow-gpu'], 'cpu': ['tensorflow']},
   zip_safe=False
)

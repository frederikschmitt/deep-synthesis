from setuptools import setup, find_packages

setup(
    name='deepsynthesis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'jupyter',
        'matplotlib',
        'numpy',
        'sly',
        'tensorflow>=2.4.0'
    ]
)

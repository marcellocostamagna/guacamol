import io
import re
from os import path
from setuptools import setup, find_packages

# Get the version from guacamol/__init__.py
__version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        io.open('guacamol/__init__.py', encoding='utf_8_sig').read()
                        ).group(1)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='guacamol',
      version=__version__,
      author='Marcello Costamagna',
      description='Package to run the Guacamol Benchmarks with DENOPTIM',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=["tests", "tests.*"]),
      license='MIT',
      install_requires=[
          'joblib>=0.12.5',
          'numpy>=1.15.2',
          'scipy>=1.1.0',
          'tqdm>=4.26.0',
          'FCD>=1.1',
          'rdkit-pypi>=2021.9.2.1',
      ],
      python_requires='>=3.6',
      extras_require={
          'rdkit': ['rdkit>=2018.09.1.0'],
      },
      include_package_data=True,
      zip_safe=False,
      )

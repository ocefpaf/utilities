# -*- coding: utf-8 -*-

import os
import re
import codecs
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

rootpath = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    return codecs.open(os.path.join(rootpath, *parts), 'r').read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

pkg_data = {'': ['data/*.csv']}

LICENSE = read('LICENSE.txt')
version = find_version('utilities', '__init__.py')
long_description = '{}\n'.format(read('README.md'))

# Hard library dependencies:
requires = ['iris', 'oceans', 'lxml', 'pandas', 'beautifulsoup4']
# Soft library dependencies:
recommended = dict(full=["ipython-notebook", "pyugrid", "folium"])
# pip install 'oceans[full]'


config = dict(name='utilities',
              version=version,
              description='Misc utilities functions for SECOORA',
              long_description=long_description,
              author='Filipe Fernandes',
              author_email='ocefpaf@gmail.com',
              license='MIT License',
              url='https://github.com/ocefpaf/utilities',
              classifiers=['Development Status :: 4 - Beta',
                           'Programming Language :: Python :: 2.7',
                           'Programming Language :: Python :: 3.3',
                           'Programming Language :: Python :: 3.4',
                           'License :: OSI Approved :: MIT License'],
              packages=['utilities'],
              package_data=pkg_data,
              install_requires=requires,
              extras_require=recommended,
              zip_safe=False)

setup(**config)

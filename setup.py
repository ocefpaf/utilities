# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


rootpath = os.path.abspath(os.path.dirname(__file__))


class PyTest(TestCommand):
    """python setup.py test"""
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


def read(*parts):
    return open(os.path.join(rootpath, *parts), 'r').read()


def extract_version():
    version = None
    fname = os.path.join(rootpath, 'utilities', '__init__.py')
    with open(fname) as f:
        for line in f:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters.
                break
    return version


email = "ocefpaf@gmail.com"
maintainer = "Filipe Fernandes"
authors = ['Rich Signell', 'Filipe Fernandes']

LICENSE = read('LICENSE.txt')
long_description = '{}\n{}'.format(read('README.txt'), read('CHANGES.txt'))


# Dependencies.
hard = ['iris', 'lxml', 'pandas', 'beautifulsoup4']
soft = dict(full=["ipython-notebook", "pyugrid", "folium", "oceans"])
tests_require = ['pytest', 'pytest-cov']


config = dict(name='utilities',
              version=extract_version(),
              packages=['utilities'],
              package_data={'': ['data/*.csv']},
              cmdclass=dict(test=PyTest),
              license=LICENSE,
              long_description=long_description,
              classifiers=['Development Status :: 4 - Beta',
                           'Environment :: Console',
                           'Intended Audience :: Science/Research',
                           'Intended Audience :: Developers',
                           'Intended Audience :: Education',
                           'License :: OSI Approved :: MIT License',
                           'Operating System :: OS Independent',
                           'Programming Language :: Python',
                           'Topic :: Education',
                           'Topic :: Scientific/Engineering'],
              description='Misc utilities functions for IOOS/SECOORA',
              authors=authors,
              author_email=email,
              maintainer='Filipe Fernandes',
              maintainer_email=email,
              url='https://github.com/pyoceans/utilities/releases/tag/v0.01',
              platforms='any',
              keywords=['oceanography', 'data analysis'],
              extras_require=soft,
              install_requires=hard,
              tests_require=tests_require,
              zip_safe=False)

setup(**config)

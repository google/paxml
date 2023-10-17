# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup.py file for paxml."""

import os
from setuptools import find_namespace_packages
from setuptools import setup

# Set this envvar to avoid installing packages from head that can overwrite
# existing installs of those packages, e.g., jax
SKIP_HEAD_INSTALLS = os.environ.get('SKIP_HEAD_INSTALLS', '')

def _get_requirements():
  """Parses requirements.txt file."""
  install_requires_tmp = []
  with open(
      os.path.join(os.path.dirname(__file__), './requirements.in'), 'r'
  ) as f:
    for line in f:
      package_name = line.strip()
      # Skip empty line or comments starting with "#".
      if (
          not package_name
          or package_name[0] == '#'
          or (' @ ' in package_name and SKIP_HEAD_INSTALLS)
      ):
        continue
      else:
        install_requires_tmp.append(package_name)
  return install_requires_tmp


install_requires = _get_requirements()

setup(
    name='paxml',
    version='1.2.0',  # use major/minor version number, e.g. "0.1.0"
    description=(
        'Framework to configure and run machine learning experiments '
        'on top of Jax.'
    ),
    author='PAX team',
    author_email='pax-dev@google.com',
    packages=find_namespace_packages(include=['paxml*']),
    python_requires='>=3.10',
    install_requires=install_requires,
    url='https://github.com/google/paxml',
    license='Apache-2.0',
    extras_require={
        'gpu': ['jsonlines==3.1.0', 'pysimdjson==5.0.2', 'zstandard==0.18.0'],
    },
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    zip_safe=False,
)

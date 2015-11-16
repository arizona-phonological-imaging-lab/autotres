from setuptools import setup
import os

# use requirements.txt as deps list
with open('install-reqs.txt') as f:
    required = f.read().splitlines()

with open('repos.txt') as f:
    repos = f.read().splitlines()

print(required)

setup(name='autotres',
      version='0.1',
      keywords='ultrasound tongue linguistics dbn',
      description='A library for automatically detecting tongue contours in ultrasound images',
      url='http://github.com/arizona-phonological-imaging-lab/autotres',
      author='APIL',
      author_email='gushahnpowell@gmail.com',
      license='Apache',
      packages=['a3'],
      install_requires=required,
      dependency_links=repos,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

from setuptools import setup, find_packages

# With this current version, can run scripts as:
# (1) python scripts/main_leafsnap.py
# (2) python -m scripts.main_leafsnap
# - Case 1. installed with pip install -e ./ => can use (1) and (2) if
# - Case 2: installed with pip install ./ =>  can only use (1)
setup(name='mce',
      version='0.1',
      description='Multi-step Contrastive Explanations',
      url='http://github.com/dmelis/mce',
      author='David Alvarez Melis',
      author_email='dalvmel@mit.edu',
      license='MIT',
      packages= find_packages(),#exclude=['js', 'node_modules', 'tests']),
      install_requires=[
          'numpy',
          'scipy',
          'torchnet',
          'matplotlib',
          'tqdm',
          'nltk',
          #'scikit-learn>=0.18',
          #'scikit-image>=0.12',
          'shapely',
          'squarify'
      ],
      include_package_data=True,
      zip_safe=False
)

# Approach two: aliasing the src
# setup(name='mce',
#       version='0.1',
#       description='Multi-step Contrastive Explanations',
#       url='http://github.com/dmelis/mce',
#       author='David Alvarez Melis',
#       author_email='dalvmel@mit.edu',
#       license='MIT',
#       scripts=['scripts/main_leafsnap.py'],
#       package_dir={'mce': 'src'},
#       packages=['mce'],
#       install_requires=[
#           'numpy',
#           'scipy',
#           'torchnet',
#           'matplotlib',
#           'tqdm',
#           'nltk',
#           #'scikit-learn>=0.18',
#           #'scikit-image>=0.12',
#           'shapely',
#           'squarify'
#       ],
#       include_package_data=True,
#       zip_safe=False
# )

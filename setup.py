from setuptools import setup, find_packages

setup(name='mce',
      version='0.1',
      description='Multi-step Contrastive Explanations',
      url='http://github.com/dmelis/mce',
      author='David Alvarez Melis',
      author_email='dalvmel@mit.edu',
      license='MIT',
      packages = ['mce']
      #packages= find_packages(exclude=['js', 'node_modules', 'tests']),
      install_requires=[
          'numpy',
          'pdb',
          'scipy',
          'matplotlib',
          'tqdm',
          #'scikit-learn>=0.18',
          #'scikit-image>=0.12',
          'shapely',
          'squarify'
      ],
      include_package_data=True,
      zip_safe=False
)

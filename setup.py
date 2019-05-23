from setuptools import setup, find_packages

contrib = [
    'Raphael Ortiz',
]

# setup.
setup(name='inter_view',
      version='0.2',
      description=
      'Utils to interactively visualize pre-computed dataframe with bokeh',
      author=', '.join(contrib),
      packages=find_packages(exclude=[
          'tests',
      ]),
      install_requires=[
          'numpy>=1.15.4',
          'matplotlib>=3.0.2',
          'bokeh>=1.0.4',
          'pandas>=0.23.4',
          'seaborn>=0.9.0',
          'Pillow>=6.0.0',
          'opencv-python>=4.1',
          'scikit-image',
      ],
      zip_safe=False)

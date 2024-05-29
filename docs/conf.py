# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import pathlib
import sys

from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))

pyproject_path = '../segmentationmetrics'
readme_path = '../README.md'

project = 'Segmentation Metrics'
copyright = f'{datetime.now().year}, Alex Daniel'
author = 'Alex Daniel'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'myst_parser']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autosummary_mock_imports = ['segmentationmetrics.surface_distance',
                            'tests']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

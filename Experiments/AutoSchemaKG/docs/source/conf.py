# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Point to your package root for auto-doc
project = 'AutoSchemaKG'
copyright = '2025, TSANG, Hong Ting'
author = 'TSANG, Hong Ting'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.autodoc',      # Auto-generate API docs from code
    # 'sphinx.ext.napoleon',     # Support Google-style docstrings
    # 'sphinx.ext.viewcode',     # Add links to source code
    'myst_parser',             # Enable Markdown support
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/atlas-rag-icon-wo-background.png"
html_favicon = "_static/atlas-rag-icon-wo-background.ico"

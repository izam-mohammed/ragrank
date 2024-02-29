import os
import sys

sys.path.insert(0, os.path.abspath("."))


# -- Project information --------
project = "Ragrank"
copyright = "2024, Izam Mohammed"
author = "Izam Mohammed"
release = "0.0.1"


# -- General configuration ---------

extensions = [
    "myst_parser",
    # 'sphinxext.opengraph',
    "sphinx_copybutton",
    "sphinx_rtd_theme",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

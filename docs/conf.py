"""Configuration file for the Sphinx documentation builder."""

import os
import sys
from importlib.metadata import version as get_version

sys.path.append(os.path.abspath("../"))
import stellarmesh

project = "Stellarmesh"
copyright = "2025, Stellarmesh Developers"  # noqa: A001
author = "Alex Koen"
release: str = get_version("package-name")

extensions = [
    "sphinx_github_style",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_tabs.tabs",
]
top_level = "stellarmesh"
linkcode_blob = "head"
linkcode_url = r"https://github.com/stellarmesh/stellarmesh"
linkcode_link_text = "Source"
add_module_names = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nbsphinx_execute = "never"  # Can't render without display

html_theme = "shibuya"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

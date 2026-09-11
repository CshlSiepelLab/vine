# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "VINE"
copyright = "2025-2026, Cold Spring Harbor Laboratory"
author = "Cold Spring Harbor Laboratory, Siepel Lab"

with open("../../version") as f:
    release = f.read().strip()
version = release

extensions = []

primary_domain = "c"
highlight_language = "console"

# -- HTML output --------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

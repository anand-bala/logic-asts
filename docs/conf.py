import importlib.metadata
from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).absolute()
DOC_SRC_DIR = CURRENT_FILE_PATH.parent
assert DOC_SRC_DIR.is_dir()
BUILD_DIR = DOC_SRC_DIR / "_build"
STATIC_DIR = DOC_SRC_DIR / "_static"


# -- Project information -----------------------------------------------------

copyright = "2026, Anand Balakrishnan"
author = "Anand Balakrishnan"
project = "logic-asts"
package_name = "logic-asts"
release = importlib.metadata.version(package_name)


# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / autosummary ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    # "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True


# -- Napoleon (Google-style docstrings) ---------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_static_path = [str(STATIC_DIR)]

html_theme_options = {
    "globaltoc_maxdepth": 4,
    "globaltoc_collapse": False,
    "navigation_with_keys": True,
    "sidebar_collapse": True,
    "show_relbars": True,
}

html_sidebars = {
    "**": [
        "about.html",
        "searchfield.html",
        "navigation.html",
        "relations.html",
        "donate.html",
    ]
}

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../takco'))


# -- Project information -----------------------------------------------------

project = 'takco'
copyright = '2020, Benno Kruit'
author = 'Benno Kruit'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinxcontrib.apidoc',
    'sphinx.ext.autodoc', 
    # 'sphinx_autodoc_napoleon_typehints',
    'sphinx.ext.coverage', 
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx_autodoc_typehints',
#     'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.linkcode',
]

apidoc_module_dir = '../takco'
apidoc_toc_file = False
apidoc_module_first = True
apidoc_separate_modules = True
apidoc_extra_args = ['-F']
autodoc_default_flags = ['members', 'titlesonly', 'show-inheritance']

# autosummary_generate = True

napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')


    import inspect
    import importlib

    module, fullname = info['module'], info['fullname']
    obj = importlib.import_module(module)
    for item in fullname.split('.'):
        obj = getattr(obj, item, None)

    if obj is None:
        return None

    # get original from decorated methods
    try: obj = getattr(obj, '_orig')
    except AttributeError: pass

    try:
        _, line = inspect.getsourcelines(obj)
    except (TypeError, IOError):
        # obj doesn't have a module, or something
        return None


    return "https://github.com/karmaresearch/takco/blob/master/%s.py#L%s" % (filename, line)

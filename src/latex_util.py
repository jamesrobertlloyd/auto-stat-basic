"""
Latex report creation routines for automatic statistician

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

import os
import subprocess
import tempfile
import shutil
import jinja2
import re

# DEBUG = True
DEBUG = False


def initialise_tex(latex_dir, suffix):
    """Copies latex_dir to a temp directory.
    Returns name of temp directory."""
    temp_dir = tempfile.mkdtemp(suffix='_' + suffix)
    os.rmdir(temp_dir)  # In preparation for copytree
    shutil.copytree(latex_dir, temp_dir)
    return temp_dir


def update_tex(temp_dir, tex):
    """Writes tex to temp_dir"""
    file_name = 'output'
    tex_file = os.path.join(temp_dir, '%s.tex' % file_name)
    with open(tex_file, 'w') as tex_fp:
        tex_fp.write(tex)


def compile_tex(temp_dir):
    """Compiles (twice) the latex into pdf"""
    file_name = 'output'
    tex_file = os.path.join(temp_dir, '%s.tex' % file_name)
    if DEBUG is False:  # then bin the output of pdflatex
        with open(os.devnull, 'w') as nullout:
            subprocess.call('pdflatex -output-directory {} {}'.format(temp_dir, tex_file),
                            shell=True,
                            stdout=nullout)
            subprocess.call('pdflatex -output-directory {} {}'.format(temp_dir, tex_file),
                            shell=True,
                            stdout=nullout)
    else:
        subprocess.call('pdflatex -output-directory {} {}'.format(temp_dir, tex_file),
                        shell=True)
        subprocess.call('pdflatex -output-directory {} {}'.format(temp_dir, tex_file),
                        shell=True)
    return os.path.join(temp_dir, '%s.pdf' % file_name)


LATEX_SUBS = (
    (re.compile(r'\\'), r'\\textbackslash'),
    (re.compile(r'([{}_#%&$])'), r'\\\1'),
    (re.compile(r'~'), r'\~{}'),
    (re.compile(r'\^'), r'\^{}'),
    (re.compile(r'"'), r"''"),
    (re.compile(r'\.\.\.+'), r'\\ldots'),
)


def escape_tex(value):
    """Filters strings for the template."""
    newval = value
    for pattern, replacement in LATEX_SUBS:
        newval = pattern.sub(replacement, newval)
    return newval


def get_latex_template(templatefl):
    """Sets up the template environment and loads the template file to create a template object."""
    # In this case, we will load templates off the filesystem.
    # This means we must construct a FileSystemLoader object.
    template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(os.path.realpath(__file__)))

    template_env = jinja2.Environment(loader=template_loader, trim_blocks=True, lstrip_blocks=True)

    # Change the jinja tags, because the defaults look like latex
    template_env.block_start_string = '((*'
    template_env.block_end_string = '*))'
    template_env.variable_start_string = '((('
    template_env.variable_end_string = ')))'
    template_env.comment_start_string = '((='
    template_env.comment_end_string = '=))'
    template_env.filters['escape_tex'] = escape_tex  # remove latex from strings passed to template

    template = template_env.get_template(templatefl)

    return template


def error_pdf(latex_dir, templatefl="./pdf_template.jinja",
              title='An error has occurred',
              body='Check the file format - is the data csv and numerical?'):
    """Make a pdf with custom error messages"""
    template = get_latex_template(templatefl)
    temp_dir = initialise_tex(latex_dir)
    update_tex(temp_dir, template.render({'title': title,
                                          'body': body}))
    outfile = compile_tex(temp_dir)
    return outfile

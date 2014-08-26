"""
Latex report creation routines for automatic statistician

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

import os
import subprocess
import tempfile
import shutil


def compile_tex(tex, latex_dir, file_name):
    temp_dir = tempfile.mkdtemp()
    os.rmdir(temp_dir) # In preparation for copytree
    shutil.copytree(latex_dir, temp_dir)
    with open(os.path.join(temp_dir, '%s.tex' % file_name), 'w') as tex_file:
        tex_file.write(tex)
    old_dir = os.getcwd()
    os.chdir(temp_dir)
    subprocess.call([os.path.join(temp_dir, 'create_all_pdf.sh')])
    os.chdir(old_dir)
    return os.path.join(temp_dir, '%s.pdf' % file_name)


def title_tex(title, body=''):
    tex = '''
\documentclass{article} %% For LaTeX2e
\usepackage{format/nips13submit_e}
\\nipsfinalcopy %% Uncomment for camera-ready version
\usepackage{times}
\usepackage{hyperref}
\usepackage{url}
\usepackage{color}
\definecolor{mydarkblue}{rgb}{0,0.08,0.45}
\hypersetup{
    pdfpagemode=UseNone,
    colorlinks=true,
    linkcolor=mydarkblue,
    citecolor=mydarkblue,
    filecolor=mydarkblue,
    urlcolor=mydarkblue,
    pdfview=FitH}

\usepackage{graphicx, amsmath, amsfonts, bm, lipsum, capt-of}

\usepackage{natbib, xcolor, wrapfig, booktabs, multirow, caption}

\usepackage{float}

\def\ie{i.e.\ }
\def\eg{e.g.\ }

\\title{%s}

\\newcommand{\\fix}{\marginpar{FIX}}
\\newcommand{\\new}{\marginpar{NEW}}

\setlength{\marginparwidth}{0.9in}
\input{include/commenting.tex}

%%%% For submission, make all render blank.
%%\\renewcommand{\LATER}[1]{}
%%\\renewcommand{\\fLATER}[1]{}
%%\\renewcommand{\TBD}[1]{}
%%\\renewcommand{\\fTBD}[1]{}
%%\\renewcommand{\PROBLEM}[1]{}
%%\\renewcommand{\\fPROBLEM}[1]{}
%%\\renewcommand{\NA}[1]{#1}  %% Note, NA's pass through!

\\begin{document}

\\allowdisplaybreaks

\maketitle

%s

\\end{document}
''' % (title, body)
    return tex


def title_pdf(latex_dir, file_name, title, body=''):
    return compile_tex(tex=title_tex(title=title, body=body),
                       latex_dir=latex_dir, file_name=file_name)

def error_pdf(latex_dir, file_name):
    return title_pdf(latex_dir=latex_dir, file_name=file_name, title='An error has occurred',
                     body='Check the file format - is the data csv and numerical?')


def waiting_pdf(latex_dir, file_name):
    return title_pdf(latex_dir=latex_dir, file_name=file_name, title='Your report is being prepared',
                     body='This usually takes a couple of minutes - at most ten minutes.')


def waiting_tex():
    return title_tex(title='Your report is being prepared',
                     body='This usually takes a couple of minutes - at most ten minutes.')


def over_capacity_pdf(latex_dir, file_name):
    return title_pdf(latex_dir=latex_dir, file_name=file_name,
                     title='The compute server is over capacity - please try again later')


def custom_pdf(latex_dir, file_name, title, body=''):
    return title_pdf(latex_dir=latex_dir, file_name=file_name,
                     title=title, body=body)
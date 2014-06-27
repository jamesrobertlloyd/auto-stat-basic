#!/bin/bash

FILES=*.tex
for f in $FILES
do
    pdflatex $f
    pdflatex $f
done

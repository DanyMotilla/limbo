#!/bin/bash

# Clean up any files in the current directory
rm -f *.aux *.bbl *.blg *.idx *.ind *.lof *.lot *.out *.toc *.acn *.acr *.alg *.glg *.glo *.gls *.ist *.fls *.log *.fdb_latexmk *.snm *.nav *.dvi *.synctex.gz *.pdf

# Ensure build directory exists
mkdir -p build

# Run latexmk with output to build directory
latexmk -pdf -outdir=build main.tex

# If successful, remove any stray files that might have been created
rm -f *.aux *.bbl *.blg *.idx *.ind *.lof *.lot *.out *.toc *.acn *.acr *.alg *.glg *.glo *.gls *.ist *.fls *.log *.fdb_latexmk *.snm *.nav *.dvi *.synctex.gz *.pdf

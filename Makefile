# Makefile for NKUA thesis compilation

MAIN = thesis_final
LATEX = pdflatex
BIBER = biber

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex bibliography.bib
	$(LATEX) $(MAIN)
	$(BIBER) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).bcf $(MAIN).blg $(MAIN).run.xml $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).out

view: $(MAIN).pdf
	open $(MAIN).pdf

# For testing without bibliography
quick:
	$(LATEX) $(MAIN)
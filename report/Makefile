# SPDX-License-Identifier: ISC

NAME       = A_Quantitative_Cache_Evaluation_of_Select_PolyBench_Kernels_-_Andrade_Canizela_Dimas_Jordão_Pellegrini
MAIN       = main
TEXT       = *.tex
ETC        = Makefile
FIG        = ../figures/*.pdf
VER        = $(shell git describe --long --tags \
              | sed 's/^v//;s/\([^-]*-g\)/r\1/;s/-/./g')
LATEX      = latexrun --latex-cmd lualatex
LATEXFLAGS = -O build
CLEANFLAGS = --clean-all -O build
RELEASE    = $(NAME)_$(VER)

.PHONY: all clean release

all: $(MAIN).pdf

release: $(RELEASE).zip $(RELEASE).tar.gz $(RELEASE).tar.xz $(RELEASE).tar.zst

clean:
	$(LATEX) $(CLEANFLAGS)
	$(RM) -r *.pdf *.ps *.idx *.bbl *.brf *.glo *.dvi *.toc *.lof *.aux \
		*.log *.blg *.ilg *.ind *.out *.wsp *.fls *.synctex* *.zip  \
		*.tar.gz *.tar.xz *.tar.zst *.*~ build/ latex.out/

$(MAIN).pdf: $(ETC) $(FIG)
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@touch $(MAIN).pdf

$(RELEASE).pdf: $(MAIN).pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
		-dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(RELEASE).pdf         \
		$(MAIN).pdf

$(RELEASE).tar: $(RELEASE).pdf
	tar -vcf               \
		$(RELEASE).tar \
		$(RELEASE).pdf

$(RELEASE).tar.gz: $(RELEASE).tar
	gzip -kf $(RELEASE).tar

$(RELEASE).tar.xz: $(RELEASE).tar
	xz -kf $(RELEASE).tar

$(RELEASE).tar.zst: $(RELEASE).tar
	zstd -qf $(RELEASE).tar

$(RELEASE).zip: $(MAIN).pdf $(RELEASE).pdf
	zip -r                 \
		$(RELEASE).zip \
		$(RELEASE).pdf

master := python-pniio-ug.tex
document := $(patsubst %.tex,%.pdf,$(master))
sources := $(wildcard tex/*.tex)

all: $(document)

$(document): $(master) $(sources)
	pdflatex --shell-escape $(master)
	pdflatex --shell-escape $(master)

.PHONY: clean

clean:
	rm -rf $(document)
	rm -rf $(patsubst %.tex,%.aux,$(master))
	rm -rf $(patsubst %.tex,%.toc,$(master))
	rm -rf $(patsubst %.tex,%.pyg,$(master))
	rm -rf $(patsubst %.tex,%.log,$(master))

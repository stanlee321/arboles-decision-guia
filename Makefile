PYTHON ?= python3
JUPYTER ?= jupyter
TECTONIC ?= tectonic

.PHONY: all data results notebook pdf slides test clean-generated

all: data results notebook pdf slides test

data:
	$(PYTHON) scripts/generate_data.py

results: data
	$(PYTHON) scripts/run_analysis.py

notebook: results
	PYTHONPATH=src $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=180 notebooks/01_arboles_decision.ipynb

pdf: results
	$(TECTONIC) docs/guide/arboles-decision-guia.tex --outdir output/pdf
	cp output/pdf/arboles-decision-guia.pdf docs/guide/arboles-decision-guia.pdf

slides: results
	$(TECTONIC) docs/slides/arboles-decision-presentacion.tex --outdir output/pdf
	cp output/pdf/arboles-decision-presentacion.pdf docs/slides/arboles-decision-presentacion.pdf

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

clean-generated:
	$(PYTHON) scripts/clean_generated.py

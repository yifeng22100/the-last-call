# Single-threaded on purpose: identical results on any machine, and avoids
# OpenMP/BLAS thread oversubscription that made multi-threaded runs crawl.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

PY ?= .venv/bin/python

.PHONY: all build verify test serve

all: build verify test

build:            ## run every experiment; writes site/data/*.json, outputs/*.csv, docs/RESULTS.md
	$(PY) -m lastcall.build
	$(PY) -m lastcall.pilot
	$(PY) -m lastcall.report

verify:           ## recompute every figure in plain Python and in R, then compare
	$(PY) -I verify/verify_stdlib.py
	Rscript verify/verify.R
	$(PY) -I verify/external_euribor.py
	$(PY) -I verify/compare.py

test:             ## invariant tests (no leakage, no peeking at the future, ...)
	$(PY) -m pytest -q tests

serve:            ## preview the website at http://localhost:8000
	cd site && $(PY) -m http.server 8000

# Prefer a Python the ML ecosystem is actually tested against. torch publishes
# 3.14 wheels now, but timm/diffusers on 3.14 is unexplored, and SETUP.md
# targets 3.11. Override with: make venv PYTHON=python3.14
PYTHON ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3)
VENV   ?= $(HOME)/.venvs/photo3d
BLENDER ?= /Applications/Blender.app/Contents/MacOS/Blender

.PHONY: help venv test smoke serve addon clean

help:
	@echo "make venv     create the solver venv and install everything"
	@echo "make weights  download the Depth Pro checkpoint (~1.9 GB)"
	@echo "make serve    run the solve daemon on :8765 (leave it running)"
	@echo "make health   ask a running daemon what it has loaded"
	@echo "make addon    build photo3d.zip for Blender's Install from Disk"
	@echo "make test     run the test suite (no Blender, no torch, ~9s)"
	@echo "make smoke    register the add-on in a real Blender and exercise it"
	@echo "make clean    remove build artefacts and __pycache__"
	@echo
	@echo "First run:  make venv && make weights && make addon && make serve"

venv:
	@echo "building $(VENV) with $(PYTHON) ($$($(PYTHON) --version))"
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e '.[depth,refine,raw,dev]'
	$(VENV)/bin/pip install git+https://github.com/apple/ml-depth-pro.git
	@echo
	@echo "Now fetch the Depth Pro weights (~1.9 GB):"
	@echo "  make weights"
	@echo
	@echo "Optional, for the M7 sun gobo:"
	@echo "  $(VENV)/bin/pip install git+https://github.com/compphoto/Intrinsic.git"

# The daemon finds the checkpoint by searching; ~/src/ml-depth-pro is the first
# place it looks, so put it there and nothing needs configuring.
weights:
	@test -d $(HOME)/src/ml-depth-pro || \
	  git clone https://github.com/apple/ml-depth-pro.git $(HOME)/src/ml-depth-pro
	cd $(HOME)/src/ml-depth-pro && bash get_pretrained_models.sh
	@ls -lh $(HOME)/src/ml-depth-pro/checkpoints/depth_pro.pt

test:
	@$(VENV)/bin/python -m pytest server/tests -q 2>/dev/null || \
	  $(PYTHON) -m pytest server/tests -q

smoke:
	$(BLENDER) --background --factory-startup --python tools/blender_smoke_test.py

# Runs in the foreground on purpose: the models stay resident in this process,
# and a cold reload per photo is the thing the whole daemon exists to avoid.
serve:
	$(VENV)/bin/python -m uvicorn server.solver_server:app --host 127.0.0.1 --port 8765

health:
	@curl -s http://127.0.0.1:8765/health | $(PYTHON) -m json.tool || \
	  echo "no daemon on :8765 — run 'make serve' in another terminal"

addon:
	$(PYTHON) tools/build_addon_zip.py

clean:
	rm -rf dist build *.egg-info photo3d.zip
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete

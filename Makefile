PYTHON ?= python3
VENV   ?= $(HOME)/.venvs/photo3d
BLENDER ?= /Applications/Blender.app/Contents/MacOS/Blender

.PHONY: help venv test smoke serve addon clean

help:
	@echo "make venv    create the solver venv and install everything"
	@echo "make test    run the test suite (no Blender, no torch, ~1s)"
	@echo "make smoke   register the add-on in a real Blender and exercise it"
	@echo "make serve   run the solve daemon on :8765"
	@echo "make addon   build photo3d.zip for Blender's Install from Disk"
	@echo "make clean   remove build artefacts and __pycache__"

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e '.[depth,refine,raw,dev]'
	@echo
	@echo "Then, separately (they are git installs, not on PyPI):"
	@echo "  $(VENV)/bin/pip install git+https://github.com/apple/ml-depth-pro.git"
	@echo "  $(VENV)/bin/pip install git+https://github.com/compphoto/Intrinsic.git"
	@echo "and fetch the Depth Pro weights with its get_pretrained_models.sh"

test:
	$(PYTHON) -m pytest server/tests -q

smoke:
	$(BLENDER) --background --factory-startup --python tools/blender_smoke_test.py

serve:
	$(PYTHON) -m uvicorn server.solver_server:app --host 127.0.0.1 --port 8765

addon:
	$(PYTHON) tools/build_addon_zip.py

clean:
	rm -rf dist build *.egg-info photo3d.zip
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete

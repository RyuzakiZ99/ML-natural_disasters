VENV_DIR = venv
PLOTS1_DIR = plots_eda
PLOTS2_DIR = plots_pre
RESUL_DIR = resultados

PIP = $(VENV_DIR)/bin/pip

.PHONY: all setup run clean

all: setup

setup: $(VENV_DIR) install_deps

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

install_deps: $(VENV_DIR) requirements.txt
	$(PIP) install -r requirements.txt

run: setup
	$(VENV_DIR)/bin/python3 src/eda.py
	$(VENV_DIR)/bin/python3 src/spot_checking.py

clean:
	rm -rf $(PLOTS1_DIR)
	rm -rf $(PLOTS2_DIR)
	rm -rf $(RESUL_DIR)
	rm -rf $(VENV_DIR)
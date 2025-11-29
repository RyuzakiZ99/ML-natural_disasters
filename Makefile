VENV_DIR = venv
ETAPA1_DIR = etapa_1

PIP = $(VENV_DIR)/bin/pip

.PHONY: all setup run clean

all: setup

setup: $(VENV_DIR) install_deps

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

install_deps: $(VENV_DIR) requirements.txt
	$(PIP) install -r requirements.txt

run: setup
	$(VENV_DIR)/bin/python3 src/etapa1.py

clean:
	rm -rf $(ETAPA1_DIR)
	rm -rf $(VENV_DIR)
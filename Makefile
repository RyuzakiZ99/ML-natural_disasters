VENV_DIR = venv
ETAPA1_DIR = etapa_1
ETAPA2_DIR = etapa_2

PIP = $(VENV_DIR)/bin/pip

.PHONY: all setup run clean

all: setup

setup: $(VENV_DIR) install_deps

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

install_deps: $(VENV_DIR) requirements.txt
	$(PIP) install -r requirements.txt --no-warn-conflicts

run: setup
	$(VENV_DIR)/bin/python3 src/etapa1.py
	$(VENV_DIR)/bin/python3 src/etapa2.py

clean:
	rm -rf $(ETAPA1_DIR)
	rm -rf $(ETAPA2_DIR)
	rm -rf $(VENV_DIR)
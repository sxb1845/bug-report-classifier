#!/bin/bash
mkdir work
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

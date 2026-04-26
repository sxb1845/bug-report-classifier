#!/bin/bash
source .venv/bin/activate || exit 1
python3 evaluator.py "$@"


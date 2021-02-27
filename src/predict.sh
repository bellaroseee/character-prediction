#!/usr/bin/env bash
set -e
set -v
python src/main.py --test_data $1 --test_output $2

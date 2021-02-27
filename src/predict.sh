#!/usr/bin/env bash
set -e
set -v
python src/languagePrediction.py > language.txt
python src/myprogram.py test --work_dir $(cat language.txt) --test_data $1 --test_output $2

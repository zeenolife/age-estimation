#!/usr/bin/env bash

cd /age-estimation
python data_preparation.py
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

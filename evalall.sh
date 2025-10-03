#!/bin/sh

source .venv/bin/activate
for f in environment/scenarios/testing/*; do
	echo "Training on $(basename "$f")"
	python evaluate.py --exp=$(basename "$f")
done




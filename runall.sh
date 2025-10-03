#!/bin/sh

source .venv/bin/activate
for f in environment/scenarios/training/*; do
	echo "Training on $(basename "$f")"
	python ppo.py --exp=$(basename "$f")
	echo "Testing on $(basename "$f")"
	python evaluate.py --exp=$(basename "$f")
done




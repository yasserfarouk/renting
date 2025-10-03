#!/bin/sh

source .venv/bin/activate
for f in environment/scenarios/training/*; do
	echo "[DEBUG] Training on $(basename "$f")"
	python ppo.py --exp=$(basename "$f") --debug
	echo "[DEBUG] Testing on $(basename "$f")"
	python evaluate.py --exp=$(basename "$f") --debug
done




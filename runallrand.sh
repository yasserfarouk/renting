#!/bin/sh

source .venv/bin/activate
for f in $(ls environment/scenarios/training | shuf); do
	if [[ $(basename "$f") == anac* ]]; then
      continue
	fi
	echo "Training on $(basename "$f")"
	python ppo.py --exp=$(basename "$f") $@
	echo "Testing on $(basename "$f")"
	python evaluate.py --exp=$(basename "$f") $@
done




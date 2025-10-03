#!/bin/sh

source .venv/bin/activate
for b in training testing; do
	for f in environment/scenarios/${b}_src/*; do
			python convert_scenarios.py "${f}" --extend
    # if [ "$(basename "$f")" = "scml_dynamic" ] || [ "$(basename "$f")" = "anac" ] || [ "$(basename "$f")" = "anac2024" ]; then
    # 				echo "Converting $f with --extend"
    #     python convert_scenarios.py "${f}" --extend
    # else
    # 				echo "Converting $f without extension"
    #     python convert_scenarios.py "${f}"
    # fi
	done
done




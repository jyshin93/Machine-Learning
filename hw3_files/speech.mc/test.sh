#!/bin/bash
#
# a shell script to automate testing. Makes batch testing simpler.
#
# written by Eric Bridgeford
# #ShellFTW

algorithm=(knn distance_knn)
options=(finance easy vision speech hard)

for opt in "${options[@]}"; do
    for algo in "${algorithm[@]}"; do
        python classify.py --mode train --algorithm $algo --model-file ${opt}.knn.model --data ${opt}.train

        python classify.py --mode test --model-file ${opt}.knn.model --data ${opt}.dev --predictions-file ${opt}.dev.predictions

        acc="$(python compute_accuracy.py ${opt}.dev ${opt}.dev.predictions)"

        echo "${opt} | $algo | $acc"

    done
done
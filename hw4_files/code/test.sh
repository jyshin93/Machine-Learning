#!/bin/bash
#
# a shell script to automate testing. Makes batch testing simpler.
#
# written by Eric Bridgeford
# #ShellFTW

algorithm=(lambda_means nb_clustering)
options=(easy finance speech vision hard)

for opt in "${options[@]}"; do
    for algo in "${algorithm[@]}"; do
        python classify.py --mode train --algorithm $algo --model-file ${opt}.perceptron.model --data ${opt}.train

        python classify.py --mode test --model-file ${opt}.perceptron.model --data ${opt}.dev --predictions-file ${opt}.dev.predictions

        acc="$(python cluster_accuracy.py ${opt}.dev ${opt}.dev.predictions)"
        ann="$(python number_clusters.py ${opt}.dev.predictions)"
        echo "${opt} | $algo | $acc | $ann"

    done
done
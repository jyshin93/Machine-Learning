#!/bin/bash
#
# a shell script to automate testing. Makes batch testing simpler.
#
# written by Eric Bridgeford
# #ShellFTW

algorithm=(lambda_means)
options=(date)

for opt in "${options[@]}"; do
    for algo in "${algorithm[@]}"; do
        python classify.py --mode train --algorithm $algo --model-file ${opt}.perceptron.model --data ${opt}.train

        python classify.py --mode test --model-file ${opt}.perceptron.model --data ${opt}.train --predictions-file ${opt}.test.predictions

        acc="$(python cluster_accuracy.py ${opt}.train ${opt}.test.predictions)"
        ann="$(python number_clusters.py ${opt}.test.predictions)"
        echo "${opt} | $algo | $acc | $ann"

    done
done
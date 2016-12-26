#!/bin/bash
#
# a shell script to automate testing. Makes batch testing simpler.
#
# written by Eric Bridgeford
# #ShellFTW

algorithm=(margin_perceptron)
options=(data1 data2 data3 data4 data5)

for opt in "${options[@]}"; do
    for algo in "${algorithm[@]}"; do
        python classify.py --mode train --algorithm $algo --model-file ${opt}.cluster.model --data ${opt}.train

        python classify.py --mode test --model-file ${opt}.cluster.model --data ${opt}.test --predictions-file ${opt}.test.predictions

        acc="$(python compute_accuracy.py ${opt}.test ${opt}.test.predictions)"

        echo "${opt} | $algo | $acc"

    done
done
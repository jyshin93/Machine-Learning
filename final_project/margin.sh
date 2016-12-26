#!/bin/bash
#
# a shell script to automate testing. Makes batch testing simpler.
#
# written by Eric Bridgeford
# #ShellFTW


# #full_clust_lambda_11001 full_clust_lambda_11002 full_clust_lambda_11003 full_clust_lambda_11004 full_clust_lambda_11005 
# 	full_clust_lambda_5001 full_clust_lambda_5002 full_clust_lambda_5003 full_clust_lambda_5004 full_clust_lambda_5005
# 	full_clust_lambda_7001 full_clust_lambda_7002 full_clust_lambda_7003 full_clust_lambda_7004 full_clust_lambda_7005
# 	full_clust_lambda_9001 full_clust_lambda_9002 full_clust_lambda_9003 full_clust_lambda_9004 full_clust_lambda_9005
algorithm=(pegasos)
options=(data1 data2 data3 data4 data5)

for opt in "${options[@]}"; do
    for algo in "${algorithm[@]}"; do
        python classify.py --mode train --algorithm $algo --model-file ${opt}.margin_perceptron.model --data ${opt}.train

        python classify.py --mode test --model-file ${opt}.margin_perceptron.model --data ${opt}.test --predictions-file ${opt}.test.predictions

        acc="$(python compute_accuracy.py ${opt}.test ${opt}.test.predictions)"

        echo "${opt} | $algo | $acc"

    done
done
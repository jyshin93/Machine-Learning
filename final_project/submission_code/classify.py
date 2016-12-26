import os
import argparse
import sys
import pickle
import numpy as np

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
from perceptron import PredictorSubclass, Averaged, Margin_Perceptron
from Pegasos import Pegasos
from knn import KNN, DISTANCE_KNN
from Lambda import LambdaMean
from NaiveCluster import NaiveClustering

def load_data(filename):
    instances = []
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue

            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            label = ClassificationLabel(int_label)
            feature_vector = FeatureVector()
            for item in split_line[1:]:
                try:
                    index = int(item.split(":")[0])
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                lastIndex = index
                if value != 0.0:
                    feature_vector.add(index, value)

            instance = Instance(feature_vector, label)
            instances.append(instance)

    return instances


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")
    parser.add_argument("--online-learning-rate", type=float, help="The learning rate for perceptron", default=1.0)
    parser.add_argument("--online-training-iterations", type=int,
                        help="The number of training iterations for online methods.", default=25)
    parser.add_argument("--pegasos-lambda", type=float, help="The regularization parameter for Pegasos.", default = 1e-4)
    parser.add_argument("--knn", type = int, help="The value of K for KNN classification.", default = 5)
    parser.add_argument("--clustering-training-iterations", type=int, help="The number of clustering iterations.", default=10)
    parser.add_argument("--num-clusters", type=int, help="The number of clusters in Naive Bayes clustering.", default=6)
    parser.add_argument("--cluster-lambda", type=float, help="The value of lambda in lambda-means.", default=1100.0)

    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--algorithm should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def train(instances, algorithm):
    args = get_args()
    rate = args.online_learning_rate
    I = args.online_training_iterations
    k = args.knn
    lamb = args.pegasos_lambda
    cluster_lambda = args.cluster_lambda
    num_clusters = args.num_clusters
    II = args.clustering_training_iterations
    if algorithm == "margin_perceptron":
        margin_percep = Margin_Perceptron(rate, I)
        margin_percep.train(instances)
        return margin_percep
    elif algorithm == "pegasos":
        pegasos = Pegasos(lamb, I)
        pegasos.train(instances)
        return pegasos
    elif algorithm == "knn":
        knn_train = KNN(k)
        knn_train.train(instances)
        return knn_train
    elif algorithm == "distance_knn":
        dist_knn = DISTANCE_KNN(k)
        dist_knn.train(instances)
        return dist_knn
    elif algorithm == "lambda_means":
        LambdaMean_train = LambdaMean(II, cluster_lambda)
        LambdaMean_train.train(instances)
        return LambdaMean_train
    elif algorithm == "nb_clustering":
        nb_train = NaiveClustering(II, num_clusters)
        nb_train.train(instances)
        return nb_train
    else:
        raise Exception("No algorithm available")


def write_predictions(predictor, instances, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for instance in instances:
                label = predictor.predict(instance)

                writer.write(str(label))
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()

    if args.mode.lower() == "train":
        # Load the training data.
        
        instances = load_data(args.data)
        # Train the model.
        predictor = train(instances, args.algorithm)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")

    elif args.mode.lower() == "test":
        # Load the test data.
        instances = load_data(args.data)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")

        write_predictions(predictor, instances, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")


if __name__ == "__main__":
    main()


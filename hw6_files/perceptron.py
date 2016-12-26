from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

import os
import numpy as np

class MultiPerceptron(Predictor):
    
    def __init__(self, I):
        self.weights = {}
        self.I = I
        self.K = {}
        self.M = 0
    def train(self, instances):
        #get maximum key --> which is M
        for instance in instances:
            temp_M = max(instance._feature_vector.sparseVec.keys())
            temp_label = instance._label.int_label()
            if self.M < temp_M:
                self.M = temp_M
            self.K.update({temp_label: 1})
        #initialize weights
        for k in self.K:
            self.weights.update({k : [0] * (self.M + 1)})
        for i in range(self.I):
            for instance in instances:
                argmax = 0 
                predicted_label = self.K.keys()[0]
                y_label = instance._label.int_label()
                for k in self.K:
                    temp_value = 0
                    weight_k = self.weights.get(k)
                    for j in range(1, self.M+1):
                        temp_value += instance._feature_vector.get(j) * weight_k[j]
                    if temp_value > argmax:
                        argmax = temp_value
                        predicted_label = k
                if not (y_label == predicted_label):
                    self.weights_update(y_label, predicted_label, instance)

    def predict(self, instance):
        argmax = 0
        predicted_label = self.K.keys()[0]
        for k in self.K:
            temp_value = 0
            weight_k = self.weights.get(k)
            for j in range(1, self.M+1):
                temp_value += instance._feature_vector.get(j) * weight_k[j]
            if temp_value > argmax:
                argmax = temp_value
                predicted_label = k
        return predicted_label
    
    def weights_update(self, y_label, predicted_label, instance):
        temp_y = self.weights.get(y_label)
        temp_predict = self.weights.get(predicted_label)
        #update rule
        for j in range(1, self.M+1):
            val = instance._feature_vector.get(j)
            temp_y[j] += val
            temp_predict[j] -= val
        self.weights.update({y_label : temp_y})
        self.weights.update({predicted_label : temp_predict})


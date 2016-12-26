from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

import os
import numpy as np

class PredictorSubclass(Predictor):
    
    def __init__(self, rate, I):
        self.weights = {}
        self.rate = rate
        self.I = I
        
    def train(self, instances):
        for i in range(self.I):
            for instance in instances:
                y = instance._label.__str__()
                y_hat = self.predict(instance)
                y_label = int(y)
                if y_hat != y_label:
                    self.weights_Addition(y_label, instance)
    
    def predict(self, instance):
        result_Value = 0
        for i in instance._feature_vector.sparseVec:
            result = self.weights.get(i,0) * instance._feature_vector.get(i)
            result_Value += result
        if result_Value >= 0:
            return 1
        else:
            return 0
    
    def weights_Addition(self, y, instance):
        if y == 1:
            y = 1
        else:
            y = -1
        for i in instance._feature_vector.sparseVec:
            add = {i : self.weights.get(i,0) + self.rate * (y) * instance._feature_vector.get(i)}
            self.weights.update(add)

class Averaged(Predictor):
    def __init__(self, rate, I):
        self.weights = {}
        self.rate = rate
        self.I = I
        self.average_weight = {}

    def train(self, instances):
        for i in range(self.I):
            for instance in instances:
                y = instance._label.__str__()
                y_hat = self.predict_inside(instance)
                y_label = int(y)
                if y_hat != y_label:
                    self.weights_Addition(y_label, y_hat, instance)
                for j in self.weights:
                    self.average_weight.update({j : self.weights.get(j,0) + self.average_weight.get(j,0)})
                
    def predict_inside(self, instance):
        result_Value = 0
        for i in instance._feature_vector.sparseVec:
            result = self.weights.get(i,0) * instance._feature_vector.get(i)
            result_Value += result
        if result_Value >= 0:
            return 1
        else:
            return 0

    def weights_Addition(self, y, y_hat, instance):
        for i in instance._feature_vector.sparseVec:
            self.weights.update({i : self.weights.get(i,0) + self.rate * (y - y_hat) * instance._feature_vector.get(i)})

    def predict(self, instance):
        result_Value = 0
        for i in instance._feature_vector.sparseVec:
            result = self.average_weight.get(i,0) * instance._feature_vector.get(i)
            result_Value += result
        if result_Value >= 0:
            return 1
        else:
            return 0

class Margin_Perceptron(Predictor):
    def __init__(self, rate, I):
        self.weights = {}
        self.rate = rate
        self.I = I

    def train(self, instances):
        for i in range(self.I):
            for instance in instances:
                y_label = instance._label.__str__()
                y = int(y_label)
                if y == 0:
                    y = -1
                if (self.cal(instance)*y) < 1:
                    self.weights_Addition(y, instance)
                
    def predict(self, instance):
        if self.cal(instance) >= 0:
            return 1
        else:
            return 0

    def cal(self, instance):
        result_Value = 0
        for i in instance._feature_vector.sparseVec:
            result = self.weights.get(i,0) * instance._feature_vector.get(i)
            result_Value += result
        return result_Value

    def weights_Addition(self, y_label, instance):
        if y_label == 1:
            y_label = 1
        else:
            y_label = -1
        for i in instance._feature_vector.sparseVec:
            add = {i : self.weights.get(i,0) + self.rate * (y_label) * instance._feature_vector.get(i)}
            self.weights.update(add)

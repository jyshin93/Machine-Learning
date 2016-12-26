from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

import os
import numpy as np

class Pegasos(Predictor):
    def __init__(self, lamb, I):
        self.weights = {}
        self.I = I
        self.lamb = lamb

    def train(self, instances):
        t = 1
        for i in range(self.I):
            for instance in instances:
                y = int(instance._label.__str__())
                if y == 0:
                    y = -1
                self.weights_Addition(instance, t, y)
                t +=1
    
    def predict(self, instance):
        result_Value = 0
        for i in instance._feature_vector.sparseVec:
            result = self.weights.get(i,0) * instance._feature_vector.get(i)
            result_Value += result
        if result_Value >= 0:
            return 1
        else:
            return 0    

    def weights_Addition(self, instance, t, y):
        indicator = self.indicator_func(instance, y)
        rate = 1.0 / (self.lamb * t)
        coeff = 1.0 - (1.0/t)
        for i in self.weights:
            update = {i : coeff * self.weights.get(i,0)}
            self.weights.update(update)                
        if indicator is not 0:
            for i in instance._feature_vector.sparseVec:
                update = {i : self.weights.get(i,0) + rate * indicator * instance._feature_vector.get(i) * y}
                self.weights.update(update)
            

    def indicator_func(self, instance, y):
        margin = 0
        for key in instance._feature_vector.sparseVec:
            margin += self.weights.get(key,0) * instance._feature_vector.get(key)
        if margin * y < 1:
            return 1
        else:
            return 0

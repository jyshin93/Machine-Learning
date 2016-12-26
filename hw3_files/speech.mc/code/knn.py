from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
import math
import os
import numpy as np

class KNN(Predictor):
    
    def __init__(self, k):
        self.k = k
        self.instances = []
        self.possibility = {}
        
    def train(self, instances):
        self.instances = instances
        for instance in self.instances:
            for key in instance._feature_vector.sparseVec:
                index = {key : 1}
                self.possibility.update(index)
                
    
    def predict(self, instance):
        temp = []
        for train_instance in self.instances:
            dist = self.Euclidean(instance, train_instance)
            y_label = int(train_instance._label.__str__())
            t = (dist, y_label)
            temp.append(t)
        temp.sort()
        result = {}
        for i in range(self.k):
            t = temp.pop(0)
            key_label = t[1]
            value = result.get(key_label, 0) + 1
            update = {key_label:value}
            result.update(update)
        maxKey = max(result, key = result.get)
        maxValue = result.get(maxKey)
        for key in result:
            if result.get(key) == maxValue and maxKey > key:
                maxKey = key
        return maxKey        
        
    def Euclidean(self, test_input, train_instance):
        dist = 0
        for i in self.possibility:
            dist += ((test_input._feature_vector.get(i) - train_instance._feature_vector.get(i))**2)
        return math.sqrt(dist)

class DISTANCE_KNN(Predictor):
    
    def __init__(self, k):
        self.k = k
        self.instances = []
        self.possibility = {}
        
    def train(self, instances):
        self.instances = instances
        for instance in self.instances:
            for key in instance._feature_vector.sparseVec:
                index = {key : 1}
                self.possibility.update(index)
    
    def predict(self, instance):
        y_hat= []
        for train_instance in self.instances:
            dist = self.Euclidean(instance, train_instance)
            y_label = int(train_instance._label.__str__())
            t = (dist, y_label)
            y_hat.append(t)
        y_hat_list = [0] * self.k
        dist_list = [0] * self.k
        y_hat.sort()
        for i in range(self.k):
            t = y_hat.pop(0)
            y_hat_list[i] = t[1]
            dist_list[i] = t[0]
        result = {}
        for i in range(len(y_hat_list)):
            value = result.get(y_hat_list[i], 0) + (1.0 / (1 + dist_list[i]**2))
            update = {y_hat_list[i] : value}
            result.update(update)
        maxKey = max(result, key = result.get)
        maxValue = result.get(maxKey)
        for key in result:
            if result.get(key) == maxValue and maxKey > key:
                maxKey = key
        return maxKey
        
    def Euclidean(self, test_input, train_instance):
        dist = 0
        for i in self.possibility:
            dist += (test_input._feature_vector.get(i) - train_instance._feature_vector.get(i))**2
        return math.sqrt(dist) 

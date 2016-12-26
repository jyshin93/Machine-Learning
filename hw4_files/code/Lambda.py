from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

import os
import numpy as np
import math

class LambdaMean(Predictor):
    
    def __init__(self, I, lambda_value):
        self.I = I
        self.K_num = 1
        self.K = list(range(self.K_num))
        self.lambda_value = lambda_value
        self.prototype = {}
        self.Rnk = []
        self.poss_index = {}
        
    def train(self, instances):
        #Initializing Cluster
        x_mean = self.Initializing_Cluster(instances)
        for k in self.K:
            update = {k : x_mean}
            self.prototype.update(update)
        if self.lambda_value == 0.0:
            self.lambda_value = self.set_Lambda(x_mean, instances)
        #Initialize Rnk
        self.Rnk = [-1] * len(instances)
        #Start Clustering Iteration
        for i in range(self.I):
            #E-Step
            n = 0
            for instance in instances:                
                Ecu_candidate = []
                for k in self.K:
                    k_proto = self.prototype.get(k)
                    Ecu_candidate.append(self.Euclidean(instance._feature_vector.sparseVec, k_proto)**2)
                arg_min_k = np.argmin(Ecu_candidate)
                if Ecu_candidate[arg_min_k] <= self.lambda_value:
                    self.Rnk[n] = arg_min_k;
                else:
                    self.K_num += 1
                    self.K.append(self.K_num - 1)
                    self.Rnk[n] = self.K_num - 1
                    self.prototype.update({self.K_num - 1: instance._feature_vector.sparseVec})
                n += 1
            #M - Step
            for key in self.prototype:
                sum_denom = 0
                k_proto = self.prototype.get(key)
                #find sum(Rnk)
                for rn in self.Rnk:
                    if key == rn:
                        sum_denom += 1
                if sum_denom == 0.0:
                    for key_in in k_proto:
                        temp = {key_in:0}
                        k_proto.update(temp)
                        self.prototype.update({key:k_proto})
                else:
                    n = 0
                    temp_mu = {}
                    for instance in instances:
                        if self.Rnk[n] == key:
                            for vecKey, vecValue in instance._feature_vector.sparseVec.iteritems():
                                temp_val = temp_mu.get(vecKey,0)
                                update_val = temp_val + ((vecValue * 1.0) / sum_denom)
                                temp_mu.update({vecKey : update_val})
                        n+=1
                    self.prototype.update({key : temp_mu})
    def predict(self, instance):
        Ecu_candidate = []
        for k in self.K:
            k_proto = self.prototype.get(k)
            Ecu_candidate.append(self.Euclidean(instance._feature_vector.sparseVec, k_proto))
        arg_min_k = np.argmin(Ecu_candidate)
        return arg_min_k
        
    def Initializing_Cluster(self, instances):
        for instance in instances:
            for key in instance._feature_vector.sparseVec:
                index = {key : 1}
                self.poss_index.update(index)
        poss_value = {}
        for instance in instances:
            for key in self.poss_index:
                if poss_value.get(key,0) == 0:
                    value = instance._feature_vector.sparseVec.get(key, 0)
                    update = {key: [value]}
                    poss_value.update(update)
                else:
                    value = instance._feature_vector.sparseVec.get(key, 0)
                    temp = poss_value.get(key)
                    temp.append(value)
                    update = {key: temp}
                    poss_value.update(update)
        x_mean = {}
        for key in poss_value:
            temp = poss_value.get(key, 0)
            avg = np.mean(temp)
            update = {key : avg}
            x_mean.update(update)
        return x_mean

    def set_Lambda(self, x_mean, instances):
        dist = []
        for instance in instances:
            dist.append((self.Euclidean(instance._feature_vector.sparseVec, x_mean))**2)
        return sum(dist) / len(instances)

    def Euclidean(self, instance, x_mean):
        dist = []
        for key in self.poss_index:
            dist.append(((instance.get(key,0) - x_mean.get(key, 0))**2))
        return math.sqrt(sum(dist))

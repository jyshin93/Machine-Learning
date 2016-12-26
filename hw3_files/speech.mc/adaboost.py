from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
import math
import os
import numpy as np

class ADABOOST(Predictor):
    
    def __init__(self, n, T):
        self.D = [1.0 / n] * n
        self.T = T
        self.alpha = [0] * T
        self.H = []
        
    def train(self, instances):
        poss_index = {}
        for instance in instances:
            for key in instance._feature_vector.sparseVec:
                index = {key : 1}
                poss_index.update(index)
        poss_value = {}
        for instance in instances:
            y_label = int(instance._label.__str__())
            for key in poss_index:
                if poss_value.get(key,0) == 0:
                    value = instance._feature_vector.sparseVec.get(key,0)
                    update = {key: [(value,y_label)]}
                    poss_value.update(update)
                else:
                    value = instance._feature_vector.sparseVec.get(key,0)
                    temp = poss_value.get(key,0)
                    tup = (value, y_label)
                    temp.append(tup)
                    update = {key: temp}
                    poss_value.update(update)
        for t in range(self.T):
            #poss_value = {j_index: [(Possible j index feature value, lable of instance)]}
            #poss_index = {1:1, 2:1, ... ALL POSSIBLE INDEX OF FEATURE VALUES]
            h_t = self.hypothesis(poss_value, poss_index)
            self.H.append(h_t)
            #add h_t
            #Terminate if 0
            if h_t[4] < 0.000001:
                self.alpha[t] = 1
                return
            else:
                self.alpha[t] = (1.0/2.0) * math.log((1-h_t[4]) / h_t[4])
            
            Z_t = 0.0
            X_value = poss_value.get(h_t[0])
            for i in range(len(self.D)):
                y_i = X_value[i][1]
                if y_i is not 1:
                    y_i = -1
                if X_value[i][0] > h_t[3]:
                    Z_t += self.D[i] * math.exp(-1.0 *self.alpha[t] * y_i * h_t[1])
                else:
                    Z_t += self.D[i] * math.exp(-1.0 *self.alpha[t] * y_i * h_t[2])                    
            for i in range(len(self.D)):
                y_i = X_value[i][1]
                if y_i is not 1:
                    y_i = -1
                if X_value[i][0] > h_t[3]:
                    self.D[i] = (1.0/Z_t) * self.D[i] * math.exp(-1.0 * self.alpha[t] * y_i * h_t[1])
                else:
                    self.D[i] = (1.0/Z_t) * self.D[i] * math.exp(-1.0 * self.alpha[t] * y_i * h_t[2])
    
    def predict(self, instance):
        y_0 = 0
        y_1 = 0
        for t in range(len(self.H)):
            h_t = self.H[t]
            x_t = instance._feature_vector.get(h_t[0])
            h_value = 0
            if x_t > h_t[3]:
                h_value = h_t[1]
            else:
                h_value = h_t[2]
            if h_value == 1:
                y_1 += self.alpha[t]
            else:
                y_0 += self.alpha[t]
        if y_1 > y_0:
            return 1
        else:
            return 0

    def hypothesis(self, poss_value, poss_index):
        h_candidate = []
        for j in poss_index:
            #Get all feature values in j index features
            h_j = []
            value_j = poss_value.get(j)
            temp_unique = sorted((value_j))
            poss_c = []
            if len(temp_unique) == 1:
                poss_c.append((temp_unique[0][0] *3) / 2.0)
            else:
                for z in range(len(temp_unique)-1):
                    poss_c.append((temp_unique[z][0] + temp_unique[z+1][0]) / 2.0)
            #Poss_c = [all possible c values in j index in all instances]
            #value_j = [(values, y_label)........]
            #h_j_c will contain all possible h_j_c in j index with (j, y_1, y_2, c, error) tuples
            for C in poss_c:
                error_1 = 0
                error_2 = 0
                m1 = 0
                m2 = 0
                for i in range(len(value_j)):
                    feature_value = value_j[i][0]
                    feature_label = value_j[i][1]
                    D_value = self.D[i]
                    #hypothesis saying that I need x>1 will have positive
                    if feature_value > C and feature_label == 0:
                        error_1 += D_value
                        m2 +=1
                    elif feature_value <= C and feature_label == 1:
                        error_1 += D_value
                        m2 +=1
                    elif feature_value > C and feature_label == 1:
                        error_2 += D_value
                        m1 +=1
                    elif feature_value <= C and feature_label == 0:
                        error_2 += D_value
                        m1 +=1
                if m1 > m2:
                    tup = (j, 1, -1, C, error_1, m1)
                else:
                    tup = (j, -1, 1, C, error_2, m2)
                h_j.append(tup)
            h_j.sort(key = lambda tup:tup[5], reverse = True)
            h_candidate.append(h_j.pop(0))
        h_candidate.sort(key = lambda tup:tup[4])
        return h_candidate.pop(0)

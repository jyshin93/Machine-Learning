from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

import os
import numpy as np
import math

class NaiveClustering(Predictor):

    def __init__(self, I, num_cluster):
        self.I = I
        self.K = num_cluster
        self.Clusters = [] #[k, mu, var, prop, size]
        self.S_j = {}
        self.poss_index = {}

    def train(self, instances):
        #Get all the possible indexes
        for instance in instances:
            for key in instance._feature_vector.sparseVec:
                index = {key : 1}
                self.poss_index.update(index)
        #initialize mu_k
        self.S_j = self.calculate_total_var(instances)
        self.initialUpdate(instances)
        #get total variane of all instances aka S_j
        for it in range(self.I):
            #E-Step
            hard_assignment = self.E_Step(instances, self.Clusters)
            #M-Step
            self.M_Step(hard_assignment, instances)

    def predict(self, instance):
        return self.maxProb(instance, self.Clusters)

    def E_Step(self, instances, Clusters):
        returnVal = []
        for instance in instances:
            prob = self.maxProb(instance, self.Clusters)
            returnVal.append([prob, instance])
        return returnVal

    def M_Step(self, hard_assignment, instances):
        temp_cluster = []
        for k in range(self.K):
            temp_cluster.append([0, []])
        for assign in hard_assignment:
            temp_cluster[assign[0]][0] += 1
            temp_cluster[assign[0]][1].append(assign[1])
        self.update(temp_cluster, instances)        

    def initialUpdate(self, instances):
        temp_cluster = []
        for k in range(self.K):
            temp_cluster.append([0, []])
        for i in range(len(instances)):
            index = i % self.K
            temp_cluster[index][0] += 1
            temp_cluster[index][1].append(instances[i])
        for k in range(self.K):
            k_instance = temp_cluster[k]
            mu = self.calculate_mu(k, k_instance[1], k_instance[0])
            var = self.calculate_var(k, k_instance[1], mu, k_instance[0])
            self.Clusters.append([k, mu, var, 0, k_instance[0]])
            self.Clusters[k][3] = k_instance[0] / (len(instances) * 1.0)
    

    def update(self, temp_cluster, instances):
        cluster_to_update = self.Clusters
        for k in range(self.K):
            k_cluster = temp_cluster[k]
            if temp_cluster[k][0] == 0:
                self.Clusters[k][4] = k_cluster[0]
                self.Clusters[k][1] = {}
                self.Clusters[k][2] = {}
            else:
                self.Clusters[k][4] = k_cluster[0]
                self.Clusters[k][1] = self.calculate_mu(k, k_cluster[1], k_cluster[0])
                self.Clusters[k][2] = self.calculate_var(k, k_cluster[1], self.Clusters[k][1], k_cluster[0])
            self.Clusters[k][3] = (k_cluster[0] + 1) / (len(instances) * 1.0 + self.K)

    def maxProb(self, instance, clusters):
        prob_list = []
        for cluster in clusters:
            sum_prob = 0
            for j in self.poss_index:
                prob = 0
                x_val = instance._feature_vector.get(j)
                mean_j = cluster[1].get(j,0)
                var_j = cluster[2].get(j,0)
                if var_j == 0.0:
                    prob = 0.0
                else:
                    f = 1 / (math.sqrt(2.0*var_j*math.pi))
                    e = -((x_val - mean_j) ** 2) / (2.0 * var_j)
                    prob = math.log(f) + e
                if prob != 0.0:
                    sum_prob += prob
                else:
                    sum_prob += float('-inf')
            sum_prob += math.log(cluster[3])    
            prob_list.append(sum_prob)
        index = np.argmax(prob_list)
        return index

    def calculate_mu(self, k, instances, size):
        denom = size
        temp = {}
        for instance in instances:
            for key, value in instance._feature_vector.sparseVec.iteritems():
                temp_value = temp.get(key, 0)
                result_value = temp_value + ((1.0*value) / denom)
                temp.update({key : result_value})
        return temp

    def calculate_var(self, k, instances, mu_k, size):
        if size < 2:
            return self.S_j
        else:
            temp = {}
            denom = size - 1
            for instance in instances:
                for key in self.poss_index:
                    value = instance._feature_vector.get(key)
                    temp_value = temp.get(key, 0)
                    result_value = temp_value + (((1.0*value) - mu_k.get(key,0))**2) / denom
                    temp.update({key : result_value})
            for key in temp.keys():
                Sj = self.S_j.get(key)
                sigma_j = temp.get(key)
                if sigma_j <= Sj:
                    sigma_j = Sj
                    temp.update({key : sigma_j})
        return temp

    def calculate_total_var(self, instances):
        temp = {}
        mu = self.calculate_total_mu(instances)
        denom = len(instances) - 1
        for instance in instances:
            for key in self.poss_index:
                value = instance._feature_vector.get(key)
                temp_value = temp.get(key, 0)
                result_value = temp_value + (((1.0*value) - mu.get(key,0))**2) / denom
                temp.update({key : result_value})
        for key in temp.keys():
            temp_value = temp.get(key)
            temp_value = temp_value * 0.01
            temp.update({key : temp_value})
        return temp

    def calculate_total_mu(self, instances):
        denom = len(instances)
        temp = {}
        for instance in instances:
            for key, value in instance._feature_vector.sparseVec.iteritems():
                temp_value = temp.get(key, 0)
                result_value = temp_value + ((1.0*value) / denom)
                temp.update({key : result_value})
        return temp
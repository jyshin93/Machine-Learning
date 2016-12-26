import math
import numpy as np

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        self.n = self._potentials.chain_length()
        self.k = self._potentials.num_x_values()
        self.FM = {} #key is n values, values are [] list of k+1 values
        self.BM = {} #key is n values, values are [] list of k+1 values
        self.initialize()

    def marginal_probability(self, x_i):
        result = [0] * (self.k + 1)
        denom = 0
        # TODO: EDIT HERE
        # should return a python list of type float, with its length=k+1, and the first value 0
        for k in range(self.k):
            k_forward = self.FM.get(x_i)
            k_backward = self.BM.get(x_i)
            result[k+1] += k_forward[k+1] * k_backward[k+1] * self._potentials.potential(x_i, k+1)
            denom += result[k+1]
        for k in range(self.k):
            result[k+1] = result[k+1] / denom
        # This code is used for testing only and should be removed in your implementation.
        # It creates a uniform distribution, leaving the first position 0   
        return result

    def initialize(self):
        #find message at node 1 to send them
        for i in range(self.n):
            self.FM.update({i+1 : [0] * (self.k + 1)})
            self.BM.update({i+1 : [0] * (self.k + 1)})
        k_val_F = [0] * (self.k + 1)
        temp_k_1 = self.FM.get(1)
        for j in range(self.k):
            temp = self._potentials.potential(1, (j+1))
            temp_k_1[(j+1)] = 1
            k_val_F[j+1] = temp
        self.FM.update({1 : temp_k_1})
        self.FactorToNodeF(k_val_F, self.n + 1) #need to Implement
        #now get end node and update
        k_val_B = [0] * (self.k + 1)
        temp_k_n = self.BM.get(self.n)
        for j in range(self.k):
            temp = self._potentials.potential(self.n, j+1)
            temp_k_n[j+1] = 1
            k_val_B[j+1] = temp
        self.BM.update({self.n : temp_k_n})
        self.FactorToNodeB(k_val_B, 2*self.n-1)

    def FactorToNodeF(self, message, f_index):
        message_var = [0] * (self.k + 1)
        for k in range(self.k):
            for k2 in range(self.k):
                message_var[k+1] += self._potentials.potential(f_index, k2+1, k+1) * message[k2+1]
        f_index = f_index + 1 - self.n
        self.FM.update({f_index : message_var})
        #update message
        for k in range(self.k):
            message[k+1] = message_var[k+1] * self._potentials.potential(f_index, k+1)
        if (f_index != self.n):
            self.FactorToNodeF(message, f_index+self.n)
        
    def FactorToNodeB(self, message, f_index):
        message_var = [0] * (self.k + 1)
        for k in range(self.k):
            for k2 in range(self.k):
                message_var[k+1] += self._potentials.potential(f_index, k+1, k2+1) * message[k2+1]
        f_index = f_index - self.n
        self.BM.update({f_index : message_var})
        #update message
        for k in range(self.k):
            message[k+1] = message_var[k+1] * self._potentials.potential(f_index, k+1)
        if (f_index != 1):
            self.FactorToNodeB(message, f_index+ self.n - 1)

###########################################################################################################################

class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        self.n = self._potentials.chain_length()
        self.k = self._potentials.num_x_values()
        self.FM = {} #key is n values, values are [] list of k+1 values
        self.BM = {} #key is n values, values are [] list of k+1 values
        self.FM_Normal = {}
        self.BM_Normal = {}

    def initialize(self):
        for i in range(self.n):
            self.FM.update({i+1 : [0] * (self.k + 1)})
            self.BM.update({i+1 : [0] * (self.k + 1)})
            self.FM_Normal.update({i+1 : [0] * (self.k + 1)})
            self.BM_Normal.update({i+1 : [0] * (self.k + 1)})
        k_val_F = [0] * (self.k + 1)
        log_val_F = [0] * (self.k + 1)
        k_val_B = [0] * (self.k + 1)
        log_val_B = [0] * (self.k + 1)
        for k in range(self.k):
            k_val_F[k+1] = self._potentials.potential(1, k+1)
            log_val_F[k+1] = math.log(self._potentials.potential(1, k+1))
            k_val_B[k+1] = self._potentials.potential(self.n, k+1)
            log_val_B[k+1] = math.log(self._potentials.potential(self.n, k+1))
        start = [1.0] * (self.k + 1)
        self.FM.update({1: log_val_F})
        self.BM.update({self.n: log_val_B})
        self.FM_Normal.update({1 : start})
        self.BM_Normal.update({self.n : start})
        self.FactorToNodeF(log_val_F, self.n + 1)
        self.FactorToNodeB(log_val_B, 2*self.n - 1)
        self.NormalF(k_val_F, self.n + 1)
        self.NormalB(k_val_B, 2 * self.n - 1)
        
    def NormalF(self, message, f_index):
        message_var = [0] * (self.k + 1)
        for k in range(self.k):
            for k2 in range(self.k):
                message_var[k+1] += self._potentials.potential(f_index, k2+1, k+1) * message[k2+1]
        f_index = f_index + 1 - self.n
        self.FM_Normal.update({f_index : message_var})
        #update message
        for k in range(self.k):
            message[k+1] = message_var[k+1] * self._potentials.potential(f_index, k+1)
        if (f_index != self.n):
            self.NormalF(message, f_index+self.n)
        
    def NormalB(self, message, f_index):
        message_var = [0] * (self.k + 1)
        for k in range(self.k):
            for k2 in range(self.k):
                message_var[k+1] += self._potentials.potential(f_index, k+1, k2+1) * message[k2+1]
        f_index = f_index - self.n
        self.BM_Normal.update({f_index : message_var})
        #update message
        for k in range(self.k):
            message[k+1] = message_var[k+1] * self._potentials.potential(f_index, k+1)
        if (f_index != 1):
            self.NormalB(message, f_index+ self.n - 1)

    def FactorToNodeF(self, message, f_index):
        newMessage = [0] * (self.k + 1)
        max_index = [0] * (self.k + 1)
        values = {}
        for k in range(self.k):
            values.update({k+1: [0]*(self.k+1)})
        for k in range(self.k):
            for k2 in range(self.k):
                temp = values.get(k2+1)
                temp[k+1] = math.log(self._potentials.potential(f_index, k+1, k2+1)) + message[k+1]
                values.update({k2+1 : temp})
        for k2 in range(self.k):
            temp = values.get(k2+1)
            temp.pop(0)
            argmax = np.argmax(temp)
            max_val = temp[argmax]
            newMessage[k2+1] = max_val
            max_index[k2+1] = argmax+1
        self.FM.update({f_index + 1 - self.n: newMessage})
        self.VariableNodeF(f_index + 1 - self.n, newMessage)

    def FactorToNodeB(self,message, f_index):
        newMessage = [0] * (self.k + 1)
        max_index = [0] * (self.k + 1)
        values = {}
        for k in range(self.k):
            values.update({k+1: [0]*(self.k+1)})
        for k in range(self.k):
            for k2 in range(self.k):
                temp = values.get(k2 + 1)
                temp[k+1] = math.log(self._potentials.potential(f_index, k2+1, k+1)) + message[k+1]
                values.update({k2+1 : temp})
        for k2 in range(self.k):
            temp = values.get(k2+1)
            temp.pop(0)
            argmax = np.argmax(temp)
            max_val = temp[argmax]
            newMessage[k2+1] = max_val
            max_index[k2+1] = argmax+1
        self.BM.update({f_index - self.n: newMessage})
        self.VariableNodeB(f_index - self.n, newMessage)

    def VariableNodeF(self, n_index, newMessage):
        store_newMessage = [0] * (self.k + 1)
        store_new_normal_message = [0] * (self.k + 1)
        if n_index != self.n:
            for k in range(self.k):
                store_newMessage[k+1] = math.log(self._potentials.potential(n_index, k+1)) + newMessage[k+1]
            self.FactorToNodeF(store_newMessage, n_index + self.n)

    def VariableNodeB(self, n_index, newMessage):
        store_newMessage = [0] * (self.k + 1)
        store_new_normal_message = [0] * (self.k + 1)
        if n_index != 1:
            for k in range(self.k):
                store_newMessage[k+1] = math.log(self._potentials.potential(n_index, k+1)) + newMessage[k+1]
            self.FactorToNodeB(store_newMessage, n_index + self.n - 1)

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        self.initialize()
        max_prob = [0] * (self.n + 1)
        total_sum = 0
        for n in range(self.n):
            storage = [0] * (self.k+1)
            if (n+1) == 1:
                Backward = self.BM.get(n+1)
                for k in range(self.k):
                    storage[k+1] = Backward[k+1] + math.log(self._potentials.potential(n+1, k+1))
            elif (n+1) == self.n:
                Forward = self.FM.get(n+1)
                for k in range(self.k):
                    storage[k+1] = Forward[k+1] + math.log(self._potentials.potential(n+1, k+1))
            else:
                Backward = self.BM.get(n+1)
                Forward = self.FM.get(n+1)
                for k in range(self.k):
                    storage[k+1] = Forward[k+1] + math.log(self._potentials.potential(n+1, k+1)) + Backward[k+1]
            storage.pop(0)
            argmax = np.argmax(storage)
            max_val = storage[argmax]
            max_prob[n+1] = max_val
            self._assignments[n+1] = argmax + 1
        normal_value = 0
        Forward = self.FM_Normal.get(x_i)
        Backward = self.BM_Normal.get(x_i)
        #print(max_prob)
        for k in range(self.k):
            normal_value += Forward[k+1] * Backward[k+1] * self._potentials.potential(x_i, k+1)
        return max_prob[x_i] - math.log(normal_value)

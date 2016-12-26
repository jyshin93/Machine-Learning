from abc import ABCMeta, abstractmethod


# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self._label = label
        
    def __str__(self):
        return str(self._label)
    def set_label(self, nlabel):
        self.label = nlabel
    
class FeatureVector:
    def __init__(self):
        #implement with sparse vector
        self.sparseVec = {}
        
    def add(self, index, value):
        dic = {index : value}
        self.sparseVec.update(dic)
        
    def get(self, index):
       return self.sparseVec.get(index,0)


    
class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

    def get_feature_vector(self):
        return self._feature_vector.sparseVec

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass


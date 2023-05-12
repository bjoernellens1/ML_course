import abc

class ContinuousDistribution(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def import_data(self, file_path):
        pass
    
    @abc.abstractmethod
    def export_data(self, data, file_path):
        pass
    
    @abc.abstractmethod
    def compute_mean(self, data):
        pass
    
    @abc.abstractmethod
    def compute_standard_deviation(self, data):
        pass
    
    @abc.abstractmethod
    def visualize(self, data=None):
        pass
    
    @abc.abstractmethod
    def generate_samples(self, n_samples):
        pass
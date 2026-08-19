class Calibration:
    
    def __init__(self):
        self.data = []
        
    def add_sample(self, sample):
        self.data.append(sample)
    
    def clear(self):
        self.data.clear()
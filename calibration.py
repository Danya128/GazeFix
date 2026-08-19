class Calibration:
    
    def __init__(self):
        self.data = []
        
    def add_sample(self, sample):
        self.data.append(sample)
        
    def prepare_data(self):
        X = []
        y = []
        
        for sample in self.data:
            X.append([
                sample["rx"], sample["ry"], sample["rz"],
                sample["tx"], sample["ty"], sample["tz"]
            ])
            
            y.append([
                sample["left_x_ratio"], sample["left_y_ratio"],
                sample["right_x_ratio"], sample["right_y_ratio"]
            ])
        return X, y
    
    def clear(self):
        self.data.clear()
        
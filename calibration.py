import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class Calibration:
    
    def __init__(self):
        self.data = []
        self.model = None
        self.X_test = None
        self.y_text = None
        
        
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
    
    
    def train_model(self):
        X, y = self.prepare_data()
        
        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.float64)
        
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
    
    
    def evaluate_model(self):
        if self.model is None:
            print("The model has not been trained yet")
            return False
        
        predictions = self.model.predict(self.X_test)
        
        mae = mean_absolute_error(self.y_test, predictions)
        print(f"Mean Absolute Error: {mae:.4f}")
        
        if mae <= 0.33:
            print("Model performance is acceptable")
            return True
        else:
            print("Model performance is unacceptable")
            return False
        
        
    def clear(self):
        self.data.clear()
        
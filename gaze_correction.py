class GazeCorrector:
    
    def process(self, frame):
        
        plain_frame = frame.copy()
        
        # Future processes
        corrected_frame = plain_frame
        
        return corrected_frame
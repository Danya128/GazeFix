from gaze_correction import GazeCorrector
from calibration import Calibration
import cv2
import numpy as np

def main():
    
    cap = cv2.VideoCapture(0)
    correction = GazeCorrector()
    calibration = Calibration()
    
    calibration_mode = False
    gaze_correction_mode = False
    frame_counter = 0
    
    if not cap.isOpened():
        print("Error with opening a webcam")
        return 0
        
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error with reading a webcam")
                break
            
            original_frame = frame.copy()
            corrected_frame, tracked_data = correction.process(frame)
            
            frame_counter += 1

            if calibration_mode and tracked_data:
                if frame_counter % 5 == 0:
                    calibration.add_sample(tracked_data)
            
            cv2.imshow("Corrected Frame", corrected_frame)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                calibration_mode = True
                print("Calibration started")
    
            elif key == ord("s"):
                calibration_mode = False
                X, y = calibration.prepare_data()
                
                X = np.array(X, dtype=np.float64)
                y = np.array(y, dtype=np.float64)
                
                print("X: ", X.shape)
                print("y: ", y.shape)
                # Train the model
                print("Model trained")
    
            elif key == ord("g"):
                gaze_correction_mode = True
                print("Gaze correction turned on")

            elif key == ord("t"):
                gaze_correction_mode = False
                print("Gaze correction turned off")
    
            elif key == 27:
                break
                

            
    finally:
        cv2.destroyAllWindows()
        cap.release()
            
if __name__ == "__main__":
    main()
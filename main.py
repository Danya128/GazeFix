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
            corrected_frame, tracked_data = correction.process(frame, calibration, gaze_correction_mode)
            
            frame_counter += 1

            # Add samples for model training
            if calibration_mode and tracked_data:
                if frame_counter % 5 == 0:
                    calibration.add_sample(tracked_data)
            
            display_width = 750
            display_height = 480
            
            original_display = cv2.resize(original_frame, (display_width, display_height))
            corrected_display = cv2.resize(corrected_frame, (display_width, display_height))
            combined_frame = cv2.hconcat([original_display,corrected_display])
            cv2.imshow("GazeFix", combined_frame)
            
            
            key = cv2.waitKey(1) & 0xFF

            # Start calibration
            if key == ord("c"):
                calibration_mode = True
                print("Calibration started")

            # Stop calibration and train the model
            elif key == ord("s"):
                calibration_mode = False
                calibration.train_model()
                model_score = calibration.evaluate_model()
                
                if model_score:
                    print("Model has been trained successfully")
                else:
                    print("Calibration needs improvements")
                
    
            elif key == ord("g"):
                gaze_correction_mode = True
                print("Gaze correction turned on")

            elif key == ord("t"):
                gaze_correction_mode = False
                print("Gaze correction turned off")
    
            elif key == 27:
                break
            
            if gaze_correction_mode == True:
                pass
                

            
    finally:
        cv2.destroyAllWindows()
        cap.release()
            
if __name__ == "__main__":
    main()
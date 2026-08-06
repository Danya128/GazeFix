from gaze_correction import GazeCorrector
import cv2

def main():
    
    cap = cv2.VideoCapture(0)
    correction = GazeCorrector()
    
    
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
            corrected_frame = correction.process(frame)
            
            cv2.imshow("Original Frame", original_frame)
            cv2.imshow("Corrected Frame", corrected_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
    finally:
        cv2.destroyAllWindows()
        cap.release()
            
if __name__ == "__main__":
    main()
import cv2

key = cv2.waitKey(1) & 0xFF
calibration_mode = False
gaze_correction_mode = False

if key == ord("c"):
    calibration_mode = True
    print("Calibration started")
    
elif key == ord("s"):
    calibration_mode = False
    # Train the model
    print("Model trained")
    
elif key == ord("g"):
    gaze_correction_mode = True
    print("Gaze correction turned on")

elif key == ord("t"):
    gaze_correction_mode = False
    print("Gaze correction turned off")
    
elif key == 27:
    # Escape from the system
    pass
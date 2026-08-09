import mediapipe as mp
import cv2
import time

class GazeCorrector:
    
    def __init__(self):
        base_options = mp.tasks.BaseOptions(
            model_asset_path="assets/models/face_landmarker.task"
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )
        
        
        
    def detect_landmarks(self, frame):
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.monotonic() * 1000)
        
        result = self.detector.detect_for_video(mp_image, timestamp)
        return result
    
    
    
    def process(self, frame):
        
        plain_frame = frame.copy()
        
        result = self.detect_landmarks(frame)
        if result.face_landmarks:
            print("Face detected")
        
        # Future processes
        corrected_frame = plain_frame
        
        return corrected_frame
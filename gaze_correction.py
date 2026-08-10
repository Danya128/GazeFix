import mediapipe as mp
import numpy as np
import cv2
import time

# Indices(positions) of specific landmarks that MediaPipe returns
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

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
    
    
    
    def get_iris_center(self, frame, face_landmarks, iris_indices):
        
        height, width = frame.shape[:2]
        iris_points = []
        
        for index in iris_indices:
            
            landmark = face_landmarks[index]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            iris_points.append((x,y))
            
        iris_points = np.array(iris_points, dtype=np.float32)
        
        # Find the smallest possible circle that contains all the given points
        (center_x, center_y), radius = cv2.minEnclosingCircle(iris_points)
        
        center = (int(center_x), int(center_y))
        
        cv2.circle(frame, center, 4, (0,255,0), -1)
        
        return center
    
    
    
    def process(self, frame):
        
        plain_frame = frame.copy()
        
        result = self.detect_landmarks(frame)
        if not result.face_landmarks:
            return plain_frame
        
        face_landmarks = result.face_landmarks[0]
        
        left_iris_center = self.get_iris_center(plain_frame, face_landmarks, LEFT_IRIS_INDICES)
        right_iris_center = self.get_iris_center(plain_frame, face_landmarks, RIGHT_IRIS_INDICES)
        
        # Future processes
        corrected_frame = plain_frame
        
        return corrected_frame
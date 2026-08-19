import mediapipe as mp
import numpy as np
import cv2
import time

# Indices(positions) of the iris landmarks
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# Left eye
LEFT_OUTER_CORNER = 33
LEFT_INNER_CORNER = 133
LEFT_UPPER_EYELID = 159
LEFT_LOWER_EYELID = 145

# Right eye
RIGHT_OUTER_CORNER = 263
RIGHT_INNER_CORNER = 362
RIGHT_UPPER_EYELID = 386
RIGHT_LOWER_EYELID = 374



class GazeCorrector:
    
    # Define FaceLandmark detector
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
        
        
        
    # Get Landmarks
    def detect_landmarks(self, frame):
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.monotonic() * 1000)
        
        result = self.detector.detect_for_video(mp_image, timestamp)
        return result
    
    
    
    # Get iris center
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
    
    
    
    # Head Position
    def get_head_pos(self, frame, face_landmarks):
        
        height, width = frame.shape[:2]
        
        image_points = np.array([
            (
                face_landmarks[1].x * width,
                face_landmarks[1].y * height
            ),# nose
            (
                face_landmarks[152].x * width,
                face_landmarks[152].y * height
            ),# chin
            (
                face_landmarks[33].x * width,
                face_landmarks[33].y * height
            ),  # left eye corner
            (
                face_landmarks[263].x * width,
                face_landmarks[263].y * height
            ),  # right eye corner
            (
                face_landmarks[61].x * width,
                face_landmarks[61].y * height
            ),  # left mouth corner
            (
                face_landmarks[291].x * width,
                face_landmarks[291].y * height
            ),  # right mouth corner

        ], dtype=np.float64)
        
        #Template of 3D face model
        model_points = np.array([
            (0.0, 0.0, 0.0),# nose
            (0.0, -63.6, -12.5),# chin
            (-43.3, 32.7, -26.0),# left eye
            (43.3, 32.7, -26.0),# right eye
            (-28.9, -28.9, -24.1),# left mouth
            (28.9, -28.9, -24.1),# right mouth
        ], dtype=np.float64)
        
        focal_length = width
        
        # 3D Matrix for 3D face template
        # Camera Intrinsic Matrix
        camera_matrix = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        distortion_coefficients = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion_coefficients
        )
        
        if not success:
            return None
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )
        
        rx = angles[0][0]
        ry = angles[1][0]
        rz = angles[2][0]
        
        tx = translation_vector[0][0]
        ty = translation_vector[1][0]
        tz = translation_vector[2][0]

        return rx, ry, rz, tx, ty, tz
    
    
    
    def get_iris_ratio(self, frame, face_landmarks, iris_center,
                       outer_corner_index, inner_corner_index, 
                       upper_eyelid_index, lower_eyelid_index):
        
        height, width = frame.shape[:2]
        
        outer_x = face_landmarks[outer_corner_index].x * width
        inner_x = face_landmarks[inner_corner_index].x * width
        
        upper_y = face_landmarks[upper_eyelid_index].y * height
        lower_y = face_landmarks[lower_eyelid_index].y * height
        
        iris_x, iris_y = iris_center
        
        # Horizontal iris position
        x_ratio = ((iris_x - outer_x) / (inner_x - outer_x))
        
        # Vertical iris position
        y_ratio = ((iris_y - upper_y) / (lower_y - upper_y))
        
        return x_ratio, y_ratio
        
    
    
    # Main Process
    def process(self, frame):
        
        plain_frame = frame.copy()
        
        result = self.detect_landmarks(frame)
        if not result.face_landmarks:
            return plain_frame
        
        face_landmarks = result.face_landmarks[0]
        
        # Iris Position
        left_iris_center = self.get_iris_center(plain_frame, face_landmarks, LEFT_IRIS_INDICES)
        right_iris_center = self.get_iris_center(plain_frame, face_landmarks, RIGHT_IRIS_INDICES)
        
        # Left iris ratios
        left_x_ratio, left_y_ratio = self.get_iris_ratio(plain_frame, face_landmarks, left_iris_center, 
                                                         LEFT_OUTER_CORNER, LEFT_INNER_CORNER, LEFT_UPPER_EYELID, LEFT_LOWER_EYELID)

        # Right iris ratios
        right_x_ratio, right_y_ratio = self.get_iris_ratio(plain_frame, face_landmarks, right_iris_center, 
                                                           RIGHT_OUTER_CORNER, RIGHT_INNER_CORNER, RIGHT_UPPER_EYELID, RIGHT_LOWER_EYELID)
        
        print(f"Left iris: x={left_x_ratio:.2f}, y={left_y_ratio:.2f}," 
              f"Right iris: x={right_x_ratio:.2f}, y={right_y_ratio:.2f}")
        
        # Head Position
        head_pos = self.get_head_pos(plain_frame, face_landmarks)
        if head_pos:
            yaw, pitch, roll = head_pos
            print(
                f"Yaw: {yaw:.1f}, "
                f"Pitch: {pitch:.1f}, "
                f"Roll: {roll:.1f}"
            )
        
        # Future processes
        corrected_frame = plain_frame
        
        return corrected_frame
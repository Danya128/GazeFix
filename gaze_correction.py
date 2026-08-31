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
        
        #Template of the 3D face model
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
        
        rx = rotation_vector[0][0]
        ry = rotation_vector[1][0]
        rz = rotation_vector[2][0]
        
        tx = translation_vector[0][0]
        ty = translation_vector[1][0]
        tz = translation_vector[2][0]

        return rx, ry, rz, tx, ty, tz
    
    
    
    # Iris Ratio
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
    
    
    
    # Target iris position
    def get_target_point(self, frame, face_landmarks, x_ratio, y_ratio, outer_corner_index,
                            inner_corner_index, upper_eyelid_index, lower_eyelid_index):
        
        height, width = frame.shape[:2]
        
        outer_x = face_landmarks[outer_corner_index].x * width
        inner_x = face_landmarks[inner_corner_index].x * width
        
        upper_y = face_landmarks[upper_eyelid_index].y * height
        lower_y = face_landmarks[lower_eyelid_index].y * height
        
        target_x = outer_x + x_ratio * (inner_x - outer_x)
        target_y = upper_y + y_ratio * (lower_y - upper_y)
        
        cv2.circle(frame, (int(target_x), int(target_y)), 7, (0, 0, 255), -1)
        
        return int(target_x), int(target_y)
    
    
    
    # Shift the iris image
    def shift_iris(self, frame, iris_center, target_point, warp_radius):
        
        iris_x, iris_y = iris_center
        target_x, target_y = target_point
        
        dx = target_x - iris_x
        dy = target_y - iris_y
        
        padding = (warp_radius + max(abs(dx), abs(dy)) + 1)

        x1 = max(0, iris_x - padding)
        y1 = max(0, iris_y - padding)
        
        x2 = min(frame.shape[1], iris_x + padding)
        y2 = min(frame.shape[0], iris_y + padding)
        
        roi = frame[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            return frame
        
        height, width = roi.shape[:2]
        
        map_x, map_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32)
        )
        
        center_x = target_x - x1
        center_y = target_y - y1
        
        # Pythagorean theorem
        distance = np.sqrt(
            (map_x - center_x) ** 2 + 
            (map_y - center_y) ** 2
        )
        
        weight = np.clip(1.0 - distance / warp_radius, 0.0, 1.0)
        
        weight = weight * (2.5 - 2 * weight)
        
        map_x = map_x - (dx * weight)*1.5
        map_y = map_y - (dy * weight)*1.5
        
        warped_roi = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        frame[y1:y2, x1:x2] = warped_roi
        
        return frame
        
    
    
    # Main Process
    def process(self, frame, calibration, gaze_correction_mode):
        
        plain_frame = frame.copy()
        height, width = plain_frame.shape[:2]
        
        # Detect face landmarks
        result = self.detect_landmarks(frame)
        if not result.face_landmarks:
            return plain_frame, None
        
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
        
        #print(f"Left iris: x={left_x_ratio:.2f}, y={left_y_ratio:.2f}," 
        #      f"Right iris: x={right_x_ratio:.2f}, y={right_y_ratio:.2f}")
        
        # Head Position
        head_pos = self.get_head_pos(plain_frame, face_landmarks)
        if head_pos is None:
            return plain_frame, None
        
        rx, ry, rz, tx, ty, tz = head_pos
        
        tracked_data = {
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "left_x_ratio": left_x_ratio,
            "left_y_ratio": left_y_ratio,
            "right_x_ratio": right_x_ratio,
            "right_y_ratio": right_y_ratio
        }
        
        # Run ML prediction only when gaze correction is enabled
        if gaze_correction_mode and calibration is not None:
            
            predicted_ratios = calibration.predict(tracked_data)
            
            if predicted_ratios is not None:
                
                (target_left_x_ratio,
                target_left_y_ratio,
                target_right_x_ratio,
                target_right_y_ratio) = predicted_ratios
                
                left_target = self.get_target_point(plain_frame, face_landmarks, target_left_x_ratio,
                target_left_y_ratio, LEFT_OUTER_CORNER, LEFT_INNER_CORNER, LEFT_UPPER_EYELID, LEFT_LOWER_EYELID)
            
                right_target = self.get_target_point(plain_frame, face_landmarks, target_right_x_ratio,
                target_right_y_ratio, RIGHT_OUTER_CORNER, RIGHT_INNER_CORNER, RIGHT_UPPER_EYELID, RIGHT_LOWER_EYELID)
        
                cv2.circle(plain_frame, left_target, 4, (0, 0, 255), -1)
                cv2.circle(plain_frame, right_target, 4, (0, 0, 255), -1)
            
                # Calculate a reasonable warp radius for shift function
                left_eye_width = int(abs(   
                    face_landmarks[LEFT_INNER_CORNER].x * width -
                    face_landmarks[LEFT_OUTER_CORNER].x * width)*0.45)

                right_eye_width = int(abs(
                    face_landmarks[RIGHT_INNER_CORNER].x * width -
                    face_landmarks[RIGHT_OUTER_CORNER].x * width)*0.45)

                # Shift left iris
                #plain_frame = self.shift_iris(
                #    plain_frame,
                #    left_iris_center,
                #    left_target,
                #    left_eye_width
                #)

                # Shift right iris
                #plain_frame = self.shift_iris(
                #    plain_frame,
                #    right_iris_center,
                #    right_target,
                #    right_eye_width
                #)
        
        
        return plain_frame, tracked_data
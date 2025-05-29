"""
Core pipeline module for pose estimation, angle calculation, and pose classification.

This module provides three main classes:
- PoseEstimator: Handles MediaPipe setup, landmark detection from images, 
                 and extraction of landmarks for pose, face, and hands.
- AngleCalculator: A utility class with static methods to calculate angles 
                   between three points and distances between two points.
- PoseClassifier: Classifies human poses (e.g., T-Pose, Warrior II, Tree Pose) 
                  based on calculated angles from landmarks. It also manages 
                  angle states over time for gesture recognition.
"""
import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from icecream import ic # For debugging, can be removed in production


class PoseEstimator:
    """
    Estimates pose, face, and hand landmarks from an image.
    """
    def __init__(self, static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.5, model_complexity=2):
        """
        Initializes MediaPipe objects.
        Args:
            static_image_mode: Whether to treat the input images as a batch of static
                               and possibly unrelated images, or a video stream.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for the
                                      detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                                       person tracking to be considered successful.
            model_complexity: Complexity of the pose landmark model: 0, 1 or 2.
        """
        self.mp_pose = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Holistic(static_image_mode=static_image_mode,
                                           min_detection_confidence=min_detection_confidence,
                                           min_tracking_confidence=min_tracking_confidence,
                                           model_complexity=model_complexity)

    def extract_keypoints(self, results):
        """
        Extracts keypoints for pose, face, and hands from MediaPipe results.
        Args:
            results: MediaPipe Holistic output.
        Returns:
            A numpy array concatenating pose, face, left hand, and right hand landmarks.
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4) # 33 landmarks * 4 values (x,y,z,visibility)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3) # 468 landmarks * 3 values (x,y,z)
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3) # 21 landmarks * 3 values (x,y,z)
        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3) # 21 landmarks * 3 values (x,y,z)
        return np.concatenate([pose, face, left_hand, right_hand])

    def extract_pose_keypoints(self, results):
        """
        Extracts only pose keypoints from MediaPipe results.
        Args:
            results: MediaPipe Holistic output.
        Returns:
            A numpy array of pose landmarks.
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        return np.concatenate([pose])
    # Removed extract_pose_keypoints_alternative_format as it's unused by the classification pipeline.

    def detectPose(self, image, display=True):
        '''
        Performs pose detection on an image and extracts landmarks.
        Args:
            image: The input image.
            display: If True, displays the original and output images with landmarks.
        Returns:
            output_image: The image with landmarks drawn.
            detected_landmarks: A dictionary containing lists of landmarks for
                                'pose', 'left_hand', and 'right_hand'.
                                Landmarks are tuples (x, y, z) scaled to image dimensions.
                                Returns None for a category if no landmarks are detected.
        '''
        output_image = image.copy()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imageRGB)

        height, width, _ = image.shape
        
        detected_landmarks = {
            'pose': None, 
            'left_hand': None, 
            'right_hand': None
        }

        # Process and draw pose landmarks
        if results.pose_landmarks:
            detected_landmarks['pose'] = []
            for landmark in results.pose_landmarks.landmark:
                detected_landmarks['pose'].append(
                    (int(landmark.x * width), int(landmark.y * height), int(landmark.z * width))
                )
            self.mp_drawing.draw_landmarks(
                image=output_image,
                landmark_list=results.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Process and draw left hand landmarks
        if results.left_hand_landmarks:
            detected_landmarks['left_hand'] = []
            for landmark in results.left_hand_landmarks.landmark:
                detected_landmarks['left_hand'].append(
                    (int(landmark.x * width), int(landmark.y * height), int(landmark.z * width))
                )
            self.mp_drawing.draw_landmarks(
                image=output_image,
                landmark_list=results.left_hand_landmarks,
                connections=self.mp_pose.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

        # Process and draw right hand landmarks
        if results.right_hand_landmarks:
            detected_landmarks['right_hand'] = []
            for landmark in results.right_hand_landmarks.landmark:
                detected_landmarks['right_hand'].append(
                    (int(landmark.x * width), int(landmark.y * height), int(landmark.z * width))
                )
            self.mp_drawing.draw_landmarks(
                image=output_image,
                landmark_list=results.right_hand_landmarks,
                connections=self.mp_pose.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
        
        # Draw face landmarks (tesselation and contours)
        # These are not returned in detected_landmarks but drawn if present
        if results.face_landmarks:
            # self.mp_drawing.draw_landmarks(image=output_image, landmark_list=results.face_landmarks,connections=self.mp_pose.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
            self.mp_drawing.draw_landmarks(image=output_image, landmark_list=results.face_landmarks,connections=self.mp_pose.FACEMESH_TESSELATION,landmark_drawing_spec=None,connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

        if display:
            plt.figure(figsize=[22,22])
            plt.subplot(121); plt.imshow(image[:,:,::-1]); plt.title("Original Image"); plt.axis('off');
            plt.subplot(122); plt.imshow(output_image[:,:,::-1]); plt.title("Output Image"); plt.axis('off');
            if results.pose_world_landmarks: # Plot 3D world landmarks if available
                self.mp_drawing.plot_landmarks(results.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS)
            plt.show()
        
        return output_image, detected_landmarks

class AngleCalculator:
    """
    Utility class for calculating angles and distances between landmarks.
    """
    @staticmethod
    def calculateAngle(landmark1, landmark2, landmark3):
        '''
        This function calculates angle between three different landmarks.
        Args:
            landmark1: The first landmark containing the x,y and z coordinates.
            landmark2: The second landmark containing the x,y and z coordinates.
            landmark3: The third landmark containing the x,y and z coordinates.
        Returns:
            angle: The calculated angle between the three landmarks.
        '''
        # Get the required landmarks coordinates.
        x1, y1, _ = landmark1
        x2, y2, _ = landmark2
        x3, y3, _ = landmark3

        # Calculate the angle between the three points
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        
        # Check if the angle is less than zero.
        if angle < 0:
            # Add 360 to the found angle.
            angle += 360
        # Return the calculated angle. This line is correctly indented to be
        # part of the calculateAngle method and at the same level as the 'if' statement.
        return angle

    @staticmethod
    def calculateDistance(landmark1, landmark2):
        x1, y1, _ = landmark1
        x2, y2, _ = landmark2

        # Calculate the Distance between the two points
        dis = math.sqrt( ((x2 - x1)**2)+((y2 - y1))**2)

        
        # Return the calculated Distance.
        return dis


class PoseClassifier:
    """
    Classifies poses based on landmark angles and manages angle states.
    """
    # --- Constants for Pose Classification Logic ---
    # Angle thresholds for T-Pose and Warrior II
    STRAIGHT_ARM_ELBOW_MIN = 165
    STRAIGHT_ARM_ELBOW_MAX = 195
    TPOSE_SHOULDER_MIN = 80
    TPOSE_SHOULDER_MAX = 110
    TPOSE_STRAIGHT_LEG_MIN = 160 # Knee angle for T-Pose
    TPOSE_STRAIGHT_LEG_MAX = 195

    WARRIOR_II_STRAIGHT_LEG_MIN = 165
    WARRIOR_II_STRAIGHT_LEG_MAX = 195
    WARRIOR_II_BENT_LEG_MIN = 90
    WARRIOR_II_BENT_LEG_MAX = 120

    # Angle thresholds for Tree Pose
    TREE_POSE_STRAIGHT_LEG_MIN = 165
    TREE_POSE_STRAIGHT_LEG_MAX = 195
    TREE_POSE_BENT_LEG_MIN_RIGHT_LEG_UP = 25 # Right leg bent, foot at left thigh
    TREE_POSE_BENT_LEG_MAX_RIGHT_LEG_UP = 45
    TREE_POSE_BENT_LEG_MIN_LEFT_LEG_UP = 315 # Left leg bent, foot at right thigh (angle > 180)
    TREE_POSE_BENT_LEG_MAX_LEFT_LEG_UP = 335
    
    # Angle diff threshold for "bye" gesture (example)
    BYE_GESTURE_ELBOW_DIFF_THRESHOLD = 10
    BYE_GESTURE_RIGHT_ELBOW_MIN = 25
    BYE_GESTURE_RIGHT_ELBOW_MAX = 110
    BYE_GESTURE_LEFT_ELBOW_MIN = 260
    BYE_GESTURE_LEFT_ELBOW_MAX = 340


    # --- Constants for Drawing Logic ---
    # Colors (BGR format)
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLUE_TEXT = (255, 0, 0) # For distance text
    
    # Font properties
    FONT = cv2.FONT_HERSHEY_PLAIN
    FONT_SCALE_ANGLES = 1.3
    FONT_SCALE_DISTANCES = 1.3
    FONT_THICKNESS = 2

    # Text positioning for angles
    TEXT_POS_X_ANGLES = 10
    TEXT_POS_Y_ANGLES_START = 20
    TEXT_STEP_Y_ANGLES = 40

    # Text positioning for distances
    TEXT_POS_X_DISTANCES = 500 # Adjusted from 10*50
    TEXT_POS_Y_DISTANCES_START = 20
    TEXT_STEP_Y_DISTANCES = 40
    
    # Label text
    LABEL_POS_X = 10
    LABEL_POS_Y = 30
    LABEL_FONT_SCALE = 2
    UNKNOWN_POSE_LABEL = 'Unknown Pose'


    def __init__(self):
        self.time_ = time.time() # Initialize time for diff calculation
        self.Angle_previous = [] # Stores history of angle sets for diff calculation (can be removed if not further used)
        self.Angle_diff = []     # Stores calculated differences (can be removed if not further used)

        # Current angles - initialized to 0
        self.left_elbow_angle = self.right_elbow_angle = 0
        self.left_shoulder_angle = self.right_shoulder_angle = 0
        self.left_wrist_angle = self.right_wrist_angle = 0
        self.left_index_mcp_angle = self.right_index_mcp_angle = 0
        self.left_pinky_tip_angle = self.right_pinky_tip_angle = 0
        self.left_knee_angle = self.right_knee_angle = 0

        # Previous angles - initialized to 0
        self.left_elbow_angle_previous = self.right_elbow_angle_previous = 0
        self.left_shoulder_angle_previous = self.right_shoulder_angle_previous = 0
        self.left_wrist_angle_previous = self.right_wrist_angle_previous = 0
        self.left_index_mcp_angle_previous = self.right_index_mcp_angle_previous = 0
        self.left_pinky_tip_angle_previous = self.right_pinky_tip_angle_previous = 0

        # Angle differences - initialized to 0
        self.left_elbow_angle_diff = self.right_elbow_angle_diff = 0
        self.left_shoulder_angle_diff = self.right_shoulder_angle_diff = 0
        self.left_wrist_angle_diff = self.right_wrist_angle_diff = 0
        self.left_index_mcp_angle_diff = self.right_index_mcp_angle_diff = 0
        self.left_pinky_tip_angle_diff = self.right_pinky_tip_angle_diff = 0
        
        self.mp_pose = mp.solutions.holistic # For accessing PoseLandmark and HandLandmark enums

    def _update_angle_history_and_diffs(self):
        """
        Updates the history of angles and calculates differences.
        This internal method is called periodically.
        """
        if (time.time() - self.time_ > 0.2):
            self.left_elbow_angle_diff = abs(self.left_elbow_angle - self.left_elbow_angle_previous)
            self.right_elbow_angle_diff = abs(self.right_elbow_angle - self.right_elbow_angle_previous)
            self.left_shoulder_angle_diff = abs(self.left_shoulder_angle - self.left_shoulder_angle_previous)
            self.right_shoulder_angle_diff = abs(self.right_shoulder_angle - self.right_shoulder_angle_previous)
            self.left_wrist_angle_diff = abs(self.left_wrist_angle - self.left_wrist_angle_previous)
            self.right_wrist_angle_diff = abs(self.right_wrist_angle - self.right_wrist_angle_previous)
            self.left_index_mcp_angle_diff = abs(self.left_index_mcp_angle - self.left_index_mcp_angle_previous)
            self.right_index_mcp_angle_diff = abs(self.right_index_mcp_angle - self.right_index_mcp_angle_previous)
            self.left_pinky_tip_angle_diff = abs(self.left_pinky_tip_angle - self.left_pinky_tip_angle_previous)
            self.right_pinky_tip_angle_diff = abs(self.right_pinky_tip_angle - self.right_pinky_tip_angle_previous)

            self.Angle_diff.append([
                self.left_elbow_angle_diff, self.right_elbow_angle_diff,
                self.left_shoulder_angle_diff, self.right_shoulder_angle_diff,
                self.left_wrist_angle_diff, self.right_wrist_angle_diff,
                self.left_index_mcp_angle_diff, self.right_index_mcp_angle_diff,
                self.left_pinky_tip_angle_diff, self.right_pinky_tip_angle_diff
            ])
            # ic(self.Angle_diff) # Commented out for now, useful for debugging angle diffs.

            # Update previous angles
            self.left_elbow_angle_previous = self.left_elbow_angle
            self.right_elbow_angle_previous = self.right_elbow_angle
            self.left_shoulder_angle_previous = self.left_shoulder_angle
            self.right_shoulder_angle_previous = self.right_shoulder_angle
            self.left_wrist_angle_previous = self.left_wrist_angle
            self.right_wrist_angle_previous = self.right_wrist_angle
            self.left_index_mcp_angle_previous = self.left_index_mcp_angle
            self.right_index_mcp_angle_previous = self.right_index_mcp_angle
            self.left_pinky_tip_angle_previous = self.left_pinky_tip_angle
            self.right_pinky_tip_angle_previous = self.right_pinky_tip_angle
            
            self.Angle_previous = [] # Reset, as previous values are now stored in instance vars
            self.Angle_diff = [] # Reset diff list for next interval
            self.time_ = time.time()


    def classifyPose(self, landmarks_dict, output_image, display=False): # Renamed 'landmarks' to 'landmarks_dict'

        '''
        This function classifies yoga poses depending upon the angles of various body joints.
        Args:
            landmarks: A list of detected landmarks of the person whose pose needs to be classified.
            output_image: A image of the person with the detected pose landmarks drawn.
            display: A boolean value that is if set to true the function displays the resultant image with the pose label 
            written on it and returns nothing.
        Returns:
            output_image: The image with the detected pose landmarks drawn and pose label written.
            label: The classified pose label of the person in the output_image.
        '''
        # landmarks_dict contains 'pose', 'left_hand', 'right_hand'
        pose_landmarks = landmarks_dict.get('pose')
        left_hand_landmarks = landmarks_dict.get('left_hand')
        right_hand_landmarks = landmarks_dict.get('right_hand')

        label = self.UNKNOWN_POSE_LABEL
        current_color = self.COLOR_RED # Default color for unknown pose
        
        # --- Calculate Body Angles (if pose landmarks are available) ---
        if pose_landmarks:
            self.right_elbow_angle = AngleCalculator.calculateAngle(
                pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                pose_landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                pose_landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value])
            
            self.left_elbow_angle = AngleCalculator.calculateAngle(
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value])   
            
            self.right_shoulder_angle = AngleCalculator.calculateAngle(
                pose_landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                pose_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])

            self.left_shoulder_angle = AngleCalculator.calculateAngle(
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value])

            self.right_knee_angle = AngleCalculator.calculateAngle(
                pose_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                pose_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                pose_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])

            self.left_knee_angle = AngleCalculator.calculateAngle(
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        else: # Reset body angles if no pose landmarks
            self.right_elbow_angle = self.left_elbow_angle = 0
            self.right_shoulder_angle = self.left_shoulder_angle = 0
            self.right_knee_angle = self.left_knee_angle = 0

        # --- Calculate Hand Angles (if hand landmarks are available) ---
        # Note: Original code had left/right hand angle calculations swapped. Corrected here.
        # For self.left_wrist_angle, we use right_hand_landmarks because it's the person's left hand from their perspective.
        if right_hand_landmarks: # Person's LEFT hand
            self.left_wrist_angle = AngleCalculator.calculateAngle(
                right_hand_landmarks[self.mp_pose.HandLandmark.THUMB_TIP.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.PINKY_TIP.value])

            self.left_index_mcp_angle = AngleCalculator.calculateAngle(
                right_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_MCP.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_TIP.value])
            
            self.left_pinky_tip_angle = AngleCalculator.calculateAngle(
                right_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.PINKY_MCP.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.PINKY_TIP.value])
        else: # Reset left hand angles
            self.left_wrist_angle = self.left_index_mcp_angle = self.left_pinky_tip_angle = 0

        if left_hand_landmarks: # Person's RIGHT hand
            self.right_wrist_angle = AngleCalculator.calculateAngle(
                left_hand_landmarks[self.mp_pose.HandLandmark.THUMB_TIP.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.PINKY_TIP.value])

            self.right_index_mcp_angle = AngleCalculator.calculateAngle(
                left_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_MCP.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_TIP.value])

            self.right_pinky_tip_angle = AngleCalculator.calculateAngle(
                left_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.PINKY_MCP.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.PINKY_TIP.value])
        else: # Reset right hand angles
            self.right_wrist_angle = self.right_index_mcp_angle = self.right_pinky_tip_angle = 0
            
        # --- Calculate Hand Distances (if hand landmarks are available) ---
        left_thumb_index_distance = right_thumb_index_distance = 0
        left_thumb_pinky_distance = right_thumb_pinky_distance = 0
        left_wrist_index_distance = right_wrist_index_distance = 0
        left_wrist_pinky_distance = right_wrist_pinky_distance = 0

        if right_hand_landmarks: # Person's LEFT hand distances
            left_thumb_index_distance = AngleCalculator.calculateDistance(
                right_hand_landmarks[self.mp_pose.HandLandmark.THUMB_TIP.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_TIP.value])
            left_thumb_pinky_distance = AngleCalculator.calculateDistance(
                right_hand_landmarks[self.mp_pose.HandLandmark.THUMB_TIP.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.PINKY_TIP.value])
            left_wrist_index_distance = AngleCalculator.calculateDistance(
                right_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_TIP.value])
            left_wrist_pinky_distance = AngleCalculator.calculateDistance(
                right_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                right_hand_landmarks[self.mp_pose.HandLandmark.PINKY_TIP.value])

        if left_hand_landmarks: # Person's RIGHT hand distances
            right_thumb_index_distance = AngleCalculator.calculateDistance(
                left_hand_landmarks[self.mp_pose.HandLandmark.THUMB_TIP.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_TIP.value])
            right_thumb_pinky_distance = AngleCalculator.calculateDistance(
                left_hand_landmarks[self.mp_pose.HandLandmark.THUMB_TIP.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.PINKY_TIP.value])
            right_wrist_index_distance = AngleCalculator.calculateDistance(
                left_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_TIP.value])
            right_wrist_pinky_distance = AngleCalculator.calculateDistance(
                left_hand_landmarks[self.mp_pose.HandLandmark.WRIST.value],
                left_hand_landmarks[self.mp_pose.HandLandmark.PINKY_TIP.value])

        # (Commented out circle drawing for brevity, can be re-added if needed)
        # if right_hand_landmarks:
        #     cirlcr1_posx ,cirlcr1_posy,_= right_hand_landmarks[self.mp_pose.HandLandmark.THUMB_TIP.value]
        #     cirlcr2_posx ,cirlcr2_posy,_= right_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_TIP.value]
        # if left_hand_landmarks:
        #     cirlcr1_posx ,cirlcr1_posy,_= left_hand_landmarks[self.mp_pose.HandLandmark.THUMB_TIP.value]
        #     cirlcr2_posx ,cirlcr2_posy,_= left_hand_landmarks[self.mp_pose.HandLandmark.INDEX_FINGER_TIP.value]
        self._draw_angle_and_distance_data(output_image, 
                                           left_thumb_index_distance, right_thumb_index_distance,
                                           left_thumb_pinky_distance, right_thumb_pinky_distance,
                                           left_wrist_index_distance, right_wrist_index_distance,
                                           left_wrist_pinky_distance, right_wrist_pinky_distance)
        
        self._update_angle_history_and_diffs() # Update historical data for next frame
        
        # --- Pose Classification Logic ---
        # Check for specific poses. Order might matter if poses have overlapping criteria.
        label = self._check_t_pose_and_warrior_ii()
        if label == self.UNKNOWN_POSE_LABEL: # Only check for Tree Pose if not already classified
            label = self._check_tree_pose()
        
        # Example "Bye" gesture detection (can be expanded or moved)
        self._check_bye_gesture() # This currently prints debug messages, doesn't set label

        if label != self.UNKNOWN_POSE_LABEL:
            current_color = self.COLOR_GREEN
        
        cv2.putText(output_image, label, (self.LABEL_POS_X, self.LABEL_POS_Y), 
                    self.FONT, self.LABEL_FONT_SCALE, current_color, self.FONT_THICKNESS)
        
        if display:
            plt.figure(figsize=[10,10])
            plt.title("Output Image"); plt.axis('off'); plt.imshow(output_image[:,:,::-1]); plt.show()
        else:
            return output_image, label

    def _check_t_pose_and_warrior_ii(self):
        """
        Checks for T-Pose and Warrior II pose.
        These are grouped as they share initial arm and shoulder conditions.
        Returns:
            str: The pose label ('T Pose', 'Warrior II Pose') or UNKNOWN_POSE_LABEL.
        """
        label = self.UNKNOWN_POSE_LABEL
        # Check if both arms are straight and shoulders are at T-pose angles
        if (self.STRAIGHT_ARM_ELBOW_MIN < self.left_elbow_angle < self.STRAIGHT_ARM_ELBOW_MAX and
            self.STRAIGHT_ARM_ELBOW_MIN < self.right_elbow_angle < self.STRAIGHT_ARM_ELBOW_MAX):
            if (self.TPOSE_SHOULDER_MIN < self.left_shoulder_angle < self.TPOSE_SHOULDER_MAX and
                self.TPOSE_SHOULDER_MIN < self.right_shoulder_angle < self.TPOSE_SHOULDER_MAX):

                # Check for Warrior II: one leg straight, one leg bent at specific angles
                is_left_leg_straight_for_warrior = self.WARRIOR_II_STRAIGHT_LEG_MIN < self.left_knee_angle < self.WARRIOR_II_STRAIGHT_LEG_MAX
                is_right_leg_straight_for_warrior = self.WARRIOR_II_STRAIGHT_LEG_MIN < self.right_knee_angle < self.WARRIOR_II_STRAIGHT_LEG_MAX
                is_left_leg_bent_for_warrior = self.WARRIOR_II_BENT_LEG_MIN < self.left_knee_angle < self.WARRIOR_II_BENT_LEG_MAX
                is_right_leg_bent_for_warrior = self.WARRIOR_II_BENT_LEG_MIN < self.right_knee_angle < self.WARRIOR_II_BENT_LEG_MAX

                if (is_left_leg_straight_for_warrior and is_right_leg_bent_for_warrior) or \
                   (is_right_leg_straight_for_warrior and is_left_leg_bent_for_warrior):
                    label = 'Warrior II Pose'
                
                # Check for T-Pose: both legs straight
                elif (self.TPOSE_STRAIGHT_LEG_MIN < self.left_knee_angle < self.TPOSE_STRAIGHT_LEG_MAX and
                      self.TPOSE_STRAIGHT_LEG_MIN < self.right_knee_angle < self.TPOSE_STRAIGHT_LEG_MAX):
                    label = 'T Pose'
        return label

    def _check_tree_pose(self):
        """
        Checks for Tree Pose.
        Returns:
            str: The pose label ('Tree Pose') or UNKNOWN_POSE_LABEL.
        """
        label = self.UNKNOWN_POSE_LABEL
        # Check if one leg is straight
        is_left_leg_straight = self.TREE_POSE_STRAIGHT_LEG_MIN < self.left_knee_angle < self.TREE_POSE_STRAIGHT_LEG_MAX
        is_right_leg_straight = self.TREE_POSE_STRAIGHT_LEG_MIN < self.right_knee_angle < self.TREE_POSE_STRAIGHT_LEG_MAX

        if is_left_leg_straight or is_right_leg_straight:
            # Check if the other leg is bent appropriately for Tree Pose
            # (foot at the inner thigh of the straight leg)
            is_right_leg_bent_for_tree = self.TREE_POSE_BENT_LEG_MIN_RIGHT_LEG_UP < self.right_knee_angle < self.TREE_POSE_BENT_LEG_MAX_RIGHT_LEG_UP
            is_left_leg_bent_for_tree = self.TREE_POSE_BENT_LEG_MIN_LEFT_LEG_UP < self.left_knee_angle < self.TREE_POSE_BENT_LEG_MAX_LEFT_LEG_UP
            
            if (is_left_leg_straight and is_right_leg_bent_for_tree) or \
               (is_right_leg_straight and is_left_leg_bent_for_tree):
                label = 'Tree Pose'
        return label

    def _check_bye_gesture(self):
        """
        Checks for a "bye" gesture based on elbow movement.
        This method currently prints debug messages but does not set a pose label.
        """
        # Check if hands are detected (wrist angles would be non-zero)
        if self.left_wrist_angle != 0 or self.right_wrist_angle != 0:
            # Check for right arm "bye"
            if (self.BYE_GESTURE_RIGHT_ELBOW_MIN < self.right_elbow_angle < self.BYE_GESTURE_RIGHT_ELBOW_MAX and
                self.right_elbow_angle_diff > self.BYE_GESTURE_ELBOW_DIFF_THRESHOLD):
                print("Debug: Potential 'bye right' gesture detected based on elbow movement.")
            
            # Check for left arm "bye"
            if (self.BYE_GESTURE_LEFT_ELBOW_MIN < self.left_elbow_angle < self.BYE_GESTURE_LEFT_ELBOW_MAX and
                self.left_elbow_angle_diff > self.BYE_GESTURE_ELBOW_DIFF_THRESHOLD):
                print("Debug: Potential 'bye left' gesture detected based on elbow movement.")

    def _draw_angle_and_distance_data(self, output_image, 
                                     left_thumb_index_dist, right_thumb_index_dist,
                                     left_thumb_pinky_dist, right_thumb_pinky_dist,
                                     left_wrist_index_dist, right_wrist_index_dist,
                                     left_wrist_pinky_dist, right_wrist_pinky_dist):
        """Draws angle and distance data onto the output image using class constants."""
        
        # Draw Body Angles
        angle_texts = [
            f"L_elbow_angle : {self.left_elbow_angle:.2f}",
            f"R_elbow_angle : {self.right_elbow_angle:.2f}",
            f"L_shoulder_angle : {self.left_shoulder_angle:.2f}",
            f"R_shoulder_angle : {self.right_shoulder_angle:.2f}",
            f"L_wrist_angle : {self.left_wrist_angle:.2f}",
            f"R_wrist_angle : {self.right_wrist_angle:.2f}",
            f"L_index_finger_angle : {self.left_index_mcp_angle:.2f}",
            f"R_index_finger_angle : {self.right_index_mcp_angle:.2f}", # Person's Right Hand
            f"L_pinky_finger_angle : {self.left_pinky_tip_angle:.2f}",   # Person's Left Hand
            f"R_pinky_finger_angle : {self.right_pinky_tip_angle:.2f}"  # Person's Right Hand
        ]
        for i, text in enumerate(angle_texts):
            cv2.putText(output_image, text, 
                        (self.TEXT_POS_X_ANGLES, self.TEXT_POS_Y_ANGLES_START + i * self.TEXT_STEP_Y_ANGLES),
                        self.FONT, self.FONT_SCALE_ANGLES, self.COLOR_WHITE, self.FONT_THICKNESS)
        
        # Draw Hand Distances
        distance_texts = [
            f"L_thumb_index_distance : {left_thumb_index_dist:.2f}",
            f"R_thumb_index_distance : {right_thumb_index_dist:.2f}",
            f"L_thumb_pinky_distance : {left_thumb_pinky_dist:.2f}",
            f"R_thumb_pinky_distance : {right_thumb_pinky_dist:.2f}",
            f"L_wrist_index_distance : {left_wrist_index_dist:.2f}",
            f"R_wrist_index_distance : {right_wrist_index_dist:.2f}",
            f"L_wrist_pinky_distance : {left_wrist_pinky_dist:.2f}",
            f"R_wrist_pinky_distance : {right_wrist_pinky_dist:.2f}"
        ]
        for i, text in enumerate(distance_texts):
            cv2.putText(output_image, text, 
                        (self.TEXT_POS_X_DISTANCES, self.TEXT_POS_Y_DISTANCES_START + i * self.TEXT_STEP_Y_DISTANCES),
                        self.FONT, self.FONT_SCALE_DISTANCES, self.COLOR_BLUE_TEXT, self.FONT_THICKNESS)
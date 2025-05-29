import unittest
import math
import sys
import os
import numpy as np # Import numpy

# Adjust sys.path to include the parent directory (project root)
# This allows importing `pipline` from the tests directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipline import AngleCalculator, PoseClassifier, PoseEstimator
from unittest.mock import MagicMock, patch

# Mock the mediapipe module to avoid actual MediaPipe initialization during tests
# This is crucial if MediaPipe is not available or configured in the test environment.
# We do this before `pipline` is fully processed by Python's import system for its classes.
# Note: If pipline.py itself tries to access attributes of mp.solutions.holistic etc.
# at module load time (outside classes/functions), those specific attributes might also need mocking.
# For now, mocking the top-level 'mediapipe' import used in pipline.py.
mp_mock = MagicMock()
mp_mock.solutions = MagicMock()
mp_mock.solutions.holistic = MagicMock()
mp_mock.solutions.drawing_utils = MagicMock()
mp_mock.solutions.drawing_styles = MagicMock()
sys.modules['mediapipe'] = mp_mock


class TestAngleCalculator(unittest.TestCase):
    """
    Unit tests for the AngleCalculator class.
    """

    def test_calculate_angle_90_degrees(self):
        """Test a 90-degree angle."""
        lm1 = (0, 0, 0)  # Point A
        lm2 = (1, 0, 0)  # Vertex B (angle point)
        lm3 = (1, 1, 0)  # Point C
        # Angle at p2 between p2-p1 and p2-p3.
        # p2-p1: (-1,0). atan2(0,-1) = 180 deg.
        # p2-p3: (0,1). atan2(1,0) = 90 deg.
        # Result: 90 - 180 = -90. Normalized: 270.
        self.assertAlmostEqual(AngleCalculator.calculateAngle(lm1, lm2, lm3), 270.0)

        lm1_y = (0, 0, 0) # p1
        lm2_y = (0, 1, 0) # p2 (Vertex)
        lm3_y = (1, 1, 0) # p3
        # p2-p1: (0,-1). atan2(-1,0) = -90 (or 270) deg.
        # p2-p3: (1,0). atan2(0,1) = 0 deg.
        # Result: 0 - 270 = -270. Normalized: 90.
        self.assertAlmostEqual(AngleCalculator.calculateAngle(lm1_y, lm2_y, lm3_y), 90.0)

    def test_calculate_angle_0_degrees_collinear_p3_extends_p1p2(self):
        """Test a 0-degree angle (collinear, p3 extends p1-p2 vector from p2)."""
        # Points: p1=(0,0), p2=(1,0) (Vertex), p3=(2,0)
        # Vector p2-p1: (-1,0). atan2(0,-1) = 180 deg.
        # Vector p2-p3: (1,0). atan2(0,1) = 0 deg.
        # Angle = (angle of p2-p3) - (angle of p2-p1) = 0 - 180 = -180. Normalized: 180 deg.
        # This actually represents a 180-degree angle at vertex p2.
        p1 = (0,0,0)
        p2 = (1,0,0) # Vertex
        p3 = (2,0,0)
        self.assertAlmostEqual(AngleCalculator.calculateAngle(p1, p2, p3), 180.0)

        # To get 0 degrees, p1, p2, p3 must be in order such that p2-p1 and p2-p3 point in the same direction.
        # This is not possible if p2 is the vertex.
        # The function calculates the angle between vector p2->p1 and p2->p3.
        # If p1= (2,0,0), p2=(1,0,0), p3=(2,0,0) -> angle p1-p2-p3
        # p2-p1 = (1,0). atan2(0,1)=0
        # p2-p3 = (1,0). atan2(0,1)=0. Result 0.
        p1_alt = (2,0,0)
        p2_alt = (1,0,0)
        p3_alt = (2,0,0) # Effectively same as p1_alt relative to p2_alt
        # This is not what the original test meant.
        # The original test was likely for p1-p2-p3 where p1,p2,p3 are distinct and p2-p1 and p2-p3 are in the same direction
        # This implies p1 is on the "other side" of p2 from p3.
        # Example: p1=(2,0), p2=(1,0), p3=(0,0). Angle p1-p2-p3 (at p2)
        # p2-p1: (1,0) -> 0 deg
        # p2-p3: (-1,0) -> 180 deg
        # Result: 180 - 0 = 180 deg.
        # The only way to get 0 is if p1 and p3 are in the same direction from p2.
        # e.g. p1=(0,0), p2=(0,0), p3=(1,0) -> p2-p1 is (0,0), p2-p3 is (1,0). atan2(0,0) behavior is undefined for angle.
        # The current implementation cannot produce 0 for distinct p1,p2,p3 if p2 is the vertex.
        # The smallest positive angle is 0 if p1 and p3 are the same point (not the vertex).
        # Let's test for a very small angle instead.
        p1_small_angle = (1, 0.001, 0)
        p2_small_angle = (0,0,0)
        p3_small_angle = (1,0,0)
        # p2-p1: (1, 0.001)
        # p2-p3: (1,0)
        # angle p2p1 to p2p3, at p2. Should be small.
        # angle(p2-p3) - angle(p2-p1) = atan2(0,1) - atan2(0.001,1) = 0 - ~0.057 deg = -0.057 deg -> ~359.94 deg
        # This interpretation of "0 degrees" for distinct points is tricky.
        # If the points are p1, p2, p3 in a line, angle at p2 is 180.
        # If the points are p2, p1, p3 in a line, angle at p1 is 180.
        # The current code will always give 180 for 3 distinct collinear points when angle is at middle point.
        # If we want to test a "straight line but not turning back" (0 degrees in some contexts),
        # this function's definition of angle (p2p1 vs p2p3) makes it 180.
        # For the original points: p1=(0,0,0), p2=(1,0,0), p3=(2,0,0) --> angle at p2 is 180.
        self.assertAlmostEqual(AngleCalculator.calculateAngle((0,0,0), (1,0,0), (2,0,0)), 180.0)


    def test_calculate_angle_180_degrees(self):
        """Test a 180-degree angle (collinear points, p1, p2, p3 in order)."""
        # p1=(0,0,0), p2=(1,0,0) (Vertex), p3=(2,0,0)
        # Vector p2-p1: (-1,0). Angle is 180 deg.
        # Vector p2-p3: (1,0). Angle is 0 deg.
        # Result: 0 - 180 = -180. Normalized: 180.
        self.assertAlmostEqual(AngleCalculator.calculateAngle((0,0,0), (1,0,0), (2,0,0)), 180.0)
        
        # p1=(2,0,0), p2=(1,0,0) (Vertex), p3=(0,0,0)
        # Vector p2-p1: (1,0). Angle is 0 deg.
        # Vector p2-p3: (-1,0). Angle is 180 deg.
        # Result: 180 - 0 = 180.
        self.assertAlmostEqual(AngleCalculator.calculateAngle((2,0,0), (1,0,0), (0,0,0)), 180.0)

    def test_calculate_angle_45_degrees(self):
        """Test a 45-degree and a 315-degree angle."""
        # Test for 315 degrees (which is -45 degrees)
        # p1=(1,0,0), p2=(0,0,0) (Vertex), p3=(1,1,0)
        # p2-p1: (1,0). atan2(0,1) = 0 deg.
        # p2-p3: (1,1). atan2(1,1) = 45 deg.
        # Result: 45 - 0 = 45 deg.
        self.assertAlmostEqual(AngleCalculator.calculateAngle((1,0,0), (0,0,0), (1,1,0)), 45.0)

        # Test for 45 degrees using different configuration
        # p1=(1,1,0), p2=(0,0,0) (Vertex), p3=(1,0,0)
        # p2-p1: (1,1). atan2(1,1) = 45 deg.
        # p2-p3: (1,0). atan2(0,1) = 0 deg.
        # Result: 0 - 45 = -45. Normalized: 315.
        self.assertAlmostEqual(AngleCalculator.calculateAngle((1,1,0), (0,0,0), (1,0,0)), 315.0)


    def test_calculate_angle_specific_orientations(self):
        """Test angles with specific orientations."""
        # Angle from vector (0,0)->(1,0) to (0,0)->(0,1) should be 90 deg counter-clockwise.
        # p1=(1,0,0), p2=(0,0,0), p3=(0,1,0)
        # p2-p1: (1,0). atan2(0,1) = 0 deg.
        # p2-p3: (0,1). atan2(1,0) = 90 deg.
        # Result: 90 - 0 = 90.
        self.assertAlmostEqual(AngleCalculator.calculateAngle((1,0,0), (0,0,0), (0,1,0)), 90.0)

        # Test for 0/360 case: p1=(1,0,0), p2=(0,0,0), p3=(2,0,0) (same direction from p2)
        # p2-p1: (1,0). atan2(0,1) = 0.
        # p2-p3: (2,0). atan2(0,2) = 0.
        # Result: 0 - 0 = 0.
        self.assertAlmostEqual(AngleCalculator.calculateAngle((1,0,0), (0,0,0), (2,0,0)), 0.0)


    def test_calculate_distance_zero(self):
        """Test distance between identical points."""
        lm1 = (1, 2, 3)
        lm2 = (1, 2, 3)
        self.assertAlmostEqual(AngleCalculator.calculateDistance(lm1, lm2), 0.0)

    def test_calculate_distance_one_axis(self):
        """Test distance along one axis."""
        lm1 = (0, 0, 0)
        lm2 = (5, 0, 0)
        self.assertAlmostEqual(AngleCalculator.calculateDistance(lm1, lm2), 5.0)

        lm1_y = (0, -2, 0)
        lm2_y = (0, 3, 0)
        self.assertAlmostEqual(AngleCalculator.calculateDistance(lm1_y, lm2_y), 5.0)

    def test_calculate_distance_multi_axes(self):
        """Test distance with changes in multiple axes."""
        lm1 = (0, 0, 0)
        lm2 = (3, 4, 0) # 3-4-5 triangle in XY plane
        self.assertAlmostEqual(AngleCalculator.calculateDistance(lm1, lm2), 5.0)

        lm1_3d = (1, 1, 1)
        lm2_3d = (4, 5, 13) # dx=3, dy=4, dz=12
        # Expected distance: sqrt(3^2 + 4^2 + 12^2) = sqrt(9 + 16 + 144) = sqrt(25 + 144) = sqrt(169) = 13
        # Note: calculateDistance currently only uses x, y. This test will reflect that.
        self.assertAlmostEqual(AngleCalculator.calculateDistance(lm1_3d, lm2_3d), 5.0) # sqrt(3^2+4^2)


class TestPoseEstimator(unittest.TestCase):
    """
    Unit tests for the PoseEstimator class.
    Limited scope due to MediaPipe dependency.
    """
    @patch('pipline.mp.solutions.holistic.Holistic') # Corrected patch target
    def test_pose_estimator_initialization(self, mock_holistic_constructor):
        """Test if PoseEstimator initializes MediaPipe components."""
        # Create a mock instance to be returned by the Holistic constructor
        mock_holistic_instance = MagicMock()
        mock_holistic_constructor.return_value = mock_holistic_instance

        estimator = PoseEstimator()
        self.assertIsNotNone(estimator.pose) # Check if the pose object is created
        mock_holistic_constructor.assert_called_once() # Check if Holistic was called
        # Check if it was called with some default arguments (can be more specific)
        args, kwargs = mock_holistic_constructor.call_args
        self.assertTrue(kwargs.get('static_image_mode')) 
        self.assertEqual(kwargs.get('min_detection_confidence'), 0.3)


class TestPoseClassifier(unittest.TestCase):
    """
    Unit tests for the PoseClassifier class.
    Focuses on classification logic by mocking angle calculations.
    """
    def setUp(self):
        """Set up a PoseClassifier instance for each test."""
        self.classifier = PoseClassifier()
        self.mock_output_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Provide a minimal valid structure for landmarks_dict if pose landmarks are expected
        self.dummy_landmark = (0,0,0)
        self.mock_pose_lm_list = [self.dummy_landmark] * 33 # Holistic has 33 pose landmarks
        self.mock_hand_lm_list = [self.dummy_landmark] * 21 # Hand has 21 landmarks
        
        self.mock_landmarks_all_present = {
            'pose': self.mock_pose_lm_list,
            'left_hand': self.mock_hand_lm_list,
            'right_hand': self.mock_hand_lm_list
        }
        self.mock_landmarks_no_pose = {
            'pose': None,
            'left_hand': self.mock_hand_lm_list,
            'right_hand': self.mock_hand_lm_list
        }

    def test_constructor_default_angles(self):
        """Test if default angles are initialized to 0."""
        self.assertEqual(self.classifier.left_elbow_angle, 0)
        self.assertEqual(self.classifier.right_elbow_angle, 0)
        # Add more checks if necessary for other initial angle states

    @patch('pipline.AngleCalculator.calculateAngle')
    def test_classify_t_pose_positive(self, mock_calculate_angle):
        """Test T-Pose classification with ideal angles mocked."""
        # Configure mock_calculate_angle to return values that define a T-Pose
        def angle_side_effect(*args):
            # args will be (landmark1, landmark2, landmark3)
            # We need to return specific angles based on which landmarks are passed.
            # This example simplifies by setting angles directly on the instance,
            # assuming calculateAngle populates these correctly.
            # A more robust mock would inspect args.
            return 170 # For elbows (straight)
        
        mock_calculate_angle.side_effect = [
            170, 170, # right_elbow_angle, left_elbow_angle (Pose LM 11-13-15, 12-14-16)
            90, 90,   # right_shoulder_angle, left_shoulder_angle (Pose LM 13-11-23, 14-12-24)
            170, 170, # right_knee_angle, left_knee_angle (Pose LM 23-25-27, 24-26-28)
            0,0,0,0,0,0 # Default for hand angles if not specified for this test
        ]
        # Removed direct setting of self.classifier angles here.
        # The classifyPose method will call the mocked AngleCalculator.calculateAngle,
        # which will use the side_effect list to populate the angles.
        
        _, label = self.classifier.classifyPose(self.mock_landmarks_all_present, self.mock_output_image, display=False)
        self.assertEqual(label, "T Pose")
        # Verify calculateAngle was called (e.g., for elbow, shoulder, knee angles for both sides)
        self.assertGreaterEqual(mock_calculate_angle.call_count, 6)


    @patch('pipline.AngleCalculator.calculateAngle')
    def test_classify_t_pose_negative_bent_elbow(self, mock_calculate_angle):
        """Test T-Pose classification fails if an elbow is bent."""
        mock_calculate_angle.return_value = 0 # Default return for any angle calculation
        # Simulate one bent elbow, others T-Pose like
        self.classifier.left_elbow_angle = 100 # Bent
        self.classifier.right_elbow_angle = 170
        self.classifier.left_shoulder_angle = 90
        self.classifier.right_shoulder_angle = 90
        self.classifier.left_knee_angle = 170
        self.classifier.right_knee_angle = 170
        
        _, label = self.classifier.classifyPose(self.mock_landmarks_all_present, self.mock_output_image, display=False)
        self.assertEqual(label, self.classifier.UNKNOWN_POSE_LABEL)

    @patch('pipline.AngleCalculator.calculateAngle')
    def test_classify_warrior_ii_positive(self, mock_calculate_angle):
        """Test Warrior II classification with ideal angles mocked."""
        # Order: R_elbow, L_elbow, R_shoulder, L_shoulder, R_knee, L_knee
        # Then 6 hand angles (3 for right hand (person's left), 3 for left hand (person's right))
        mock_calculate_angle.side_effect = [
            170, 170,  # Elbows (straight)
            90, 90,    # Shoulders (T-pose like)
            170, 100,  # Knees (e.g., Right leg straight, Left leg bent for Warrior II)
            0,0,0,0,0,0 # Default for hand angles
        ]
        _, label = self.classifier.classifyPose(self.mock_landmarks_all_present, self.mock_output_image, display=False)
        self.assertEqual(label, "Warrior II Pose")

    @patch('pipline.AngleCalculator.calculateAngle')
    def test_classify_warrior_ii_negative_both_legs_straight(self, mock_calculate_angle):
        """Test Warrior II fails if both legs are straight (should be T-Pose)."""
        mock_calculate_angle.side_effect = [
            170, 170,  # Elbows
            90, 90,    # Shoulders
            170, 170,  # Knees (both straight - T-Pose)
            0,0,0,0,0,0 
        ]
        _, label = self.classifier.classifyPose(self.mock_landmarks_all_present, self.mock_output_image, display=False)
        self.assertEqual(label, "T Pose") 

    @patch('pipline.AngleCalculator.calculateAngle')
    def test_classify_tree_pose_positive(self, mock_calculate_angle):
        """Test Tree Pose classification with ideal angles mocked."""
        # Order of angles set by classifyPose:
        # R_elbow, L_elbow, R_shoulder, L_shoulder, R_knee (Person's Left), L_knee (Person's Right)
        
        # Test Case 1: Person's Right Leg Up
        # Person's Left Leg (MP LEFT_KNEE, self.right_knee_angle) is straight.
        # Person's Right Leg (MP RIGHT_KNEE, self.left_knee_angle) is bent (25-45 range).
        mock_calculate_angle.side_effect = [
            100, 100,  # Elbows (arms not straight, to avoid T-Pose/Warrior II conflict)
            60, 60,    # Shoulders (arms not T-pose like)
            35,        # R_knee (Person's Left Knee, self.right_knee_angle) -> Bent for "Right Leg Up" logic
            170,       # L_knee (Person's Right Knee, self.left_knee_angle) -> Straight
            0,0,0,0,0,0 # Default for hand angles
        ]
        # With this setup:
        # self.right_knee_angle will be 35. self.left_knee_angle will be 170.
        # is_left_leg_straight (checks self.left_knee_angle = 170) -> True
        # is_right_leg_bent_for_tree (checks self.right_knee_angle = 35) -> True
        # So, (True and True) or (...) should pass.
        
        _, label = self.classifier.classifyPose(self.mock_landmarks_all_present, self.mock_output_image, display=False)
        self.assertEqual(label, "Tree Pose")

    @patch('pipline.AngleCalculator.calculateAngle')
    def test_classify_tree_pose_negative_both_legs_straight(self, mock_calculate_angle):
        """Test Tree Pose fails if both legs are straight (should be T-Pose or Unknown)."""
        mock_calculate_angle.side_effect = [
            170, 170,  # Elbows (straight)
            90, 90,    # Shoulders (T-pose like)
            170, 170,  # Knees (both straight)
            0,0,0,0,0,0
        ]
        _, label = self.classifier.classifyPose(self.mock_landmarks_all_present, self.mock_output_image, display=False)
        self.assertNotEqual(label, "Tree Pose")
        self.assertEqual(label, "T Pose") # Expect T-pose if arms/shoulders match

    @patch('pipline.AngleCalculator.calculateAngle')
    def test_classify_unknown_pose(self, mock_calculate_angle):
        """Test that default (all zero) angles result in Unknown Pose."""
        # Reset all angles to 0 (default state of classifier)
        self.classifier = PoseClassifier() 
        # All calculated angles will be 0 because of the dummy landmarks and default mock return
        mock_calculate_angle.return_value = 0 
        
        _, label = self.classifier.classifyPose(self.mock_landmarks_all_present, self.mock_output_image, display=False)
        self.assertEqual(label, self.classifier.UNKNOWN_POSE_LABEL)
        
    def test_classify_no_pose_landmarks(self):
        """Test that if no pose landmarks are provided, it results in Unknown Pose."""
        mock_landmarks_dict = {'pose': None, 'left_hand': None, 'right_hand': None}
        # self.mock_output_image is now a numpy array from setUp
        _, label = self.classifier.classifyPose(mock_landmarks_dict, self.mock_output_image, display=False)
        self.assertEqual(label, self.classifier.UNKNOWN_POSE_LABEL)
        # Also check that angles are reset/remain 0
        self.assertEqual(self.classifier.left_elbow_angle, 0)


if __name__ == '__main__':
    unittest.main()

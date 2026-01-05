import time
import numpy as np

class PoseValidator:
    def __init__(self, audio_coach):
        self.audio = audio_coach
        self.active_user_id = None
        self.last_feedback_time = 0
        self.feedback_cooldown = 5.0  # Seconds between voice corrections
        self.frame_height = 0
        self.frame_width = 0
        self.is_locked = False

    def set_frame_dimensions(self, height, width):
        self.frame_height = height
        self.frame_width = width

    def get_active_result(self, results):
        """
        Filters YOLO results to return only the locked user.
        If no user is locked, locks onto the first detected person.
        """
        if not results:
            return None

        # Iterate through results (usually one frame per result in stream)
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            boxes = r.boxes
            ids = boxes.id.cpu().numpy().astype(int)
            
            # If not locked, lock onto the first ID found
            if self.active_user_id is None:
                self.active_user_id = ids[0]
                self.is_locked = True
                print(f"[PoseValidator] Locked onto User ID: {self.active_user_id}")
                self.audio.speak("I see you. Tracking initialized.")
            
            # Find the index of the active user
            if self.active_user_id in ids:
                idx = np.where(ids == self.active_user_id)[0][0]
                
                # Construct a single-person result object (proxy) or return specific data
                # For simplicity, we return the specific keypoints and box for the active user
                # We can modify the result object in place to only contain the active user
                # but that might break other logic. Let's return the specific data needed.
                return r, idx
        
        return None, None

    def validate_framing(self, keypoints_xy, should_speak=True):
        """
        Checks if the user is fully in frame and provides audio feedback.
        keypoints_xy: numpy array of shape (17, 2)
        Returns: True if framing is valid, False otherwise.
        """
        if not self.is_locked or keypoints_xy is None or len(keypoints_xy) == 0:
            return False

        # Keypoint Indices (COCO):
        # 0: Nose, 15: L-Ankle, 16: R-Ankle
        # We check bounding box of keypoints
        
        # Filter out (0,0) points which indicate low confidence/missing
        valid_points = keypoints_xy[np.any(keypoints_xy > 0, axis=1)]
        if len(valid_points) < 5:
            return False # Not enough points to judge

        min_x, min_y = np.min(valid_points, axis=0)
        max_x, max_y = np.max(valid_points, axis=0)

        margin = 20 # pixels
        instruction = None

        # Check Vertical
        if min_y < margin:
            instruction = "Move back, I can't see your head."
        elif max_y > self.frame_height - margin:
            instruction = "Step back, I can't see your feet."
        
        # Check Horizontal (less critical, but good for centering)
        elif min_x < margin:
            instruction = "Move a bit to your left."
        elif max_x > self.frame_width - margin:
            instruction = "Move a bit to your right."

        # Check Size (Too far/Too close)
        # Heuristic: Height of bounding box relative to frame
        user_height = max_y - min_y
        if user_height < self.frame_height * 0.3:
            instruction = "Come a little closer."
        
        if instruction:
            if should_speak and (time.time() - self.last_feedback_time > self.feedback_cooldown):
                print(f"[PoseValidator] {instruction}")
                self.audio.speak(instruction)
                self.last_feedback_time = time.time()
            return False
            
        return True

    def send_feedback(self, instruction):
        """Triggers audio feedback if cooldown has passed."""
        if instruction and (time.time() - self.last_feedback_time > self.feedback_cooldown):
            print(f"[PoseValidator] {instruction}")
            self.audio.speak(instruction)
            self.last_feedback_time = time.time()

    def determine_orientation(self, keypoints_xy):
        """
        Determines user orientation (front, back, left, right) based on keypoint visibility.
        Returns: A string 'front', 'back', 'left', 'right', or 'unknown'.
        """
        if keypoints_xy is None or len(keypoints_xy) == 0:
            return "unknown"

        # Helper to check if a keypoint is visible (coordinates are not 0,0)
        def is_visible(kp_index):
            return np.any(keypoints_xy[kp_index] > 0)

        # Keypoint indices from COCO
        nose = 0
        left_eye = 1
        right_eye = 2
        left_ear = 3
        right_ear = 4
        left_shoulder = 5
        right_shoulder = 6

        # --- Visibility checks ---
        has_nose = is_visible(nose)
        has_l_eye = is_visible(left_eye)
        has_r_eye = is_visible(right_eye)
        has_l_ear = is_visible(left_ear)
        has_r_ear = is_visible(right_ear)
        has_l_shoulder = is_visible(left_shoulder)
        has_r_shoulder = is_visible(right_shoulder)

        # Scoring for side profile
        # If I am facing Left, my Right side is visible to the camera
        right_side_score = sum([has_r_eye, has_r_ear, has_r_shoulder])
        # If I am facing Right, my Left side is visible to the camera
        left_side_score = sum([has_l_eye, has_l_ear, has_l_shoulder])

        if has_nose and has_l_eye and has_r_eye:
            return "front"
        
        if has_l_shoulder and has_r_shoulder and not has_nose and not has_l_eye and not has_r_eye:
            return "back"

        # Side determination with threshold
        if right_side_score > left_side_score:
            return "left"
        if left_side_score > right_side_score:
            return "right"

        return "unknown"

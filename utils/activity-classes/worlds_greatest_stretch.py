from utils.rep_counter import BaseActivity

class WorldsGreatestStretchActivity(BaseActivity):
    def match(self, exercise_name):
        return "world greatest stretch" in exercise_name.lower()

    def process(self, keypoints):
        """
        Tracks states for World Greatest Stretch.
        States: lunge elbow down, rotation hand up
        Sequence required for rep: Down -> Up -> Down
        """
        if len(keypoints) < 17:
            return False

        # Keypoints
        # 9: L-Wrist, 10: R-Wrist
        # 11: L-Hip, 12: R-Hip
        # 13: L-Knee, 14: R-Knee
        
        l_wrist, r_wrist = keypoints[9][:2], keypoints[10][:2]
        l_knee, r_knee = keypoints[13][:2], keypoints[14][:2]
        
        l_wrist_vis = self.is_visible(l_wrist)
        r_wrist_vis = self.is_visible(r_wrist)
        
        # Need at least one wrist to judge arm position
        if not (l_wrist_vis or r_wrist_vis):
            return False

        # Determine reference height (Knees)
        # Y-axis increases downwards (0 is top of screen)
        ref_y = 0
        count_ref = 0
        if self.is_visible(l_knee):
            ref_y += l_knee[1]
            count_ref += 1
        if self.is_visible(r_knee):
            ref_y += r_knee[1]
            count_ref += 1
            
        if count_ref == 0:
            # Fallback to hips if knees missing
            if self.is_visible(keypoints[11][:2]):
                ref_y += keypoints[11][1] + 100 
                count_ref += 1
            elif self.is_visible(keypoints[12][:2]):
                ref_y += keypoints[12][1] + 100
                count_ref += 1
        
        if count_ref == 0:
            return False
            
        avg_knee_y = ref_y / count_ref

        # State Detection
        is_rotated = False
        is_down = False

        # Rotation Logic: Hand high (low Y) or large vertical spread
        # Threshold: Wrist significantly above knee (e.g. 150px) or spread > 150px
        if l_wrist_vis and r_wrist_vis:
            if abs(l_wrist[1] - r_wrist[1]) > 150:
                is_rotated = True
        
        if not is_rotated:
            if l_wrist_vis and l_wrist[1] < avg_knee_y - 150:
                is_rotated = True
            elif r_wrist_vis and r_wrist[1] < avg_knee_y - 150:
                is_rotated = True

        # Down Logic: Hands below knee level
        hands_low = True
        if l_wrist_vis and l_wrist[1] < avg_knee_y: hands_low = False
        if r_wrist_vis and r_wrist[1] < avg_knee_y: hands_low = False
        
        if hands_low:
            is_down = True

        # State Machine Update
        if is_rotated:
            self.current_state = "rotation hand up"
            if self.stage == "DOWN":
                self.stage = "UP"
        
        elif is_down:
            self.current_state = "lunge elbow down"
            if self.stage == "UP":
                self.stage = "DOWN"
                self.reps += 1
                return True
            else:
                self.stage = "DOWN"
        
        return False

    def check_orientation(self, keypoints, current_orientation):
        """
        Ensures the user is facing the camera with their open side during the stretch.
        """
        # 9: L-Wrist, 10: R-Wrist
        l_wrist = keypoints[9]
        r_wrist = keypoints[10]
        
        # Check which hand is down (larger Y value means lower on screen)
        # We assume the hand on the floor is significantly lower than the other or hips
        if self.is_visible(l_wrist) and self.is_visible(r_wrist):
            if r_wrist[1] > l_wrist[1] + 50: # Right hand down
                # If Right hand is down, we rotate Left. We need to see the Left side (Orientation 'right')
                if current_orientation == "left":
                    return "Turn around, I need to see your open side."
            elif l_wrist[1] > r_wrist[1] + 50: # Left hand down
                if current_orientation == "right":
                    return "Turn around, I need to see your open side."
        return None
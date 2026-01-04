import numpy as np

class RepCounter:
    def __init__(self):
        self.reps = 0
        self.stage = "UP"  # UP or DOWN
        self.current_exercise = ""
        self.anchor_y = None # For vertical movement tracking

    def calculate_angle(self, a, b, c):
        """Calculates angle between three points (a, b, c)."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        return angle

    def count_squat(self, keypoints):
        """Counts squats using hip-knee-ankle angle."""
        # Use Left side (Hip-Knee-Ankle)
        # Ensure confidence is high enough (omitted for brevity, assume valid)
        if len(keypoints) > 15:
            hip = keypoints[11][:2]
            knee = keypoints[13][:2]
            ankle = keypoints[15][:2]
            
            angle = self.calculate_angle(hip, knee, ankle)
            
            # Squat Logic
            if angle > 160:
                self.stage = "UP"
            if angle < 90 and self.stage == "UP":
                self.stage = "DOWN"
                self.reps += 1
                return True # Rep counted
        return False

    def count_hinge(self, keypoints):
        """Counts hinge movements (RDL, Good Morning) using Shoulder-Hip-Knee angle."""
        # Shoulder (5), Hip (11), Knee (13)
        if len(keypoints) > 13:
            shoulder = keypoints[5][:2]
            hip = keypoints[11][:2]
            knee = keypoints[13][:2]
            
            angle = self.calculate_angle(shoulder, hip, knee)
            
            # Hinge Logic: Standing (180) -> Bent (<120) -> Standing
            if angle > 160:
                self.stage = "UP"
            if angle < 130 and self.stage == "UP":
                self.stage = "DOWN"
                self.reps += 1
                return True
        return False

    def count_bridge(self, keypoints):
        """Counts glute bridges using Shoulder-Hip-Knee angle."""
        if len(keypoints) > 13:
            shoulder = keypoints[5][:2]
            hip = keypoints[11][:2]
            knee = keypoints[13][:2]
            
            angle = self.calculate_angle(shoulder, hip, knee)
            
            # Bridge Logic: Floor (Flexed ~130) -> Bridge (Extended ~180)
            # Logic is inverse of squat/hinge (starts down/flexed, goes up/extended).
            if angle < 145:
                self.stage = "DOWN"
            if angle > 170 and self.stage == "DOWN":
                self.stage = "UP"
                self.reps += 1
                return True
        return False

    def count_curl(self, keypoints):
        """Counts hamstring curls using Hip-Knee-Ankle angle."""
        if len(keypoints) > 15:
            hip = keypoints[11][:2]
            knee = keypoints[13][:2]
            ankle = keypoints[15][:2]
            
            angle = self.calculate_angle(hip, knee, ankle)
            
            # Curl Logic: Straight (180) -> Curled (<90)
            if angle > 160:
                self.stage = "UP"
            if angle < 100 and self.stage == "UP":
                self.stage = "DOWN"
                self.reps += 1
                return True
        return False

    def count_calf_raise(self, keypoints):
        """Counts calf raises using vertical displacement of the nose/head."""
        # Simple heuristic: if nose goes up significantly then down.
        # This is hard without a fixed camera reference, but we assume static camera.
        # For now, we'll skip complex calibration and rely on visual cues or time, 
        # or just map to squat if the user moves their knees slightly.
        return False

    def process_pose(self, keypoints, exercise_name):
        """
        Heuristic rep counting based on exercise name.
        Keypoints expected: COCO format (17 keypoints).
        """
        # Map COCO Keypoints (Ultralytics default)
        # 5: L-Shoulder, 6: R-Shoulder
        # 11: L-Hip, 12: R-Hip
        # 13: L-Knee, 14: R-Knee
        # 15: L-Ankle, 16: R-Ankle
        
        ex_name = exercise_name.lower()
        
        if "squat" in ex_name or "lunge" in ex_name or "stretch" in ex_name:
            return self.count_squat(keypoints)
        elif "dead lift" in ex_name or "hinge" in ex_name or "row" in ex_name:
            return self.count_hinge(keypoints)
        elif "bridge" in ex_name:
            return self.count_bridge(keypoints)
        elif "curl" in ex_name:
            return self.count_curl(keypoints)
        elif "calf" in ex_name:
            return self.count_calf_raise(keypoints)
        
        return False
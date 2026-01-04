from utils.rep_counter import BaseActivity

class SquatActivity(BaseActivity):
    def match(self, exercise_name):
        ex_name = exercise_name.lower()
        # Exclude world greatest stretch which has its own handler
        if "world greatest stretch" in ex_name:
            return False
        return "squat" in ex_name or "lunge" in ex_name or "stretch" in ex_name

    def process(self, keypoints):
        """Counts squats using hip-knee-ankle angle."""
        # Use Left side (Hip-Knee-Ankle)
        if len(keypoints) > 15:
            hip = keypoints[11][:2]
            knee = keypoints[13][:2]
            ankle = keypoints[15][:2]
            
            if not (self.is_visible(hip) and self.is_visible(knee) and self.is_visible(ankle)):
                return False
            
            angle = self.calculate_angle(hip, knee, ankle)
            self.last_angle = angle
            
            # State Detection
            if angle > 160:
                self.current_state = "standing"
            elif angle < 100:
                self.current_state = "squat hold"
            elif self.stage == "UP":
                self.current_state = "descending"
            elif self.stage == "DOWN":
                self.current_state = "ascending"

            # Squat Logic
            if angle > 160:
                self.stage = "UP"
            if angle < 100 and self.stage == "UP": # Adjusted threshold slightly to match hold state
                self.stage = "DOWN"
                self.reps += 1
                return True # Rep counted
        return False

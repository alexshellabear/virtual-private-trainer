# Objective
A virtual trainer that explains my workout, gives me cues, counts my weight/reps/rest, provides cues based upon form.

# Method
- Use the inbuilt laptop camera and python to start recording me workout in my garage gym.
- Work outs will be stored in a json format with a folder structure to indicate different workout types and timing. The integer at the front of the json file or folder will determine order
    input-workouts | the parent folder
        ->  workout-name
            -> (INT)-(DAY/WEEK)
                -> (INT)-workout-description.json
- The workout-history folder will be used to save outputs such as an mp4 files with video snippets, a json file that logs events
    workout-history
        -> YYYY-MM-DD-HH-MM-SS-workout-name-(INT)-(DAY/WEEK)-(INT)
            -> workout.json | the running log file
- The program will have two states, "rest" and "active", it starts in rest and once the audio has entered an activity then it will be active.
- For rep counting and observing me work out use yolo11-pose-nano, this will keep track of counts, exercise type, rest etc.
- For heavy analytics the Google Cloud Run will be used to run VitPose++ to get highly accurate results as my laptop doesn't have good enough specs. The video will be ccropped and sent likely when there is system idle such as rest etc. The video file will be sent via post and it return json of pose data back.
- Gemini will be used to provide text and prompts to keep things lively and gTTS for the voice. At the start cues and scripts will be created. During rest periods and after deeper analytics gemini will be used to provide cues. The general format for the workout and voice should be stored in the input-workouts json. This will have the audio tag and follow a similar format to workout-audio-gen https://github.com/alexshellabear/workout-audio-gen
- cache common phrases from TTS engine to save time generating it later. Use the package hashlib to name the file name and store under audio-lib, use text-to-audio-dict.json to create lookup
    audio-lib
        ->  0a543cee85bbba1917dbe839bc863a02.file extension
            text-to-audio-dict.json | {"0a543cee85bbba1917dbe839bc863a02": {"text","Good morning!","config": "gTTS native"}}
- selecting the workout. it should ask the user via the terminal to select the workout, then pass the int for the week selection then the int for the specific workout for the first time. Once there is workout history the program will go to the next one (+1) but also give the user an option to change via the terminal asking to select current and print the name or re-select workout
- markdown is provided and stored as this is a good reference which can be converted into json manually
- pose_validator.py, this module will check if all joints are captured in the video. It will give audio instructions to frame the video.
    - I'm likely to start the video with one person looking at the camera, capture their face and store it for object tracking, potentially my wife might walk into the video field of view later, ignore the second person detected by YOLO
    - I want all joint positions in the video, give instructions of where i should move or to move the camera, maybe towards the camera, away from the camera etc.
    - Add audio to audio que
    - Pose validator has to be fired first in order to progress to the next section, in order to move from rest to active state the pose validator must be valid. Let the audio que finish before inserting new commands to adjust the screen. Also when active don't provide audio adjustments, only for the transition period from rest to active.
    - determine if I'm side or face on, different angles will affect rep counting. Determine if I am my left side, right side,front or back towards the camera.
    - Audio to play guiding me through each movement in real time and use this to generate training data.
        - input json audio is an array that is an array of strings with occasional dictionaries that have different actions
        - {"break":int}, play silence for this int period of time 
        - Audibly count the reps for the user, "1, 5 more to go", "2, 4 more to go"
        - When resting indicate to the user how long the rest is, aka "2 minutes rest"
        - {"identifier": "text"} - These are custom identifiers used by the rep counter class to provide cues and give input to the user, wait until that position has been achieved before playing the next
            e.g. for a squat the cues might be
                ...
                { "standing": "Stand with feet shoulder-width apart, core engaged." },
                { "descending": "Sit back into your lowest position, keeping your chest up." },
                { "squat hold": "Hold for a second at the bottom." },
                { "ascending": "Stand up activating your glutes and driving through your heels."},
                ...
                The audio would be read and then waiting a few seconds for the user to enter that portion of the exercise
- update worlds_greatest_stretch and other relevant code files so that if I start with my right hand down then I should be prompted to be facing my left hand side to the camera. Likewise the opposite if it is the opposite side. 
- for reps_per_side_grouped all x reps should be finished before switching to the next side.
- for single sided exercises there will also be a tag "side": "left" or "side": "right" added to the audio elements

# TODO
- Implement training routine so that if an activity does not have a name 
- reduce size of text on printout screen, if activity name is long the reps are printed off screen, put reps somewhere else to not suffer from overflow
- orientation doesn't update for each frame
- 
- Should get wide lense so you don't have to stand in the doorway so far away from the screen.
- use https://github.com/SajjadAemmi/YOLOv8-Pose-Classification for pose classification and rep counting. YOLO-Pose-Classification & Rep-Counting: This project uses a k-Nearest Neighbors (k-NN) algorithm rather than hardcoded angles. It is excellent for generalized movement because you can "train" it on new exercises by simply recording a few frames of the "up" and "down" positions. 
    - for each activity that doesn't have rep_counter get the user to record themselves doing the exercise and use the microphone to listen to what the name of the state is
    - for anything that is predefined in here https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/workouts-monitoring.md use this as the first pass
    - Classes will be stored in utils/activity-classes and if they are a ML approach then use utils/activity-data for storing the recorded data


# Long term TODO
- use Anrold Schwarztnegers voice

# Assumptions
- There will only be one person in a frame
- laptop system specs - lenovo thinkpad e16 gen 2
    Processor: Intel(R) Core(TM) Ultra 5 125U (1.30 GHz)
    RAM: 16.0 GB (15.5 GB usable)
    NPU: Intel(R) AI Boost, total memory 8.8Gb / shared memory 8.8Gb
    GPU: Intel(R) Graphics, total memory 8.8Gb / shared memory 8.8Gb
    Integrated camera: SunplusIT, can do 1080p 16:9 30fps
- I like the metric system, I don't want to see any pounds or fathoms in my code base

# Potential future expansions
- Use multiple cameras over wifi
- Use a better TTS system such as Google Text to Speech or something better on device

# Set up issues
- ran into not having a dll when trying to import YOLO, ensure you've downloaded C++ https://aka.ms/vc14/vc_redist.x64.exe
- when I plug in the HDMI cord it switched to a different speaker that may have been connected to the screen
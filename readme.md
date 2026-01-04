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

# TODO
- pose_validator.py, this module will check if all joints are captured in the video. It will give audio instructions to frame the video.
    - I'm likely to start the video with one person looking at the camera, capture their face and store it for object tracking, potentially my wife might walk into the video field of view later, ignore the second person detected by YOLO
    - I want all joint positions in the video, give instructions of where i should move or to move the camera, maybe towards the camera, away from the camera etc.
    - Add audio to audio que
    - Pose validator has to be fired first in order to progress to the next section, in order to move from rest to active state the pose validator must be valid. Let the audio que finish before inserting new commands to adjust the screen. Also when active don't provide audio adjustments, only for the transition period from rest to active.
- Audio to play guiding me through each movement in real time.

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
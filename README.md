Fall Detection Using Human Pose Estimation

Objective:

Fall detection project using MediaPipe Pose and OpenCV. The script processes video frame-by-frame, 
extracts skeleton keypoints, calculates vertical distances and pose angles, and detects fall events 
with timestamps. Includes annotated video output, fall detection logs, and a detailed report of 
results. 

Tech Stack:

•	Python

•	MediaPipe Pose – Human pose and skeleton keypoint extraction

•	OpenCV – Video processing and annotation

•	NumPy – Mathematical computations and confidence analysis

Project Structure:

├pose_detection.py          # Main pose & fall detection script

├requirements.txt          # Project dependencies

├input_video.mp4        # Input video given

├annotated_video.mp4        # Output video with skeleton overlay

├fall_detection_log.txt     # Detected falls with timestamps

├detection_report.txt       # Analysis & reliability report

├README.md                  # Project documentation

How the System Works:

1. The video is processed frame-by-frame
2. MediaPipe Pose extracts human body keypoints
3. Important joints are analyzed:

   o	Shoulders
  	
   o	Hips
  	
   o	Knees
  	
   o	Ankles
  	
4. The system computes:
   
    o Body angle using shoulder–hip orientation
  	
    o Vertical distance between head and feet
  	
5. A fall is detected when:

    o The body becomes nearly horizontal
  	
    o The person is close to the ground
  	
    o The posture is classified as lying
  	
6. All detected falls are logged and analyzed for confidence and reliability

Fall Detection Logic:

The system uses an explainable, rule-based approach instead of a black-box model.

Detection Criteria:

• Body_Angle

   o	Standing → small angle
    
   o	Lying → large angle (~90°)
 
• Vertical Distance (Head to Feet)

   o	Large → standing

   o	Small → person on ground

• Posture Classification

  o	Standing
    
  o	Bent
    
  o	Lying

A fall is detected only when all criteria are satisfied, reducing false positives.

How to Run the Project:

1.Clone the Repository

  git clone https://github.com/Shekinah579/fall-detection
    
  cd fall-detection
    
2.Install Dependencies

  pip install -r requirements.txt
    
3.Run the Script

   python pose_detection.py --video input_video.mp4 
   
4.Generated Outputs
   
 After execution, the following files will be created:
   
 • annotated_video.mp4
   
 • fall_detection_log.txt
   
 • detection_report.txt

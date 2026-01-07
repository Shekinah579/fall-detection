
"""
Fall Detection Using Human Pose Estimation
Task 2 Implementation - MediaPipe Pose Analysis
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import os
import argparse
from datetime import datetime

class FallDetector:
    def __init__(self, video_path, output_video_path="annotated_video.mp4"):
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.fall_events = []
        self.fall_cooldown = 0
        self.cooldown_frames = 30
        
        # Initialize MediaPipe Pose Landmarker
        model_path = 'pose_landmarker_lite.task'
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(base_options=base_options)
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        
        # Keypoint indices for MediaPipe Pose
        self.KEYPOINTS = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }

    def extract_keypoints(self, landmarks):
        """Extract body keypoints (shoulders, hips, knees, ankles)"""
        if not landmarks:
            return None
        
        keypoints = {}
        for name, idx in self.KEYPOINTS.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                keypoints[name] = {
                    'x': lm.x, 'y': lm.y, 'z': lm.z,
                    'visibility': lm.visibility
                }
        return keypoints

    def calculate_vertical_distance(self, keypoints):
        """Calculate vertical distance between head and feet"""
        if not keypoints or 'nose' not in keypoints:
            return None
        
        head_y = keypoints['nose']['y']
        
        # Get feet positions (ankles)
        feet_points = []
        for ankle in ['left_ankle', 'right_ankle']:
            if ankle in keypoints and keypoints[ankle]['visibility'] > 0.5:
                feet_points.append(keypoints[ankle]['y'])
        
        if not feet_points:
            return None
        
        # Use lowest foot point
        lowest_foot_y = max(feet_points)
        vertical_distance = lowest_foot_y - head_y
        
        return vertical_distance

    def calculate_body_angle(self, keypoints):
        """Calculate body orientation angle"""
        if not keypoints:
            return None
        
        required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        if not all(point in keypoints for point in required_points):
            return None
        
        # Calculate center points
        shoulder_center_x = (keypoints['left_shoulder']['x'] + keypoints['right_shoulder']['x']) / 2
        shoulder_center_y = (keypoints['left_shoulder']['y'] + keypoints['right_shoulder']['y']) / 2
        hip_center_x = (keypoints['left_hip']['x'] + keypoints['right_hip']['x']) / 2
        hip_center_y = (keypoints['left_hip']['y'] + keypoints['right_hip']['y']) / 2
        
        # Calculate angle from vertical
        dx = shoulder_center_x - hip_center_x
        dy = shoulder_center_y - hip_center_y
        
        angle = abs(math.degrees(math.atan2(dx, -dy)))
        return angle

    def classify_posture(self, keypoints):
        """Analyze pose angles to classify postures (standing, bent, lying)"""
        body_angle = self.calculate_body_angle(keypoints)
        
        if body_angle is None:
            return "unknown", 0.0
        
        # Calculate confidence based on keypoint visibility
        visibilities = []
        for point in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            if point in keypoints:
                visibilities.append(keypoints[point]['visibility'])
        
        confidence = np.mean(visibilities) if visibilities else 0.0
        
        # Classify posture
        if body_angle > 70:
            return "lying", confidence
        elif body_angle > 30:
            return "bent", confidence
        else:
            return "standing", confidence

    def detect_fall(self, keypoints, frame_number, fps):
        """Detect falls (person near ground = fall event)"""
        if not keypoints:
            return False, None
        
        vertical_distance = self.calculate_vertical_distance(keypoints)
        body_angle = self.calculate_body_angle(keypoints)
        posture, confidence = self.classify_posture(keypoints)
        
        fall_info = {
            'frame': frame_number,
            'timestamp': frame_number / fps,
            'posture': posture,
            'body_angle': body_angle,
            'vertical_distance': vertical_distance,
            'confidence': confidence
        }
        
        # Fall detection criteria - Stricter thresholds to reduce false positives
        fall_detected = False
        if self.fall_cooldown == 0:
            # Criterion 1: Body angle indicates severe lying (>80° instead of 60°)
            angle_criterion = body_angle is not None and body_angle > 80
            
            # Criterion 2: Small vertical distance (person near ground)
            distance_criterion = vertical_distance is not None and vertical_distance < 0.3
            
            # Criterion 3: Posture classified as lying
            posture_criterion = posture == "lying"
            
            # Fall detected only if ALL criteria met (stricter)
            criteria_met = sum([angle_criterion, distance_criterion, posture_criterion])
            
            if criteria_met >= 3:  # All 3 criteria must be met
                fall_detected = True
                self.fall_cooldown = 60  # Longer cooldown
                fall_info['criteria_met'] = criteria_met
                fall_info['detection_confidence'] = min(1.0, criteria_met / 3 * confidence)
        else:
            self.fall_cooldown -= 1
        
        return fall_detected, fall_info

    def draw_skeleton_overlay(self, frame, landmarks):
        """Generate skeleton overlay on video frames"""
        if not landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        # Define skeleton connections
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (23, 25), (25, 27), (24, 26), (26, 28)   # Legs
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                start_x, start_y = int(start.x * w), int(start.y * h)
                end_x, end_y = int(end.x * w), int(end.y * h)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Draw keypoints
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        return frame

    def process_video(self):
        """Process video frame-by-frame"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return None
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer for annotated output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        frame_number = 0
        print(f"Processing {total_frames} frames at {fps} FPS...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect pose
            try:
                result = self.detector.detect(mp_image)
                
                fall_detected = False
                posture = "no_person"
                
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    keypoints = self.extract_keypoints(landmarks)
                    
                    # Detect falls
                    fall_detected, fall_info = self.detect_fall(keypoints, frame_number, fps)
                    
                    if fall_info:
                        posture = fall_info['posture']
                    
                    if fall_detected:
                        self.fall_events.append(fall_info)
                        print(f"FALL DETECTED at {fall_info['timestamp']:.2f}s!")
                    
                    # Draw skeleton overlay
                    frame = self.draw_skeleton_overlay(frame, landmarks)
                
                # Add annotations
                if fall_detected:
                    cv2.putText(frame, "FALL DETECTED!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                
                cv2.putText(frame, f"Posture: {posture}", (10, height - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Time: {frame_number/fps:.1f}s", (10, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}")
            
            # Write annotated frame
            out.write(frame)
            frame_number += 1
            
            # Progress indicator
            if frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        print(f"\\nProcessing complete! Processed {frame_number} frames")
        print(f"Falls detected: {len(self.fall_events)}")
        print(f"Annotated video saved: {self.output_video_path}")
        
        return {
            'total_frames': frame_number,
            'fps': fps,
            'duration': frame_number / fps,
            'fall_events': self.fall_events
        }

    def generate_fall_log(self, log_path="fall_detection_log.txt"):
        """Log all detected falls with timestamps"""
        with open(log_path, 'w') as f:
            f.write("FALL DETECTION LOG\\n")
            f.write("=" * 50 + "\\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Video: {self.video_path}\\n")
            f.write(f"Total Falls: {len(self.fall_events)}\\n\\n")
            
            if self.fall_events:
                for i, event in enumerate(self.fall_events, 1):
                    f.write(f"Fall #{i}:\\n")
                    f.write(f"  Timestamp: {event['timestamp']:.2f} seconds\\n")
                    f.write(f"  Frame: {event['frame']}\\n")
                    f.write(f"  Posture: {event['posture']}\\n")
                    f.write(f"  Body Angle: {event['body_angle']:.1f}°\\n")
                    f.write(f"  Confidence: {event.get('detection_confidence', 0):.2f}\\n\\n")
            else:
                f.write("No falls detected.\\n")
        
        print(f"Fall log saved: {log_path}")

    def generate_report(self, results, report_path="detection_report.txt"):
        """Generate analysis report with confidence/reliability"""
        with open(report_path, 'w') as f:
            f.write("FALL DETECTION ANALYSIS REPORT\\n")
            f.write("=" * 60 + "\\n\\n")
            
            # Video Information
            f.write("VIDEO ANALYSIS:\\n")
            f.write(f"  File: {self.video_path}\\n")
            f.write(f"  Duration: {results['duration']:.1f} seconds\\n")
            f.write(f"  Total Frames: {results['total_frames']}\\n")
            f.write(f"  Frame Rate: {results['fps']:.1f} FPS\\n\\n")
            
            # Fall Detection Results
            f.write("FALL DETECTION RESULTS:\\n")
            f.write(f"  Number of falls detected: {len(self.fall_events)}\\n")
            
            if self.fall_events:
                f.write(f"  Fall rate: {len(self.fall_events) / (results['duration'] / 60):.2f} falls/minute\\n\\n")
                
                f.write("TIMESTAMPS OF EACH FALL:\\n")
                for i, event in enumerate(self.fall_events, 1):
                    f.write(f"  {i}. {event['timestamp']:.2f}s (Frame {event['frame']})\\n")
                
                f.write("\\nCONFIDENCE/RELIABILITY ANALYSIS:\\n")
                confidences = [e.get('detection_confidence', 0) for e in self.fall_events]
                avg_confidence = np.mean(confidences) if confidences else 0
                f.write(f"  Average Detection Confidence: {avg_confidence:.2%}\\n")
                f.write(f"  Minimum Confidence: {min(confidences):.2%}\\n")
                f.write(f"  Maximum Confidence: {max(confidences):.2%}\\n")
                
                # Reliability assessment
                if avg_confidence > 0.8:
                    reliability = "HIGH"
                elif avg_confidence > 0.6:
                    reliability = "MEDIUM"
                else:
                    reliability = "LOW"
                f.write(f"  Overall Reliability: {reliability}\\n")
            else:
                f.write("  No falls detected in this video.\\n")
            
            f.write("\\nDETECTION METHODOLOGY:\\n")
            f.write("  - MediaPipe Pose estimation for keypoint extraction\\n")
            f.write("  - Body angle analysis (>60° indicates lying)\\n")
            f.write("  - Vertical distance measurement (head to feet)\\n")
            f.write("  - Multi-criteria fall detection algorithm\\n")
            f.write("  - Cooldown period to prevent duplicate detections\\n")
        
        print(f"Detection report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Fall Detection Using Human Pose Estimation')
    parser.add_argument('--video', '-v', default='input_video.mp4', help='Input video file')
    parser.add_argument('--output', '-o', default='annotated_video.mp4', help='Output annotated video')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found!")
        return
    
    # Initialize fall detector
    detector = FallDetector(args.video, args.output)
    
    # Process video
    results = detector.process_video()
    
    if results:
        # Generate all required deliverables
        detector.generate_fall_log("fall_detection_log.txt")
        detector.generate_report(results, "detection_report.txt")
        
        print("\\n" + "="*50)
        print("ALL DELIVERABLES GENERATED:")
        print("✓ pose_detection.py - Main script")
        print("✓ fall_detection_log.txt - Fall timestamps")
        print("✓ annotated_video.mp4 - Video with skeleton overlay")
        print("✓ detection_report.txt - Analysis report")
        print("="*50)

if __name__ == "__main__":
    main()


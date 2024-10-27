import os
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Dict, List
import json
from tqdm import tqdm
import pandas as pd
from scipy import stats

@dataclass
class AngleStats:
    mean: float
    std: float
    min: float
    max: float
    confidence_interval: tuple
    
class YogaAngleAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # Define the angles we want to track
        self.angle_definitions = {
            # Arm angles
            'right_arm': [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_WRIST
            ],
            'left_arm': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST
            ],
            
            # Leg angles
            'right_leg': [
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ],
            'left_leg': [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE
            ],
            
            # Hip angles
            'right_hip': [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE
            ],
            'left_hip': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE
            ],
            
            # Back angles
            'upper_back': [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ],
            'lower_back': [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ],
            
            # Shoulder angles
            'right_shoulder': [
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ],
            'left_shoulder': [
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP
            ],
            
            # Additional angles
            'hip_width': [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE
            ],
            'shoulder_width': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW
            ],
            'torso_twist': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ],
            'neck_tilt': [
                self.mp_pose.PoseLandmark.RIGHT_EAR,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ],
            
            # Ankle and wrist flexion
            'right_ankle_flex': [
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
            ],
            'left_ankle_flex': [
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX
            ],
            'right_wrist_flex': [
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_INDEX
            ],
            'left_wrist_flex': [
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.LEFT_INDEX
            ],
            
            # Pelvis and spine
            'pelvis_tilt': [
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER
            ],
            'spine_alignment': [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP
            ]
        }
        
    def calculate_angle(self, p1, p2, p3) -> float:
        """Calculate angle between three points in 3D space"""
        a = np.array([p1.x, p1.y, p1.z])
        b = np.array([p2.x, p2.y, p2.z])
        c = np.array([p3.x, p3.y, p3.z])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def process_image(self, image_path: str) -> Dict[str, float]:
        """Process a single image and return all calculated angles"""
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        if not results.pose_landmarks:
            return None
            
        angles = {}
        for angle_name, landmarks in self.angle_definitions.items():
            try:
                angle = self.calculate_angle(
                    results.pose_landmarks.landmark[landmarks[0].value],
                    results.pose_landmarks.landmark[landmarks[1].value],
                    results.pose_landmarks.landmark[landmarks[2].value]
                )
                angles[angle_name] = angle
            except:
                angles[angle_name] = None
                
        return angles
    
    def analyze_dataset(self, data_dir: str) -> Dict[str, Dict[str, AngleStats]]:
        """Analyze entire dataset and compute statistics for each pose"""
        pose_data = {}
        
        # Process all images
        print("Processing images...")
        for pose_name in os.listdir(data_dir):
            pose_path = os.path.join(data_dir, pose_name)
            if not os.path.isdir(pose_path):
                continue
                
            pose_angles = {angle_name: [] for angle_name in self.angle_definitions.keys()}
            
            # Process each image in the pose directory
            for img_name in tqdm(os.listdir(pose_path), desc=f"Processing {pose_name}"):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(pose_path, img_name)
                angles = self.process_image(img_path)
                
                if angles is not None:
                    for angle_name, angle_value in angles.items():
                        if angle_value is not None:
                            pose_angles[angle_name].append(angle_value)
            
            # Calculate statistics for each angle
            pose_stats = {}
            for angle_name, angle_values in pose_angles.items():
                if angle_values:
                    mean = np.mean(angle_values)
                    std = np.std(angle_values)
                    ci = stats.t.interval(0.95, len(angle_values)-1, loc=mean, scale=std/np.sqrt(len(angle_values)))
                    pose_stats[angle_name] = AngleStats(
                        mean=float(mean),
                        std=float(std),
                        min=float(np.min(angle_values)),
                        max=float(np.max(angle_values)),
                        confidence_interval=(float(ci[0]), float(ci[1]))
                    )
            
            pose_data[pose_name] = pose_stats
        
        return pose_data
    
    def generate_report(self, pose_data: Dict[str, Dict[str, AngleStats]], output_file: str):
        """Generate a detailed report of the angle analysis"""
        report = []
        report.append("# Yoga Pose Angle Analysis Report\n")
        
        for pose_name, pose_stats in pose_data.items():
            report.append(f"\n## {pose_name}\n")
            report.append("| Angle | Mean | Std Dev | Range | 95% Confidence Interval |")
            report.append("|-------|------|---------|--------|------------------------|")
            
            for angle_name, stats in pose_stats.items():
                report.append(
                    f"| {angle_name} | {stats.mean:.1f}° | {stats.std:.1f}° | "
                    f"{stats.min:.1f}° - {stats.max:.1f}° | "
                    f"{stats.confidence_interval[0]:.1f}° - {stats.confidence_interval[1]:.1f}° |"
                )
        
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
            
        # Also save as CSV for easier data analysis
        csv_data = []
        for pose_name, pose_stats in pose_data.items():
            for angle_name, stats in pose_stats.items():
                csv_data.append({
                    'pose': pose_name,
                    'angle': angle_name,
                    'mean': stats.mean,
                    'std': stats.std,
                    'min': stats.min,
                    'max': stats.max,
                    'ci_lower': stats.confidence_interval[0],
                    'ci_upper': stats.confidence_interval[1]
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file.replace('.md', '.csv'), index=False)
        
        # Save raw data as JSON
        json_data = {
            pose_name: {
                angle_name: vars(stats)
                for angle_name, stats in pose_stats.items()
            }
            for pose_name, pose_stats in pose_data.items()
        }
        
        with open(output_file.replace('.md', '.json'), 'w') as f:
            json.dump(json_data, f, indent=2)

    def visualize_pose(self, image_path: str, output_path: str = None):
        """Visualize the pose landmarks and angles on an image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        if not results.pose_landmarks:
            return None
            
        # Draw landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # Calculate and draw angles
        angles = self.process_image(image_path)
        if angles:
            height, width = image.shape[:2]
            for angle_name, angle_value in angles.items():
                if angle_value is not None:
                    # Get middle point of the angle
                    landmarks = self.angle_definitions[angle_name]
                    mid_point = results.pose_landmarks.landmark[landmarks[1]]
                    x = int(mid_point.x * width)
                    y = int(mid_point.y * height)
                    
                    # Draw angle value
                    cv2.putText(
                        image,
                        f'{angle_name}: {angle_value:.1f}°',
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
        
        # Convert back to BGR for saving
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image

def main():
    # Initialize analyzer
    analyzer = YogaAngleAnalyzer()
    
    # Set your dataset directory
    data_dir = "./yoga_dataset"
    
    # Analyze dataset
    pose_data = analyzer.analyze_dataset(data_dir)
    
    # Generate report
    analyzer.generate_report(pose_data, "yoga_pose_analysis.md")
    
    # Print some example insights
    print("\nExample optimal angles for poses:")
    for pose_name, pose_stats in pose_data.items():
        print(f"\n{pose_name}:")
        for angle_name, stats in pose_stats.items():
            print(f"  {angle_name}: {stats.mean:.1f}° ± {stats.std:.1f}° "
                  f"(range: {stats.min:.1f}° - {stats.max:.1f}°)")

if __name__ == "__main__":
    main()
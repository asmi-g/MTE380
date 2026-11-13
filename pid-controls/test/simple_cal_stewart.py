# Simple Auto Calibration System for Stewart Platform Control
# Interactive calibration tool for geometry, platform leveling, and 3-servo limits
# Generates config.json file compatible with ball tracking controller

import cv2
import numpy as np
import json
import math
import serial
import time
from datetime import datetime

class StewartAutoCalibrator:
    """Interactive calibration system for 3-DOF Stewart platform setup."""
    
    def __init__(self):
        # Physical system parameters (Platform / Workspace radius or reference length)
        self.PLATFORM_RADIUS_M = 0.15  # Adjust based on the actual platform's radius
        
        # Camera configuration
        self.CAM_INDEX = 1  # 0 for laptop cam, 1 for external cam
        self.FRAME_W, self.FRAME_H = 640, 480
        
        # Calibration state tracking
        self.current_frame = None
        self.phase = "geometry"  # "geometry" -> "neutral_tuning" -> "limits" -> "complete"
        
        # Geometry calibration data (e.g., center reference or platform boundaries)
        self.peg_points = []
        self.pixel_to_meter_ratio = None
        
        # Servo hardware configuration matching teammate's .ino structure
        self.servo = None
        self.servo_port = "COM5"  # Update to your port
        self.baud_rate = 115200   # Matches teammate's baud rate
        
        # 3-Servo Angle States (Neutral set around 90 degrees based on your arm attachment)
        self.neutral_angles = [90.0, 90.0, 90.0]
        self.current_angles = [90.0, 90.0, 90.0]
        
        # Position limit results
        self.position_min = None
        self.position_max = None

    def connect_servo(self):
        """Establish serial connection to Arduino running servo_controller_stewart.ino."""
        try:
            self.servo = serial.Serial(self.servo_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Allow Arduino auto-reset / connection to stabilize
            print("[SERVO] Connected successfully on", self.servo_port)
            return True
        except Exception as e:
            print(f"[SERVO] Failed to connect: {e}")
            return False

    def send_servo_angles(self, angles):
        """Send 3 absolute joint angles using teammate's 'A,angle1,angle2,angle3' protocol."""
        if self.servo and self.servo.is_open:
            # Clip angles to safe mechanical range so that the motor doesnt over-rotates
            clipped = [np.clip(a, 60.0, 120.0) for a in angles]
            command = f"A,{clipped[0]:.2f},{clipped[1]:.2f},{clipped[2]:.2f}\n"
            self.servo.write(command.encode('utf-8'))

    def send_neutral(self):
        """Send neutral command 'N' to Arduino."""
        if self.servo and self.servo.is_open:
            self.servo.write(b"N\n") #all servos to go back to default pos

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for geometry reference selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "geometry" and len(self.peg_points) < 2:
                self.peg_points.append((x, y))
                print(f"[GEO] Point {len(self.peg_points)} selected at ({x}, {y})")
                if len(self.peg_points) == 2:
                    self.calculate_geometry()

    def calculate_geometry(self):
        """Calculate pixel-to-meter conversion ratio."""
        p1, p2 = self.peg_points
        pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if pixel_distance > 0:
            self.pixel_to_meter_ratio = self.PLATFORM_RADIUS_M / pixel_distance
            print(f"[GEO] Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f}")
        self.phase = "neutral_tuning"

    def detect_ball_position(self, frame):
        """Detect ball position using teammate's color thresholds from config or defaults."""
        if self.pixel_to_meter_ratio is None: 
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Default orange HSV range matching ball_detection_stewart.py
        lower = np.array([5, 150, 150], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        if radius < 5:
            return None
        
        center_x = frame.shape[1] // 2
        pixel_offset = x - center_x
        return pixel_offset * self.pixel_to_meter_ratio

    def find_limits_automatically(self):
        """Sweep all 3 servos simultaneously to evaluate workspace range."""
        if not self.servo:
            print("[LIMITS] No servo connection - skipping sweep.")
            self.position_min = -0.05
            self.position_max = 0.05
            return
        
        print("[LIMITS] Sweeping 3-servo platform to collect data...")
        positions = []
        
        # Test offset angles relative to neutral
        test_offsets = [-5.0, 0.0, 5.0]
        for offset in test_offsets:
            target = [n + offset for n in self.neutral_angles]
            self.send_servo_angles(target)
            time.sleep(1.5)  # Wait for platform to settle
            
            angle_positions = []
            start_time = time.time()
            while time.time() - start_time < 1.0:
                ret, frame = self.cap.read()
                if ret:
                    pos = self.detect_ball_position(frame)
                    if pos is not None:
                        angle_positions.append(pos)
                time.sleep(0.05)
            
            if angle_positions:
                avg_pos = np.mean(angle_positions)
                positions.append(avg_pos)
                print(f"[LIMITS] Offset {offset:+4.1f}deg: Measured pos {avg_pos:.4f}m")
        
        self.send_neutral()
        
        if len(positions) >= 2:
            self.position_min = min(positions)
            self.position_max = max(positions)
            print(f"[LIMITS] Workspace Range: {self.position_min:.4f}m to {self.position_max:.4f}m")
        else:
            self.position_min = -0.05
            self.position_max = 0.05
            print("[LIMITS] Insufficient ball tracking data during sweep; using default bounds.")

    def save_config(self):
        """Save calibration output to config.json for teammate's modules."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "camera": {
                "index": int(self.CAM_INDEX),
                "frame_width": int(self.FRAME_W),
                "frame_height": int(self.FRAME_H)
            },
            "calibration": {
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None,
                "position_min_m": float(self.position_min) if self.position_min else None,
                "position_max_m": float(self.position_max) if self.position_max else None
            },
            "servo": {
                "port": str(self.servo_port),
                "neutral_angles": self.neutral_angles
            }
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("[SAVE] Configuration saved to config.json successfully.")

    def draw_overlay(self, frame):
        """Render GUI instructions and real-time status."""
        overlay = frame.copy()
        phase_instructions = {
            "geometry": "Click 2 reference points on platform workspace bounds.",
            "neutral_tuning": "Use W/S, A/D, Z/C to level platform. Press 'l' when level.",
            "limits": "Press 'l' to run automatic 3-servo sweep limits.",
            "complete": "Calibration complete! Press 's' to save config."
        }
        
        cv2.putText(overlay, f"Phase: {self.phase}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, phase_instructions.get(self.phase, ""), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if self.phase == "neutral_tuning":
            status_str = f"Angles -> S1:{self.current_angles[0]:.1f} S2:{self.current_angles[1]:.1f} S3:{self.current_angles[2]:.1f}"
            cv2.putText(overlay, status_str, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(overlay, "Controls: [1/2] S1 +/- | [3/4] S2 +/- | [5/6] S3 +/-", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        for i, pt in enumerate(self.peg_points):
            cv2.circle(overlay, pt, 6, (0, 255, 0), -1)
            cv2.putText(overlay, f"Pt {i+1}", (pt[0]+8, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if len(self.peg_points) == 2:
            cv2.line(overlay, self.peg_points[0], self.peg_points[1], (255, 0, 0), 2)
            
        return overlay

    def run(self):
        self.cap = cv2.VideoCapture(self.CAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        
        cv2.namedWindow("Stewart Calibration")
        cv2.setMouseCallback("Stewart Calibration", self.mouse_callback)
        
        self.connect_servo()
        
        print("[INFO] Stewart Platform Calibration Started")
        print("Step 1: Click 2 points on the frame for geometry scale.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.current_frame = frame
            
            display = self.draw_overlay(frame)
            cv2.imshow("Stewart Calibration", display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif self.phase == "neutral_tuning":
                # Manual fine-tuning for leveling all 3 servos individually
                step = 1.0
                if key == ord('1'): self.current_angles[0] += step
                elif key == ord('2'): self.current_angles[0] -= step
                elif key == ord('3'): self.current_angles[1] += step
                elif key == ord('4'): self.current_angles[1] -= step
                elif key == ord('5'): self.current_angles[2] += step
                elif key == ord('6'): self.current_angles[2] -= step
                elif key == ord('l'):
                    # Lock current fine-tuned angles as the true neutral level position
                    self.neutral_angles = list(self.current_angles)
                    print(f"[INFO] Neutral level locked at: {self.neutral_angles}")
                    self.phase = "limits"
                    continue
                self.send_servo_angles(self.current_angles)
                
            elif key == ord('l') and self.phase == "limits":
                self.find_limits_automatically()
                self.phase = "complete"
            elif key == ord('s') and self.phase == "complete":
                self.save_config()
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        if self.servo and self.servo.is_open:
            self.servo.close()

if __name__ == "__main__":
    calibrator = StewartAutoCalibrator()
    calibrator.run()
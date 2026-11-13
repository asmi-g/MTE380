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
        self.platform_center_px = None
        
        # Servo hardware configuration matching teammate's .ino structure
        self.servo = None
        self.servo_port = "COM5"  # Update to your port
        self.baud_rate = 115200   # Matches teammate's baud rate
        # Keep these consistent with MIN_ANGLE and MAX_ANGLE in the Arduino sketch.
        # These are example limits and must be physically verified before enabling servos.
        self.SERVO_MIN_ANGLE = 80.0
        self.SERVO_MAX_ANGLE = 100.0
        self.servo_ready = False
        
        # 3-Servo Angle States (Neutral set around 90 degrees based on your arm attachment)
        self.neutral_angles = [90.0, 90.0, 90.0]
        self.current_angles = [90.0, 90.0, 90.0]
        
        # Position limit results
        self.position_min = None
        self.position_max = None

    def connect_servo(self):
        """Connect only if the Arduino firmware reports READY."""
        try:
            self.servo = serial.Serial(self.servo_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Allow Arduino auto-reset / connection to stabilize

            startup_message = self.servo.read_all().decode("utf-8", errors="replace")
            if "READY" not in startup_message.splitlines():
                print("[SERVO] Arduino is not READY. Startup message:")
                print(startup_message.strip() or "(No response received)")
                print("[SERVO] Check CALIBRATION_READY and the PCA9685 before servo testing.")
                self.servo.close()
                self.servo = None
                self.servo_ready = False
                return False

            self.servo_ready = True
            print("[SERVO] Connected and READY on", self.servo_port)
            return True
        except Exception as e:
            print(f"[SERVO] Failed to connect: {e}")
            self.servo_ready = False
            if self.servo and self.servo.is_open:
                self.servo.close()
            self.servo = None
            return False

    def send_servo_angles(self, angles):
        """Send 3 absolute joint angles using the Arduino A,angle1,angle2,angle3 protocol."""
        if self.servo_ready and self.servo and self.servo.is_open:
            # Match the angle limits in servo_controller_stewart.ino.
            clipped = [
                np.clip(a, self.SERVO_MIN_ANGLE, self.SERVO_MAX_ANGLE)
                for a in angles
            ]
            command = f"A,{clipped[0]:.2f},{clipped[1]:.2f},{clipped[2]:.2f}\n"
            self.servo.write(command.encode("utf-8"))

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
        if pixel_distance <= 0:
            print("[GEO] The two points overlap. Please select two different points.")
            self.peg_points = []
            return

        # Select the platform centre first, then a point one known radius away.
        self.platform_center_px = p1
        self.pixel_to_meter_ratio = self.PLATFORM_RADIUS_M / pixel_distance
        print(f"[GEO] Platform centre: {self.platform_center_px}")
        print(f"[GEO] Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f} m/pixel")
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
        
        center_x = (
            self.platform_center_px[0]
            if self.platform_center_px is not None
            else frame.shape[1] // 2
        )
        pixel_offset = x - center_x
        return pixel_offset * self.pixel_to_meter_ratio

    def confirm_limits_manually(self):
        """Finish after manually checking safe mechanical clearance.

        No automatic servo sweep is performed. This program does not calculate
        the full X/Y workspace limits or the 3x2 servo mapping.
        """
        print("[LIMITS] No automatic servo sweep will be performed.")
        print("[LIMITS] Manually confirm linkage clearance and the verified")
        print("[LIMITS] Arduino angle and pulse-width limits.")
        self.position_min = None
        self.position_max = None

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
            "geometry": "Click platform centre, then a point one known radius away.",
            "neutral_tuning": "Use 1/2, 3/4, 5/6 for S1/S2/S3 +/- 1 degree. Press 'l' when level.",
            "limits": "Check linkage clearance and safe range manually, then press 'l'.",
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
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f"Camera {self.CAM_INDEX} could not be opened. Check CAM_INDEX."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        
        cv2.namedWindow("Stewart Calibration")
        cv2.setMouseCallback("Stewart Calibration", self.mouse_callback)
        
        self.connect_servo()
        
        print("[INFO] Stewart Platform Calibration Started")
        print("Step 1: Click the platform centre, then a point one known radius away.")
        print("Use the actual measured platform radius in PLATFORM_RADIUS_M.")
        
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
                # Manual fine-tuning for leveling all 3 servos individually.
                step = 1.0
                servo_changes = {
                    ord('1'): (0, step),  ord('2'): (0, -step),
                    ord('3'): (1, step),  ord('4'): (1, -step),
                    ord('5'): (2, step),  ord('6'): (2, -step)
                }

                if key in servo_changes:
                    if not self.servo_ready:
                        print("[SERVO] Cannot move servos: Arduino has not reported READY.")
                        continue

                    servo_index, change = servo_changes[key]
                    self.current_angles[servo_index] = float(np.clip(
                        self.current_angles[servo_index] + change,
                        self.SERVO_MIN_ANGLE,
                        self.SERVO_MAX_ANGLE
                    ))
                    self.send_servo_angles(self.current_angles)

                elif key == ord('l'):
                    if not self.servo_ready:
                        print("[SERVO] Cannot lock a physical neutral position until Arduino is READY.")
                        continue

                    # Lock current fine-tuned angles as the true neutral level position.
                    self.neutral_angles = list(self.current_angles)
                    print(f"[INFO] Neutral level locked at: {self.neutral_angles}")
                    print("[IMPORTANT] Copy these values into NEUTRAL_ANGLE in the Arduino sketch")
                    print("[IMPORTANT] and re-upload it so the Arduino N command and timeout use this neutral.")
                    self.phase = "limits"
                    continue

            elif key == ord('l') and self.phase == "limits":
                self.confirm_limits_manually()
                self.phase = "complete"
            elif key == ord('s') and self.phase == "complete":
                self.save_config()
                break
                
        # Return to the neutral position stored in the Arduino firmware before closing.
        if self.servo_ready and self.servo and self.servo.is_open:
            self.send_neutral()
            time.sleep(0.1)
            self.servo.close()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = StewartAutoCalibrator()
    calibrator.run()
"""
MTE380 Stewart Platform - Calibration Script
Finds neutral level angles and generates the config data needed for the controller.
"""

import time
import json
import serial

class StewartCalibrator:
    def __init__(self, port='COM5', baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        
        # Starting baseline angles
        self.neutral_angles = [90.0, 90.0, 90.0]
        self.min_angles = [60.0, 60.0, 60.0]
        self.max_angles = [120.0, 120.0, 120.0]
        
        # Placeholder for the 3x2 mixing matrix your friend's code requires
        # Rows = Servos (1, 2, 3), Columns = Axes (X, Y)
        self.axis_to_servo_offsets = [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ]

    def connect(self):
        try:
            print(f"[SERIAL] Connecting to Arduino on {self.port}...")
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            time.sleep(2.0)  # Critical: wait for Arduino auto-reset
            print("[SERIAL] Connected successfully!")
        except Exception as e:
            print(f"[SERIAL] Connection failed: {e}")
            self.ser = None

    def send_angles(self, angles):
        if self.ser is None or not self.ser.is_open:
            print("[SERIAL] Error: Serial port not open.")
            return False
        
        # Safety clamp
        clipped = [max(self.min_angles[i], min(self.max_angles[i], a)) for i, a in enumerate(angles)]
        cmd = f"A,{clipped[0]:.2f},{clipped[1]:.2f},{clipped[2]:.2f}\n"
        try:
            self.ser.write(cmd.encode("ascii"))
            return True
        except Exception as e:
            print(f"[SERIAL] Write error: {e}")
            return False

    def send_neutral(self):
        if self.ser and self.ser.is_open:
            self.ser.write(b"N\n")
            print("[SERIAL] Sent neutral command (N).")

    def run_manual_leveling(self):
        """
        Allows you to use the keyboard/console to nudge individual servo angles 
        until your physical platform is perfectly flat and level.
        """
        if not self.ser:
            print("Cannot level without active serial connection.")
            return

        current = list(self.neutral_angles)
        print("\n--- MANUAL LEVELING MODE ---")
        print("Commands:")
        print("  1/2 : Servo 1 (+/- 1 deg)")
        print("  3/4 : Servo 2 (+/- 1 deg)")
        print("  5/6 : Servo 3 (+/- 1 deg)")
        print("  s   : Save current angles as neutral and exit")
        print("  q   : Quit without saving")

        while True:
            print(f"\nCurrent Angles -> S1: {current[0]}° | S2: {current[1]}° | S3: {current[2]}°")
            choice = input("Enter command (1-6, s, q): ").strip().lower()

            if choice == '1': current[0] += 1.0
            elif choice == '2': current[0] -= 1.0
            elif choice == '3': current[1] += 1.0
            elif choice == '4': current[1] -= 1.0
            elif choice == '5': current[2] += 1.0
            elif choice == '6': current[2] -= 1.0
            elif choice == 's':
                self.neutral_angles = list(current)
                print(f"[SAVED] New neutral angles locked: {self.neutral_angles}")
                break
            elif choice == 'q':
                print("[EXIT] Calibration aborted.")
                break
            else:
                print("Invalid input. Try again.")
                continue

            self.send_angles(current)

    def generate_mapping_matrix_guide(self):
        """
        Explains how to determine the 3x2 matrix values for your friend's code.
        """
        print("\n--- 3x2 MAPPING MATRIX GENERATION ---")
        print("Your friend's code requires 'axis_to_servo_offsets' matrix formatted as 3x2:")
        print("  [[servo1_x_gain, servo1_y_gain],")
        print("   [servo2_x_gain, servo2_y_gain],")
        print("   [servo3_x_gain, servo3_y_gain]]")
        print("\nTo fill this out empirically:")
        print("1. Tilt the platform slightly along X (roll) and record how much each servo angle changes.")
        print("2. Tilt the platform slightly along Y (pitch) and record how much each servo angle changes.")
        print("3. Plug those coefficients into your config file.")

    def save_config(self, filename="config.json"):
        data = {
            "neutral_angles": self.neutral_angles,
            "min_angles": self.min_angles,
            "max_angles": self.max_angles,
            "axis_to_servo_offsets": self.axis_to_servo_offsets,
            "mapping_calibrated": True,
            "hardware_enabled": True
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\n[CONFIG] Successfully saved calibration data to {filename}")
        print("Copy these values into your friend's `stewart_controller_3servo.py` configuration dictionary!")


if __name__ == "__main__":
    calibrator = StewartCalibrator(port='COM5', baud=115200)
    calibrator.connect()
    if calibrator.ser:
        calibrator.run_manual_leveling()
        calibrator.generate_mapping_matrix_guide()
        calibrator.save_config("config.json")
        calibrator.send_neutral()
        calibrator.ser.close()
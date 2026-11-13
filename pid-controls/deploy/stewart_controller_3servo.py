"""
MTE380 Stewart Platform - 2-axis ball controller with 
automated coordinated servo calibration and JSON configuration.
"""

import csv
import json
import math
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
import serial


class AxisPD:
    """PD controller using measured velocity for the derivative term."""

    def __init__(self, velocity_tau=0.10):
        self.prev_position = None
        self.filtered_velocity = 0.0
        self.velocity_tau = float(velocity_tau)

    def reset(self):
        self.prev_position = None
        self.filtered_velocity = 0.0

    def calculate(self, target, position, dt, kp, kd):
        dt = max(0.005, min(float(dt), 0.15))
        if self.prev_position is None:
            raw_velocity = 0.0
        else:
            raw_velocity = (position - self.prev_position) / dt

        alpha = dt / (self.velocity_tau + dt)
        self.filtered_velocity += alpha * (raw_velocity - self.filtered_velocity)
        self.prev_position = float(position)

        return kp * (target - position) - kd * self.filtered_velocity


class StewartController:
    def __init__(self):
        self.config_path = Path(__file__).with_name("config.json")
        self.cfg = self._load_or_create_config()

        self._validate_config()
        c = self.cfg
        s = c["servo"]

        self.neutral = np.asarray(s["neutral_angles"], dtype=float)
        self.minimum = np.asarray(s["min_angles"], dtype=float)
        self.maximum = np.asarray(s["max_angles"], dtype=float)
        self.mix = np.asarray(s["axis_to_servo_offsets"], dtype=float)

        self.mapping_rank = int(np.linalg.matrix_rank(self.mix)) if self.mix.size > 0 else 0
        self.mapping_calibrated = bool(s.get("mapping_calibrated", False))
        self.hardware_enabled = bool(s.get("hardware_enabled", True))
        self.can_use_hardware = (
            self.hardware_enabled
            and self.mapping_calibrated
            and self.mapping_rank == 2
        )
        self.dry_run_allowed = bool(c.get("dry_run", True))

        self.kp = float(c["control"].get("kp", 2.0))
        self.kd = float(c["control"].get("kd", 0.4))
        self.loop_hz = float(c["control"].get("loop_hz", 30))
        self.ball_timeout = float(c["control"].get("ball_timeout_s", 0.20))
        self.position_tau = float(c["control"].get("position_filter_tau_s", 0.05))
        velocity_tau = float(c["control"].get("velocity_filter_tau_s", 0.10))

        self.camera = cv2.VideoCapture(c["camera"]["index"], cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(c["camera"]["index"])
        if not self.camera.isOpened():
            raise RuntimeError(
                f"Camera {c['camera']['index']} not found. Check camera.index."
            )

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, c["camera"].get("width", 640))
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, c["camera"].get("height", 480))
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lower_hsv = np.asarray(c["ball"]["lower_hsv"], dtype=np.uint8)
        self.upper_hsv = np.asarray(c["ball"]["upper_hsv"], dtype=np.uint8)

        self.ser = None
        self.running = True
        self.lock = threading.Lock()
        self.ball_measurement = None
        self.last_detection = None
        
        # Camera-to-platform calibration state
        self.platform_center = None
        self.platform_radius_px = None
        self.calibration_points = []
        self.calibration_step = 0

        self.filtered_position = None
        self.target = [0.0, 0.0]
        self.last_angles = self.neutral.copy()
        self.last_u = [0.0, 0.0]
        self.hardware_armed = False
        self.preview_enabled = False
        self.mode = "CAMERA READY — CLICK 3 EDGES"

        self.ax_x = AxisPD(velocity_tau)
        self.ax_y = AxisPD(velocity_tau)
        self.log = []
        self.log_start = time.monotonic()

        self.root = tk.Tk()
        self.root.title("MTE380 — Stewart Platform 2D Ball Controller")
        self.root.geometry("720x720")
        self._build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        
        cv2.namedWindow("Stewart Platform — Ball Tracking")
        cv2.setMouseCallback("Stewart Platform — Ball Tracking", self.on_camera_click)

        if self.can_use_hardware:
            self.connect_serial()
        else:
            self.connect_serial_only()

        self.thread = threading.Thread(target=self.control_loop, daemon=True)
        self.thread.start()
        self.root.after(20, self.camera_tick)
        self.root.after(100, self.gui_tick)

    def _load_or_create_config(self):
        default_cfg = {
            'camera': {'index': 0, 'width': 640, 'height': 480, 'center_x': None, 'center_y': None},
            'ball': {'lower_hsv': [5, 150, 150], 'upper_hsv': [20, 255, 255], 'min_area_px': 75, 'min_circularity': 0.55, 'min_radius_px': 4, 'max_radius_px': 80},
            'servo': {
                'hardware_enabled': True, 
                'mapping_calibrated': False, 
                'port': 'COM5', 
                'neutral_angles': [90.0, 90.0, 90.0], 
                'min_angles': [80.0, 80.0, 80.0], 
                'max_angles': [100.0, 100.0, 100.0], 
                'axis_to_servo_offsets': [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 
                'max_offset_deg': 3.0, 
                'max_step_deg': 0.5, 
                'jog_step_deg': 1.0
            },
            'control': {'kp': 2.0, 'kd': 0.4, 'loop_hz': 30, 'ball_timeout_s': 0.2, 'position_filter_tau_s': 0.05, 'velocity_filter_tau_s': 0.1},
            'dry_run': True
        }
        if self.config_path.exists():
            try:
                with self.config_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default_cfg
        else:
            self._save_config(default_cfg)
            return default_cfg

    def _save_config(self, cfg_data):
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=4)

    def _validate_config(self):
        c = self.cfg
        for key in ("camera", "ball", "servo", "control"):
            if key not in c:
                raise ValueError(f"Missing '{key}' section in configuration")

    def _build_gui(self):
        panel = ttk.Frame(self.root, padding=12)
        panel.pack(fill="both", expand=True)

        ttk.Label(
            panel, text="Stewart Platform — 2-Axis Ball Balancing",
            font=("Arial", 16, "bold")
        ).pack(anchor="w")

        self.status_label = ttk.Label(panel, text=self.mode, wraplength=680)
        self.status_label.pack(anchor="w", pady=(6, 10))

        self.kp_var = tk.DoubleVar(value=self.kp)
        self.kd_var = tk.DoubleVar(value=self.kd)
        self.tx_var = tk.DoubleVar(value=0.0)
        self.ty_var = tk.DoubleVar(value=0.0)

        for label, variable, low, high in [
            ("Kp", self.kp_var, 0.0, 5.0),
            ("Kd", self.kd_var, 0.0, 2.0),
            ("Target X (− left, + right)", self.tx_var, -0.7, 0.7),
            ("Target Y (− down, + up)", self.ty_var, -0.7, 0.7),
        ]:
            ttk.Label(panel, text=label).pack(anchor="w")
            ttk.Scale(
                panel, from_=low, to=high, variable=variable,
                orient="horizontal"
            ).pack(fill="x", pady=(0, 7))

        self.calibration_label = ttk.Label(panel, text="STEP 1: Click THREE widely spaced points on platform EDGE", wraplength=680)
        self.calibration_label.pack(anchor="w", pady=(2, 5))
        
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(anchor="w", pady=4)

        ttk.Button(btn_frame, text="RECALIBRATE CAMERA", command=self.reset_camera_calibration).pack(side="left", padx=2)
        
        self.auto_cal_button = ttk.Button(btn_frame, text="RUN AUTO SERVO CALIBRATION", command=self.run_auto_calibration)
        self.auto_cal_button.pack(side="left", padx=2)
        if not self.can_use_hardware:
            # Enable auto calibration button if serial is open even if uncalibrated
            pass

        self.position_label = ttk.Label(panel, text="Ball position: not detected")
        self.position_label.pack(anchor="w", pady=3)
        self.output_label = ttk.Label(panel, text="PD output: X=0.000, Y=0.000")
        self.output_label.pack(anchor="w", pady=3)
        self.servo_label = ttk.Label(panel, text="Servo preview: 90°, 90°, 90°")
        self.servo_label.pack(anchor="w", pady=3)

        buttons = ttk.Frame(panel)
        buttons.pack(anchor="w", pady=12)

        self.arm_button = ttk.Button(
            buttons, text="ARM HARDWARE", command=self.arm
        )
        self.arm_button.pack(side="left", padx=4)

        ttk.Button(buttons, text="DISARM / NEUTRAL", command=self.disarm).pack(side="left", padx=4)
        ttk.Button(buttons, text="CENTRE TARGET", command=self.center_target).pack(side="left", padx=4)
        ttk.Button(buttons, text="EXPORT CSV", command=self.export_log).pack(side="left", padx=4)

        if not self.can_use_hardware:
            self.arm_button.state(["disabled"])

        self.mapping_label = ttk.Label(
            panel,
            text=f"Mapping rank: {self.mapping_rank}/2 | "
                 f"Mapping calibrated: {self.mapping_calibrated}"
        )
        self.mapping_label.pack(anchor="w", pady=3)

    def connect_serial_only(self):
        try:
            self.ser = serial.Serial(
                self.cfg["servo"]["port"], 115200, timeout=0.2, write_timeout=0.2
            )
            time.sleep(2.0)
            self.mode = "ARDUINO CONNECTED — PENDING CALIBRATION"
        except Exception as exc:
            self.mode = "SERIAL FAILED: " + str(exc)
            self.ser = None

    def connect_serial(self):
        try:
            self.ser = serial.Serial(
                self.cfg["servo"]["port"], 115200, timeout=0.2, write_timeout=0.2
            )
            time.sleep(2.0)
            self.mode = "ARDUINO CONNECTED — READY"
        except Exception as exc:
            self.mode = "SERIAL FAILED: " + str(exc)
            self.ser = None

    def send_angles(self, angles):
        if self.ser is None or not self.ser.is_open:
            return False
        a = np.asarray(angles, dtype=float)
        command = f"A,{a[0]:.2f},{a[1]:.2f},{a[2]:.2f}\n"
        try:
            self.ser.write(command.encode("ascii"))
            return True
        except (serial.SerialException, OSError):
            return False

    def send_neutral(self):
        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.write(b"N\n")
            except (serial.SerialException, OSError):
                pass

    def run_auto_calibration(self):
        """
        Automated Coordinated Servo Calibration Routine:
        1. Moves all 3 servos together in patterned sine/cosine multi-axis steps.
        2. Tracks ball or platform displacement via camera to map actuator changes.
        3. Computes the 3x2 matrix via least squares, saves config.json, and hols neutral.
        """
        if self.ser is None or not self.ser.is_open:
            messagebox.showerror("Serial Error", "Arduino not connected.")
            return
        if self.platform_center is None:
            messagebox.showwarning("Edge Calibration Required", "Please click 3 edge points first.")
            return

        def calibration_task():
            self.mode = "CALIBRATING: Sweeping Servos Coordinated..."
            print("[CALIBRATION] Starting coordinated 3-servo sweep...")
            
            input_commands = []
            measured_outputs = []

            # Coordinated multi-servo excitation pattern
            steps = 30
            for i in range(steps):
                # Move all 3 servos together with phase offsets to excite 2D tilt
                t_val = i / float(steps) * 2.0 * math.pi
                d1 = 3.0 * math.sin(t_val)
                d2 = 3.0 * math.sin(t_val + 2.0 * math.pi / 3.0)
                d3 = 3.0 * math.sin(t_val + 4.0 * math.pi / 3.0)
                
                angles = self.neutral + np.array([d1, d2, d3])
                self.send_angles(angles)
                time.sleep(0.1)

                with self.lock:
                    sample = self.ball_measurement
                if sample is not None:
                    # Record relative motion inputs and measured ball displacements
                    input_commands.append([d1, d2, d3])
                    measured_outputs.append([sample[0], sample[1]])

            # Return servos to neutral position
            self.send_neutral()
            time.sleep(0.5)

            if len(measured_outputs) < 10:
                self.mode = "CALIBRATION FAILED: Ball not tracked reliably."
                messagebox.showerror("Calibration Failed", "Could not track ball reliably during sweep. Ensure ball is on platform.")
                return

            # Solve via Least Squares: Mapping matrix mapping 2D position error back to 3 servos
            # Here we approximate a pseudo-inverse / regression fit for 3x2 matrix
            X = np.array(measured_outputs) # shape (N, 2)
            Y = np.array(input_commands)   # shape (N, 3)
            
            # Least squares solution: Y = X * M  => M = (X^T * X)^-1 * X^T * Y  (transposed for 3x2)
            try:
                # Ridge regression robust fit
                mix_matrix, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                mix_matrix = mix_matrix.T # Should result in shape (3, 2)
                
                if mix_matrix.shape != (3, 2):
                    raise ValueError("Calculated matrix shape invalid.")

                self.mix = mix_matrix
                self.mapping_rank = int(np.linalg.matrix_rank(self.mix))
                self.mapping_calibrated = True
                
                # Update configuration dictionary and save config.json
                self.cfg["servo"]["axis_to_servo_offsets"] = self.mix.tolist()
                self.cfg["servo"]["mapping_calibrated"] = True
                self._save_config(self.cfg)

                self.can_use_hardware = self.hardware_enabled and self.mapping_calibrated and self.mapping_rank == 2
                if self.can_use_hardware:
                    self.arm_button.state(["!disabled"])

                self.mode = "CALIBRATION DONE! config.json generated."
                print(f"[CALIBRATION] Done! 3x2 Matrix:\n{self.mix}")
                messagebox.showinfo("Success", "Calibration Done! config.json generated successfully.")
            except Exception as e:
                self.mode = "CALIBRATION ERROR"
                messagebox.showerror("Error", f"Matrix calculation failed: {e}")

        threading.Thread(target=calibration_task, daemon=True).start()

    def arm(self):
        if not self.can_use_hardware:
            messagebox.showwarning("Hardware locked", "Complete edge selection and auto-calibration first.")
            return
        with self.lock:
            self.hardware_armed = True
            self.mode = "ARMED — TRACKING"

    def disarm(self):
        with self.lock:
            self.hardware_armed = False
            self.mode = "DISARMED"
        self.send_neutral()

    def center_target(self):
        self.tx_var.set(0.0)
        self.ty_var.set(0.0)

    def reset_camera_calibration(self):
        self.disarm()
        self.platform_center = None
        self.platform_radius_px = None
        self.calibration_step = 0
        self.calibration_points = []
        print("[CAMERA] Click THREE points on platform edge.")

    def on_camera_click(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.calibration_step < 3:
            self.calibration_points.append((float(x), float(y)))
            self.calibration_step = len(self.calibration_points)
            print(f"[CAMERA] Edge point {self.calibration_step}/3 = ({x}, {y})")

        if self.calibration_step == 3:
            (x1, y1), (x2, y2), (x3, y3) = self.calibration_points
            det = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if abs(det) < 100.0:
                self.reset_camera_calibration()
                return
            s1, s2, s3 = x1**2 + y1**2, x2**2 + y2**2, x3**2 + y3**2
            cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / det
            cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / det
            radius = sum(math.hypot(px - cx, py - cy) for px, py in self.calibration_points) / 3.0
            
            self.platform_center = (cx, cy)
            self.platform_radius_px = radius
            self.mode = "CAMERA CALIBRATED — Ready for Auto Calibration"
            print(f"[CAMERA] Center=({cx:.1f}, {cy:.1f}), Radius={radius:.1f}px")

    def detect_ball(self, frame):
        if self.calibration_step < 3:
            for idx, pt in enumerate(self.calibration_points):
                cv2.circle(frame, (round(pt[0]), round(pt[1])), 5, (255, 255, 0), -1)
            cv2.putText(frame, f"CLICK {3 - self.calibration_step} MORE EDGE POINTS", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            return None, frame

        cx, cy = self.platform_center
        cv2.circle(frame, (round(cx), round(cy)), round(self.platform_radius_px), (255, 255, 0), 2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg["ball"]["min_area_px"]:
                continue
            (bx, by), br = cv2.minEnclosingCircle(c)
            if math.hypot(bx - cx, by - cy) > self.platform_radius_px * 1.15:
                continue
            if best is None or area > best[0]:
                best = (area, bx, by, br)

        if best is None:
            return None, frame

        _, bx, by, br = best
        nx = (bx - cx) / self.platform_radius_px
        ny = (cy - by) / self.platform_radius_px
        cv2.circle(frame, (round(bx), round(by)), round(br), (0, 255, 0), 2)
        return (float(nx), float(ny)), frame

    def camera_tick(self):
        if not self.running:
            return
        ok, frame = self.camera.read()
        now = time.monotonic()
        if not ok:
            return

        position, annotated = self.detect_ball(frame)
        with self.lock:
            if position is not None:
                self.last_detection = position
                self.ball_measurement = (position[0], position[1], now)
            else:
                self.last_detection = None

        cv2.imshow("Stewart Platform — Ball Tracking", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.stop()
            return
        if key == ord("r"):
            self.reset_camera_calibration()

        self.root.after(20, self.camera_tick)

    def gui_tick(self):
        if not self.running:
            return
        with self.lock:
            self.kp = max(0.0, float(self.kp_var.get()))
            self.kd = max(0.0, float(self.kd_var.get()))
            self.target = [float(self.tx_var.get()), float(self.ty_var.get())]
            mode = self.mode
            detection = self.last_detection
            angles = self.last_angles.copy()
            u = self.last_u[:]

        self.status_label.config(text="Status: " + mode)
        if detection is not None:
            self.position_label.config(text=f"Ball position: X={detection[0]:+.3f}, Y={detection[1]:+.3f}")
        self.output_label.config(text=f"PD output: X={u[0]:+.3f}, Y={u[1]:+.3f}")
        self.servo_label.config(text="Servo angles: " + ", ".join(f"{a:.2f}°" for a in angles))
        self.root.after(100, self.gui_tick)

    def control_loop(self):
        period = 1.0 / max(1.0, self.loop_hz)
        previous = time.monotonic()
        next_tick = previous

        while self.running:
            now = time.monotonic()
            dt = max(0.005, min(0.15, now - previous))
            previous = now

            with self.lock:
                sample = self.ball_measurement
                armed = self.hardware_armed
                target = self.target[:]
                kp, kd = self.kp, self.kd

            if not armed or sample is None or (now - sample[2]) > self.ball_timeout:
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
                continue

            x, y, _ = sample
            raw_position = np.array([x, y], dtype=float)
            if self.filtered_position is None:
                self.filtered_position = raw_position.copy()
            else:
                alpha = dt / (self.position_tau + dt)
                self.filtered_position += alpha * (raw_position - self.filtered_position)

            px, py = self.filtered_position
            ux = self.ax_x.calculate(target[0], px, dt, kp, kd)
            uy = self.ax_y.calculate(target[1], py, dt, kp, kd)

            offsets = self.mix @ np.array([ux, uy], dtype=float)
            desired = np.clip(self.neutral + offsets, self.minimum, self.maximum)

            if armed:
                success = self.send_angles(desired)
                if success:
                    with self.lock:
                        self.last_angles = desired
                        self.last_u = [ux, uy]
                        self.mode = "ARMED — TRACKING & BALANCING"

            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def export_log(self):
        destination = Path(__file__).with_name("stewart_results.csv")
        with destination.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "raw_x", "raw_y", "servo1_deg", "servo2_deg", "servo3_deg"])
            writer.writerows(self.log)
        messagebox.showinfo("Exported", f"Saved to {destination}")

    def stop(self):
        self.running = False
        self.send_neutral()
        if self.ser:
            self.ser.close()
        self.camera.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = StewartController()
    app.run()
"""
MTE380 Stewart Platform - 2-axis ball controller (pre-calibration version).

This version is prepared for camera testing and safe dry-run testing.
Physical servo control remains locked until BOTH:
  1) servo.hardware_enabled = true
  2) servo.mapping_calibrated = true and the 3x2 mapping has rank 2

The 3x2 mapping is an empirical local mapping. It is NOT the full Stewart
inverse-kinematics solution. Determine it experimentally on the assembled rig.

Install:
    pip install opencv-python numpy pyserial

Run:
    python "stewart_controller_3servo(2).py"
"""

import csv
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
        # All settings are embedded here so this project needs only this
        # Python file (plus the separate Arduino .ino firmware).
        self.cfg = {'camera': {'index': 1, 'width': 640, 'height': 480, 'center_x': None, 'center_y': None}, 'ball': {'lower_hsv': [5, 150, 150], 'upper_hsv': [20, 255, 255], 'min_area_px': 75, 'min_circularity': 0.55, 'min_radius_px': 4, 'max_radius_px': 80}, 'servo': {'hardware_enabled': False, 'mapping_calibrated': False, 'port': 'COM5', 'neutral_angles': [90.0, 90.0, 90.0], 'min_angles': [80.0, 80.0, 80.0], 'max_angles': [100.0, 100.0, 100.0], 'axis_to_servo_offsets': [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 'max_offset_deg': 3.0, 'max_step_deg': 0.5, 'jog_step_deg': 1.0}, 'control': {'kp': 2.0, 'kd': 0.4, 'loop_hz': 30, 'ball_timeout_s': 0.2, 'position_filter_tau_s': 0.05, 'velocity_filter_tau_s': 0.1}, 'dry_run': True}

        self._validate_config()
        c = self.cfg
        s = c["servo"]

        self.neutral = np.asarray(s["neutral_angles"], dtype=float)
        self.minimum = np.asarray(s["min_angles"], dtype=float)
        self.maximum = np.asarray(s["max_angles"], dtype=float)
        self.mix = np.asarray(s["axis_to_servo_offsets"], dtype=float)

        self.mapping_rank = int(np.linalg.matrix_rank(self.mix))
        self.mapping_calibrated = bool(s.get("mapping_calibrated", False))
        self.hardware_enabled = bool(s.get("hardware_enabled", False))
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
        self.ball_measurement = None  # x, y, monotonic timestamp
        self.last_detection = None
        # Camera-to-platform calibration (session only): click three points on the rim.
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
        self.mode = "CAMERA READY — DISARMED"

        self.ax_x = AxisPD(velocity_tau)
        self.ax_y = AxisPD(velocity_tau)
        self.log = []
        self.log_start = time.monotonic()

        self.root = tk.Tk()
        self.root.title("MTE380 — Stewart Platform 2D Ball Controller")
        self.root.geometry("720x670")
        self._build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        # OpenCV mouse callback runs when the user clicks inside the live camera image.
        cv2.namedWindow("Stewart Platform — Ball Tracking")
        cv2.setMouseCallback("Stewart Platform — Ball Tracking", self.on_camera_click)

        if self.can_use_hardware:
            self.connect_serial()
        else:
            self.mode = self._lock_reason()

        self.thread = threading.Thread(target=self.control_loop, daemon=True)
        self.thread.start()
        self.root.after(20, self.camera_tick)
        self.root.after(100, self.gui_tick)

    def _validate_config(self):
        c = self.cfg
        for key in ("camera", "ball", "servo", "control"):
            if key not in c:
                raise ValueError(f"Missing '{key}' section in embedded configuration")

        s = c["servo"]
        neutral = np.asarray(s["neutral_angles"], dtype=float)
        minimum = np.asarray(s["min_angles"], dtype=float)
        maximum = np.asarray(s["max_angles"], dtype=float)
        mix = np.asarray(s["axis_to_servo_offsets"], dtype=float)

        if neutral.shape != (3,) or minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("Servo neutral/min/max settings must each contain 3 values.")
        if mix.shape != (3, 2):
            raise ValueError("axis_to_servo_offsets must be a 3x2 matrix.")
        if not all(np.all(np.isfinite(a)) for a in (neutral, minimum, maximum, mix)):
            raise ValueError("Servo settings must contain finite numeric values.")
        if np.any(minimum >= maximum) or np.any(neutral < minimum) or np.any(neutral > maximum):
            raise ValueError("Servo neutral angles must lie within min/max limits.")

        for key in ("lower_hsv", "upper_hsv"):
            if len(c["ball"][key]) != 3:
                raise ValueError(f"ball.{key} must contain 3 HSV values.")

    def _lock_reason(self):
        if not self.hardware_enabled:
            return "DRY RUN — hardware_enabled is false"
        if not self.mapping_calibrated:
            return "HARDWARE LOCKED — servo mapping not calibrated"
        if self.mapping_rank != 2:
            return "HARDWARE LOCKED — 3x2 mapping must have rank 2"
        return "HARDWARE LOCKED — check Arduino connection"

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

        self.calibration_label = ttk.Label(panel, text="CAMERA CALIBRATION: click THREE widely spaced points on the platform EDGE", wraplength=680)
        self.calibration_label.pack(anchor="w", pady=(2, 5))
        ttk.Button(panel, text="RECALIBRATE CAMERA (or press R in camera window)",
                   command=self.reset_camera_calibration).pack(anchor="w", pady=(0, 8))

        self.position_label = ttk.Label(panel, text="Ball position: not detected")
        self.position_label.pack(anchor="w", pady=3)
        self.output_label = ttk.Label(panel, text="PD output: X=0.000, Y=0.000")
        self.output_label.pack(anchor="w", pady=3)
        self.servo_label = ttk.Label(panel, text="Servo preview: 90°, 90°, 90°")
        self.servo_label.pack(anchor="w", pady=3)

        buttons = ttk.Frame(panel)
        buttons.pack(anchor="w", pady=12)

        self.preview_button = ttk.Button(
            buttons, text="START DRY-RUN PREVIEW", command=self.toggle_preview
        )
        self.preview_button.pack(side="left", padx=4)

        self.arm_button = ttk.Button(
            buttons, text="ARM HARDWARE", command=self.arm
        )
        self.arm_button.pack(side="left", padx=4)

        ttk.Button(
            buttons, text="DISARM / NEUTRAL", command=self.disarm
        ).pack(side="left", padx=4)
        ttk.Button(
            buttons, text="CENTRE TARGET", command=self.center_target
        ).pack(side="left", padx=4)
        ttk.Button(
            buttons, text="EXPORT CSV", command=self.export_log
        ).pack(side="left", padx=4)

        if not self.dry_run_allowed:
            self.preview_button.state(["disabled"])
        if not self.can_use_hardware:
            self.arm_button.state(["disabled"])

        warning = (
            "Dry-run mode never sends servo movement commands. "
            "The zero mapping is intentionally left uncalibrated."
        )
        ttk.Label(panel, text=warning, wraplength=680).pack(anchor="w", pady=4)

        self.mapping_label = ttk.Label(
            panel,
            text=f"Mapping rank: {self.mapping_rank}/2 | "
                 f"Mapping calibrated: {self.mapping_calibrated}"
        )
        self.mapping_label.pack(anchor="w", pady=3)

        ttk.Label(
            panel,
            text="Camera: click THREE points on platform edge; R recalibrates; Q / ESC exits. "
                 "Check detection before physical servo testing.",
            wraplength=680
        ).pack(anchor="w", pady=6)

    def connect_serial(self):
        try:
            self.ser = serial.Serial(
                self.cfg["servo"]["port"], 115200, timeout=0.2,
                write_timeout=0.2
            )
            time.sleep(2.0)
            received = self.ser.read_all().decode("utf-8", errors="replace")
            if "READY" not in received.splitlines():
                raise RuntimeError(
                    "Arduino did not report READY. "
                    f"Received startup text: {received!r}"
                )
            self.mode = "ARDUINO CONNECTED — DISARMED"
        except Exception as exc:
            self.mode = "SERIAL FAILED: " + str(exc)
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
            self.arm_button.state(["disabled"])

    def send_angles(self, angles):
        if self.ser is None or not self.ser.is_open:
            return False

        a = np.asarray(angles, dtype=float)
        if (
            a.shape != (3,)
            or not np.all(np.isfinite(a))
            or np.any(a < self.minimum)
            or np.any(a > self.maximum)
        ):
            return False

        command = f"A,{a[0]:.2f},{a[1]:.2f},{a[2]:.2f}\n"
        try:
            self.ser.write(command.encode("ascii"))
            return True
        except (serial.SerialException, OSError) as exc:
            self.mode = "SERIAL ERROR: " + str(exc)
            self.hardware_armed = False
            return False

    def send_neutral(self):
        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.write(b"N\n")
            except (serial.SerialException, OSError):
                pass

    def toggle_preview(self):
        if not self.dry_run_allowed:
            return
        with self.lock:
            self.preview_enabled = not self.preview_enabled
            self.ax_x.reset()
            self.ax_y.reset()
            self.filtered_position = None
            if self.preview_enabled:
                self.mode = "DRY RUN — calculating only; no servo commands sent"
                self.preview_button.config(text="STOP DRY-RUN PREVIEW")
            else:
                self.mode = "CAMERA READY — PREVIEW STOPPED"
                self.preview_button.config(text="START DRY-RUN PREVIEW")

    def arm(self):
        if not self.can_use_hardware:
            messagebox.showwarning(
                "Hardware locked",
                "Hardware is locked. Calibrate the 3x2 servo mapping, "
                "set mapping_calibrated=true, verify servo limits, and enable hardware."
            )
            return
        if self.ser is None or not self.ser.is_open:
            messagebox.showerror("No Arduino", "Arduino is not connected and READY.")
            return
        if self.platform_radius_px is None:
            messagebox.showwarning("Calibrate camera", "Click platform centre and then its edge before arming.")
            return
        with self.lock:
            sample = self.ball_measurement
        if sample is None or time.monotonic() - sample[2] > self.ball_timeout:
            messagebox.showwarning(
                "Ball not detected",
                "The camera must detect the ball before arming."
            )
            return
        if messagebox.askyesno(
            "Enable servo motion",
            "Confirm all three servo limits, neutral angles, mapping directions, "
            "linkage clearance and emergency stop have been checked. Enable motion?"
        ):
            with self.lock:
                self.ax_x.reset()
                self.ax_y.reset()
                self.filtered_position = None
                self.last_angles = self.neutral.copy()
                self.hardware_armed = True
                self.preview_enabled = False
                self.mode = "ARMED — WAITING FOR CONTROL LOOP"
                self.preview_button.config(text="START DRY-RUN PREVIEW")

    def disarm(self):
        with self.lock:
            self.hardware_armed = False
            self.preview_enabled = False
            self.ax_x.reset()
            self.ax_y.reset()
            self.filtered_position = None
            self.last_angles = self.neutral.copy()
            self.mode = "DISARMED"
            self.preview_button.config(text="START DRY-RUN PREVIEW")
        self.send_neutral()

    def center_target(self):
        self.tx_var.set(0.0)
        self.ty_var.set(0.0)

    def reset_camera_calibration(self):
        """Reset camera reference; no servo movement is permitted during calibration."""
        self.disarm()
        self.platform_center = None
        self.platform_radius_px = None
        self.calibration_step = 0
        with self.lock:
            self.ball_measurement = None
            self.last_detection = None
            self.last_u = [0.0, 0.0]
        self.calibration_points = []
        print("[CAMERA] Click THREE widely spaced points on the circular platform edge.")

    def on_camera_click(self, event, x, y, flags, userdata):
        """Three-point calibration of the visible platform circle (session only)."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.calibration_step == 0:
            self.disarm()
            self.platform_center = None
            self.platform_radius_px = None
            self.calibration_points = []

        if self.calibration_step < 3:
            self.calibration_points.append((float(x), float(y)))
            self.calibration_step = len(self.calibration_points)
            print(
                f"[CAMERA] Edge point {self.calibration_step}/3 = ({x}, {y})."
            )

        if self.calibration_step < 3:
            return

        # Calculate the circumcentre of the triangle formed by the 3 rim points.
        (x1, y1), (x2, y2), (x3, y3) = self.calibration_points
        determinant = 2.0 * (
            x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        )

        # A small determinant means the points are nearly collinear and
        # the resulting circle would be numerically unstable.
        if abs(determinant) < 100.0:
            print(
                "[CAMERA] These points are too close to a straight line. "
                "Press R and select three more widely spaced edge points."
            )
            self.calibration_points = []
            self.calibration_step = 0
            self.platform_center = None
            self.platform_radius_px = None
            return

        s1 = x1 * x1 + y1 * y1
        s2 = x2 * x2 + y2 * y2
        s3 = x3 * x3 + y3 * y3

        cx = (
            s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)
        ) / determinant
        cy = (
            s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)
        ) / determinant

        radii = [
            math.hypot(px - cx, py - cy)
            for px, py in self.calibration_points
        ]
        radius = sum(radii) / 3.0

        if not math.isfinite(radius) or radius < 20.0:
            print("[CAMERA] Calculated radius is invalid. Press R and try again.")
            self.calibration_points = []
            self.calibration_step = 0
            self.platform_center = None
            self.platform_radius_px = None
            return

        # Reject visibly inconsistent clicks (e.g. one point not on the rim).
        radius_error = max(abs(r - radius) for r in radii)
        if radius_error > max(8.0, 0.08 * radius):
            print(
                "[CAMERA] Edge points do not fit a consistent circle closely. "
                "Press R and click the visible rim more carefully."
            )
            self.calibration_points = []
            self.calibration_step = 0
            self.platform_center = None
            self.platform_radius_px = None
            return

        self.platform_center = (cx, cy)
        self.platform_radius_px = radius
        self.calibration_step = 3

        with self.lock:
            self.ball_measurement = None
            self.last_detection = None
            self.last_u = [0.0, 0.0]
        self.ax_x.reset()
        self.ax_y.reset()
        self.filtered_position = None

        print(
            f"[CAMERA] Calibrated centre=({cx:.1f},{cy:.1f}), "
            f"radius={radius:.1f}px."
        )
        print("[CAMERA] Ball positions are normalised to the platform radius.")

    def detect_ball(self, frame):
        # Require three widely spaced clicks on the visible platform rim.
        if self.calibration_step < 3:
            for index, point in enumerate(self.calibration_points):
                cv2.circle(frame, (round(point[0]), round(point[1])),
                           5, (255, 255, 0), -1)
                cv2.putText(
                    frame, str(index + 1),
                    (round(point[0]) + 8, round(point[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )
            prompt = f"CLICK {3 - self.calibration_step} MORE EDGE POINT(S)"
            cv2.putText(frame, prompt, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            return None, frame

        cx, cy = self.platform_center
        cv2.drawMarker(frame, (round(cx), round(cy)), (255, 255, 255),
                       cv2.MARKER_CROSS, 18, 2)

        radius_px = self.platform_radius_px
        cv2.circle(frame, (round(cx), round(cy)), round(radius_px), (255, 255, 0), 2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_cfg = self.cfg["ball"]
        min_area = float(ball_cfg.get("min_area_px", 75))
        min_circularity = float(ball_cfg.get("min_circularity", 0.55))
        min_radius = float(ball_cfg.get("min_radius_px", 4))
        max_radius = float(ball_cfg.get("max_radius_px", 80))

        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue
            (bx, by), ball_radius = cv2.minEnclosingCircle(contour)
            if not (min_radius <= ball_radius <= max_radius):
                continue
            # Ignore orange objects clearly outside the calibrated plate.
            if math.hypot(bx - cx, by - cy) > radius_px + 0.15 * radius_px:
                continue
            if best is None or area > best[0]:
                best = (area, bx, by, ball_radius)

        if best is None:
            cv2.putText(frame, "BALL NOT DETECTED", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            return None, frame

        _, bx, by, ball_radius = best
        # +X = right; +Y = up. Circle radius corresponds to 1.0 in either axis.
        nx = (bx - cx) / radius_px
        ny = (cy - by) / radius_px
        if not (math.isfinite(nx) and math.isfinite(ny)):
            return None, frame

        cv2.circle(frame, (round(bx), round(by)), round(ball_radius), (0, 255, 0), 2)
        cv2.circle(frame, (round(bx), round(by)), 3, (0, 255, 0), -1)
        cv2.putText(frame, f"X={nx:+.3f}, Y={ny:+.3f} (plate radii)", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return (float(nx), float(ny)), frame

    def camera_tick(self):
        if not self.running:
            return
        ok, frame = self.camera.read()
        now = time.monotonic()

        if not ok:
            self.mode = "CAMERA ERROR — DISARMED"
            self.disarm()
            self.stop()
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
        if self.calibration_step < 3:
            remaining = 3 - self.calibration_step
            self.calibration_label.config(
                text=f"CAMERA: Click {remaining} more widely spaced point(s) "
                     "on the PLATFORM EDGE"
            )
        else:
            cx, cy = self.platform_center
            self.calibration_label.config(
                text=f"CAMERA CALIBRATED: centre=({cx:.0f},{cy:.0f}), "
                     f"radius={self.platform_radius_px:.1f}px | Press R to redo"
            )
        if detection is None:
            self.position_label.config(text="Ball position: not detected yet")
        else:
            self.position_label.config(
                text=f"Ball position: X={detection[0]:+.3f}, Y={detection[1]:+.3f}"
            )
        self.output_label.config(text=f"PD output: X={u[0]:+.3f}, Y={u[1]:+.3f}")
        self.servo_label.config(
            text="Servo preview: " + ", ".join(f"{a:.2f}°" for a in angles)
        )
        self.root.after(100, self.gui_tick)

    def control_loop(self):
        period = 1.0 / max(1.0, self.loop_hz)
        previous = time.monotonic()
        next_tick = previous
        was_active = False

        while self.running:
            now = time.monotonic()
            dt = max(0.005, min(0.15, now - previous))
            previous = now

            with self.lock:
                sample = self.ball_measurement
                armed = self.hardware_armed
                preview = self.preview_enabled
                target = self.target[:]
                kp, kd = self.kp, self.kd

            active = armed or preview
            lost = (self.platform_radius_px is None or sample is None or
                    (now - sample[2]) > self.ball_timeout)

            if not active:
                was_active = False
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
                if next_tick < time.monotonic() - period:
                    next_tick = time.monotonic()
                continue

            if lost:
                if armed:
                    with self.lock:
                        self.hardware_armed = False
                        self.last_angles = self.neutral.copy()
                        self.mode = "BALL LOST — DISARMED"
                    self.send_neutral()
                else:
                    with self.lock:
                        self.mode = "DRY RUN — waiting for ball"
                self.ax_x.reset()
                self.ax_y.reset()
                self.filtered_position = None
                was_active = False
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
                if next_tick < time.monotonic() - period:
                    next_tick = time.monotonic()
                continue

            x, y, stamp = sample
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
            max_offset = float(self.cfg["servo"].get("max_offset_deg", 3.0))
            offsets = np.clip(offsets, -max_offset, max_offset)
            desired = np.clip(self.neutral + offsets, self.minimum, self.maximum)

            max_step = float(self.cfg["servo"].get("max_step_deg", 0.5))
            with self.lock:
                previous_angles = self.last_angles.copy()
            command_angles = previous_angles + np.clip(
                desired - previous_angles, -max_step, max_step
            )

            if armed:
                success = self.send_angles(command_angles)
                if success:
                    with self.lock:
                        self.last_angles = command_angles
                        self.last_u = [ux, uy]
                        self.mode = "ARMED — TRACKING"
                    self.log.append([
                        now - self.log_start, x, y, px, py,
                        target[0], target[1], ux, uy,
                        *command_angles.tolist(), "HARDWARE"
                    ])
                else:
                    with self.lock:
                        self.hardware_armed = False
                        self.last_angles = self.neutral.copy()
                        self.mode = "COMMAND REJECTED — DISARMED"
                    self.send_neutral()
            else:
                # Dry-run: calculate and display, but never send servo commands.
                with self.lock:
                    self.last_angles = command_angles
                    self.last_u = [ux, uy]
                    if self.mapping_rank < 2 or not self.mapping_calibrated:
                        self.mode = "DRY RUN — mapping not calibrated; no servo output"
                    else:
                        self.mode = "DRY RUN — preview only; no servo commands sent"
                self.log.append([
                    now - self.log_start, x, y, px, py,
                    target[0], target[1], ux, uy,
                    *command_angles.tolist(), "DRY_RUN"
                ])

            was_active = True
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
            if next_tick < time.monotonic() - period:
                next_tick = time.monotonic()

        self.send_neutral()

    def export_log(self):
        destination = Path(__file__).with_name("stewart_results.csv")
        with destination.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s", "raw_x", "raw_y", "filtered_x", "filtered_y",
                "target_x", "target_y", "u_x", "u_y",
                "servo1_deg", "servo2_deg", "servo3_deg", "mode"
            ])
            with self.lock:
                writer.writerows(self.log)
        messagebox.showinfo(
            "CSV exported",
            f"Saved {len(self.log)} samples to:\n{destination}"
        )

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.hardware_armed = False
        self.send_neutral()

        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass

        self.camera.release()
        cv2.destroyAllWindows()
        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = StewartController()
        app.run()
    except Exception as error:
        messagebox.showerror("Controller startup error", str(error))
        raise

"""
MTE380 Stewart Platform - camera PD control with slide-based inverse kinematics.

Install: pip install opencv-python numpy pyserial
Run:     python stewart_controller_3servo.py
Check the mathematics without a camera/Arduino:
         python stewart_controller_3servo.py --self-test

SETUP: edit CONFIG below. All linkage dimensions are in millimetres.
  base_radii_mm        = [l1, l2, l3], origin to each motor's rotation axis
  crank_lengths_mm     = [l11, l21, l31], motor axis to elbow pin centre
  rod_lengths_mm       = [l12, l22, l32], elbow to platform ball-joint centre
  platform_joint_radius_mm = level plate centre to each of its three joints
                             (NOT the outer radius used by camera calibration)
  neutral_height_mm    = level joint plane height above the motor-axis plane
Dimensions are None until measured; no example geometry is sent to hardware.

PD outputs -> downhill tilt in camera X/Y (degrees) -> camera/base alignment
-> rigid plate joint positions -> geometric motor angles -> calibrated servo
commands. Positive X/Y tilt means downhill toward positive X/Y. These are
directional slope angles, not an assumed Euler roll/pitch convention.

The same two-link equations are used for all legs
  r = b + a*cos(theta) - c*cos(phi)
  z =     a*sin(theta) + c*sin(phi)
  theta = atan2(z, r-b) +/- acos(((r-b)^2 + z^2 + a^2 - c^2)
                                / (2*a*hypot(r-b, z)))


First fill dimensions and use START DRY-RUN PREVIEW. Then verify neutral,
servo order/directions, elbow branches, camera alignment and mechanical limits.
Set ik.calibrated=True and servo.hardware_enabled=True only after those checks.
ARM HARDWARE also requires an Arduino READY response and a detected ball.
Limits reduce the requested tilt along a continuous pose path, preserving
plate rigidity; they do not independently clip the three calculated angles.
"""

import csv
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np


CONFIG = {
    "camera": {"index": 1, "width": 640, "height": 480},
    "ball": {
        "lower_hsv": [5, 150, 150], "upper_hsv": [20, 255, 255],
        "min_area_px": 75, "min_circularity": 0.55,
        "min_radius_px": 4, "max_radius_px": 80,
    },
    "ik": {
        "base_radii_mm": [None, None, None],
        "crank_lengths_mm": [None, None, None],
        "rod_lengths_mm": [None, None, None],
        "platform_joint_radius_mm": None,
        "neutral_height_mm": None,
        "branch_signs": [-1, -1, -1],
        "leg_to_servo": [0, 1, 2],
        "calibrated": False,
    },
    "servo": {
        "hardware_enabled": False, "port": "COM5",
        "neutral_angles": [90.0, 90.0, 90.0],
        "directions": [1, 1, 1],
        "min_angles": [80.0, 80.0, 80.0],
        "max_angles": [100.0, 100.0, 100.0],
        "max_offset_deg": 3.0, "max_step_deg": 0.5,
    },
    "control": {
        "kp": 2.0, "kd": 0.4, "loop_hz": 30,
        "ball_timeout_s": 0.2, "position_filter_tau_s": 0.05,
        "velocity_filter_tau_s": 0.1,
        "tilt_deg_per_u": 1.0, "max_tilt_deg": 5.0,
        "camera_to_base_deg": 0.0, "camera_axis_signs": [1, 1],
    },
    "dry_run": True,
}


class IKError(ValueError):
    """Missing geometry, unreachable pose or a singular linkage."""


class StewartIK:
    """Slide linkage model with a rigid equilateral plate in radial planes."""

    def __init__(self, cfg):
        def lengths(key, count):
            value = cfg[key]
            if value is None or (isinstance(value, (list, tuple)) and None in value):
                raise IKError(f"Measure and fill CONFIG['ik']['{key}'].")
            result = np.asarray(value, dtype=float).reshape(-1)
            if result.size != count or not np.all(np.isfinite(result)) or np.any(result <= 0):
                raise IKError(f"ik.{key} requires {count} positive finite value(s).")
            return result

        self.base = lengths("base_radii_mm", 3)
        self.crank = lengths("crank_lengths_mm", 3)
        self.rod = lengths("rod_lengths_mm", 3)
        self.radius = float(lengths("platform_joint_radius_mm", 1)[0])
        self.height = float(lengths("neutral_height_mm", 1)[0])
        self.branches = np.asarray(cfg["branch_signs"], dtype=float)
        if self.branches.shape != (3,) or not np.all(np.isin(self.branches, [-1, 1])):
            raise IKError("ik.branch_signs must contain three +1/-1 values.")
        order = cfg["leg_to_servo"]
        if sorted(order) != [0, 1, 2] or any(type(i) is not int for i in order):
            raise IKError("ik.leg_to_servo must be a permutation of [0, 1, 2].")
        self.order = np.asarray(order, dtype=int)
        self.radial = np.array([[0.0, 1.0],
                                [math.sqrt(3) / 2, -0.5],
                                [-math.sqrt(3) / 2, -0.5]])
        self.level_motor_deg = None
        self.level_motor_deg = self.solve([0.0, 0.0])["motor_deg"].copy()

    def platform_points(self, tilt_deg):
        """Solve three radial coordinates from rigid edge-length constraints.

        Work in units of platform joint radius for numerical conditioning.
        z = h - tan(tx)*(x-xc) - tan(ty)*(y-yc); (xc,yc) is the centroid.
        Each edge length must equal sqrt(3)*radius. Newton's method with
        backtracking follows the small-tilt solution connected to level.
        """
        tilt = np.asarray(tilt_deg, dtype=float)
        if tilt.shape != (2,) or not np.all(np.isfinite(tilt)):
            raise IKError("Tilt must contain two finite angles in degrees.")
        if np.linalg.norm(tilt) > 15.0:
            raise IKError("This local plate solver supports tilt vectors up to 15 degrees.")
        slopes = np.tan(np.radians(tilt))
        edges = ((0, 1), (1, 2), (2, 0))

        def evaluate(radii):
            xy = radii[:, None] * self.radial
            z = -(xy - xy.mean(axis=0)) @ slopes
            points = np.column_stack((xy, z))
            residual = np.array([np.dot(points[i] - points[j], points[i] - points[j]) - 3.0
                                 for i, j in edges])
            # Each vertex derivative includes the centroid-height correction.
            derivative = np.zeros((3, 3, 3))
            for k in range(3):
                derivative[k, :2, k] = self.radial[k]
                projected = float(self.radial[k] @ slopes)
                derivative[:, 2, k] = projected / 3.0
                derivative[k, 2, k] -= projected
            jacobian = np.array([2.0 * (points[i] - points[j]) @
                                 (derivative[i] - derivative[j]) for i, j in edges])
            return residual, jacobian, points

        radii = np.ones(3)
        for _ in range(20):
            residual, jacobian, points = evaluate(radii)
            error = float(np.max(np.abs(residual)))
            if error < 1e-11:
                points *= self.radius
                points[:, 2] += self.height
                return points
            try:
                step = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError as exc:
                raise IKError("Singular rigid-plate position solve.") from exc
            fraction = 1.0
            for _ in range(15):
                candidate = radii + fraction * step
                if np.all(candidate > 0) and np.max(np.abs(evaluate(candidate)[0])) < error:
                    radii = candidate
                    break
                fraction *= 0.5
            else:
                raise IKError("Rigid-plate position solve did not converge.")
        raise IKError("Rigid-plate position solve did not converge.")

    def solve(self, tilt_deg):
        points = self.platform_points(tilt_deg)
        radii = np.sum(points[:, :2] * self.radial, axis=1)
        d = radii - self.base
        z = points[:, 2]
        distance = np.hypot(d, z)
        if np.any(distance < 1e-9) or np.any(z <= 0):
            raise IKError("Joint at/below the motor plane or at a singular motor centre.")
        cosine = (distance**2 + self.crank**2 - self.rod**2) / (2 * self.crank * distance)
        if np.any(np.abs(cosine) > 1.0 + 1e-10):
            raise IKError("Requested joint position is outside a leg's reach.")
        cosine = np.clip(cosine, -1.0, 1.0)  # Floating-point roundoff only.
        if np.any(1.0 - cosine**2 < 1e-10):
            raise IKError("A leg is too close to a straight/folded singularity.")
        theta = np.arctan2(z, d) + self.branches * np.arccos(cosine)
        phi = np.arctan2(z - self.crank * np.sin(theta),
                         self.base + self.crank * np.cos(theta) - radii)
        motor_deg = np.degrees(theta)
        if self.level_motor_deg is not None:
            motor_deg = self.level_motor_deg + (motor_deg - self.level_motor_deg + 180) % 360 - 180
        # Verify against the forward equations, independently of angle inverses.
        radial_fk = self.base + self.crank * np.cos(theta) - self.rod * np.cos(phi)
        z_fk = self.crank * np.sin(theta) + self.rod * np.sin(phi)
        if not (np.allclose(radial_fk, radii, rtol=0, atol=1e-7) and
                np.allclose(z_fk, z, rtol=0, atol=1e-7)):
            raise IKError("Forward-kinematics check failed.")
        return {"points_mm": points, "motor_deg": motor_deg,
                "passive_deg": np.degrees(phi)}


class IKServoMapper:
    """Convert PD output to a reachable, calibrated and rate-limited pose."""

    def __init__(self, ik, cfg):
        self.ik = ik
        s, c = cfg["servo"], cfg["control"]
        self.neutral = np.asarray(s["neutral_angles"], dtype=float)
        self.minimum = np.asarray(s["min_angles"], dtype=float)
        self.maximum = np.asarray(s["max_angles"], dtype=float)
        self.directions = np.asarray(s["directions"], dtype=float)
        self.axis_signs = np.asarray(c["camera_axis_signs"], dtype=float)
        for name, value in (("neutral", self.neutral), ("minimum", self.minimum),
                            ("maximum", self.maximum)):
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise IKError(f"Servo {name} must contain three finite values.")
        if (np.any(self.minimum >= self.maximum) or np.any(self.neutral < self.minimum)
                or np.any(self.neutral > self.maximum)):
            raise IKError("Servo neutral must be inside the min/max limits.")
        if self.directions.shape != (3,) or not np.all(np.isin(self.directions, [-1, 1])):
            raise IKError("servo.directions must contain three +1/-1 values.")
        if self.axis_signs.shape != (2,) or not np.all(np.isin(self.axis_signs, [-1, 1])):
            raise IKError("control.camera_axis_signs must contain two +1/-1 values.")
        angle = math.radians(float(c["camera_to_base_deg"]))
        if not math.isfinite(angle):
            raise IKError("camera_to_base_deg must be finite.")
        self.rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                  [math.sin(angle), math.cos(angle)]])
        self.tilt_gain = float(c["tilt_deg_per_u"])
        self.max_tilt = float(c["max_tilt_deg"])
        self.max_offset = float(s["max_offset_deg"])
        self.max_step = float(s["max_step_deg"])
        values = [self.tilt_gain, self.max_tilt, self.max_offset, self.max_step]
        if not np.all(np.isfinite(values)) or min(values) <= 0 or self.max_tilt > 15:
            raise IKError("IK gains/limits must be positive; max_tilt_deg must be <= 15.")

    def angles_for_tilt(self, tilt_deg):
        delta = self.ik.solve(tilt_deg)["motor_deg"] - self.ik.level_motor_deg
        offsets = np.empty(3)
        offsets[self.ik.order] = delta
        return self.neutral + self.directions * offsets

    def command(self, ux, uy, previous_tilt):
        u = np.array([ux, uy], dtype=float)
        if not np.all(np.isfinite(u)):
            raise IKError("PD output is not finite.")
        requested = self.rotation @ (self.axis_signs * u * self.tilt_gain)
        norm = float(np.linalg.norm(requested))
        limited = norm > self.max_tilt
        if limited:
            requested *= self.max_tilt / norm
        previous_tilt = np.asarray(previous_tilt, dtype=float)
        previous_angles = self.angles_for_tilt(previous_tilt)

        def feasible(tilt):
            try:
                angles = self.angles_for_tilt(tilt)
            except IKError:
                return None
            if (np.any(angles < self.minimum) or np.any(angles > self.maximum) or
                    np.any(np.abs(angles - self.neutral) > self.max_offset + 1e-9) or
                    np.any(np.abs(angles - previous_angles) > self.max_step + 1e-9)):
                return None
            return angles

        angles = feasible(requested)
        if angles is not None:
            return angles, requested, limited
        # Reduce the pose change, never clip the three joints independently.
        angles = feasible(previous_tilt)
        if angles is None:
            raise IKError("Previous commanded pose is outside the configured limits.")
        low, high = 0.0, 1.0
        accepted_tilt = previous_tilt.copy()
        for _ in range(18):
            fraction = (low + high) / 2.0
            candidate = previous_tilt + fraction * (requested - previous_tilt)
            candidate_angles = feasible(candidate)
            if candidate_angles is None:
                high = fraction
            else:
                low, angles, accepted_tilt = fraction, candidate_angles, candidate
        return angles, accepted_tilt, True



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
        # Keep the mathematical solver/self-test usable without camera packages.
        global cv2, serial, tk, ttk, messagebox
        import cv2
        import serial
        import tkinter as tk
        from tkinter import ttk, messagebox

        self.cfg = CONFIG

        self._validate_config()
        c = self.cfg
        s = c["servo"]

        self.neutral = np.asarray(s["neutral_angles"], dtype=float)
        self.minimum = np.asarray(s["min_angles"], dtype=float)
        self.maximum = np.asarray(s["max_angles"], dtype=float)
        self.ik = None
        self.ik_mapper = None
        self.ik_error = ""
        try:
            self.ik = StewartIK(c["ik"])
            self.ik_mapper = IKServoMapper(self.ik, c)
        except (IKError, ValueError, TypeError) as exc:
            self.ik_error = str(exc)
        self.ik_calibrated = bool(c["ik"].get("calibrated", False))
        self.hardware_enabled = bool(s.get("hardware_enabled", False))
        self.can_use_hardware = (
            self.hardware_enabled
            and self.ik_calibrated
            and self.ik_mapper is not None
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
        self.last_tilt = np.zeros(2)
        self.hardware_armed = False
        self.preview_enabled = False
        self.mode = "CAMERA READY — DISARMED"

        self.ax_x = AxisPD(velocity_tau)
        self.ax_y = AxisPD(velocity_tau)
        self.log = []
        self.log_start = time.monotonic()

        self.root = tk.Tk()
        self.root.title("MTE380 — Stewart Platform 2D Ball Controller")
        self.root.geometry("780x730")
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
        for key in ("camera", "ball", "servo", "control", "ik"):
            if key not in c:
                raise ValueError(f"Missing '{key}' section in embedded configuration")

        s = c["servo"]
        neutral = np.asarray(s["neutral_angles"], dtype=float)
        minimum = np.asarray(s["min_angles"], dtype=float)
        maximum = np.asarray(s["max_angles"], dtype=float)

        if neutral.shape != (3,) or minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("Servo neutral/min/max settings must each contain 3 values.")
        if not all(np.all(np.isfinite(a)) for a in (neutral, minimum, maximum)):
            raise ValueError("Servo settings must contain finite numeric values.")
        if np.any(minimum >= maximum) or np.any(neutral < minimum) or np.any(neutral > maximum):
            raise ValueError("Servo neutral angles must lie within min/max limits.")

        for key in ("lower_hsv", "upper_hsv"):
            if len(c["ball"][key]) != 3:
                raise ValueError(f"ball.{key} must contain 3 HSV values.")

    def _lock_reason(self):
        if self.ik_mapper is None:
            return "IK NOT CONFIGURED - " + self.ik_error
        if not self.hardware_enabled:
            return "DRY RUN — hardware_enabled is false"
        if not self.ik_calibrated:
            return "HARDWARE LOCKED - IK geometry and servo calibration not confirmed"
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
            "Enter measured IK geometry in CONFIG and verify servo/camera alignment."
        )
        ttk.Label(panel, text=warning, wraplength=680).pack(anchor="w", pady=4)

        self.ik_label = ttk.Label(
            panel,
            text=(f"IK geometry ready: {self.ik_mapper is not None} | "
                  f"Calibration confirmed: {self.ik_calibrated}"),
            wraplength=720,
        )
        self.ik_label.pack(anchor="w", pady=3)

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
        # Switching to preview always disarms physical output first.
        if self.hardware_armed:
            self.disarm()
        with self.lock:
            self.preview_enabled = not self.preview_enabled
            self.ax_x.reset()
            self.ax_y.reset()
            self.filtered_position = None
            self.last_angles = self.neutral.copy()
            self.last_tilt = np.zeros(2)
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
                self._lock_reason() + "\nFill measured IK geometry, verify the "
                "servo/camera calibration, then set ik.calibrated=True and enable hardware."
            )
            return
        if self.ser is None or not self.ser.is_open:
            messagebox.showerror("No Arduino", "Arduino is not connected and READY.")
            return
        if self.platform_radius_px is None:
            messagebox.showwarning("Calibrate camera", "Click THREE widely spaced platform edge points before arming.")
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
            "Confirm measured geometry, elbow branches, servo order/directions, "
            "camera axes, servo limits, neutral angles, "
            "linkage clearance and emergency stop have been checked. Enable motion?"
        ):
            with self.lock:
                self.ax_x.reset()
                self.ax_y.reset()
                self.filtered_position = None
                self.last_angles = self.neutral.copy()
                self.last_tilt = np.zeros(2)
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
            self.last_tilt = np.zeros(2)
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
        # Detect on the original pixels; overlays can obscure the ball at centre.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.drawMarker(frame, (round(cx), round(cy)), (255, 255, 255),
                       cv2.MARKER_CROSS, 18, 2)

        radius_px = self.platform_radius_px
        cv2.circle(frame, (round(cx), round(cy)), round(radius_px), (255, 255, 0), 2)

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
            tilt = self.last_tilt.copy()

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
        self.output_label.config(
            text=f"PD: X={u[0]:+.3f}, Y={u[1]:+.3f} | "
                 f"Applied base tilt: X={tilt[0]:+.2f}, Y={tilt[1]:+.2f} deg"
        )
        self.servo_label.config(
            text="Servo preview: " + ", ".join(f"{a:.2f}°" for a in angles)
        )
        self.root.after(100, self.gui_tick)

    def control_loop(self):
        period = 1.0 / max(1.0, self.loop_hz)
        previous = time.monotonic()
        next_tick = previous
        while self.running:
            now = time.monotonic()
            dt = max(0.005, min(0.15, now - previous))
            previous = now
            self.control_tick(now, dt)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
            if next_tick < time.monotonic() - period:
                next_tick = time.monotonic()
        self.send_neutral()

    def control_tick(self, now, dt):
        # Serialize computation/output with disarm, so an old A command cannot
        # be sent after the GUI has cleared hardware_armed and sent N.
        with self.lock:
            if not self.running or not (self.hardware_armed or self.preview_enabled):
                return
            sample = self.ball_measurement
            armed = self.hardware_armed
            target = self.target[:]
            lost = (self.platform_radius_px is None or sample is None or
                    (now - sample[2]) > self.ball_timeout)
            if lost:
                if armed:
                    self.hardware_armed = False
                    self.mode = "BALL LOST — DISARMED"
                    self.send_neutral()
                else:
                    self.mode = "DRY RUN — waiting for ball"
                self.last_angles = self.neutral.copy()
                self.last_tilt = np.zeros(2)
                self.last_u = [0.0, 0.0]
                self.ax_x.reset()
                self.ax_y.reset()
                self.filtered_position = None
                return
            x, y, stamp = sample
            raw_position = np.array([x, y], dtype=float)
            if self.filtered_position is None:
                self.filtered_position = raw_position.copy()
            else:
                alpha = dt / (self.position_tau + dt)
                self.filtered_position += alpha * (raw_position - self.filtered_position)

            px, py = self.filtered_position
            ux = self.ax_x.calculate(target[0], px, dt, self.kp, self.kd)
            uy = self.ax_y.calculate(target[1], py, dt, self.kp, self.kd)
            self.last_u = [ux, uy]
            if self.ik_mapper is None:
                self.mode = "IK NOT CONFIGURED - " + self.ik_error
                if armed:
                    self.hardware_armed = False
                    self.send_neutral()
                return
            try:
                command_angles, applied_tilt, limited = self.ik_mapper.command(
                    ux, uy, self.last_tilt
                )
            except IKError as exc:
                self.mode = "IK ERROR - " + str(exc)
                if armed:
                    self.hardware_armed = False
                    self.send_neutral()
                self.last_angles = self.neutral.copy()
                self.last_tilt = np.zeros(2)
                self.ax_x.reset()
                self.ax_y.reset()
                self.filtered_position = None
                return
            if armed:
                if not self.send_angles(command_angles):
                    self.hardware_armed = False
                    self.last_angles = self.neutral.copy()
                    self.last_tilt = np.zeros(2)
                    self.mode = "COMMAND REJECTED — DISARMED"
                    self.send_neutral()
                    return
                self.mode = "ARMED — TRACKING"
            else:
                self.mode = "DRY RUN — IK preview only; no servo commands sent"
            if limited:
                self.mode += " | tilt limited"
            self.last_angles = command_angles
            self.last_tilt = applied_tilt
            self.log.append([
                now - self.log_start, x, y, px, py,
                target[0], target[1], ux, uy,
                *command_angles.tolist(), *applied_tilt.tolist(),
                "HARDWARE" if armed else "DRY_RUN"
            ])

    def export_log(self):
        destination = Path(__file__).with_name("stewart_results.csv")
        with destination.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s", "raw_x", "raw_y", "filtered_x", "filtered_y",
                "target_x", "target_y", "u_x", "u_y",
                "servo1_deg", "servo2_deg", "servo3_deg",
                "tilt_base_x_deg", "tilt_base_y_deg", "mode"
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
        with self.lock:
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


def run_self_test():
    """Synthetic geometry only. Never opens a camera, window or serial port."""
    import copy

    cfg = copy.deepcopy(CONFIG)
    # These are TEST FIXTURES, not dimensions assumed for the user's platform.
    cfg["ik"].update({
        "base_radii_mm": [80.0, 80.0, 80.0],
        "crank_lengths_mm": [35.0, 35.0, 35.0],
        "rod_lengths_mm": [100.0, 100.0, 100.0],
        "platform_joint_radius_mm": 90.0, "neutral_height_mm": 105.0,
    })
    checked_poses = 0
    rng = np.random.default_rng(380)
    tilts = [[0.0, 0.0], [5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [0.0, -5.0]]
    tilts += [t.tolist() for t in rng.uniform(-5, 5, size=(200, 2))]
    for branches in ([-1, -1, -1], [1, 1, 1], [-1, 1, -1]):
        geometry = copy.deepcopy(cfg["ik"])
        geometry["branch_signs"] = branches
        # Different link lengths check that the correct first/second link is used.
        geometry["crank_lengths_mm"] = [35.0, 36.0, 34.0]
        geometry["rod_lengths_mm"] = [100.0, 102.0, 98.0]
        ik = StewartIK(geometry)
        for tilt in tilts:
            result = ik.solve(tilt)
            p = result["points_mm"]
            theta = np.radians(result["motor_deg"])
            phi = np.radians(result["passive_deg"])
            r = ik.base + ik.crank * np.cos(theta) - ik.rod * np.cos(phi)
            z = ik.crank * np.sin(theta) + ik.rod * np.sin(phi)
            forward = np.column_stack((r[:, None] * ik.radial, z))
            np.testing.assert_allclose(forward, p, rtol=0, atol=1e-7)
            # Independently check both physical link lengths.
            bases = np.column_stack((ik.base[:, None] * ik.radial, np.zeros(3)))
            elbows = bases + np.column_stack((
                (ik.crank * np.cos(theta))[:, None] * ik.radial,
                ik.crank * np.sin(theta)))
            np.testing.assert_allclose(np.linalg.norm(elbows - bases, axis=1), ik.crank)
            np.testing.assert_allclose(np.linalg.norm(p - elbows, axis=1), ik.rod)
            for i, j in ((0, 1), (1, 2), (2, 0)):
                np.testing.assert_allclose(np.linalg.norm(p[i] - p[j]),
                                           math.sqrt(3) * ik.radius, atol=1e-7)
            n = np.r_[np.tan(np.radians(tilt)), 1.0]
            np.testing.assert_allclose((p - p.mean(axis=0)) @ n, 0.0, atol=1e-8)
            np.testing.assert_allclose(p[:, 2].mean(), ik.height, atol=1e-8)
            np.testing.assert_allclose(p[:, 0] * ik.radial[:, 1] -
                                       p[:, 1] * ik.radial[:, 0], 0.0, atol=1e-8)
            checked_poses += 1

    # Closed-form reference: horizontal crank and vertical upper rod.
    analytic = copy.deepcopy(cfg["ik"])
    analytic.update(platform_joint_radius_mm=115.0, neutral_height_mm=100.0)
    np.testing.assert_allclose(StewartIK(analytic).level_motor_deg, [0, 0, 0], atol=1e-10)
    for key, value in (("neutral_height_mm", 300.0),
                       ("crank_lengths_mm", [0, 35, 35]),
                       ("platform_joint_radius_mm", None)):
        bad = copy.deepcopy(cfg["ik"])
        bad[key] = value
        try:
            StewartIK(bad)
        except IKError:
            pass
        else:
            raise AssertionError(f"Invalid geometry accepted: {key}")

    ik = StewartIK(cfg["ik"])
    mapper = IKServoMapper(ik, cfg)
    np.testing.assert_allclose(mapper.angles_for_tilt([0, 0]), cfg["servo"]["neutral_angles"])
    # +X downhill must lower the +X vertex; +Y downhill must lower leg 1.
    assert ik.solve([2, 0])["points_mm"][1, 2] < ik.solve([2, 0])["points_mm"][2, 2]
    assert ik.solve([0, 2])["points_mm"][0, 2] < ik.height
    remapped = copy.deepcopy(cfg)
    remapped["ik"]["leg_to_servo"] = [2, 0, 1]
    remapped["servo"]["directions"] = [-1, 1, -1]
    remapped["servo"]["neutral_angles"] = [89, 91, 92]
    other_ik = StewartIK(remapped["ik"])
    other_mapper = IKServoMapper(other_ik, remapped)
    delta = other_ik.solve([1, -1])["motor_deg"] - other_ik.level_motor_deg
    expected = np.array([89, 91, 92]) + np.array([-delta[1], delta[2], -delta[0]])
    np.testing.assert_allclose(other_mapper.angles_for_tilt([1, -1]), expected)
    rotated = copy.deepcopy(cfg)
    rotated["control"].update(camera_to_base_deg=90, camera_axis_signs=[-1, 1])
    _, tilt, _ = IKServoMapper(ik, rotated).command(0.01, 0, [0, 0])
    np.testing.assert_allclose(tilt, [0, -0.01], atol=1e-12)

    previous_tilt = np.zeros(2)
    previous_angles = np.array(cfg["servo"]["neutral_angles"])
    for target in ([10, 0], [-10, 10], [0, -10], [0, 0]):
        for _ in range(25):
            angles, tilt, limited = mapper.command(*target, previous_tilt)
            assert np.max(np.abs(angles - previous_angles)) <= mapper.max_step + 1e-8
            assert np.max(np.abs(angles - mapper.neutral)) <= mapper.max_offset + 1e-8
            assert np.all(angles >= mapper.minimum) and np.all(angles <= mapper.maximum)
            assert np.linalg.norm(tilt) <= mapper.max_tilt + 1e-8
            np.testing.assert_allclose(angles, mapper.angles_for_tilt(tilt), atol=1e-8)
            previous_angles, previous_tilt = angles, tilt
    np.testing.assert_allclose(previous_tilt, [0, 0], atol=1e-8)

    class RecordingSerial:
        is_open = True

        def __init__(self):
            self.messages = []

        def write(self, data):
            self.messages.append(data)

    # Exercise the actual control_tick and serial formatter without hardware.
    controller = StewartController.__new__(StewartController)
    controller.cfg = cfg
    controller._validate_config()
    controller.running = True
    controller.lock = threading.Lock()
    controller.hardware_armed = False
    controller.preview_enabled = True
    controller.target = [0, 0]
    controller.kp, controller.kd = 2.0, 0.4
    controller.platform_radius_px = 200
    controller.ball_timeout, controller.position_tau = 0.2, 0.05
    controller.ax_x, controller.ax_y = AxisPD(), AxisPD()
    controller.filtered_position = None
    controller.neutral = mapper.neutral.copy()
    controller.minimum, controller.maximum = mapper.minimum, mapper.maximum
    controller.last_angles, controller.last_tilt = mapper.neutral.copy(), np.zeros(2)
    controller.last_u = [0, 0]
    controller.ik_mapper, controller.ik_error = mapper, ""
    controller.log, controller.log_start = [], 1.0
    controller.ser = RecordingSerial()
    controller.ball_measurement = (0.5, 0.0, 1.0)
    controller.control_tick(1.01, 1 / 30)
    assert not controller.ser.messages, "Dry-run sent a hardware command."
    assert controller.last_u[0] < 0 and controller.last_tilt[0] < 0
    assert not np.allclose(controller.last_angles, controller.neutral)
    controller.hardware_armed, controller.preview_enabled = True, False
    controller.control_tick(1.02, 1 / 30)
    assert controller.ser.messages[-1].startswith(b"A,")
    assert len(controller.log[-1]) == 15
    controller.control_tick(1.3, 1 / 30)
    assert controller.ser.messages[-1] == b"N\n" and not controller.hardware_armed
    count = len(controller.ser.messages)
    controller.ball_measurement = (0, 0, 1.31)
    controller.control_tick(1.32, 1 / 30)
    assert len(controller.ser.messages) == count, "Ball recovery rearmed hardware."
    controller.preview_enabled = True
    controller.ik_mapper = None
    controller.ik_error = "Missing dimensions"
    controller.control_tick(1.33, 1 / 30)
    assert len(controller.ser.messages) == count and "IK NOT CONFIGURED" in controller.mode
    print(f"PASS: {checked_poses} poses: forward IK, rigid plate, radial planes, fixed height.")
    print("PASS: analytic pose, invalid geometry, servo offsets/order/signs, camera alignment.")
    print("PASS: 100 limited moves, dry-run, serial formatting and ball-loss disarm.")
    print("Synthetic geometry only; physical dimensions and hardware remain unverified.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_self_test()
    else:
        try:
            app = StewartController()
            app.run()
        except Exception as error:
            print(f"Controller startup error: {error}")
            raise

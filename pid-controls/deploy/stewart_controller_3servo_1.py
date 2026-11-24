"""
Stewart ball balancer: three-click calibration, online velocity planning,
priority edge recovery, one vector PID and exact inverse kinematics.

Run: python stewart_edge_controller.py
Dependencies: numpy, opencv-python, pyserial (unchanged).
Keep the included config.json next to this file. Read README.md before testing.

Positions are outer-platform-radius units, right/up positive. The ball detector,
three-click circle calibration and IK are retained from the user's original
Nov 21 uploads. The new planner forecasts stopping distance in full 2-D and
overrides the target when the forecast reaches an inner circular boundary.
This is a heuristic with unmeasured braking/delay assumptions, not guaranteed
physical containment. Actual platform/servo angles are not measured.

Geometry is in the constants below; no JSON geometry overrides are used.
Arduino: READY handshake, 115200 baud, A,a1,a2,a3 degree commands, N for neutral.
Edit US_PER_DEGREE only in the Arduino sketch and upload again to recalibrate.
"""

import csv
import json
import math
from pathlib import Path
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
import serial


# Geometry: all lengths in millimetres, measured between pivot centres.
BASE_RADII_MM = [53.2, 53.2, 53.2]
CRANK_LENGTHS_MM = [80.0, 80.0, 80.0]
ROD_LENGTHS_MM = [97.6]*3
PLATFORM_JOINT_RADIUS_MM = 140.0
NEUTRAL_HEIGHT_MM = 110.0  # REPLACE with the measured joint-centre height.
ELBOW_BRANCHES = [-1, -1, -1]  # Outward-elbow solution; must match assembly.
LEG_TO_SERVO = [0, 1, 2]       # Serial slot for each geometric leg.
MAX_TILT_DEG = 5.0
MAX_FRAME_GAP_S = 0.25
WINDOW = "Stewart Platform - Ball Tracking"

DEFAULT_CONFIG = {
    "camera": {"index": 0, "width": 640, "height": 480},
    "ball": {"lower_hsv": [5, 150, 150], "upper_hsv": [20, 255, 255], "min_area_px": 75},
    "servo": {"port": "COM5", "neutral_angles": [90.0] * 3,
              "min_angles": [0.0] * 3, "max_angles": [120.0] * 3,
              "directions": [1, 1, 1], "max_offset_deg": 5.0,
              "max_speed_deg_s": 30.0},
    "control": {"kp": 2.0, "ki": 0.0, "kd": 3.4,
                "position_filter_tau_s": 0.015, "velocity_filter_tau_s": 0.05},
    "motion": {
        "target_radius_fraction": 0.45,
        "max_speed_radius_s": 0.2,
        "max_accel_radius_s2": 0.3,
        "deceleration_radius_s2": 0.3
    },
    "edge": {
        "ball_radius_fraction": 0.15,
        "margin_fraction": 0.1,
        "braking_accel_radius_s2": 0.3,
        "delay_s": 0.2,
        "release_radius_fraction": 0.5,
        "release_speed_radius_s": 0.1,
        "release_hold_s": 0.25,
        "recovery_speed_radius_s": 0.15,
        "recovery_velocity_gain_multiplier": 1.5,
        "recovery_max_tilt_deg": 2.5
    },
    "logging": {"enabled": True},
}


def load_config(path):
    """Load supported fields; earlier trajectory/approach settings are obsolete."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        for section, values in config.items():
            for key in values:
                if key in saved.get(section, {}):
                    values[key] = saved[section][key]
    else:
        path.write_text(json.dumps(config, indent=4), encoding="utf-8")
    for key in ("kp", "ki", "kd", "position_filter_tau_s", "velocity_filter_tau_s"):
        value = float(config["control"][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"control.{key} must be finite and nonnegative.")
        config["control"][key] = value
    if config["control"]["kp"] <= 0 or config["control"]["kd"] <= 0:
        raise ValueError("Edge recovery requires control.kp > 0 and control.kd > 0.")
    for section in ("motion", "edge"):
        for key in config[section]:
            value = float(config[section][key])
            zero_allowed = key in ("ball_radius_fraction", "margin_fraction", "delay_s")
            if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
                raise ValueError(f"{section}.{key} must be finite and " + ("nonnegative." if zero_allowed else "positive."))
            config[section][key] = value
    safe_radius = 1-config["edge"]["ball_radius_fraction"]-config["edge"]["margin_fraction"]
    if not (0 < config["motion"]["target_radius_fraction"] < config["edge"]["release_radius_fraction"] < safe_radius < 1):
        raise ValueError("Require 0 < target radius < release radius < inner boundary radius < 1.")
    if config["edge"]["recovery_max_tilt_deg"] > MAX_TILT_DEG:
        raise ValueError("Recovery tilt cannot exceed MAX_TILT_DEG.")
    if not isinstance(config["logging"]["enabled"], bool):
        raise ValueError("logging.enabled must be true or false.")
    return config


def limit_vector(vector, limit):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    return vector * (limit / length) if length > limit else vector.copy()


def circle_from_three_points(points):
    (x1, y1), (x2, y2), (x3, y3) = points
    det = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(det) < 100:
        raise ValueError("Choose three widely spaced edge points; these are too close to a line.")
    s1, s2, s3 = x1*x1 + y1*y1, x2*x2 + y2*y2, x3*x3 + y3*y3
    cx = (s1 * (y2-y3) + s2 * (y3-y1) + s3 * (y1-y2)) / det
    cy = (s1 * (x3-x2) + s2 * (x1-x3) + s3 * (x2-x1)) / det
    radius = sum(math.hypot(x-cx, y-cy) for x, y in points) / 3
    if not np.all(np.isfinite([cx, cy, radius])) or radius < 5:
        raise ValueError("Invalid platform circle; click the three edge points again.")
    return np.array([cx, cy]), radius


def alignment_from_clicks(points, center):
    """Fit an orthogonal camera-to-base map, including possible image mirroring."""
    camera = (np.asarray(points, float) - center) * [1., -1.]
    camera /= np.linalg.norm(camera, axis=1)[:, None]
    base = np.array([[0., 1.], [math.sqrt(3)/2, -.5], [-math.sqrt(3)/2, -.5]])
    left, _, right = np.linalg.svd(base.T @ camera)
    transform = left @ right
    agreement = np.sum((camera @ transform.T) * base, axis=1)
    if np.min(agreement) < math.cos(math.radians(15)):
        raise ValueError("Clicks must follow three leg directions roughly 120 degrees apart. Try again.")
    return transform


class EdgeAwarePlanner:
    """Replan a 2-D velocity trajectory from measured state on every frame.

    The stop forecast assumes constant velocity for delay_s, followed by
    constant deceleration opposite the full velocity vector. It is a heuristic,
    not a proven invariant or a measurement of available braking authority.
    """

    def __init__(self, motion, edge):
        self.motion, self.edge = motion, edge
        self.safe_radius = 1.0 - edge["ball_radius_fraction"] - edge["margin_fraction"]
        self.reset()

    def reset(self):
        self.velocity = np.zeros(2)
        self.goal = np.zeros(2)
        self.stop_point = np.zeros(2)
        self.predicted_radius = 0.0
        self.current_radius = 0.0
        self.recovering = False
        self.release_elapsed = 0.0
        self.limited = False
        self.target_clamped = False

    @staticmethod
    def stopping_speed(distance, acceleration, delay):
        distance = max(0.0, float(distance))
        denominator = math.sqrt((acceleration*delay)**2 + 2*acceleration*distance) + acceleration*delay
        return 0.0 if denominator == 0 else 2*acceleration*distance / denominator

    def stopping_prediction(self, position, velocity):
        speed = float(np.linalg.norm(velocity))
        delayed = position + self.edge["delay_s"] * velocity
        stop = delayed + velocity * speed / (2*self.edge["braking_accel_radius_s2"])
        # The forecast is a straight segment; the convex norm peaks at an endpoint.
        return stop, max(float(np.linalg.norm(position)), float(np.linalg.norm(stop)))

    def edge_speed_limit(self, position, direction):
        """Distance along a full XY direction to the inner circular boundary."""
        dot = float(position @ direction)
        discriminant = dot*dot + self.safe_radius**2 - float(position @ position)
        distance = max(0.0, -dot + math.sqrt(max(0.0, discriminant)))
        return self.stopping_speed(distance, self.edge["braking_accel_radius_s2"], self.edge["delay_s"])

    def update(self, goal, position, raw_position, velocity, nominal_velocity, dt, kp, kd):
        self.goal = limit_vector(goal, self.motion["target_radius_fraction"])
        self.target_clamped = np.linalg.norm(self.goal-goal) > 1e-10
        forecasts = [self.stopping_prediction(p, velocity) for p in (position, raw_position)]
        self.stop_point, self.predicted_radius = max(forecasts, key=lambda item: item[1])
        self.current_radius = max(float(np.linalg.norm(position)), float(np.linalg.norm(raw_position)))
        risk = self.predicted_radius >= self.safe_radius
        if risk:
            self.recovering = True
            self.release_elapsed = 0.0
        elif self.recovering:
            clear = (self.current_radius <= self.edge["release_radius_fraction"] and
                     self.predicted_radius <= self.edge["release_radius_fraction"] and
                     np.linalg.norm(velocity) <= self.edge["release_speed_radius_s"])
            self.release_elapsed = self.release_elapsed + dt if clear else 0.0
            if self.release_elapsed >= self.edge["release_hold_s"]:
                self.recovering = False
                self.release_elapsed = 0.0
                # Resume from the measured motion, not a stale pre-recovery plan.
                self.velocity = limit_vector(velocity, self.motion["max_speed_radius_s"])

        if self.recovering:
            # Take the more outward position for inward direction near the rim.
            recovery_position = max((position, raw_position), key=lambda p: np.linalg.norm(p))
            distance = float(np.linalg.norm(recovery_position))
            speed = min(self.edge["recovery_speed_radius_s"],
                        (kp/kd)*distance,
                        self.stopping_speed(distance, self.edge["braking_accel_radius_s2"], self.edge["delay_s"]))
            self.velocity = (np.zeros(2) if distance < 1e-12 else -recovery_position * speed/distance)
            self.limited = True
            return self.velocity.copy()

        distance = float(np.linalg.norm(self.goal-position))
        goal_speed = self.stopping_speed(distance, self.motion["deceleration_radius_s2"], self.edge["delay_s"])

        def constrain(candidate):
            speed = float(np.linalg.norm(candidate))
            if speed <= 1e-12:
                return np.zeros(2)
            direction = candidate / speed
            edge_speed = min(self.edge_speed_limit(p, direction) for p in (position, raw_position))
            return limit_vector(candidate, min(self.motion["max_speed_radius_s"], goal_speed, edge_speed))

        desired = constrain(nominal_velocity)
        # Normal changes have bounded vector acceleration. New stopping constraints
        # may bypass this ramp so it cannot delay braking at a boundary or target.
        ramped = self.velocity + limit_vector(desired-self.velocity, self.motion["max_accel_radius_s2"]*dt)
        self.velocity = constrain(ramped)
        self.limited = np.linalg.norm(self.velocity-nominal_velocity) > 1e-10
        return self.velocity.copy()


class VectorPID:
    """Shared vector gains, measured velocity, online planning and priority recovery."""

    def __init__(self, control, motion, edge):
        self.velocity_tau = float(control["velocity_filter_tau_s"])
        self.planner = EdgeAwarePlanner(motion, edge)
        self.reset()

    def reset(self):
        self.previous_position = None
        self.velocity = np.zeros(2)
        self.integral = np.zeros(2)
        self.desired_velocity = np.zeros(2)
        self.planner.reset()

    def calculate(self, target, position, dt, kp, ki, kd, freeze_integral=False, raw_position=None):
        position, target = np.asarray(position, float), np.asarray(target, float)
        raw_position = position if raw_position is None else np.asarray(raw_position, float)
        if (any(v.shape != (2,) for v in (position, target, raw_position)) or
                not np.all(np.isfinite([position, target, raw_position])) or
                not math.isfinite(dt) or dt <= 0 or not np.all(np.isfinite([kp,ki,kd])) or
                kp <= 0 or kd <= 0 or ki < 0):
            raise ValueError("Edge control requires finite XY vectors, dt > 0, Kp > 0, Kd > 0 and Ki >= 0.")
        raw_velocity = (np.zeros(2) if self.previous_position is None else (position-self.previous_position)/dt)
        self.velocity += dt/(self.velocity_tau+dt) * (raw_velocity-self.velocity)
        self.previous_position = position.copy()
        goal = limit_vector(target, self.planner.motion["target_radius_fraction"])
        error = goal-position
        if ki == 0:
            self.integral[:] = 0
        nominal = (kp*error + ki*self.integral)/kd
        self.desired_velocity = self.planner.update(target, position, raw_position, self.velocity,
                                                    nominal, dt, kp, kd)
        if self.planner.recovering:
            self.integral[:] = 0
            gain = kd*self.planner.edge["recovery_velocity_gain_multiplier"]
            # No normal trajectory ramp on recovery. Tangential/inward momentum
            # is damped as well as outward motion; this is not radial-only braking.
            return limit_vector(gain*(self.desired_velocity-self.velocity),
                                self.planner.edge["recovery_max_tilt_deg"])
        raw = kp*error + ki*self.integral - kd*self.velocity
        output = limit_vector(kd*(self.desired_velocity-self.velocity), MAX_TILT_DEG)
        # Conditional integration applies on the next frame. Do not wind up
        # against the velocity plan, tilt cap, prior actuator limit or recovery.
        if ki > 0 and not freeze_integral and error @ (raw-output) <= 1e-12:
            self.integral = limit_vector(self.integral + error*dt, 2.0/ki)
        return output


class PlatformIK:
    def __init__(self, servo):
        self.base = np.asarray(BASE_RADII_MM, float)
        self.crank = np.asarray(CRANK_LENGTHS_MM, float)
        self.rod = np.asarray(ROD_LENGTHS_MM, float)
        self.radius, self.height = PLATFORM_JOINT_RADIUS_MM, NEUTRAL_HEIGHT_MM
        self.branches = np.asarray(ELBOW_BRANCHES, float)
        self.order = np.asarray(LEG_TO_SERVO)
        self.neutral = np.asarray(servo["neutral_angles"], float)
        self.minimum = np.asarray(servo["min_angles"], float)
        self.maximum = np.asarray(servo["max_angles"], float)
        self.directions = np.asarray(servo["directions"], float)
        self.max_offset = float(servo["max_offset_deg"])
        self.max_speed = float(servo["max_speed_deg_s"])
        for values in (self.base, self.crank, self.rod, self.branches, self.neutral,
                       self.minimum, self.maximum, self.directions):
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError("Geometry and servo arrays need three finite numbers each.")
        if (np.any(self.base <= 0) or np.any(self.crank <= 0) or np.any(self.rod <= 0) or
                not np.all(np.isfinite([self.radius, self.height, self.max_offset, self.max_speed])) or
                min(self.radius, self.height, self.max_offset, self.max_speed) <= 0):
            raise ValueError("Geometry lengths and motion limits must be positive.")
        if (not np.all(np.isin(self.branches, [-1, 1])) or
                not np.all(np.isin(self.directions, [-1, 1])) or
                self.order.shape != (3,) or self.order.dtype.kind not in "iu" or
                sorted(self.order.tolist()) != [0, 1, 2]):
            raise ValueError("Branches/directions must be +/-1; leg order must permute 0,1,2.")
        if np.any(self.neutral < self.minimum) or np.any(self.neutral > self.maximum):
            raise ValueError("Neutral servo angles are outside configured limits.")
        self.radial = np.array([[0., 1.], [math.sqrt(3)/2, -.5], [-math.sqrt(3)/2, -.5]])
        self.level_angles = self.motor_angles([0., 0.])

    def platform_points(self, tilt):
        """Rigid plate joints in their radial planes at the requested downhill tilt."""
        tilt = np.asarray(tilt, float)
        if tilt.shape != (2,) or not np.all(np.isfinite(tilt)) or np.linalg.norm(tilt) > 15:
            raise ValueError("Expected a finite XY tilt vector of at most 15 degrees.")
        magnitude = np.linalg.norm(tilt)
        slope = np.zeros(2) if magnitude == 0 else math.tan(math.radians(magnitude)) * tilt / magnitude
        edges = ((0, 1), (1, 2), (2, 0))

        def evaluate(radii):
            xy = radii[:, None] * self.radial
            z = -(xy - xy.mean(axis=0)) @ slope
            points = np.column_stack((xy, z))
            residual = np.array([np.sum((points[i] - points[j])**2) - 3 for i, j in edges])
            derivative = np.zeros((3, 3, 3))
            for k in range(3):
                derivative[k, :2, k] = self.radial[k]
                derivative[:, 2, k] = (self.radial[k] @ slope) / 3
                derivative[k, 2, k] -= self.radial[k] @ slope
            jacobian = np.array([2 * (points[i]-points[j]) @ (derivative[i]-derivative[j])
                                 for i, j in edges])
            return residual, jacobian, points

        radii = np.ones(3)
        for _ in range(20):
            residual, jacobian, points = evaluate(radii)
            error = np.max(np.abs(residual))
            if error < 1e-11:
                points *= self.radius
                points[:, 2] += self.height
                return points
            step = np.linalg.solve(jacobian, -residual)
            fraction = 1.
            for _ in range(15):
                trial = radii + fraction * step
                if np.all(trial > 0) and np.max(np.abs(evaluate(trial)[0])) < error:
                    radii = trial
                    break
                fraction *= .5
            else:
                break
        raise ValueError("Rigid platform position solve did not converge.")

    def motor_angles(self, tilt):
        points = self.platform_points(tilt)
        d = np.sum(points[:, :2] * self.radial, axis=1) - self.base
        z = points[:, 2]
        distance = np.hypot(d, z)
        if np.any(z <= 0) or np.any(distance < 1e-9):
            raise ValueError("Invalid linkage target.")
        unreachable = (distance >= self.crank + self.rod - 1e-6) | (distance <= np.abs(self.crank-self.rod) + 1e-6)
        if np.any(unreachable):
            leg = int(np.flatnonzero(unreachable)[0])
            raise ValueError(f"Leg {leg+1}: target distance {distance[leg]:.1f} mm is outside "
                             f"the open reach interval {abs(self.crank[leg]-self.rod[leg]):.1f}.."
                             f"{self.crank[leg]+self.rod[leg]:.1f} mm. Check NEUTRAL_HEIGHT_MM "
                             "and the joint geometry.")
        cosine = (distance**2 + self.crank**2 - self.rod**2) / (2*self.crank*distance)
        return np.degrees(np.arctan2(z, d) + self.branches * np.arccos(np.clip(cosine, -1, 1)))

    def servo_angles(self, tilt):
        delta = (self.motor_angles(tilt) - self.level_angles + 180) % 360 - 180
        angles = self.neutral.copy()
        angles[self.order] += self.directions[self.order] * delta
        return angles

    def command(self, requested, previous, dt):
        """Reduce the tilt change when a leg, servo limit or rate limit is reached."""
        target = limit_vector(requested, MAX_TILT_DEG)
        previous = np.asarray(previous, float)
        old_angles = self.servo_angles(previous)

        def feasible(tilt):
            try:
                angles = self.servo_angles(tilt)
            except (ValueError, np.linalg.LinAlgError):
                return None
            if (np.any(angles < self.minimum) or np.any(angles > self.maximum) or
                    np.any(np.abs(angles-self.neutral) > self.max_offset) or
                    np.any(np.abs(angles-old_angles) > self.max_speed * dt)):
                return None
            return angles

        angles = feasible(target)
        if angles is not None:
            return angles, target, False
        angles = feasible(previous)
        if angles is None:
            raise ValueError("Previous commanded pose is outside configured limits.")
        low, high = 0., 1.
        for _ in range(14):
            fraction = (low+high)/2
            candidate = feasible(previous + fraction*(target-previous))
            if candidate is None:
                high = fraction
            else:
                low, angles = fraction, candidate
        return angles, previous + low*(target-previous), True


class StewartController:
    def __init__(self):
        self.cfg = load_config(Path(__file__).with_name("config.json"))
        self.neutral = np.asarray(self.cfg["servo"]["neutral_angles"], float)
        if self.neutral.shape != (3,) or not np.all(np.isfinite(self.neutral)):
            raise ValueError("servo.neutral_angles must contain three finite angles.")
        self.pid = VectorPID(self.cfg["control"], self.cfg["motion"], self.cfg["edge"])
        self.log_file, self.log_writer = None, None
        self.log_started = time.monotonic()
        self.position_tau = float(self.cfg["control"]["position_filter_tau_s"])
        self.lower_hsv = np.asarray(self.cfg["ball"]["lower_hsv"], dtype=np.uint8)
        self.upper_hsv = np.asarray(self.cfg["ball"]["upper_hsv"], dtype=np.uint8)
        self.tilt_sign = np.ones(2)
        self.camera_to_base = np.eye(2)  # Fitted from the three labeled clicks before arming.
        self.ik, self.ik_error = None, ""
        try:
            self.ik = PlatformIK(self.cfg["servo"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.ik_error = str(exc)
        self.platform_center, self.platform_radius_px = None, None
        self.calibration_points = []
        self.last_detection = None
        self.previous_frame_time = None
        self.running, self.hardware_armed, self.neutral_sent = True, False, False
        self.ser, self.camera, self.root = None, None, None
        self.mode = "CLICK THREE PLATFORM EDGE POINTS"
        self.reset_control()
        try:
            camera_cfg = self.cfg["camera"]
            self.camera = cv2.VideoCapture(camera_cfg["index"], cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera.release()
                self.camera = cv2.VideoCapture(camera_cfg["index"])
            if not self.camera.isOpened():
                raise RuntimeError("Camera not found. Check camera.index in config.json.")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["width"])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["height"])
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.root = tk.Tk()
            self.root.title("Stewart Platform - Edge Recovery and Vector PID")
            self.root.geometry("760x780")
            self._build_gui()
            self.root.protocol("WM_DELETE_WINDOW", self.stop)
            cv2.namedWindow(WINDOW)
            cv2.setMouseCallback(WINDOW, self.on_camera_click)
            self.open_log()
            self.connect_serial()
            self.root.after(10, self.camera_tick)
        except BaseException:
            self.stop()
            raise

    def _gain_row(self, panel, name, gain, minimum):
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=name, width=4).pack(side="left")
        field = tk.StringVar(value=f"{gain.get():.2f}")
        slider = tk.Scale(row, from_=minimum, to=10, resolution=.01, showvalue=False,
                          orient="horizontal", variable=gain, takefocus=True)
        slider.pack(side="left", fill="x", expand=True)
        entry = ttk.Entry(row, textvariable=field, width=8)
        entry.pack(side="right", padx=4)

        def display(*args):
            field.set(f"{gain.get():.2f}")

        def apply(event):
            try:
                value = float(field.get())
                if not minimum <= value <= 10:
                    raise ValueError
            except ValueError:
                entry.bell()
                self.gain_hint.set(f"{name}: enter {minimum:.2f} to 10.00. Previous gain retained.")
            else:
                gain.set(round(value, 2))
                self.gain_hint.set("Enter or Tab applies a typed gain (0.01 steps).")
            display()

        gain.trace_add("write", display)
        entry.bind("<Return>", apply)
        entry.bind("<FocusOut>", apply)

    def _build_gui(self):
        panel = ttk.Frame(self.root, padding=12)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Stewart Platform - Edge Recovery + Vector PID + IK",
                  font=("Arial", 15, "bold")).pack(anchor="w")
        self.status_label = ttk.Label(panel, text=self.mode, wraplength=730)
        self.status_label.pack(anchor="w", pady=6)
        self.kp_var = tk.DoubleVar(value=self.cfg["control"]["kp"])
        self.ki_var = tk.DoubleVar(value=self.cfg["control"]["ki"])
        self.kd_var = tk.DoubleVar(value=self.cfg["control"]["kd"])
        self.tx_var, self.ty_var = tk.DoubleVar(value=0.), tk.DoubleVar(value=0.)
        self.gain_hint = tk.StringVar(value="Enter or Tab applies a typed gain (0.01 steps).")
        for name, gain, minimum in (("Kp", self.kp_var, .05), ("Ki", self.ki_var, 0), ("Kd", self.kd_var, .05)):
            self._gain_row(panel, name, gain, minimum)
        ttk.Label(panel, textvariable=self.gain_hint).pack(anchor="w")
        for label, variable, low, high in (
                ("Target X (- left, + right)", self.tx_var, -.7, .7),
                ("Target Y (- down, + up)", self.ty_var, -.7, .7)):
            ttk.Label(panel, text=label).pack(anchor="w")
            ttk.Scale(panel, from_=low, to=high, variable=variable, orient="horizontal").pack(fill="x", pady=(0, 4))
        self.calibration_label = ttk.Label(panel, text="Click outer rim along each leg in channel order: 0, 1, 2.")
        self.calibration_label.pack(anchor="w", pady=8)
        direction_row = ttk.Frame(panel)
        direction_row.pack(anchor="w")
        ttk.Label(direction_row, text="Reverse downhill command (camera axes):").pack(side="left")
        self.reverse_vars = [tk.BooleanVar(value=False), tk.BooleanVar(value=False)]
        self.direction_boxes = []
        for axis, variable in zip(("X", "Y"), self.reverse_vars):
            box = ttk.Checkbutton(direction_row, text=axis, variable=variable)
            box.pack(side="left", padx=4)
            self.direction_boxes.append(box)
        buttons = ttk.Frame(panel)
        buttons.pack(anchor="w", pady=5)
        for label, command in (("RECALIBRATE CAMERA", self.reset_camera_calibration),
                               ("CONNECT ARDUINO", self.connect_serial), ("CENTRE TARGET", self.center_target)):
            ttk.Button(buttons, text=label, command=command).pack(side="left", padx=3)
        buttons = ttk.Frame(panel)
        buttons.pack(anchor="w", pady=5)
        for label, command in (("ARM HARDWARE", self.arm), ("DISARM / NEUTRAL", self.disarm)):
            ttk.Button(buttons, text=label, command=command).pack(side="left", padx=3)
        self.edge_label = ttk.Label(panel, text="Edge forecast: waiting for ball", wraplength=730)
        self.edge_label.pack(anchor="w", pady=3)
        self.log_label = ttk.Label(panel, text="", wraplength=730)
        self.log_label.pack(anchor="w", pady=3)
        self.position_label = ttk.Label(panel, text="Ball: not detected")
        self.output_label = ttk.Label(panel, text="PID tilt: X=0, Y=0 deg")
        self.servo_label = ttk.Label(panel, text="Servo commands: neutral")
        for label in (self.position_label, self.output_label, self.servo_label):
            label.pack(anchor="w", pady=3)
        geometry = ("IK ready" if self.ik is not None else "IK unavailable: " + self.ik_error)
        ttk.Label(panel, text=f"Joint height: {NEUTRAL_HEIGHT_MM:g} mm. {geometry}", wraplength=730).pack(anchor="w", pady=6)

    def reset_control(self):
        self.pid.reset()
        self.filtered_position = None
        self.last_u, self.last_tilt = np.zeros(2), np.zeros(2)
        self.last_angles = self.neutral.copy()
        self.limited = False

    def open_log(self):
        if not self.cfg["logging"]["enabled"]:
            self.log_label.config(text="CSV logging off")
            return
        try:
            folder = Path(__file__).with_name("logs_test")
            folder.mkdir(exist_ok=True)
            path = folder / ("edge_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".csv")
            self.log_file = path.open("w", newline="", encoding="utf-8")
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow([
                "elapsed_s", "dt_s", "state", "requested_target_x", "requested_target_y",
                "accepted_target_x", "accepted_target_y", "ball_x", "ball_y", "velocity_x", "velocity_y",
                "planned_vx", "planned_vy", "current_radius", "predicted_radius", "inner_boundary_radius",
                "predicted_stop_x", "predicted_stop_y", "edge_recovery", "target_clamped", "plan_limited",
                "requested_tilt_x_deg", "requested_tilt_y_deg", "feasible_tilt_base_x_deg", "feasible_tilt_base_y_deg",
                "servo_1_deg", "servo_2_deg", "servo_3_deg", "ik_limited", "kp", "ki", "kd"])
            self.log_file.flush()
            self.log_label.config(text=f"CSV: logs_test/{path.name}")
        except OSError as exc:
            self.disable_log(exc)

    def disable_log(self, error):
        if self.log_file is not None:
            try:
                self.log_file.close()
            except OSError:
                pass
        self.log_file, self.log_writer = None, None
        self.log_label.config(text=f"CSV unavailable: {error}")

    def record_sample(self, dt):
        if self.log_writer is None:
            return
        try:
            plan = self.pid.planner
            ball = ["", ""] if self.filtered_position is None else self.filtered_position
            self.log_writer.writerow([
                time.monotonic()-self.log_started, dt, self.mode, self.tx_var.get(), self.ty_var.get(),
                *plan.goal, *ball, *self.pid.velocity, *self.pid.desired_velocity,
                plan.current_radius, plan.predicted_radius, plan.safe_radius, *plan.stop_point,
                plan.recovering, plan.target_clamped, plan.limited, *self.last_u, *self.last_tilt,
                *self.last_angles, self.limited, self.kp_var.get(), self.ki_var.get(), self.kd_var.get()])
            if time.monotonic()-getattr(self, "last_log_flush", 0) >= 1.0:
                self.log_file.flush()
                self.last_log_flush = time.monotonic()
        except OSError as exc:
            self.disable_log(exc)

    def connect_serial(self):
        self.disarm()
        if self.ser is not None:
            self.ser.close()
            self.ser = None
        try:
            self.ser = serial.Serial(self.cfg["servo"]["port"], 115200, timeout=.1, write_timeout=.2)
            deadline, startup = time.monotonic()+3, []
            while time.monotonic() < deadline:
                line = self.ser.readline().decode("ascii", errors="replace").strip()
                if line == "READY":
                    self.neutral_sent = False
                    if not self.send_neutral():
                        raise RuntimeError("Could not send the initial neutral command.")
                    self.mode = "ARDUINO READY - click three edge points, then ARM"
                    break
                if line:
                    startup.append(line)
            else:
                raise RuntimeError("No READY from Arduino: " + " | ".join(startup))
        except Exception as exc:
            if self.ser is not None:
                self.ser.close()
            self.ser = None
            self.mode = "SERIAL FAILED: " + str(exc)
        self.status_label.config(text=self.mode)

    def send_angles(self, angles):
        a = np.asarray(angles, float)
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("Arduino is not connected.")
        if a.shape != (3,) or not np.all(np.isfinite(a)):
            raise ValueError("Servo commands require three finite angles.")
        data = ("A," + ",".join(f"{v:.2f}" for v in a) + "\n").encode("ascii")
        if self.ser.write(data) != len(data):
            raise IOError("Incomplete servo command write.")
        self.neutral_sent = False

    def send_neutral(self):
        self.reset_control()
        if self.ser is not None and self.ser.is_open and not self.neutral_sent:
            try:
                if self.ser.write(b"N\n") != 2:
                    raise IOError("Incomplete neutral command write.")
                self.neutral_sent = True
            except Exception as exc:
                self.hardware_armed = False
                self.mode = "SERIAL FAILED: " + str(exc)
                self.ser.close()
                self.ser = None
                return False
        return True

    def arm(self):
        if self.kp_var.get() <= 0 or self.kd_var.get() <= 0:
            messagebox.showwarning("Gains required", "Edge recovery requires Kp > 0 and Kd > 0.")
            return
        if self.platform_center is None:
            messagebox.showwarning("Camera calibration", "Click three platform edge points first.")
            return
        if self.ik is None:
            messagebox.showwarning("Geometry needs correction", self.ik_error)
            return
        if self.ser is None or not self.ser.is_open:
            messagebox.showwarning("Arduino connection", "Connect the Arduino and wait for READY first.")
            return
        self.reset_control()
        self.tilt_sign = np.array([-1.0 if var.get() else 1.0 for var in self.reverse_vars])
        for box in self.direction_boxes:
            box.state(["disabled"])
        self.hardware_armed = True
        self.mode = "ARMED - waiting for ball"

    def disarm(self):
        self.hardware_armed = False
        for box in self.direction_boxes:
            box.state(["!disabled"])
        if self.send_neutral():
            self.mode = "DISARMED / NEUTRAL"

    def center_target(self):
        self.tx_var.set(0.)
        self.ty_var.set(0.)

    def reset_camera_calibration(self):
        self.disarm()
        self.platform_center, self.platform_radius_px = None, None
        self.calibration_points = []
        self.last_detection = None
        self.mode = "CLICK THREE PLATFORM EDGE POINTS"

    def on_camera_click(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or len(self.calibration_points) >= 3:
            return
        self.calibration_points.append((float(x), float(y)))
        if len(self.calibration_points) == 3:
            try:
                center, radius = circle_from_three_points(self.calibration_points)
                self.camera_to_base = alignment_from_clicks(self.calibration_points, center)
                self.platform_center, self.platform_radius_px = center, radius
                self.mode = "CAMERA CALIBRATED - ready to ARM"
            except ValueError as exc:
                self.reset_camera_calibration()
                self.mode = str(exc)

    def detect_ball(self, frame):
        # Preserve the uploaded script's area and platform-region filters.
        overlay = frame.copy()
        if self.platform_center is None:
            for x, y in self.calibration_points:
                cv2.circle(overlay, (round(x), round(y)), 5, (255, 255, 0), -1)
            cv2.putText(overlay, f"Click rim in direction of CH {LEG_TO_SERVO[len(self.calibration_points)]}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
            return None, overlay
        cx, cy = self.platform_center
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.cfg["ball"]["min_area_px"]:
                continue
            (bx, by), br = cv2.minEnclosingCircle(contour)
            if math.hypot(bx-cx, by-cy) > self.platform_radius_px * 1.15:
                continue
            if best is None or area > best[0]:
                best = area, bx, by, br
        # Draw only after detection so calibration graphics cannot alter the mask.
        cv2.circle(overlay, (round(cx), round(cy)), round(self.platform_radius_px), (255, 255, 0), 2)
        if best is None:
            return None, overlay
        _, bx, by, br = best
        cv2.circle(overlay, (round(bx), round(by)), round(br), (0, 255, 0), 2)
        return np.array([(bx-cx)/self.platform_radius_px, (cy-by)/self.platform_radius_px]), overlay

    def update_control(self, position, dt):
        if not self.hardware_armed:
            return
        if position is None or not math.isfinite(dt) or dt <= 0 or dt > MAX_FRAME_GAP_S:
            if self.send_neutral():
                self.mode = "ARMED - ball lost; neutral" if position is None else "ARMED - frame delay; neutral"
            return
        if self.filtered_position is None:
            self.filtered_position = np.asarray(position, float).copy()
        else:
            self.filtered_position += dt/(self.position_tau+dt) * (position-self.filtered_position)
        self.last_u = self.pid.calculate([self.tx_var.get(), self.ty_var.get()], self.filtered_position,
                                        dt, self.kp_var.get(), self.ki_var.get(), self.kd_var.get(), self.limited,
                                        raw_position=position)
        angles, tilt, limited = self.ik.command(self.camera_to_base @ (self.tilt_sign * self.last_u), self.last_tilt, dt)
        self.send_angles(angles)
        self.last_angles, self.last_tilt, self.limited = angles, tilt, limited
        self.mode = "ARMED - EDGE RECOVERY" if self.pid.planner.recovering else "ARMED - TARGET TRACKING"
        if self.pid.planner.target_clamped:
            self.mode += " (target clamped inward)"
        if limited:
            self.mode += " (actuator limited)"

    def camera_tick(self):
        if not self.running:
            return
        dt = 0.0
        try:
            ok, frame = self.camera.read()
            now = time.monotonic()
            dt = 1/30 if self.previous_frame_time is None else max(now-self.previous_frame_time, 1e-4)
            self.previous_frame_time = now
            if not ok:
                self.last_detection = None
                if self.send_neutral():
                    self.mode = "CAMERA FRAME FAILED - NEUTRAL"
            else:
                position, overlay = self.detect_ball(frame)
                self.last_detection = position
                self.update_control(position, dt)
                # Overlays are added AFTER the unchanged detector has finished.
                if self.platform_center is not None:
                    cx, cy = self.platform_center
                    radius = self.platform_radius_px
                    cv2.circle(overlay, (round(cx), round(cy)), round(radius*self.pid.planner.safe_radius), (0, 210, 255), 1)
                    goal = limit_vector([self.tx_var.get(), self.ty_var.get()], self.cfg["motion"]["target_radius_fraction"])
                    if self.pid.planner.recovering:
                        goal = np.zeros(2)
                    cv2.drawMarker(overlay, (round(cx+goal[0]*radius), round(cy-goal[1]*radius)),
                                   (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
                cv2.imshow(WINDOW, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    self.stop()
                    return
                if key == ord("r"):
                    self.reset_camera_calibration()
        except Exception as exc:
            self.disarm()
            self.mode = "STOPPED: " + str(exc)
        self.record_sample(dt)
        plan = self.pid.planner
        self.edge_label.config(text=(f"Distance from center: {plan.current_radius:.3f}; forecast: {plan.predicted_radius:.3f}; "
                                    f"inner boundary: {plan.safe_radius:.3f}. "
                                    f"Ball speed: {np.linalg.norm(self.pid.velocity):.3f} radius/s"))
        self.status_label.config(text=self.mode)
        self.calibration_label.config(text=(f"Click the outer rim on the radial line through servo channel {LEG_TO_SERVO[len(self.calibration_points)]}" if self.platform_center is None
                                            else f"Camera calibrated: radius {self.platform_radius_px:.1f} px"))
        pos = self.last_detection
        self.position_label.config(text="Ball: not detected" if pos is None else f"Ball: X={pos[0]:+.3f}, Y={pos[1]:+.3f} (radius units)")
        self.output_label.config(text=f"Vector PID tilt: X={self.last_u[0]:+.3f}, Y={self.last_u[1]:+.3f} deg")
        self.servo_label.config(text="Servo commands: " + ", ".join(f"{v:.2f} deg" for v in self.last_angles))
        self.root.after(10, self.camera_tick)

    def stop(self):
        self.running = False
        self.hardware_armed = False
        try:
            self.send_neutral()
        finally:
            if self.log_file is not None:
                try:
                    self.log_file.close()
                except OSError:
                    pass
                self.log_file, self.log_writer = None, None
            if self.ser is not None:
                self.ser.close()
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            if self.root is not None:
                self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        finally:
            if self.running:
                self.stop()


if __name__ == "__main__":
    try:
        StewartController().run()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
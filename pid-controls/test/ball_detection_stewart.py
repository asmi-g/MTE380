# Ball Detection Module for Computer Vision Ball Tracking System
# Detects colored balls in video frames using HSV color space filtering
# Provides both class-based and legacy function interfaces

import cv2
import numpy as np
import json
import os

class BallDetector:
    """Computer vision ball detector using HSV color space filtering."""
    
    def __init__(self, config_file="config.json"):
        """Initialize ball detector with HSV bounds from config file.
        
        Args:
            config_file (str): Path to JSON config file with HSV bounds and calibration
        """
        # Default HSV bounds for orange ball detection
        self.lower_hsv = np.array([5, 150, 150], dtype=np.uint8)  # Orange lower bound
        self.upper_hsv = np.array([20, 255, 255], dtype=np.uint8)  # Orange upper bound
        self.scale_factor = 1.0  # Conversion factor from normalized coords to meters
        
        # Load configuration from file if it exists
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Extract HSV color bounds from config
                if 'ball_detection' in config:
                    if config['ball_detection']['lower_hsv']:
                        self.lower_hsv = np.array(config['ball_detection']['lower_hsv'], dtype=np.uint8)
                    if config['ball_detection']['upper_hsv']:
                        self.upper_hsv = np.array(config['ball_detection']['upper_hsv'], dtype=np.uint8)
                
                # Extract scale factor for position conversion from pixels to meters
                if 'calibration' in config and 'pixel_to_meter_ratio' in config['calibration']:
                    if config['calibration']['pixel_to_meter_ratio']:
                        frame_width = config.get('camera', {}).get('frame_width', 640)
                        self.scale_factor = config['calibration']['pixel_to_meter_ratio'] * (frame_width / 2)
                
                print(f"[BALL_DETECT] Loaded HSV bounds: {self.lower_hsv} to {self.upper_hsv}")
                print(f"[BALL_DETECT] Scale factor: {self.scale_factor:.6f} m/normalized_unit")
                
            except Exception as e:
                print(f"[BALL_DETECT] Config load error: {e}, using defaults")
        else:
            print("[BALL_DETECT] No config file found, using default HSV bounds")

    def detect_ball(self, frame):
        """Detect ball in frame and return detection results.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            found (bool): True if ball detected
            center (tuple): (x, y) pixel coordinates of ball center
            radius (float): Ball radius in pixels
            position_m (float): Ball position in meters from center
        """
        # Convert frame from BGR to HSV color space for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create binary mask using HSV color bounds
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Clean up mask using morphological operations
        mask = cv2.erode(mask, None, iterations=2)  # Remove noise
        mask = cv2.dilate(mask, None, iterations=2)  # Fill gaps
        
        # Find all contours in the cleaned mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None, None, 0.0
        
        # Select the largest contour (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum enclosing circle around the contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Filter out detections that are too small or too large
        if radius < 5 or radius > 100:
            return False, None, None, 0.0
        
        # Convert pixel position to meters from center
        center_x = frame.shape[1] // 2  # Frame center x-coordinate
        normalized_x = (x - center_x) / center_x  # Normalize to -1 to +1 range
        position_m = normalized_x * self.scale_factor  # Convert to meters
        
        return True, (int(x), int(y)), radius, position_m

    def draw_detection_xy(self, frame, show_info=True):
        """Detect ball and draw normalized x and y positions."""
        # Perform ball detection
        found, center, radius, _ = self.detect_ball(frame)
        overlay = frame.copy()

        # Draw center reference lines
        height, width = frame.shape[:2]
        cx, cy = width / 2.0, height / 2.0
        cv2.line(overlay, (int(cx), 0), (int(cx), height - 1),
                 (255, 255, 255), 1)
        cv2.line(overlay, (0, int(cy)), (width - 1, int(cy)),
                 (255, 255, 255), 1)
        if not found:
            return overlay, False, None, None

        # Normalize position from center: positive x right, positive y up
        x_normalized = (center[0] - cx) / cx
        y_normalized = (cy - center[1]) / cy

        # Draw ball and position information
        cv2.circle(overlay, center, int(radius), (0, 255, 0), 2)
        cv2.circle(overlay, center, 3, (0, 255, 0), -1)
        if show_info:
            label = f"x: {x_normalized:+.3f}  y: {y_normalized:+.3f} (normalized)"
            cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
        return overlay, True, x_normalized, y_normalized

    def draw_detection(self, frame, show_info=True):
        """Detect ball and draw detection overlay on frame.
        
        Args:
            frame: Input BGR image frame
            show_info (bool): Whether to display position information text
            
        Returns:
            frame_with_overlay: Frame with detection drawn
            found: True if ball detected
            position_m: Ball position in meters
        """
        # Perform ball detection
        found, center, radius, position_m = self.detect_ball(frame)
        
        # Create overlay copy for drawing
        overlay = frame.copy()
        
        # Draw vertical center reference line
        height, width = frame.shape[:2]
        center_x = width // 2
        cv2.line(overlay, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        cv2.putText(overlay, "Center", (center_x + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if found:
            # Draw circle around detected ball
            cv2.circle(overlay, center, int(radius), (0, 255, 0), 2)  # Green circle
            cv2.circle(overlay, center, 3, (0, 255, 0), -1)  # Green center dot
            
            if show_info:
                # Display ball position information
                cv2.putText(overlay, f"x: {center[0]}", (center[0] - 30, center[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(overlay, f"pos: {position_m:.4f}m", (center[0] - 40, center[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return overlay, found, position_m

def detect_ball_xy(frame, detector=None):
    """Return detection status, normalized x and y positions, and overlay."""
    # Create detector if one was not provided
    if detector is None:
        detector = BallDetector()

    # Get detection results with visual overlay
    overlay, found, x, y = detector.draw_detection_xy(frame)
    return found, x, y, overlay


# Legacy function for backward compatibility with existing code
def detect_ball_x(frame):
    """Legacy function that matches the old ball_detection.py interface.
    
    This function maintains compatibility with existing code that expects
    the original function signature and return format.
    
    Args:
        frame: Input BGR image frame
        
    Returns:
        found (bool): True if ball detected
        x_normalized (float): Normalized x position (-1 to +1)
        vis_frame (array): Frame with detection overlay
    """
    # Create detector instance using default config
    detector = BallDetector()
    
    # Get detection results with visual overlay
    vis_frame, found, position_m = detector.draw_detection(frame)
    
    if found:
        # Convert back to normalized coordinates for legacy compatibility
        x_normalized = position_m / detector.scale_factor if detector.scale_factor != 0 else 0.0
        x_normalized = np.clip(x_normalized, -1.0, 1.0)  # Ensure within bounds
    else:
        x_normalized = 0.0
    
    return found, x_normalized, vis_frame

# For testing/calibration when run directly
def main():
    """Test ball detection with current config."""
    detector = BallDetector()
    cap = cv2.VideoCapture(0)  # Use default camera
    
    print("Ball Detection Test")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Get detection results with overlay
        vis_frame, found, x, y = detector.draw_detection_xy(frame)
        
        # Show detection info in console
        if found:
            print(f"Ball: x={x:+.3f}, y={y:+.3f} (normalized)")
        
        # Display frame with detection overlay
        cv2.imshow("Ball Detection Test", vis_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
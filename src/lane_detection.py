import cv2
import numpy as np
from typing import List, Tuple, Optional, Union


class LaneDetector:
    """Lane detection pipeline for video streams."""
    
    def __init__(self, 
                 canny_low: int = 150, 
                 canny_high: int = 150,
                 hough_rho: float = 2.0,
                 hough_theta: float = np.pi/180,
                 hough_threshold: int = 100,
                 min_line_length: int = 40,
                 max_line_gap: int = 5,
                 lane_ratio: float = 3/5,
                 line_color: Tuple[int, int, int] = (0, 0, 255),
                 line_thickness: int = 10,
                 blend_weights: Tuple[float, float] = (0.8, 1.0)):
        
        # Canny parameters
        self.canny_low = canny_low
        self.canny_high = canny_high
        
        # Hough transform parameters
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        
        # Lane parameters
        self.lane_ratio = lane_ratio
        
        # Visualization parameters
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.blend_weights = blend_weights
        
    def apply_canny(self, image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection to the input image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges
    
    def get_region_of_interest(self, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """Create a triangular region of interest mask."""
        height = image_shape[0]
        width = image_shape[1]
        
        # Define a triangular region for lane detection
        # These coordinates can be adjusted based on your video
        region_points = np.array([
            [(int(width * 0.1), height),           # Bottom left
             (int(width * 0.9), height),           # Bottom right
             (int(width * 0.55), int(height * 0.6)), # Top right
             (int(width * 0.45), int(height * 0.6))] # Top left
        ], dtype=np.int32)
        
        return region_points
    
    def apply_region_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply region of interest mask to the image."""
        mask = np.zeros_like(image)
        region_points = self.get_region_of_interest(image.shape)
        cv2.fillPoly(mask, region_points, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    def detect_lines(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect lines using Hough transform."""
        lines = cv2.HoughLinesP(
            image, 
            self.hough_rho, 
            self.hough_theta, 
            self.hough_threshold,
            np.array([]), 
            minLineLength=self.min_line_length, 
            maxLineGap=self.max_line_gap
        )
        return lines
    
    def calculate_line_parameters(self, line: np.ndarray) -> Tuple[float, float]:
        """Calculate slope and intercept for a line."""
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        return slope, intercept
    
    def extend_line_to_image(self, image_shape: Tuple[int, int, int], 
                            slope: float, intercept: float) -> np.ndarray:
        """Extend a line to cover the entire image height."""
        height = image_shape[0]
        y1 = height  # Bottom of image
        y2 = int(y1 * self.lane_ratio)  # Upper point
        
        # Avoid division by zero
        if slope == 0:
            slope = 0.001
            
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        
        return np.array([x1, y1, x2, y2])
    
    def average_lane_lines(self, image_shape: Tuple[int, int, int], 
                          lines: np.ndarray) -> Optional[np.ndarray]:
        """Average detected lines into left and right lanes."""
        if lines is None or len(lines) == 0:
            return None
            
        left_lane_params = []
        right_lane_params = []
        
        for line in lines:
            slope, intercept = self.calculate_line_parameters(line)
            
            # Classify lines based on slope
            if slope < 0:
                left_lane_params.append((slope, intercept))
            else:
                right_lane_params.append((slope, intercept))
        
        averaged_lines = []
        
        # Process left lane
        if left_lane_params:
            avg_left = np.average(left_lane_params, axis=0)
            left_line = self.extend_line_to_image(image_shape, avg_left[0], avg_left[1])
            averaged_lines.append(left_line)
        
        # Process right lane
        if right_lane_params:
            avg_right = np.average(right_lane_params, axis=0)
            right_line = self.extend_line_to_image(image_shape, avg_right[0], avg_right[1])
            averaged_lines.append(right_line)
        
        return np.array(averaged_lines) if averaged_lines else None
    
    def draw_lines(self, image: np.ndarray, lines: Optional[np.ndarray]) -> np.ndarray:
        """Draw detected lines on a blank image."""
        line_overlay = np.zeros_like(image)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(line_overlay, (x1, y1), (x2, y2), 
                        self.line_color, self.line_thickness)
        
        return line_overlay
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the lane detection pipeline."""
        # Edge detection
        edges = self.apply_canny(frame)
        
        # Apply region of interest
        roi_edges = self.apply_region_mask(edges)
        
        # Detect lines
        raw_lines = self.detect_lines(roi_edges)
        
        if raw_lines is not None and len(raw_lines) > 0:
            # Average lines into lanes
            averaged_lanes = self.average_lane_lines(frame.shape, raw_lines)
            
            # Draw lanes
            lane_overlay = self.draw_lines(frame, averaged_lanes)
            
            # Combine with original frame
            weight_frame, weight_lanes = self.blend_weights
            result = cv2.addWeighted(frame, weight_frame, lane_overlay, weight_lanes, 1)
        else:
            result = frame.copy()
        
        return result
    
    def process_video(self, video_path: str, display: bool = True, 
                     output_path: Optional[str] = None) -> None:
        """Process a video file for lane detection."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Lane Detection', processed_frame)
                    
                    # Add frame counter
                    frame_count += 1
                    print(f"Processing frame {frame_count}", end='\r')
                
                # Write output
                if out:
                    out.write(processed_frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {frame_count} frames")


def main():
    """Main function to run lane detection."""
    # Configuration
    VIDEO_PATH = '../res/countryroad.mp4'
    OUTPUT_PATH = '../res/countryroad_lanes.mp4'  # Set to None to disable saving
    
    # Create lane detector with custom parameters
    detector = LaneDetector(
        canny_low=150,
        canny_high=150,
        hough_rho=2.0,
        hough_threshold=100,
        min_line_length=40,
        max_line_gap=5,
        lane_ratio=3/5,
        line_color=(0, 0, 255),  # Red lines
        line_thickness=10,
        blend_weights=(0.8, 1.0)
    )
    
    # Process video
    try:
        detector.process_video(VIDEO_PATH, display=True, output_path=OUTPUT_PATH)
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
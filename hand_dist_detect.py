import cv2
import mediapipe as mp
import numpy as np
import winsound
import logging

# --- Setup Logging ---
# Configure logging for informative messages, warnings, and errors.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Distance Calculation ---
# Average adult palm width in centimeters. This value might need calibration.
KNOWN_PALM_WIDTH_CM = 8.0
# Pre-calibrated focal length of the camera in pixels.
# This is crucial for accurate distance measurement and should be determined
# experimentally for your specific camera.
FOCAL_LENGTH_PIXELS = 500

# --- Mediapipe Initialization ---
# Initialize MediaPipe Hands solution.
# min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful.
# min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for hand landmarks to be tracked successfully.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
# Initialize MediaPipe drawing utilities for rendering landmarks and connections.
mp_drawing = mp.solutions.drawing_utils

# --- Function to Calculate Distance ---
def calculate_distance(perceived_width_pixels: int) -> float:
    """
    Calculates the distance of an object from the camera based on its
    perceived width in pixels, a known real-world width, and the camera's
    focal length.

    Args:
        perceived_width_pixels (int): The width of the object as detected in pixels.

    Returns:
        float: The estimated distance of the object from the camera in centimeters.
              Returns 0 if perceived_width_pixels is zero to avoid division by zero.
    """
    if perceived_width_pixels == 0:
        logging.warning("Perceived width is zero, cannot calculate distance. Returning 0.")
        return 0.0
    # Formula: Distance = (Known_Width * Focal_Length) / Perceived_Width
    distance_cm = (KNOWN_PALM_WIDTH_CM * FOCAL_LENGTH_PIXELS) / perceived_width_pixels
    return distance_cm

# --- Main Application Logic ---
def run_hand_distance_measurement():
    """
    Runs the real-time hand distance measurement application.
    It captures video from the webcam, detects hands, calculates their distance,
    displays the information, and plays an alert sound if a hand is too close.
    """
    # Initialize webcam capture. 0 typically refers to the default camera.
    cap = cv2.VideoCapture(0)

    # Check if the webcam was opened successfully.
    if not cap.isOpened():
        logging.error("Error: Could not open webcam. Please ensure it is connected and not in use.")
        return # Exit if the webcam is not available

    logging.info("Webcam opened successfully. Hand distance measurement started. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam.
        # 'ret' is a boolean indicating success, 'frame' is the captured image.
        ret, frame = cap.read()

        # If the frame could not be read, break the loop.
        if not ret:
            logging.warning("Failed to grab frame from webcam. Exiting application.")
            break

        # Convert the BGR (OpenCV default) frame to RGB (MediaPipe required format).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        frame_rgb.flags.writeable = False
        
        # Process the RGB frame to detect hand landmarks.
        results = hands.process(frame_rgb)

        # Re-enable writability of the frame.
        frame_rgb.flags.writeable = True

        # Check if any hands were detected.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Initialize min/max coordinates for bounding box calculation.
                x_coords = []
                y_coords = []

                # Extract x and y coordinates for each landmark and store them.
                for lm in hand_landmarks.landmark:
                    # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates.
                    x_coords.append(int(lm.x * frame.shape[1]))
                    y_coords.append(int(lm.y * frame.shape[0]))

                # Calculate the bounding box coordinates.
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)

                # Calculate the perceived width of the hand in pixels.
                perceived_width = x_max - x_min

                # Calculate the estimated distance to the hand.
                distance = calculate_distance(perceived_width)

                # Draw the bounding box around the detected hand.
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Green rectangle

                # Display the calculated distance on the frame.
                distance_text = f'Distance: {distance:.2f} cm'
                cv2.putText(frame, distance_text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Green text

                # Draw the hand landmarks and connections.
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Play a sound alert if the hand is too close.
                # Threshold for proximity alert (e.g., less than 20 cm).
                PROXIMITY_THRESHOLD_CM = 20
                if distance < PROXIMITY_THRESHOLD_CM and distance > 0: # Ensure valid distance
                    logging.info(f"Hand detected at {distance:.2f} cm. Proximity alert!")
                    try:
                        winsound.Beep(1000, 100) # Frequency (Hz), Duration (ms)
                    except Exception as e:
                        logging.error(f"Could not play sound: {e}. Ensure you are on Windows.")
        else:
            # Optionally display a message if no hands are detected.
            no_hand_text = "No hand detected"
            cv2.putText(frame, no_hand_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Red text

        # Resize the frame for consistent display.
        # Calculate height to maintain aspect ratio, preventing distortion.
        display_width = 640
        display_height = int(frame.shape[0] * (display_width / frame.shape[1]))
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Display the processed frame in a window.
        cv2.imshow('Hand Distance Measurement', resized_frame)

        # Wait for 1 millisecond for a key press. If 'q' is pressed, break the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("'q' pressed. Exiting application.")
            break

    # --- Release Resources ---
    # Release the webcam resource.
    cap.release()
    # Destroy all OpenCV windows.
    cv2.destroyAllWindows()
    logging.info("Webcam released and windows closed. Application terminated.")

# Ensures the main function runs only when the script is executed directly.
if __name__ == "__main__":
    run_hand_distance_measurement()
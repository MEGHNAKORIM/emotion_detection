import cv2
import mediapipe as mp
import logging

# --- Setup Logging ---
# Configure logging for better error messages and information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_face_mesh_detection():
    """
    Initializes the webcam, performs real-time face mesh detection using MediaPipe,
    and displays the annotated video feed.
    """
    # --- MediaPipe Face Mesh Initialization ---
    # Initialize the FaceMesh solution from MediaPipe
    # 'static_image_mode=False' for video processing, 'max_num_faces=1' to detect one face
    # 'min_detection_confidence' and 'min_tracking_confidence' ensure robust detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --- OpenCV Video Capture Initialization ---
    # Create an object to capture video from the default webcam (0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        logging.error("Error: Could not open webcam. Please check if it's connected and not in use.")
        return # Exit if the webcam can't be accessed

    logging.info("Webcam opened successfully. Displaying real-time face mesh. Press 'q' to quit.")

    # --- Main Video Processing Loop ---
    while True:
        # Read a frame from the webcam
        # 'ret' (return value) is True if the frame was read successfully, 'frame' is the image itself
        ret, frame = cap.read()

        # If frame reading failed, break the loop
        if not ret:
            logging.warning("Failed to grab frame from webcam. Exiting application.")
            break

        # --- Pre-processing for MediaPipe ---
        # MediaPipe expects RGB images, but OpenCV reads in BGR format.
        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        rgb_frame.flags.writeable = False

        # --- Face Mesh Processing ---
        # Process the RGB frame to detect face landmarks
        results = face_mesh.process(rgb_frame)

        # Mark the frame as writeable again
        rgb_frame.flags.writeable = True

        # --- Drawing Landmarks on the Frame ---
        # Check if any face landmarks were detected
        if results.multi_face_landmarks:
            # Iterate through each detected face (though we set max_num_faces=1)
            for face_landmarks in results.multi_face_landmarks:
                # Iterate through each landmark point in the face mesh
                for landmark_point in face_landmarks.landmark:
                    # Convert normalized landmark coordinates (0 to 1) to pixel coordinates
                    # Multiply by frame width (shape[1]) and height (shape[0])
                    x = int(landmark_point.x * frame.shape[1])
                    y = int(landmark_point.y * frame.shape[0])

                    # Draw a small circle at each landmark point on the original BGR frame
                    # (
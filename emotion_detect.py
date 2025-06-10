import cv2
from deepface import DeepFace
import logging

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_emotion_recognition():
    """
    Initializes the webcam, performs real-time emotion detection using DeepFace,
    and displays the results.
    """
    # Open the default webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        logging.error("Error: Could not open webcam.")
        return

    logging.info("Webcam opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            logging.warning("Failed to grab frame, exiting...")
            break

        try:
            # Analyze emotions in the current frame
            # enforce_detection=False allows analysis even if no face is perfectly detected,
            # which can be useful for robustness but might yield less accurate results
            # if no face is present or clearly visible.
            results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

            if results:
                # Get the dominant emotion from the first detected face
                dominant_emotion = results[0]['dominant_emotion']
                # Define text properties for display
                text = f'Emotion: {dominant_emotion}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (0, 255, 0)  # Green color for the text
                thickness = 2
                org = (50, 50) # Position of the text

                # Put the emotion text on the frame
                cv2.putText(frame, text, org, font, font_scale, color, thickness)
            else:
                # If no face is detected or DeepFace doesn't return results
                logging.info("No face detected or emotion analysis results found.")
                text = "No face detected"
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Red for no face

        except Exception as e:
            logging.error(f"Error during DeepFace analysis: {e}")
            text = "Analysis error"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # Resize the frame for consistent display
        display_width = 800
        display_height = int(frame.shape[0] * (display_width / frame.shape[1])) # Maintain aspect ratio
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Display the frame with emotion
        cv2.imshow("Real-time Emotion Recognition", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("'q' pressed, exiting application.")
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Resources released. Application terminated.")

if __name__ == "__main__":
    run_emotion_recognition()
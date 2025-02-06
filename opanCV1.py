import cv2
import time

# Initialize video capture
video = cv2.VideoCapture(0)

# Initialize the first frame variable
first_frame = None

while True:
    # Capture frame-by-frame
    check, frame = video.read()

    if not check:  # If no frame is captured, break the loop
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Set the first frame
    if first_frame is None:
        first_frame = gray_blurred
        continue

    # Calculate the difference between the current frame and the first frame
    delta_frame = cv2.absdiff(first_frame, gray_blurred)

    # Apply thresholding to highlight the motion areas
    _, threshold_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 1000:
            continue

        # Get the bounding box coordinates for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the current frame
    cv2.imshow("Motion Detection", frame)

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()


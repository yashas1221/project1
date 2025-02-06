import cv2
import time

# Open the video capture
video = cv2.VideoCapture(0)

first_frame = None

while True:
    # Capture each frame
    check, frame = video.read()
    if not check:
        print("Failed to capture video frame. Exiting...")
        break

    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Set the first frame as the reference frame
    if first_frame is None:
        first_frame = gray
        continue

    # Calculate the difference between the first frame and the current frame
    delta_frame = cv2.absdiff(first_frame, gray)

    # Apply a threshold to highlight differences
    _, threshold_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow("Video Frame", frame)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()

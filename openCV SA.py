from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import time
import csv

# Initialize video capture
video = cv2.VideoCapture(0)

# List to store student names
students = []

# Read student names from CSV file
with open("1.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        students.append(row[1])  # Corrected 'appennd' to 'append'

while True:
    # Capture frame-by-frame
    check, frame = video.read()

    if not check:  # If no frame is captured, break the loop
        print("Failed to capture video frame.")
        break

    # Decode the QR code from the frame
    decoded_objects = decode(frame)
    try:
        for obj in decoded_objects:
            name = obj.data.decode()  # Decode the data from the QR code
            if name in students:
                students.remove(name)
                print(f"Marked as present: {name}")

    except Exception as e:
        print(f"Error: {e}")

    # Display the video feed
    cv2.imshow("Attendance", frame)

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        print("Remaining students:", students)
        break

# Release the video capture object and close OpenCV windows
video.release()
cv2.destroyAllWindows()

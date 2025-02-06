import dlib
print(dlib.__version__)
import cv2
import numpy as np
import dlib
import open3d as o3d
import pyrealsense2 as rs

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Configure RealSense depth camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def get_3d_face_points(depth_frame, face_rect):
    """Extracts 3D face landmarks from the depth frame."""
    depth_image = np.asanyarray(depth_frame.get_data())
    face_landmarks = []

    for d in detector(depth_image, 1):
        shape = predictor(depth_image, d)
        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            depth_value = depth_image[y, x]
            face_landmarks.append((x, y, depth_value))

    return np.array(face_landmarks)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = get_3d_face_points(depth_frame, face)
        
        # Draw 3D points on the face
        for (x, y, z) in landmarks:
            cv2.circle(color_image, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("3D Face Recognition", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()

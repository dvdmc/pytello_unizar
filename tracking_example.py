from djitellopy import tello
import cv2
import threading
from ultralytics import YOLO
import numpy as np
import time

SPEED = 10

IMAGE_SIZE = (940, 720)
F_X = 940
CONTROL_HZ = 10 # Hz

MODEL_NAME = "yolov8n.pt"
DETECTION_QUERY = "cup"

# We compute the estimated distance based on the estimated size of the object
ESTIMATED_HEIGHT = 0.12
ESTIMATED_WIDTH = 0.08

# We only move the drone if the control signal is greater than 
# this threshold in meters
DESIRED_DISTANCE = 0.5

class DroneControl:
    def __init__(self):
        self.tello = tello.Tello()
        self.tello.connect()
        self.tello.set_speed(SPEED)

        self.bbox = None  # Bounding box coordinates on the frame
        self.bbox_mutex = threading.Lock()

        self.control = np.zeros((4,)) # x, y, z, yaw
        self.running = False

    def takeoff(self):
        self.tello.takeoff()

    def _checkSendControl(self):
        # Transform command to cm
        self.control *= 100

        # The command is only valid between -20cm and 20cm
        is_x_valid = abs(self.control[0]) > 20
        is_y_valid = abs(self.control[1]) > 20

        if is_x_valid and is_y_valid:
            self.tello.go_xyz_speed(int(self.control[0]), int(self.control[1]), int(self.control[2]), SPEED)
        else:
            if is_x_valid:
                self._moveX(int(self.control[0]))
            if is_y_valid:
                self._moveY(int(self.control[1]))

        # TODO: Keep the object in view
        # if np.linalg.norm(self.control[3]) > ANG_TH:
        #     self.tello.rotate_clockwise(self.control[3])

    def _moveX(self, dist):
        if dist > 0:
            self.tello.move_forward(abs(dist))
        else:
            self.tello.move_back(abs(dist))

    def _moveY(self, dist):
        if dist > 0:
            self.tello.move_left(abs(dist))
        else:
            self.tello.move_right(abs(dist))

    def _loopHeightLocked(self):
        """
            Execute the control policy. If the BBox moves left or right, apply that control.
            If the BBox moves up or down, since the drone is locked in height, move backward.
            Try to keep the estimated distance based on the size of the detection.
        """
        while self.running:
            if self.bbox is not None:
                x, y, w, h = self.bbox
                center_x = x + w // 2
                center_y = y + h // 2

                # The estimated distance can be computed from the est height and width
                # and the actual detection height and width
                est_dist = min(ESTIMATED_WIDTH / w, ESTIMATED_HEIGHT / h) * F_X

                u_x = IMAGE_SIZE[0] // 2 - center_x
                self.control[1] = u_x * est_dist / (np.sqrt(F_X**2 + u_x**2))
                # self.control[0] = u_y * est_dist / (np.sqrt(F_X**2 + u_y**2))

                self.control[0] = - (DESIRED_DISTANCE - est_dist)
                # self._checkSendControl()

            time.sleep(1/CONTROL_HZ)

    def start(self):
        self.running = True
        self.control_thread = threading.Thread(target=self._loopHeightLocked)
        self.control_thread.start()

    def stop(self):
        self.running = False
        self.tello.streamoff()
        self.control_thread.join()
                

class VideoDetector:

    def __init__(self, drone_control):
        self.drone_control = drone_control
        self.drone_control.tello.streamon()
        self.webcam = cv2.VideoCapture(0)
        self.video_mutex = threading.Lock()
        self.current_frame = None
        self.model = YOLO(MODEL_NAME)
        self.running = True  # Control flag
        
        self.video_thread = threading.Thread(target=self.stream_video)
        self.video_thread.start()

        self.detection_thread = threading.Thread(target=self.detect)
        self.detection_thread.start()
        

    def stream_video(self):
        """
            This has a latency on the ordert of 1-2 ms
            djitellopy opens a video stream and delaying it will only make
            frames arrive slower and fill a buffer but it won't slow down the video.
        """
        while self.running:
                frame = self.drone_control.tello.get_frame_read().frame
                
                # ret, frame = self.webcam.read()
                with self.video_mutex:
                    self.current_frame = frame
                frame_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("Frame", frame_cv2)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    self.stop("video")

    def detect(self):
        """
            YOLO has a latency on the order of 45-50 ms
            on an Intel i9 CPU
        """
        # TODO: Include a Kalman filter
        while self.running:
            if self.current_frame is not None:
                is_plot = False
                with self.video_mutex:
                    frame = self.current_frame
                results = self.model(frame, verbose=False)
                for result in results:
                    boxes = result.boxes

                    set_bbox = False
                    
                    for box in boxes:
                        if result.names[box.cls.item()] == DETECTION_QUERY:
                            is_plot = True
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            if not set_bbox:
                                with self.drone_control.bbox_mutex:
                                    self.drone_control.bbox = (x1, y1, x2 - x1, y2 - y1)
                if is_plot:
                    frame_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imshow("Detection", frame_cv2)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                        self.stop("detection")

    def stop(self, thread):
        """Stops the detection and video threads."""
        self.running = False
        if thread == "video":
            self.detection_thread.join()
        elif thread == "detection":
            self.video_thread.join()

        if self.drone_control.running:
            self.drone_control.stop()

        # self.webcam.release()
        cv2.destroyAllWindows()
        exit()

if __name__ == "__main__":
    drone_control = DroneControl()
    # drone_control.takeoff()
    video_detector = VideoDetector(drone_control)
    drone_control.start()

    
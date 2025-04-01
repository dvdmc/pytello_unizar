import threading
import numpy as np
import time
from djitellopy import tello
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SPEED = 10
FLIGHT_ENABLED = True

IMAGE_SIZE = (940, 720)
F_X = 940
CONTROL_HZ = 10  # Hz

MODEL_NAME = "yolov8m.pt"

DETECTION_DICTIONARY = {
    "bottle": {"name": "bottle", "height": 0.205, "width": 0.07},
    "orange": {"name": "orange", "height": 0.035, "width": 0.05},
    "sports ball": {"name": "sports ball", "height": 0.20, "width": 0.20},
}
SELECTED_CLASS = "sports ball"

DETECTION_QUERY = DETECTION_DICTIONARY[SELECTED_CLASS]["name"]
# We compute the estimated distance based on the estimated size of the object
ESTIMATED_HEIGHT = DETECTION_DICTIONARY[SELECTED_CLASS]["height"]
ESTIMATED_WIDTH = DETECTION_DICTIONARY[SELECTED_CLASS]["width"]

# We only move the drone if the control signal is greater than
# this threshold in meters
DESIRED_DISTANCE = 0.75


class DroneControl:
    def __init__(self):
        self.tello = tello.Tello()
        self.tello.connect()
        self.tello.set_speed(SPEED)

        self.bbox = None  # Bounding box coordinates on the frame
        self.bbox_mutex = threading.Lock()

        self.control = np.zeros((4,))  # x, y, z, yaw
        self.running = False

    def takeoff(self):
        self.tello.takeoff()

    def _checkSendControl(self):
        if not FLIGHT_ENABLED:
            return

        # Transform command to cm
        self.control[:2] *= 100
        self.control[3] = self.control[3] * 180 / np.pi

        # The command is only valid between -20cm and 20cm
        is_x_valid = abs(self.control[0]) > 20
        is_y_valid = abs(self.control[1]) > 20

        if is_x_valid and is_y_valid:
            print("Sent go_xyz_speed command")
            self.tello.go_xyz_speed(
                int(self.control[0]), int(self.control[1]), int(self.control[2]), SPEED
            )
        else:
            if is_x_valid:
                self._moveX(int(self.control[0]))
            if is_y_valid:
                self._moveY(int(self.control[1]))

        # TODO: Keep the object in view
        print(f"Yaw: {self.control[3]}")
        if abs(self.control[3]) > 5:
            print("Sent rotate command")
            self.tello.rotate_counter_clockwise(int(self.control[3]))

    def _moveX(self, dist):
        if dist > 0:
            print("Sent forward command")
            self.tello.move_forward(abs(dist))
        else:
            print("Sent backward command")
            self.tello.move_back(abs(dist))

    def _moveY(self, dist):
        if dist > 0:
            print("Sent left command")
            self.tello.move_left(abs(dist))
        else:
            print("Sent right command")
            self.tello.move_right(abs(dist))

    def _loopHeightLocked(self):
        """
        Execute the control policy. If the BBox moves left or right, apply that control.
        If the BBox moves up or down, since the drone is locked in height, move backward.
        Try to keep the estimated distance based on the size of the detection.
        """
        while self.running:
            self.control = np.zeros((4,))
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

                self.control[0] = -(DESIRED_DISTANCE - est_dist)

                self.control[3] = np.arcsin(
                    (u_x * est_dist / (np.sqrt(F_X**2 + u_x**2))) / est_dist
                )

                # If center is too low, command to land
                print(center_y)
                if (center_y > IMAGE_SIZE[1] * 0.8) and FLIGHT_ENABLED:
                    self.tello.land()

                # print(f"Control: {self.control}. Yaw: {self.control[3] * 180 / np.pi}")

                self._checkSendControl()

            time.sleep(1 / CONTROL_HZ)

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

        # self.webcam = cv2.VideoCapture(0)
        self.model = YOLO(MODEL_NAME)
        self.running = True  # Control flag

    def run(self):
        """
        This has a latency on the ordert of 1-2 ms
        djitellopy opens a video stream and delaying it will only make
        frames arrive slower and fill a buffer but it won't slow down the video.
        """

        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots()
        img_display = self.ax.imshow(
            np.zeros((720, 940, 3), dtype=np.uint8)
        )  # Placeholder image
        self.detection_rect = patches.Rectangle(  # Dummy patch
            (0, 0),
            0,
            0,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        self.ax.add_patch(self.detection_rect)  # Add the rectangle to the plot

        while self.running:
            frame = self.drone_control.tello.get_frame_read().frame
            detection_frame = self.detect(frame)
            img_display.set_data(detection_frame)  # Update displayed image
            plt.draw()  # Trigger the plot to update
            plt.pause(0.001)

    def detect(self, frame):
        """
        YOLO has a latency on the order of 45-50 ms
        on an Intel i9 CPU
        """
        # TODO: Include a Kalman filter
        results = self.model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            set_bbox = False

            for box in boxes:
                if result.names[box.cls.item()] == DETECTION_QUERY:
                    is_plot = True
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    self.detection_rect.set_xy((x1, y1))
                    self.detection_rect.set_width(x2 - x1)
                    self.detection_rect.set_height(y2 - y1)

                    if not set_bbox:
                        with self.drone_control.bbox_mutex:
                            self.drone_control.bbox = (x1, y1, x2 - x1, y2 - y1)
                            set_bbox = True

            if not set_bbox:
                with self.drone_control.bbox_mutex:
                    self.drone_control.bbox = None

        return frame

    def stop(self):
        self.running = False
        if self.drone_control.running:
            self.drone_control.stop()
        # self.webcam.release()
        exit()


if __name__ == "__main__":
    drone_control = DroneControl()
    if FLIGHT_ENABLED:
        drone_control.takeoff()
    video_detector = VideoDetector(drone_control)
    drone_control.start()
    video_detector.run()

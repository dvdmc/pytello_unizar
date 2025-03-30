# PyTello Unizar

This is a basic example on how to program a DJI Tello Drone with camera interaction.
The `tracking_example.py` opens a video connection, detects objects and commands the drone
to keep the detected object at the center and at a desired distance.

## Install 

Create a Python environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows
# .venv/Scripts/activate
```

Install the requirements:

```bash
python -m pip -r requirements.txt install
```

## Usage

Configure the parameters in the `tracking_example.py` script. Important ones are:
- `DETECTION_QUERY`: The name of the class to detect. When using YOLO, the available classes are in `classes.txt`.
- `ESTIMATED_HEIGHT` and `ESTIMATED_WIDTH`: These are the estimated height and width of the detected object. This will be used to estimate the distance to the object and compute the commands for the drone.
- `DESIRED_DISTANCE`: The desired distance from the drone to the object.

**Notice:** The Tello drone is not able to perform movements below 20 cm. Thus, if the desired distance is too small, the drone might not be able to center the object and keep it in frame at the same time. The minimum recommended distance is 0.5 m

Turn on the Tello drone, connect from the computer to the WiFi point it creates (e.g., ). Place the drone in a safe space for *taking off*, it will raise 1 m. Then, run the example:

```bash
python tracking_example.py
```
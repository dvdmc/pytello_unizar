# PyTello Unizar

This is a basic example on how to program a DJI Tello Drone with camera interaction.
The `tracking_example.py` opens a video connection, detects objects and commands the drone
to keep the detected object at the center and at a desired distance.

## Install 

Clone the repository:

```bash
git clone https://github.com/dvdmc/pytello_unizar
cd pytello_unizar
```

Create a Python environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows
# .venv/Scripts/activate
```

Install the requirements:

```bash
python -m pip install -r requirements.txt
```

Finally, to download the YOLO model, just execute the `download_yolo.py` script. Make sure that the YOLO model name in this file and `tracking_example.py` are the same.
```bash
python download_yolo.py
```

## Usage

Configure the parameters in the `tracking_example.py` script. Adjust  Important ones are:
- **Detection parameters:** They are configured by using the `DETECTION_DICTIONARY` with a list of possible objects, and the `SELECTED_CLASS` to choose one from the dictionary. You will need to add the name of the class from `classes.txt`, and the estimated height and width of the object. These will be used to estimate the distance to the object and compute the commands for the drone.
- **Desired distance:** Adjust the variable `DESIRED_DISTANCE` from the drone to the object.

**Notice:** The Tello drone is not able to perform movements below 20 cm. Thus, if the desired distance is too small, the drone might not be able to center the object and keep it in frame at the same time. The minimum recommended distance is 0.5 m, the best distance is 1 m if the object is large enough.

### Demo
Assuming you are in the repository's top folder:

1. If you didn't do it, source in the Python virtual environment:
```bash
source .venv/bin/activate
```
2. Turn on the Tello drone and connect from the computer to the WiFi point it creates (e.g., TELLO-XXXX). Then, run the script to check if you can connect, the camera turns on, and the drone detects the object.

3. Change the variable `FLIGHT_ENABLED` to `True`. Place the drone in a safe space for *taking off*, it will raise 1 m. Then, run the example:

```bash
python tracking_example.py
```

You should show the Tello the object to track. Notice that in order for the object to be in the frame, you need to place it lower than expected.

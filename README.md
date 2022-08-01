# Edge Detection on Conveyor Line Images
The script finds you the edges (or corners) of the conveyors images. 
One can see the algorithm bellow:

1. Get the Image from the user.
2. Apply a Gaussian Filter that has big window size.
3. Apply Otsu's Method to make the image binary.
4. Firstly dilate, and then erode the binary image to remove unnecessary objects.
5. Use Canny's Algorithm to detect the borders of the remaining objects.
6. Use Hough transform to find line ones in that borders.
7. Put the lines that is found using Hough transform to the original image.

## Usage
As a first step, clone this repository into your host machine.
```bash
~$ cd ConvayorBandEdgeDetection # The directory of the repo in your host machine.
~$ virtualenv . # Create a virtual environment.
~$ source bin/activate # Activate the virtual environment. (if GNU/Linux)
(venv) ~$ pip install -r requirements.txt # Install the requirement modules.
# Installation has completed. 
(venv) ~$ python detect_edges.py --inputFile file/path/to/image.jpg
# Since it is a OpenCV window, you can close it with pressing ESC.
(venv) ~$ deactivate # To deactivate the virtual environment.
```
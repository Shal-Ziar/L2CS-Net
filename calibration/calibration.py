""" This function allows for a 9 point calibration of the user
 the callibration locations are as follows
1. top left, 2. middle left, 3. bottom left, 4. middle top, 5. middle middle, 6. middle bottom
7. right top, 8 right middle, 9 right bottom.
The user is tasked with looking at each individual point in order and pressing the spacebar to advance.
When the spacebar is pressed, the gaze-models output is stored as a calibration point [time, point, values]
This continues until all the points have been tested.

Interface
A full-screen interface that autoscales based on the size of the screen, or a pre-set size.
all point will be given as gray crosshairs on a white background. The active point will be a red crosshair.

 """

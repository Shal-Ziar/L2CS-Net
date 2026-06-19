This document decribe building virtual cursor

# Needed:
1. calibration function
    - This function map gaze to point on screen and tells how big screen is
    - give mapping from yaw/pitch to x, y pixel coord
2. l2cs gaze model should run
    Gaze model give gaze angles needed for cursor, gives gaze angles
3. pygame screen to show cursor on
    Use size of screen from calibration, full-screen allowed
    Visualise cursor as 2d ellipse with width based on error.
    calculate error based on standard deviation of gaze.
    calculate and visualise this error continuously

# next steps
1. create a simple trial to gauge the accuracy of the cursor
    - This be done by drawing simple shapes, square, rectangle circle on screen
    - ask user to look at start point, if converged slowly follow shape
    - calculate error metrics at end of trial, show heatmap of error, give rms-error of path


# Goal
1. Have a simple cursor visualiser
2. measure virtual cursor accuracy

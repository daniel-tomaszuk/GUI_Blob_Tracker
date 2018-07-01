# GUI_Blob_Tracker

## Demo: https://www.youtube.com/watch?v=HYkQH48euM8&t=

## Description: https://github.com/dghy/GUI_Blob_Tracker/blob/master/Algorithm%20description.pdf

Python GUI Multiple Blob Tracker created in PyForms. Tracker uses Kalman filter and Munkres algorithm for multiple
blobs tracking. Preview of video allows to check how different operations influence preprocessed frames.
List of possible operations:
- selection of color channel, 
- threshold,
- CLAHE (Contrast Limited Adaptive Histogram Equalization),
- ROI (Region Of Interest) indication,
- morphological operations: dilatation, erosion, opening, closing with different kernel sizes and types,
- Laplacian Of Gaussian (LoG) filtration with different kernel sizes.

It is a must to use LoG for now, because algorithm looks for local image maximas (which are treated as blob positions)
which LoG generates.

GUI uses OpenCV for image processing. As output, program generates indexed blob trajectories.
The example of multi blob video is attached in the project folder: CIMG4027.MOV. Parameters for CIMG4027.MOV video are visible in the 'parameters.png' picture. 

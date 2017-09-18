#!/usr/bin/python
# -*- coding: utf-8 -*-

from pyforms import BaseWidget
from pyforms.Controls import ControlText
from pyforms.Controls import ControlButton
from pyforms.Controls import ControlSlider
from pyforms.Controls import ControlFile
from pyforms.Controls import ControlPlayer
import pyforms
import cv2
from functions import select_frames, read_image, blob_detect, get_log_kernel
from functions import img_inv, local_maxima, pair, video_analise, kalman
from functions import plot_points
from functions import *


class MultipleBlobDetection(BaseWidget):
    def __init__(self):
        super(MultipleBlobDetection, self).__init__(
            'Multiple Blob Detection')

        # Definition of the forms fields
        self._videofile = ControlFile('Video')
        self._outputfile = ControlText('Results output file')
        self._threshold = ControlSlider('Binary Threshold', 114, 0, 255)
        # self._blobsize = ControlSlider('Minimum blob size', 100, 100, 2000)
        self._player = ControlPlayer('Player')
        self._runbutton = ControlButton('Run')

        # Define the function that will be called when a file is selected
        self._videofile.changed_event = self.__videoFileSelectionEvent
        # Define the event that will be called when the run button is processed
        self._runbutton.value = self.__runEvent
        # Define the event called before showing the image in the player
        self._player.process_frame_event = self.__processFrame
        # Define the organization of the Form Controls
        self.formset = [
            ('_videofile', '_outputfile'),
            '_threshold',
            '_runbutton',
            '_player'
        ]

    def __videoFileSelectionEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self._player.value = self._videofile.value

    def __processFrame(self, frame):
        """
        Do some processing to the frame and return the result frame
        """
        # kernel for morphological operations
        # check cv2.getStructuringElement() doc for more info
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray_frame, self._threshold.value, 255,
                                 cv2.THRESH_BINARY)
        # prepare image - morphological operations
        erosion = cv2.erode(th1, erosion_kernel, iterations=1)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, opening_kernel)
        dilate = cv2.dilate(opening, dilate_kernel, iterations=2)
        # create LoG kernel for finding local maximas
        log_img = cv2.filter2D(dilate, cv2.CV_32F, get_log_kernel(30, 15))
        # log_img = cv2.filter2D(dilate, -1, get_log_kernel(30, 15))
        log_img *= 255
        # remove near 0 floats
        log_img[log_img < 0] = 0
        return log_img

    def __runEvent(self):
        """
        After setting the best parameters run the full algorithm
        """
        print('RUN pressed')
        print('threshold:', self._threshold)
        pass


# Execute the application
if __name__ == "__main__":
    pyforms.start_app(MultipleBlobDetection)

print('EOF')

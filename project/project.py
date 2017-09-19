#!/usr/bin/python
# -*- coding: utf-8 -*-

from pyforms import BaseWidget
from pyforms.Controls import ControlText
from pyforms.Controls import ControlButton
from pyforms.Controls import ControlSlider
from pyforms.Controls import ControlFile
from pyforms.Controls import ControlPlayer
from pyforms.Controls import ControlCheckBox
from pyforms.Controls import ControlCombo
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
        self._start_frame = ControlText('Start Frame')
        self._stop_frame = ControlText('Stop Frame')

        self._dilate = ControlCheckBox('Dilate')
        self._dilate_type = ControlCombo('Dilate Kernel Type')
        self._dilate_type.add_item('RECTANGLE', cv2.MORPH_RECT)
        self._dilate_type.add_item('ELLIPSE', cv2.MORPH_ELLIPSE)
        self._dilate_type.add_item('CROSS', cv2.MORPH_CROSS)
        self._dilate_size = ControlSlider('Dilate Kernel Size', 3, 1, 10)


        self._erode = ControlCheckBox('Erode')
        self._erode_type = ControlCombo('Erode Kernel Type')
        self._erode_type.add_item('RECTANGLE', cv2.MORPH_RECT)
        self._erode_type.add_item('ELLIPSE', cv2.MORPH_ELLIPSE)
        self._erode_type.add_item('CROSS', cv2.MORPH_CROSS)
        self._erode_size = ControlSlider('Open Kernel Size', 5, 1, 10)


        self._open = ControlCheckBox('Open')
        self._open_type = ControlCombo('Open Kernel Type')
        self._open_type.add_item('RECTANGLE', cv2.MORPH_RECT)
        self._open_type.add_item('ELLIPSE', cv2.MORPH_ELLIPSE)
        self._open_type.add_item('CROSS', cv2.MORPH_CROSS)
        self._open_size = ControlSlider('Open Kernel Size', 19, 1, 40)

        self._close = ControlCheckBox('Close')
        self._close_type = ControlCombo('Close Kernel Type')
        self._close_type.add_item('RECTANGLE', cv2.MORPH_RECT)
        self._close_type.add_item('ELLIPSE', cv2.MORPH_ELLIPSE)
        self._close_type.add_item('CROSS', cv2.MORPH_CROSS)
        self._close_size = ControlSlider('Close Kernel Size', 19, 1, 40)

        self._LoG = ControlCheckBox('LoG - Laplacian of Gaussian')

        # Define the function that will be called when a file is selected
        self._videofile.changed_event = self.__videoFileSelectionEvent
        # Define the event that will be called when the run button is processed
        self._runbutton.value = self.__runEvent
        # Define the event called before showing the image in the player
        self._player.process_frame_event = self.__processFrame

        # Define the organization of the Form Controls
        self.formset = [
            ('_videofile', '_outputfile'),
            ('_start_frame', '_stop_frame'),
            '_threshold',
            ('_dilate', '_erode', '_open', '_close'),
            ('_dilate_type', '_erode_type', '_open_type', '_close_type'),
            ('_dilate_size', '_erode_size', '_open_size', '_close_size'),
            '_LoG',
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
        # kernels for morphological operations
        # check cv2.getStructuringElement() doc for more info
        opening_kernel = cv2.getStructuringElement(self._open_type.value,
                                                   (self._open_size.value,
                                                    self._open_size.value))
        close_kernel = cv2.getStructuringElement(self._close_type.value,
                                                 (self._close_size.value,
                                                  self._close_size.value))
        erosion_kernel = cv2.getStructuringElement(self._erode_type.value,
                                                   (self._erode_size.value,
                                                    self._erode_size.value))
        dilate_kernel = cv2.getStructuringElement(self._dilate_type.value,
                                                  (self._dilate_size.value,
                                                   self._dilate_size.value))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(gray_frame, self._threshold.value, 255,
                                 cv2.THRESH_BINARY)
        # prepare image - morphological operations
        if self._erode.value:
            frame = cv2.erode(frame, erosion_kernel, iterations=1)
        if self._open.value:
            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, opening_kernel)
        if self._close.value:
            frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, close_kernel)

        if self._dilate.value:
            frame = cv2.dilate(frame, dilate_kernel, iterations=1)
        # create LoG kernel for finding local maximas
        if self._LoG.value:
            frame = cv2.filter2D(frame, cv2.CV_32F, get_log_kernel(30, 15))
            # log_img = cv2.filter2D(dilate, -1, get_log_kernel(30, 15))
            frame *= 255
            # remove near 0 floats
            frame[frame < 0] = 0
        return frame

    def __runEvent(self):
        """
        After setting the best parameters run the full algorithm
        """
        start_frame = int(self._start_frame.value)
        stop_frame = int(self._stop_frame.value)
        # pass object, not string
        my_video = self._player.value
        # ####################################################################
        maxima_points, vid_fragment = video_analise(my_video, start_frame,
                                                    stop_frame)
        x_est, y_est, est_number = kalman(maxima_points, stop_frame,
                                          vid_fragment)

        print('\nFinal estimates number:', est_number)
        plot_points(vid_fragment, maxima_points, x_est, y_est, est_number)
        print('EOF - DONE')

# Execute the application
if __name__ == "__main__":
    pyforms.start_app(MultipleBlobDetection)

print('EOF')

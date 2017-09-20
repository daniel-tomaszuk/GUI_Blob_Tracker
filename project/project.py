#!/usr/bin/python
# -*- coding: utf-8 -*-

from pyforms import BaseWidget
from pyforms.Controls import ControlText
from pyforms.Controls import ControlButton
from pyforms.Controls import ControlSlider
from pyforms.Controls import ControlFile
from pyforms.Controls import ControlPlayer
from pyforms.Controls import ControlCheckBox
from pyforms.Controls import ControlCheckBoxList
from pyforms.Controls import ControlCombo
from pyforms.Controls import ControlProgress
import pyforms
import cv2
from functions import *


class MultipleBlobDetection(BaseWidget):
    def __init__(self):
        super(MultipleBlobDetection, self).__init__(
            'Multiple Blob Detection')

        # Definition of the forms fields
        self._videofile = ControlFile('Video')
        self._outputfile = ControlText('Results output file')

        self._threshold_box = ControlCheckBox('Threshold')
        self._threshold = ControlSlider('Binary Threshold', 114, 0, 255)

        # self._blobsize = ControlSlider('Minimum blob size', 100, 100, 2000)
        self._player = ControlPlayer('Player')
        self._runbutton = ControlButton('Run')
        self._start_frame = ControlText('Start Frame')
        self._stop_frame = ControlText('Stop Frame')

        self._color_list = ControlCombo('Color channels')
        self._color_list.add_item('Red Image Channel', 2)
        self._color_list.add_item('Green Image Channel', 1)
        self._color_list.add_item('Blue Image Channel', 0)

        self._clahe = ControlCheckBox('CLAHE - Adaptive contrast correction')
        self._dilate = ControlCheckBox('Morphological Dilation')
        self._dilate_type = ControlCombo('Dilation Kernel Type')
        self._dilate_type.add_item('RECTANGLE', cv2.MORPH_RECT)
        self._dilate_type.add_item('ELLIPSE', cv2.MORPH_ELLIPSE)
        self._dilate_type.add_item('CROSS', cv2.MORPH_CROSS)
        self._dilate_size = ControlSlider('Dilation Kernel Size', 3, 1, 10)

        self._erode = ControlCheckBox('Morphological Erosion')
        self._erode_type = ControlCombo('Erode Kernel Type')
        self._erode_type.add_item('RECTANGLE', cv2.MORPH_RECT)
        self._erode_type.add_item('ELLIPSE', cv2.MORPH_ELLIPSE)
        self._erode_type.add_item('CROSS', cv2.MORPH_CROSS)
        self._erode_size = ControlSlider('Erode Kernel Size', 5, 1, 10)

        self._open = ControlCheckBox('Morphological Opening')
        self._open_type = ControlCombo('Open Kernel Type')
        self._open_type.add_item('RECTANGLE', cv2.MORPH_RECT)
        self._open_type.add_item('ELLIPSE', cv2.MORPH_ELLIPSE)
        self._open_type.add_item('CROSS', cv2.MORPH_CROSS)
        self._open_size = ControlSlider('Open Kernel Size', 19, 1, 40)

        self._close = ControlCheckBox('Morphological Closing')
        self._close_type = ControlCombo('Close Kernel Type')
        self._close_type.add_item('RECTANGLE', cv2.MORPH_RECT)
        self._close_type.add_item('ELLIPSE', cv2.MORPH_ELLIPSE)
        self._close_type.add_item('CROSS', cv2.MORPH_CROSS)
        self._close_size = ControlSlider('Close Kernel Size', 19, 1, 40)

        self._LoG = ControlCheckBox('LoG - Laplacian of Gaussian')
        self._LoG_size = ControlSlider('LoG Kernel Size', 30, 1, 60)

        # self._load_bar = ControlProgress()

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
            ('_color_list', '_clahe'),
            ('_threshold_box', '_threshold'),
            ('_dilate', '_erode', '_open', '_close'),
            ('_dilate_type', '_erode_type', '_open_type', '_close_type'),
            ('_dilate_size', '_erode_size', '_open_size', '_close_size'),
            ('_LoG', '_LoG_size'),
            '_runbutton',
            '_player'
        ]

    def __color_channel(self, frame):
        """
        Returns only one color channel of input frame.
        Output is in grayscale.
        """
        frame = frame[:, :, self._color_list.value]
        return frame

    def __create_kernels(self):
        """
        Creates kernels for morphological operations.
        Check cv2.getStructuringElement() doc for more info:
        http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/
        py_morphological_ops/py_morphological_ops.html

        Assumed that all kernels (except LoG kernel) are square.
        Example of use:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        :return: _opening_kernel, _close_kernel, _erosion_kernel, \
            _dilate_kernel, _LoG_kernel
        """
        _opening_kernel = cv2.getStructuringElement(self._open_type.value,
                                                    (self._open_size.value,
                                                     self._open_size.value))
        _close_kernel = cv2.getStructuringElement(self._close_type.value,
                                                  (self._close_size.value,
                                                   self._close_size.value))
        _erosion_kernel = cv2.getStructuringElement(self._erode_type.value,
                                                    (self._erode_size.value,
                                                     self._erode_size.value))
        _dilate_kernel = cv2.getStructuringElement(self._dilate_type.value,
                                                   (self._dilate_size.value,
                                                    self._dilate_size.value))
        _LoG_kernel = get_log_kernel(self._LoG_size.value,
                                     int(self._LoG_size.value * 0.5))
        return _opening_kernel, _close_kernel, _erosion_kernel, \
            _dilate_kernel, _LoG_kernel

    def __morphological(self, frame):
        """
        Apply morphological operations selected by the user.
        :param frame: input frame of selected video.
        :return: preprocessed frame.
        """
        opening_kernel, close_kernel, erosion_kernel, \
            dilate_kernel, log_kernel = self.__create_kernels()
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
            frame = cv2.filter2D(frame, cv2.CV_32F, log_kernel)
            frame *= 255
            # remove near 0 floats
            frame[frame < 0] = 0
        return frame

    def __videoFileSelectionEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self._player.value = self._videofile.value

    def __processFrame(self, frame):
        """
        Do some processing to the frame and return the result frame
        """
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = self.__color_channel(frame)
        if self._clahe.value:
            clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
            frame = clahe.apply(frame)
        if self._threshold_box.value:
            ret, frame = cv2.threshold(frame, self._threshold.value, 255,
                                       cv2.THRESH_BINARY)
            frame = self.__morphological(frame)
        return frame

    def __runEvent(self):
        """
        After setting the best parameters run the full algorithm
        """
        if not self._start_frame.value or not self._stop_frame.value or \
                self._start_frame.value >= self._stop_frame.value:
            raise ValueError('Wrong start or stop frame!')
        start_frame = int(self._start_frame.value)
        stop_frame = int(self._stop_frame.value)
        # pass cv2.VideoCapture object, not string
        # my_video = self._player.value
        video = self._player.value
        # self._load_bar.__init__('Processing..')
        vid_fragment = select_frames(video, start_frame, stop_frame)
        try:
            height = vid_fragment[0].shape[0]
            width = vid_fragment[0].shape[1]
        except IndexError:
            raise IndexError('No video loaded. Check video path.')

        i = 0
        bin_frames = []
        # preprocess image loop
        for frame in vid_fragment:
            # if cv2.waitKey(15) & 0xFF == ord('q'):
            #     break
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = self.__color_channel(frame)
            for m in range(height):  # height
                for n in range(width):  # width
                    if n > 385 or m > 160:
                        gray_frame[m][n] = 120

            # create a CLAHE object (Arguments are optional)
            if self._clahe.value:
                clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
                gray_frame = clahe.apply(gray_frame)
            ret, th1 = cv2.threshold(gray_frame, self._threshold.value, 255,
                                     cv2.THRESH_BINARY)
            # frame_thresh1 = otsu_binary(cl1)
            bin_frames.append(th1)
            if i % 10 == 0:
                print(i)
            i += 1
        ######################################################################
        i = 0
        maxima_points = []
        # gather measurements loop
        for frame in bin_frames:
            frame = self.__morphological(frame)
            # get local maximas of filtered image per frame
            maxima_points.append(local_maxima(frame))
            if i % 10 == 0:
                print(i)
            i += 1


        x_est, y_est, est_number = kalman(maxima_points, stop_frame,
                                          vid_fragment)

        print('\nFinal estimates number:', est_number)
        plot_points(vid_fragment, maxima_points, x_est, y_est, est_number)
        print('EOF - DONE')

# Execute the application
if __name__ == "__main__":
    pyforms.start_app(MultipleBlobDetection)

print('EOF')

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

        self._roi_x_min = ControlSlider('ROI x top', 0, 0, 1000)
        self._roi_x_max = ControlSlider('ROI x bottom', 1000, 0, 1000)

        self._roi_y_min = ControlSlider('ROI y left', 0, 0, 1000)
        self._roi_y_max = ControlSlider('ROI y right', 1000, 0, 1000)

        # self._blobsize = ControlSlider('Minimum blob size', 100, 100, 2000)
        self._player = ControlPlayer('Player')
        self._runbutton = ControlButton('Run')
        self._start_frame = ControlText('Start Frame')
        self._stop_frame = ControlText('Stop Frame')

        self._color_list = ControlCombo('Color channels')
        self._color_list.add_item('Red Image Channel', 2)
        self._color_list.add_item('Green Image Channel', 1)
        self._color_list.add_item('Blue Image Channel', 0)

        self._clahe = ControlCheckBox('CLAHE      ')
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

        self._progress_bar = ControlProgress('Progress Bar')

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
            ('_color_list', '_clahe', '_roi_x_min', '_roi_y_min'),
            ('_threshold_box', '_threshold', '_roi_x_max', '_roi_y_max'),
            ('_dilate', '_erode', '_open', '_close'),
            ('_dilate_type', '_erode_type', '_open_type', '_close_type'),
            ('_dilate_size', '_erode_size', '_open_size', '_close_size'),
            ('_LoG', '_LoG_size'),
            '_runbutton',
            '_progress_bar',
            '_player'
        ]

    def __videoFileSelectionEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self._player.value = self._videofile.value

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

    def __roi(self, frame):
        """
        Define image region of interest.
        """
        # ROI
        height, width = frame.shape
        self._roi_x_max.min = int(height / 2)
        self._roi_x_max.max = height
        self._roi_y_max.min = int(width / 2)
        self._roi_y_max.max = width

        self._roi_x_min.min = 0
        self._roi_x_min.max = int(height / 2)
        self._roi_y_min.min = 0
        self._roi_y_min.max = int(width / 2)
        # x axis
        frame[:int(self._roi_x_min.value)][::] = 255
        frame[int(self._roi_x_max.value)::][::] = 255
        # y axis
        for m in range(height):  # height
            for n in range(width):  # width
                if n > self._roi_y_max.value or n < self._roi_y_min.value:
                    frame[m][n] = 255

        # frame[0::][:int(self._roi_y_min.value)] = 255
        # frame[0::][int(self._roi_y_max.value):] = 255
        return frame

    def _kalman(self, max_points, stop_frame, vid_fragment):
        """
        Kalman Filter function. Takes measurements from video analyse function
        and estimates positions of detected objects. Munkres algorithm is used
        for assignments between estimates (states) and measurements.
        :param max_points: measurements.
        :param stop_frame: number of frames to analise
        :param vid_fragment: video fragment for estimates displaying
        :return: x_est, y_est - estimates of x and y positions in the following
                 format: x_est[index_of_object][frame] gives x position of object
                 with index = [index_of_object] in the frame = [frame]. The same
                 goes with y positions.
        """
        # font for displaying info on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        index_error = 0
        value_error = 0
        # step of filter
        dt = 1.
        R_var = 1  # measurements variance between x-x and y-y
        # Q_var = 0.1  # model variance
        # state covariance matrix - no initial covariances, variances only
        # [10^2 px, 10^2 px, ..] -
        P = np.diag([100, 100, 10, 10, 1, 1])
        # state transition matrix for 6 state variables
        # (position - velocity - acceleration,
        # x, y)
        F = np.array([[1, 0, dt, 0, 0.5 * pow(dt, 2), 0],
                      [0, 1, 0, dt, 0, 0.5 * pow(dt, 2)],
                      [0, 0, 1, 0, dt, 0],
                      [0, 0, 0, 1, 0, dt],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        # x and y coordinates only - measurements matrix
        H = np.array([[1., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0.]])
        # no initial corelation between x and y positions - variances only
        R = np.array(
            [[R_var, 0.], [0., R_var]])  # measurement covariance matrix
        # Q must be the same shape as P
        Q = np.diag([100, 100, 10, 10, 1, 1])  # model covariance matrix

        # create state vectors, max number of states - as much as frames
        x = np.zeros((stop_frame, 6))
        # state initialization - initial state is equal to measurements
        m = 0
        try:
            for i in range(len(max_points[0])):
                if max_points[0][i][0] > 0 and max_points[0][i][1] > 0:
                    x[m] = [max_points[0][i][0], max_points[0][i][1],
                            0, 0, 0, 0]
                    m += 1
        # required for django runserver tests
        except IndexError:
            index_error = 1

        est_number = 0
        # number of estimates at the start
        try:
            for point in max_points[::][0]:
                if point[0] > 0 and point[1] > 0:
                    est_number += 1
        except IndexError:
            index_error = 1

        # history of new objects appearance
        new_obj_hist = [[]]
        # difference between position of n-th object in m-1 frame and position
        # of the same object in m frame
        diff_2 = [[]]
        # for how many frames given object was detected
        frames_detected = []
        # x and y posterior positions (estimates) for drawnings
        x_est = [[] for i in range(stop_frame)]
        y_est = [[] for i in range(stop_frame)]

        # variable for counting frames where object has no measurement
        striked_tracks = np.zeros(stop_frame)
        removed_states = []
        new_detection = []
        ff_nr = 0  # frame number

        self._progress_bar.label = '3/4: Generating position estimates..'
        self._progress_bar.value = 0

        # kalman filter loop
        for frame in range(stop_frame):
            self._progress_bar.value = 100 * (ff_nr / stop_frame)
            # measurements in one frame
            try:
                frame_measurements = max_points[::][frame]
            except IndexError:
                index_error = 1

            measurements = []
            # make list of lists, not tuples; don't take zeros,
            # assuming it's image
            if not index_error:
                for meas in frame_measurements:
                    if meas[0] > 0 and meas[1] > 0:
                        measurements.append([meas[0], meas[1]])
            # count prior
            for i in range(est_number):
                x[i][::] = dot(F, x[i][::])
            P = dot(F, P).dot(F.T) + Q
            S = dot(H, P).dot(H.T) + R
            K = dot(P, H.T).dot(inv(S))
            ##################################################################
            # prepare for update phase -> get (prior - measurement) assignment
            posterior_list = []
            for i in range(est_number):
                if not np.isnan(x[i][0]) and not np.isnan(x[i][1]):
                    posterior_list.append(i)
                    # print(i)
            # print(posterior_list)
            #
            # print('state\n', x[0:est_number, 0:2])
            # print('\n')
            #    temp_matrix = np.array(x[0:est_number, 0:2])
            try:
                temp_matrix = np.array(x[posterior_list, 0:2])
                temp_matrix = np.append(temp_matrix, measurements, axis=0)
            except ValueError:
                value_error = 1

            # print(temp_matrix)
            distance = pdist(temp_matrix, 'euclidean')  # returns vector

            # make square matrix out of vector
            distance = squareform(distance)
            temp_distance = distance
            # remove elements that are repeated - (0-1), (1-0) etc.
            #    distance = distance[est_number::, 0:est_number]
            distance = distance[0:len(posterior_list), len(posterior_list)::]

            # munkres
            row_index, column_index = linear_sum_assignment(distance)
            final_cost = distance[row_index, column_index].sum()
            unit_cost = []
            index = []
            for i in range(len(row_index)):
                # index(object, measurement)
                index.append([row_index[i], column_index[i]])
                unit_cost.append(distance[row_index[i], column_index[i]])

            ##################################################################
            # index correction - take past states into account
            removed_states.sort()
            for removed_index in removed_states:
                for i in range(len(index)):
                    if index[i][0] >= removed_index:
                        index[i][0] += 1
            ##################################################################
            # find object to reject
            state_list = [index[i][0] for i in range(len(index))]
            reject = np.ones(len(posterior_list))
            i = 0
            for post_index in posterior_list:
                if post_index not in state_list:
                    reject[i] = 0
                i += 1
            # check if distance (residual) isn't to high for assignment
            for i in range(len(unit_cost)):
                if unit_cost[i] > 20:
                    print('cost to high, removing', i)
                    reject[i] = 0

            ##################################################################
            # update phase
            for i in range(len(index)):
                # find object that should get measurement next
                # count residual y: measurement - state
                if index[i][1] >= 0:
                    y = np.array([measurements[index[i][1]] -
                                  dot(H, x[index[i][0], ::])])
                    # posterior
                    x[index[i][0], ::] = x[index[i][0], ::] + dot(K, y.T).T
                    # append new positions
                #        if x[i][0] and x[i][1]:
                x_est[index[i][0]].append([x[index[i][0], 0]])
                y_est[index[i][0]].append([x[index[i][0], 1]])
            # posterior state covariance matrix
            P = dot(np.identity(6) - dot(K, H), P)
            print('posterior\n', x[0:est_number, 0:2])
            ##################################################################
            # find new objects and create new states for them
            new_index = []
            measurement_indexes = []
            for i in range(len(index)):
                if index[i][1] >= 0.:
                    # measurements that have assignment
                    measurement_indexes.append(index[i][1])

            for i in range(len(measurements)):
                if i not in measurement_indexes:
                    # find measurements that don't have assignments
                    new_index.append(i)
            new_detection.append([measurements[new_index[i]]
                                  for i in range(len(new_index))])
            # for every detections in the last frame
            for i in range(len(new_detection[len(new_detection) - 1])):
                if new_detection[frame][i] and \
                                new_detection[frame][i][0] > 380:
                    x[est_number, ::] = [new_detection[frame][i][0],
                                         new_detection[frame][i][1], 0, 0, 0,
                                         0]
                    est_number += 1
                    # print('state added', est_number)
                    # print('new posterior\n', x[0:est_number, 0:2])
            ##################################################################
            # find states without measurements and remove them
            no_track_list = []
            for i in range(len(reject)):
                if not reject[i]:
                    no_track_list.append(posterior_list[i])
                    #    print('no_trk_list', no_track_list)
            for track in no_track_list:
                if track >= 0:
                    striked_tracks[track] += 1
                    print('track/strikes', track, striked_tracks[track])
            for i in range(len(striked_tracks)):
                if striked_tracks[i] >= 1:
                    x[i, ::] = [None, None, None, None, None, None]
                    if i not in removed_states:
                        removed_states.append(i)
                    print('state_removed', i)
                ff_nr += 1
                # print(removed_states)
                # print(index)
        return x_est, y_est, est_number

    def _plot_points(self, vid_frag, max_points, x_est, y_est, est_number):
        self._progress_bar.label = '4/4: Plotting - measurements..'
        self._progress_bar.value = 0
        # plot raw measurements
        for frame_positions in max_points:
            for pos in frame_positions:
                plt.plot(pos[0], pos[1], 'r.')
        # try:
        plt.axis([0, vid_frag[0].shape[1], vid_frag[0].shape[0], 0])
        # except IndexError:
        #     index_error = 1
        plt.xlabel('width [px]')
        plt.ylabel('height [px]')
        plt.title('Objects raw measurements')
        ######################################################################
        # image border - 10 px
        x_max = vid_frag[0].shape[1] - 10
        y_max = vid_frag[0].shape[0] - 10

        self._progress_bar.label = '4/4: Plotting - estimates..'
        self._progress_bar.value = 0
        i = 0
        # plot estimated trajectories
        for ind in range(est_number):
            self._progress_bar.value = 100 * (i / est_number)
            i += 1
            # if estimate exists
            if len(x_est[ind]):
                for pos in range(len(x_est[ind])):
                    # don't draw near 0 points and near max points
                    if not np.isnan(x_est[ind][pos][0]) and \
                                    x_est[ind][pos][0] > 10 and \
                                    y_est[ind][pos][0] > 10 and \
                                    x_est[ind][pos][0] < x_max - 10 and \
                                    y_est[ind][pos][0] < y_max - 10:
                        plt.plot(x_est[ind][pos][0], y_est[ind][pos][0], 'g.')
                        # plt.plot(x_est[ind][::], y_est[ind][::], 'g-')
        # print(frame)
        #  [xmin xmax ymin ymax]
        # try:
        plt.axis([0, vid_frag[0].shape[1], vid_frag[0].shape[0], 0])
        # except IndexError:
        #     index_error = 1
        plt.xlabel('width [px]')
        plt.ylabel('height [px]')
        plt.title('Objects estimated trajectories')
        plt.grid()
        plt.show()

    def __processFrame(self, frame):
        """
        Do some processing to the frame and return the result frame
        """
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = self.__color_channel(frame)

        if self._clahe.value:
            clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
            frame = clahe.apply(frame)

        frame = self.__roi(frame)

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
        self._progress_bar.label = '1/4: Creating BW frames..'
        self._progress_bar.value = 0
        for frame in vid_fragment:
            gray_frame = self.__color_channel(frame)
            # create a CLAHE object (Arguments are optional)
            if self._clahe.value:
                clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
                gray_frame = clahe.apply(gray_frame)

            # ROI
            gray_frame = self.__roi(gray_frame)
            ret, th1 = cv2.threshold(gray_frame, self._threshold.value, 255,
                                     cv2.THRESH_BINARY)
            # frame_thresh1 = otsu_binary(cl1)
            bin_frames.append(th1)
            self._progress_bar.value = 100*(i/len(vid_fragment))
            i += 1
        ######################################################################
        i = 0
        maxima_points = []
        # gather measurements loop

        self._progress_bar.label = '2/4: Finding local maximas..'
        self._progress_bar.value = 0
        for frame in bin_frames:
            frame = self.__morphological(frame)
            # get local maximas of filtered image per frame
            maxima_points.append(local_maxima(frame))
            self._progress_bar.value = 100 * (i / len(bin_frames))
            i += 1

        x_est, y_est, est_number = self._kalman(maxima_points, stop_frame,
                                                vid_fragment)

        print('\nFinal estimates number:', est_number)
        self._plot_points(vid_fragment, maxima_points, x_est, y_est,
                          est_number)
        print('EOF - DONE')

# Execute the application
if __name__ == "__main__":
    pyforms.start_app(MultipleBlobDetection)

print('EOF')

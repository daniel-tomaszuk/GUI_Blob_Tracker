import pyforms
from pyforms import BaseWidget
from pyforms.controls import ControlPlayer


class VideoWindow(BaseWidget):

    def __init__(self):

        BaseWidget.__init__(self, 'Video window')

        self._player = ControlPlayer('Player')

        self.formset = ['_player']


if __name__ == "__main__":   pyforms.start_app(VideoWindow)

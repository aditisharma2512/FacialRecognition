"""
This script initialises the open cv library for python and builds a basic facial recognition class.
This class will then be further extended to include expression recognition and facial similarity index in further expansions
"""

__author__ = "Aditi Sharma"  # Created base class
__editor__ = "Saksham Nagpal"  # Added functionality for Facial Recognition

import cv2


class Recognizer:
    """
    This class creates a Facial recognizer using the opencv library. Once a face is recognized, it will present a green
    rectangle around the face to denote it.
    This class will be extended by future classes to classify faces and recognize emotions
    """
    def __init__(self):
        """
        The constructor for the class, reads the XML file provided by https://github.com/shantnu to provide a model to
        read the faces, and creates a cascade classifier object
        """
        self.cascade_path = 'haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(self.cascade_path)
        self.video_capture = None

    def __begin_capture(self) -> None:
        """
        This method starts the video capture.
        :return: None
        """
        self.video_capture = cv2.VideoCapture(0)

    @staticmethod
    def __draw_rectangles(frame, faces) -> None:
        """
        This method draws rectangles around the faces that are recognised
        :param frame: Video frame being read
        :param faces: A list of faces detected
        :return: None
        """
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    @staticmethod
    def __show_image(frame) -> None:
        """
        Present the captured frame on the screen to the user
        :param frame: Video frame being read
        :return: None
        """
        cv2.imshow('Video', frame)

    def begin_recognition(self) -> None:
        """
        This method controls the workflow of the program.
        :return: None
        """
        self.__begin_capture()
        ret, frame = self.video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        self.__draw_rectangles(frame, faces)
        self.__show_image(frame)

    def start_camera(self) -> None:
        """
        This method controls the user's interaction with the program and calls the internal methods
        :return: None
        """
        while True:
            self.begin_recognition()
            if cv2.waitKey(1) and (0xFF == ord('q') or 0xFF == ord('Q')):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = Recognizer()
    fr.start_camera()

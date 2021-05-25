from time import monotonic

import cv2
import depthai as dai
import numpy as np

from neocam.utils.analysis import Analysis
from neocam.utils.detections import filter_body_detections, filter_face_detections
from neocam.utils.frame import to_planar, display_frame


class Device(dai.Device):
    body_detections: list = []
    face_detections: list = []
    anonymize_method: str = "none"
    window: str = ""

    def __init__(self, pipeline: dai.Pipeline, anonymize_method: str = "none"):
        super(Device, self).__init__(pipeline)

        # Define anonymization method
        self.anonymize_method = anonymize_method

        # Start pipeline
        self.startPipeline()

        # Input queue will be used to send video frames to the device.
        self.q_in = self.getInputQueue(name=pipeline.in_stream)
        # Output queue will be used to get nn data from the video frames.
        self.q_body = self.getOutputQueue(
            name=pipeline.out_body, maxSize=4, blocking=False
        )
        self.q_face = self.getOutputQueue(
            name=pipeline.out_face, maxSize=4, blocking=False
        )

        # Initialize analysis
        self.analysis = Analysis()

    @property
    def detections(self):
        return self.body_detections + self.face_detections

    def _send_frame_to_network(self, frame: np.ndarray):
        """Sends the frame as input to the networks"""
        img = dai.ImgFrame()
        img.setData(to_planar(frame, (300, 300)))
        img.setTimestamp(monotonic())
        img.setWidth(300)
        img.setHeight(300)
        self.q_in.send(img)

    def _get_face_detections(self):
        """Gets the detected faces from the network"""
        in_face = self.q_face.tryGet()

        if in_face:
            self.face_detections = filter_face_detections(in_face.detections)

    def _get_body_detections(self):
        """Gets the detected bodies from the network"""
        in_body = self.q_body.tryGet()

        if in_body is not None:
            self.body_detections = filter_body_detections(in_body.detections)
            self.analysis.update(self.body_detections)
        else:
            self.analysis.update(None)

    def _display_frame(self, frame: np.ndarray):
        """Displays given frame in opened window"""
        if frame is not None:
            display_frame(
                self.window,
                frame,
                self.detections,
                anonymize_method=self.anonymize_method,
            )
            cv2.waitKey(5)

    def _process_frame(self, cap) -> bool:
        """Tries to process a frame"""
        read_correctly, frame = cap.read()
        if not read_correctly:
            return False

        self._send_frame_to_network(frame)
        self._get_face_detections()
        self._get_body_detections()
        self._display_frame(frame)

        if cv2.waitKey(1) == ord("q"):
            return False

        if cv2.waitKey(1) == ord("b"):
            self.anonymize_method = "blur"

        if cv2.waitKey(1) == ord("f"):
            self.anonymize_method = "filled"

        if cv2.waitKey(1) == ord("p"):
            self.anonymize_method = "pixelate"

        if cv2.waitKey(1) == ord("n"):
            self.anonymize_method = "none"

        return True

    def stream_video(self, path_video: str, name: str = ""):
        """Streams a video and processes it"""
        self.window = name
        cap = cv2.VideoCapture(path_video)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        while cap.isOpened():
            keep = self._process_frame(cap)
            if not keep:
                break

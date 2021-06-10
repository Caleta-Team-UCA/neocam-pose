from time import monotonic
from typing import List

import cv2
import depthai as dai
import numpy as np

from neocam.pose.analysis import Analysis
from neocam.utils.detections import filter_body_detections, filter_face_detections
from neocam.utils.frame import to_planar, display_frame


class Device(dai.Device):
    body_detections: List[dai.RawImgDetections] = []
    face_detections: List[dai.RawImgDetections] = []
    anonymize_method: str = "none"
    window: str = ""
    width: int = 300
    height: int = 300

    def __init__(self, pipeline: dai.Pipeline, anonymize_method: str = "none"):
        """Loads a pipeline to an OAK-D device, and starts processing a video

        Parameters
        ----------
        pipeline : depthai.Pipeline
            Detection pipeline
        anonymize_method : str, optional
            Face anonymization method, by default "none". Can be modified by keypressing
            during streaming (try holding P)
        """
        super(Device, self).__init__(pipeline)

        # Define anonymization method
        self.anonymize_method = anonymize_method

        # Start pipeline
        self.startPipeline()

        # Input queue will be used to send video frames to the device.
        self.q_in = self.getInputQueue(name=pipeline.input)
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
    def detections(self) -> List[dai.RawImgDetections]:
        """Returns all the detections"""
        return self.body_detections + self.face_detections

    def _send_frame_to_network(self, frame: np.ndarray):
        """Sends the frame as input to the networks

        Parameters
        ----------
        frame : numpy.ndarray
        """
        img = dai.ImgFrame()
        img.setData(to_planar(frame, (self.width, self.height)))
        img.setTimestamp(monotonic())
        img.setWidth(self.width)
        img.setHeight(self.height)
        self.q_in.send(img)

    def _get_face_detections(self):
        """Gets the detected faces from the network"""
        in_face = self.q_face.tryGet()

        if in_face is not None:
            self.face_detections = filter_face_detections(in_face.detections)

    def _get_body_detections(self):
        """Gets the detected bodies from the network"""
        in_body = self.q_body.tryGet()

        if in_body is not None:
            self.body_detections = filter_body_detections(in_body.detections)

    def _display_frame(self, frame: np.ndarray):
        """Displays given frame in opened window

        Parameters
        ----------
        frame : numpy.ndarray
            Image
        """
        if frame is not None:
            display_frame(
                self.window,
                frame,
                self.detections,
                anonymize_method=self.anonymize_method,
            )
            cv2.waitKey(5)

    def _process_frame(self, frame: np.ndarray) -> bool:
        """Process an input frame:
            - Sends the frame to the networks
            - Gets the face detections
            - Gets the body detections
            - Updates the pose analysis and plots it
            - Display the video

        Parameters
        ----------
        frame : numpy.ndarray
            Image

        Returns
        -------
        bool
            True if the frame was processed correctly, otherwise False
        """
        self._get_face_detections()
        self._get_body_detections()
        self.analysis.update(self.body_detections, self.face_detections)
        frame = self.analysis.plot(frame)
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
        """Streams a video and processes it

        Parameters
        ----------
        path_video : str
            Path to the video file
        name : str, optional
            Name of the output video, by default ""
        """
        self.window = name
        cap = cv2.VideoCapture(path_video)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        while cap.isOpened():
            read_correctly, frame = cap.read()

            if not read_correctly:
                break
            self._send_frame_to_network(frame)
            keep = self._process_frame(frame)
            if not keep:
                break

    def stream_cam(self):
        """Streams from camera and processes it"""
        self.window = "rgb"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cam_out = self.getOutputQueue("cam_out", 1, True)
        self.analysis.dummy.resize(self.width, self.height)
        keep = True
        while keep:
            frame = (
                np.array(cam_out.get().getData())
                .reshape((3, self.height, self.width))
                .transpose(1, 2, 0)
                .astype(np.uint8)
            )
            keep = self._process_frame(frame)

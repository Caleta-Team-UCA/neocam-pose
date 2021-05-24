from time import monotonic

import cv2
import depthai as dai

from neocam.pipeline.base import Pipeline
from neocam.utils.frame import to_planar, display_frame


def run_pipeline_in_device(
    path_video: str, pipeline: Pipeline, name: str = "", anonymize_method: str = None
):
    # Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        # Start pipeline
        device.startPipeline()

        # Input queue will be used to send video frames to the device.
        q_in = device.getInputQueue(name=pipeline.in_stream)
        # Output queue will be used to get nn data from the video frames.
        q_det = device.getOutputQueue(
            name=pipeline.out_stream, maxSize=4, blocking=False
        )

        detections = []

        # nn data, being the bounding box locations, are in <0..1> range
        # they need to be normalized with frame width/height

        cap = cv2.VideoCapture(path_video)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        while cap.isOpened():
            read_correctly, frame = cap.read()
            # orig = frame.copy()
            if not read_correctly:
                break

            img = dai.ImgFrame()
            img.setData(to_planar(frame, (300, 300)))
            img.setTimestamp(monotonic())
            img.setWidth(300)
            img.setHeight(300)
            q_in.send(img)

            in_det = q_det.tryGet()

            if in_det is not None:
                detections = in_det.detections

            if frame is not None:
                display_frame(
                    name, frame, detections, anonymize_method=anonymize_method
                )
                cv2.waitKey(5)

            if cv2.waitKey(1) == ord("q"):
                break

            if cv2.waitKey(1) == ord("b"):
                anonymize_method = "blur"

            if cv2.waitKey(1) == ord("f"):
                anonymize_method = "filled"

            if cv2.waitKey(1) == ord("p"):
                anonymize_method = "pixelate"

            if cv2.waitKey(1) == ord("n"):
                anonymize_method = "none"

from typing import Iterable

import depthai as dai

LIST_LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def filter_body_detections(
    detections: Iterable[dai.RawImgDetections], target: str = "person"
) -> Iterable[dai.RawImgDetections]:
    """Takes a list of detections, returns only those with target label

    Parameters
    ----------
    detections : Iterable
        List of depthai.RawImgDetections
    target : str, optional
        Target label, by default "person"

    Returns
    -------
    Iterable
        List of depthai.RawImgDetections with label `target`

    """
    new_detections = []
    for detection in detections:
        label = LIST_LABELS[detection.label]
        if label == target:
            detection.label = 0
            new_detections.append(detection)
    return new_detections


def filter_face_detections(
    detections: Iterable[dai.RawImgDetections],
) -> Iterable[dai.RawImgDetections]:
    """Takes a list of detections, returns the first

    Parameters
    ----------
    detections : Iterable
        List of depthai.RawImgDetections

    Returns
    -------
    Iterable
        List of depthai.RawImgDetections with label `target`

    """
    try:
        detection = detections[0]
        detection.label = 1
        return [detection]
    except IndexError:
        return []

from typing import List

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
    "person",  # index 15
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "face",  # index 21, added by us
]


def filter_body_detections(
    detections: List[dai.RawImgDetections], target: str = "person"
) -> List[dai.RawImgDetections]:
    """Takes a list of detections, returns only those with target label

    Parameters
    ----------
    detections : List
        List of depthai.RawImgDetections
    target : str, optional
        Target label, by default "person"

    Returns
    -------
    List
        List of depthai.RawImgDetections with label `target`

    """
    new_detections = []
    for detection in detections:
        label = LIST_LABELS[detection.label]
        if label == target:
            new_detections.append(detection)
    return new_detections


def filter_face_detections(
    detections: List[dai.RawImgDetections],
) -> List[dai.RawImgDetections]:
    """Takes a list of detections, returns the first

    Parameters
    ----------
    detections : List
        List of depthai.RawImgDetections

    Returns
    -------
    List
        List of depthai.RawImgDetections with label `target`

    """
    try:
        detection = detections[0]
        detection.label = 21
        return [detection]
    except IndexError:
        return []

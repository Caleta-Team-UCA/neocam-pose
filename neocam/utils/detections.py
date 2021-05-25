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


def filter_body_detections(detections: list, target: str = "person") -> list:
    """Takes a list of detections, returns only those with target label

    Parameters
    ----------
    detections : list
        List of depthai.RawImgDetections
    target : str, optional
        Target label, by default "person"

    Returns
    -------
    list
        List of depthai.RawImgDetections with label `target`

    """
    new_detections = []
    for detection in detections:
        label = LIST_LABELS[detection.label]
        if label == target:
            new_detections.append(detection)
    return new_detections


def filter_face_detections(detections: list) -> list:
    """Takes a list of detections, returns the first

    Parameters
    ----------
    detections : list
        List of depthai.RawImgDetections

    Returns
    -------
    list
        List of depthai.RawImgDetections with label `target`

    """
    try:
        return [detections[0]]
    except IndexError:
        return []

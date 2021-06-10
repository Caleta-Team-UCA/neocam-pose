import cv2
import numpy as np

from neocam.utils.detections import LIST_LABELS


def frame_norm(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Normalizes the frame"""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def anonymize(frame: np.ndarray, bbox: np.ndarray, anonymize_method: str = None):
    """Anonymize the face"""
    # Locate the face
    y1, y2, x1, x2 = int(bbox[1]), int(bbox[3]), int(bbox[0]), int(bbox[2])

    if anonymize_method == "blur":
        face = frame[y1:y2, x1:x2]
        face = anonymize_shape_simple(face, factor=3.0)
        frame[y1:y2, x1:x2] = face
        # otherwise, we must be applying the "filled" face
        # anonymization method

    elif anonymize_method == "pixelate":
        face = frame[y1:y2, x1:x2]
        face = anonymize_shape_pixelate(face, blocks=3)
        frame[y1:y2, x1:x2] = face
        # otherwise, we must be applying the "filled" face
        # anonymization method
    elif anonymize_method == "filled":
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

    return frame


def display_frame(
    name: str, frame: np.ndarray, detections: list, anonymize_method: str = None
):
    """Displays the frame on screen"""
    for detection in detections:
        bbox = frame_norm(
            frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        )
        # Anonymize face
        if LIST_LABELS[detection.label] == "face":
            frame = anonymize(frame, bbox, anonymize_method=anonymize_method)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        cv2.putText(
            frame,
            LIST_LABELS[detection.label],
            (bbox[0] + 10, bbox[1] + 30),
            cv2.FONT_HERSHEY_TRIPLEX,
            1.5,
            255,
        )
        cv2.putText(
            frame,
            f"{int(detection.confidence * 100)}%",
            (bbox[0] + 10, bbox[1] + 60),
            cv2.FONT_HERSHEY_TRIPLEX,
            1.5,
            255,
        )
    cv2.resizeWindow(name, 700, 600)
    cv2.imshow(name, frame)


def anonymize_shape_simple(face, factor=3.0):
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = face.shape[:2]
    k_w = int(w / factor)
    k_h = int(h / factor)
    # ensure the width of the kernel is odd
    if k_w % 2 == 0:
        k_w -= 1
    # ensure the height of the kernel is odd
    if k_h % 2 == 0:
        k_h -= 1
    # apply a Gaussian blur to the input image using our computed
    # kernel size
    return cv2.GaussianBlur(face, (k_w, k_h), 0)


def anonymize_shape_pixelate(face, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = face.shape[:2]
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = face[start_y:end_y, start_x:end_x]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(face, (start_x, start_y), (end_x, end_y), (B, G, R), -1)
    # return the pixelated blurred image
    return face

import typer

from neocam.utils.pipeline import Pipeline
from neocam.utils.device import Device


def main(
    path_model_detection: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    path_model_face: str = "models/face-detection-openvino_2021.2_4shave.blob",
    path_video: str = "data/18-center.mp4",
    name: str = "rgb",
    anonymize_method: str = None,
):
    pipeline = Pipeline(
        path_model_detection=path_model_detection, path_model_face=path_model_face
    )
    device = Device(pipeline, anonymize_method=anonymize_method)
    device.stream_video(path_video, name=name)


if __name__ == "__main__":
    typer.run(main)

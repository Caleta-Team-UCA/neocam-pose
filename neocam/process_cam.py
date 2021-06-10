import typer

from neocam.pipeline.pipeline_cam import PipelineCam
from neocam.utils.device import Device


def main(
    path_model_detection: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    path_model_face: str = "models/face-detection-openvino_2021.2_4shave.blob",
    anonymize_method: str = None,
):
    pipeline = PipelineCam(
        path_model_body=path_model_detection, path_model_face=path_model_face
    )
    device = Device(pipeline, anonymize_method=anonymize_method)
    device.stream_cam()


if __name__ == "__main__":
    typer.run(main)

import typer

from neocam.pipeline.detection import DetectionPipeline
from neocam.utils.device import run_pipeline_in_device


def main(
    path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    path_video: str = "data/18-center.mp4",
    name: str = "rgb",
    anonymize_method: str = None,
):
    pipeline = DetectionPipeline()
    pipeline.create_neural_network(path_model)

    run_pipeline_in_device(
        path_video, pipeline, name=name, anonymize_method=anonymize_method
    )


if __name__ == "__main__":
    typer.run(main)

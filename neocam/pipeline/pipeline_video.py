import depthai as dai

from neocam.pipeline.pipeline_wrapper import PipelineWrapper


class PipelineVideo(PipelineWrapper):
    input: str = "inFrame"
    in_frame: dai.XLinkIn = None

    def __init__(
        self,
        path_model_body: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
        path_model_face: str = "models/face-detection-openvino_2021.2_4shave.blob",
    ):
        """Basic pipeline for DepthAI

        Parameters
        ----------
        path_model_body : str, optional
            Path to the model blob file, used to detect full bodies
        path_model_face : str, optional
            Path to the model blob file, used to detect faces
        """
        super(PipelineVideo, self).__init__(
            path_model_body=path_model_body, path_model_face=path_model_face
        )

    def _create_in_stream(self):
        self.in_frame = self.createXLinkIn()
        self.in_frame.setStreamName(self.input)

    def _link_input(self, nn: dai.MobileNetDetectionNetwork):
        self.in_frame.out.link(nn.input)

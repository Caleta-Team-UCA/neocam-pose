import depthai as dai

from neocam.pipeline.pipeline_wrapper import PipelineWrapper


class PipelineCam(PipelineWrapper):
    input: str = "control"
    control_in: dai.XLinkIn = None
    cam: dai.ColorCamera = None
    width: int = 300
    height: int = 300

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
        super(PipelineCam, self).__init__(
            path_model_body=path_model_body, path_model_face=path_model_face
        )

    def _create_in_stream(self):
        # Use color camera
        self.cam = self.createColorCamera()
        self.cam.setPreviewSize(self.width, self.height)
        self.cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam.setInterleaved(False)
        self.cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        # Define output stream
        cam_xout = self.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        self.cam.preview.link(cam_xout.input)
        # Define control stream
        self.control_in = self.createXLinkIn()
        self.control_in.setStreamName(self.input)
        self.control_in.out.link(self.cam.inputControl)

    def _link_input(self, nn: dai.MobileNetDetectionNetwork):
        self.cam.preview.link(nn.input)

import depthai as dai


class Pipeline(dai.Pipeline):
    nn_body: dai.MobileNetDetectionNetwork = None
    nn_face: dai.NeuralNetwork = None
    in_stream: str = "inFrame"
    out_body: str = "nn_body"
    out_face: str = "nn_face"
    in_frame: dai.XLinkIn = None

    def __init__(
        self,
        path_model_detection: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
        path_model_face: str = "models/face-detection-openvino_2021.2_4shave.blob",
    ):
        """Basic pipeline for DepthAI"""
        super(Pipeline, self).__init__()
        self.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_1)
        self._create_in_stream()
        self.create_body_network(path_model_detection)
        self.create_face_network(path_model_face)

    def _create_in_stream(self, name: str = None):
        """Create xLink input to which host will send frames from the video file"""
        self.in_frame = self.createXLinkIn()
        if name is None:
            name = self.in_stream
        self.in_frame.setStreamName(name)

    def _link_output(self, nn, name: str):
        """Assigns an output stream to given Neural Network"""
        nn_out = self.createXLinkOut()
        nn_out.setStreamName(name)
        nn.out.link(nn_out.input)

    def create_body_network(self, path_model: str):
        """Initializes a neural network used for detecting newborns"""
        # Define a neural network that will make predictions based on the source frames
        self.nn_body = self.createMobileNetDetectionNetwork()
        self.nn_body.setConfidenceThreshold(0.7)
        self.nn_body.setBlobPath(path_model)
        self.nn_body.setNumInferenceThreads(2)
        self.nn_body.input.setBlocking(False)

        # Assign input and output frames
        self.in_frame.out.link(self.nn_body.input)
        self._link_output(self.nn_body, self.out_body)

    def create_face_network(self, path_model: str):
        # Network architecture
        self.nn_face = self.createMobileNetDetectionNetwork()
        self.nn_face.setBlobPath(path_model)

        # Assign input and output frames
        self.in_frame.out.link(self.nn_face.input)
        self._link_output(self.nn_face, self.out_face)

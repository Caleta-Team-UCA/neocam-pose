from abc import abstractmethod

import depthai as dai


class PipelineWrapper(dai.Pipeline):
    nn_body: dai.MobileNetDetectionNetwork = None
    nn_face: dai.NeuralNetwork = None
    input: str = "inFrame"
    out_body: str = "nn_body"
    out_face: str = "nn_face"

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
        super(PipelineWrapper, self).__init__()
        self.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_1)
        self._create_in_stream()
        self.create_body_network(path_model_body)
        self.create_face_network(path_model_face)

    @abstractmethod
    def _create_in_stream(self):
        """Create xLink input to which host will send frames from the video file"""
        pass

    @abstractmethod
    def _link_input(self, nn: dai.MobileNetDetectionNetwork):
        """Assigns an input stream to given Neural Network

        Parameters
        ----------
        nn : depthai.MobileNetDetectionNetwork
            Neural Network

        """
        pass

    def _link_output(self, nn: dai.MobileNetDetectionNetwork, name: str):
        """Assigns an output stream to given Neural Network

        Parameters
        ----------
        nn : depthai.MobileNetDetectionNetwork
            Neural Network
        name : str
            Label of the output stream
        """
        nn_out = self.createXLinkOut()
        nn_out.setStreamName(name)
        nn.out.link(nn_out.input)

    def create_body_network(self, path_model: str):
        """Initializes a neural network used for detecting bodies

        Parameters
        ----------
        path_model : str
            Path to the blob model
        """
        # Define a neural network that will make predictions based on the source frames
        self.nn_body = self.createMobileNetDetectionNetwork()
        self.nn_body.setConfidenceThreshold(0.7)
        self.nn_body.setBlobPath(path_model)
        self.nn_body.setNumInferenceThreads(2)
        self.nn_body.input.setBlocking(False)

        # Assign input and output frames
        self._link_input(self.nn_body)
        self._link_output(self.nn_body, self.out_body)

    def create_face_network(self, path_model: str):
        """Initializes a neural network used for detecting faces

        Parameters
        ----------
        path_model : str
            Path to the blob file
        """
        # Network architecture
        self.nn_face = self.createMobileNetDetectionNetwork()
        self.nn_face.setBlobPath(path_model)

        # Assign input and output frames
        self._link_input(self.nn_face)
        self._link_output(self.nn_face, self.out_face)

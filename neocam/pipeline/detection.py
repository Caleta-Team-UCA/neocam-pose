from neocam.pipeline.base import Pipeline


class DetectionPipeline(Pipeline):
    def __init__(self):
        super(DetectionPipeline, self).__init__()

    def create_neural_network(self, path_model: str):
        # Create xLink input to which host will send frames from the video file
        xin_frame = self.createXLinkIn()
        xin_frame.setStreamName(self.in_stream)

        # Define a neural network that will make predictions based on the source frames
        self.nn = self.createMobileNetDetectionNetwork()
        self.nn.setConfidenceThreshold(0.7)
        self.nn.setBlobPath(path_model)
        self.nn.setNumInferenceThreads(2)
        self.nn.input.setBlocking(False)
        xin_frame.out.link(self.nn.input)

        # Create output
        nn_out = self.createXLinkOut()
        nn_out.setStreamName(self.out_stream)
        self.nn.out.link(nn_out.input)

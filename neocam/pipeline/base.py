from abc import abstractmethod

import depthai as dai


class Pipeline(dai.Pipeline):
    nn: dai.MobileNetDetectionNetwork = None
    in_stream: str = "inFrame"
    out_stream: str = "nn"

    def __init__(self):
        """Basic pipeline for DepthAI"""
        super(Pipeline, self).__init__()

    @abstractmethod
    def create_neural_network(self, path_model: str):
        pass

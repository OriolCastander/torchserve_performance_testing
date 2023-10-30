from ts.torch_handler.base_handler import BaseHandler
import torch
import os
import json

from neuralNetworkModel import NeuralNetworkModel


class NeuralNetworkHanlder(BaseHandler):

    def __init__(self):
        super(NeuralNetworkHanlder, self).__init__()
        

    def initialize(self, context):
        
        properties = context.system_properties

        model_dir = properties.get("model_dir")
        mapping_file_path = os.path.join(model_dir, "neuralNetwork.json")

        with open(mapping_file_path, "r") as f:
            self.mapping = json.load(f)

        self.model = NeuralNetworkModel.load(os.path.join(model_dir,"neuralNetwork.pth"), NeuralNetworkModel,
                                                self.mapping["input_dim"],
                                                self.mapping["hidden_dims"],
                                                self.mapping["output_dim"]
                                            )
        if self.device == 'cuda':
            self.model.to('cuda')


    def preprocess(self, requests):
        ##TODO: IMPROVE THIS, CREATE THE INPUT DIRECTLY USING NUMPY
        input = []
        for request in requests:
            for row in json.loads(request['body'])["data"]:
                input.append(row)

        tensor = torch.tensor(input, dtype=torch.float)

        if self.device == 'cuda':
            tensor = tensor.to('cuda')
        
        ##TODO: CHECK THAT TENSOR HAS THE APPROPIATE SHAPE (SHAPE SHOULD BE INCLUDED IN MAPPING)

        return tensor

    def inference(self, x):
        return self.model.predict(x)
    
    def postprocess(self, preds):
        if self.device == 'cuda':
            preds = preds.to('cpu')

        print(preds.detach().numpy().tolist())

        return [preds.detach().numpy().tolist()]
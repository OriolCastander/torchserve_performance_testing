import torch
try:
    from model import Model
except:
    from models.model import Model


class ConvolutionNeuralNetworkModel(Model):

    """
    Example of basic CNN for image classifying. 
    Currently only accepting 32x32 images with 2 convolution networks
    TODO: test with flexible networks
    """

    def __init__(self):
        super(ConvolutionNeuralNetworkModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,6,5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

        self.config: dict = {
            "shape": (3,32,32)
        }

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
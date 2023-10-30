import torch
try:
    from model import Model
except:
    from models.model import Model

class NeuralNetworkModel(Model):
    
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super(NeuralNetworkModel, self).__init__()

        layers = [torch.nn.Linear(input_dim, hidden_dims[0]), torch.nn.ReLU()]
        for i in range(0, len(hidden_dims) - 1):
            layers += [torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hidden_dims[-1], output_dim)]

        self.network = torch.nn.Sequential(*layers)
        self.config: dict[str, any] = {
            "input_dim": input_dim,
            "hidden_dims":hidden_dims,
            "output_dim":output_dim,
            "shape": (input_dim)
            }

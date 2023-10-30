import torch
import json


class Model(torch.nn.Module):
    """A mock model for testing purposes, not even trainable (simulates an already trained model)"""

    def __init__(self) -> None:
        super(Model,self).__init__()

    @staticmethod
    def load(path: str, modelClass: any, *args, **kwargs) -> "Model":
        model = modelClass(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        return model
    
    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        return self.network(input)
    
    def saveConfigJson(self, name: str, otherConfigs: dict) -> None:
        with open(f"./configs/{name}.json", "w") as file:
            json.dump({**self.config, **otherConfigs}, file)
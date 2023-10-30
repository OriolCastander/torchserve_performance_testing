# torchserve_performance_testing

Installation of required dependencies (in a conda env) is as follows:
-  conda install openjdk
-  Installing torchserve as explained [here](https://pytorch.org/serve/getting_started.html)
-  Run export CUDA_AVAILABLE_DEVICES=""

This code is the base architecture for torchserve model serving, currently being used for performance testing. .mar files are created automatically, bundling all necessary files. torchserve --start is currently run in a separate terminal. There are a couple of mock models already (with which the tests are run on), but technically any model that extends torch.nn.Module can be added and "should run". Each model needs 3 files:
-  The .py model class in the models folder, saved with the same name as the class (but starting lowercase). Both should end in ..Model.
-  A .py torchserve handler file in the handlers folder, saved with the same name as the .py model file but replacing Model with Handler
-  The .pth trained model, saved with any name
-  If needed, a json config file saved with the same name as the pth, where relevant information for the handler should be stored, such as model init parameters

To me, it seems that torchserve adequately distributes the computing load to all available CPUs and GPUs automatically, without the need of "manually" deciding where everything should go.

TODO:
- Create a generic torchserve handler that other handlers can be derived from
- Automatic splitting of massive input tensors that would not fit in gpu memory
- Further testing

FURTHER TESTING:
-  Image processing on the handler after sending raw files via (curl -T dog.png) instead of images having to be converted to pseudo-tensors in the client
-  Multiple requests vs one big single request with all the batches
-  Serving multiple models in a single torchserve instance

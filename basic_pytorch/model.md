# Model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

In pytorch, a model is a user defined torch.nn.Module which contains a set of torch.Module objects. And 
the set of torch.nn.Module objects form a tree structure with the model Module as the root. In this 
tutorial section, a close look of the torch.Module has been paid attention.


### Define a model as a subclass of torch.nn.Module
The pytorch class Module is defined in [torch.nn package](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). To 
define a module, one just need to extend the torch.nn.Module class and implement two methods,
```python
   __init__(self)
   forward(self, x)
```
The method forward() defines what the model does. The simple example is HelloWorldModel defined as following
```python
import torch

class HelloWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        print("Hello World!")

def main():
    model = HelloWorldModel()
    model(None)

if __name__ == "__main__":
    main()
```
To execute the [code](./hello_world_model.py), one can get
```
$ python hello_world_model.py
Hello World!
```

Certainly, above example does not serve the purpose of torch.nn.Module but just simply how 
to define a model by extending torch.nn.Module class. One interesting thing is that the forward()
method is not invoked directly instead just simply treat model as a function. Often, we call
model as a functor or function object.

Now, let's look a simple trainable or learnable model. 
```
import torch

class SimpleLinearModel(torch.nn.Module):
    def __init__(self, dim : int):
        super().__init__()
        weight_param = torch.nn.parameter.Parameter(torch.ones(dim, dim))
        bias_param = torch.nn.parameter.Parameter(torch.zeros(dim, 1))
        self.register_parameter("weight", weight_param)
        self.register_parameter("bias", bias_param)

    def forward(self, x):
        return self.weight.mm(x) + self.bias
```

With torch.nn.parameter.Parameter, the pytorch Module framework allows us
to retrieve it with its name and save it into persistent media. Further, 
if the SimpleLinearModel is considered as a SimpleLinearLayer, 
a simple multi-layer model can be defined as following,
```
import torch

class SimpleLinearLayer(torch.nn.Module):
    def __init__(self, dim : int):
        super().__init__()
        weight_param = torch.nn.parameter.Parameter(torch.ones(dim, dim))
        bias_param = torch.nn.parameter.Parameter(torch.zeros(dim, 1))
        self.register_parameter("weight", weight_param)
        self.register_parameter("bias", bias_param)

    def forward(self, x):
        return self.weight.mm(x) + self.bias

class SimpleLinearModel(torch.nn.Module):
    def __init__(self, dim : int):
        super().__init__()
        layer0 = SimpleLinearLayer(dim)
        self.add_module("layer0", layer0)
        layer1 = SimpleLinearLayer(dim)
        self.add_module("layer1", layer1)

    def forward(self, x):
        return self.layer1(self.layer0(x))
```
### torch.nn.Module APIs
For the comprehensive Module APIs, one can reference to the 
[pytorch link](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

##### Storage location movement
##### Persistence (Serializatoin and Deserialization)
##### Datatype Conversion
##### Tree structure traverse
##### Hooks

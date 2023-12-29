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
A pytorch model can be executed in CPUs, or GPU, or IPU, or XPU. The APIs to move a model to
different devices are,
```python
   cpu()
   cuda(device=None)
   ipu(device=None)
   xpu(device-None)
   share_memory()
```

##### Persistence (Serializatoin and Deserialization)
From the model definitions, one may notice that the most important things of a model are its
attributes, such as, weight parameters, bias parameters, etc. Also, they are dynamical. So,
APIs to save and retrieve them are quite essential. Here they are,
```python
    state_dict(*, prefix: str = '', keep_vars: bool = False) -> Dict[str, Any]
    load_state_dict(state_dict, strict=True,assign=False)
```
The first API, state_dict(), returns the current state of the model as a dictionary object. Then, 
the dictionary object can be saved into a persistent media, such as, a disk file, or send to different
address space (aka, processes) via inter-process communication. 

The second API, load_state_dict(), loads the data in the dictionary data, state_dict, into the model. 
The dictionary data, state_dict, could be loaded from a persistence media, such as a dick file, or from
a remote process via inter-process communication, such as, TCP/IP, etc.
##### Datatype Conversion
The state data of a model can be a different data type, such as, float, double, int. Even for float, 
the data type can be 16-bits float or 32-bits float.  Various data types have big impact on the memory usage, 
computation precision, computation performance and computation cost. Here are APIs to cast the state data 
to various data type,
```python
   Module.float()
   Module.bfloat16()
   Module.double()
   Module.half()
   Module.to(dtype)
```
##### Tree structure traverse
A complicated model could have many layers and each layer could have different Modules. How to traverse all
Modules within a model is very important for both debugging and validation. Pytorch provides a set of APIs to
serve this purpose.

The APIs to navigate Modules.
```python
   Module.children()        # Returns an iterator of its immediate children Modules. This allows layer by layer
                            # travers
   Module.named_children()  # Returns the same immediate children modules as children() but with names too
   Module.modules()         # Returns an interator for all Modules in the model.
   Module.named_modules()   # Returns the same all Modules as modules() but with names too
```

The API to get a submodule with a full qualified target name
```python
   Module.get_submodule(target_name) -> torch.nn.Module
```

APIs to access a Module's state
```python
   Module.parameters(recurse=True)      # Returns an iterator for all parameters in this module. If recurse=True,
                                        # all pararmeters in submodules are included.
   Module.named_parameters()            # Returns the same all parameters as parameters() but with names too.
   Module.get_parameter(target_name)    # Returns the parameter given by the fully qualified name path
```
##### Hooks
torch.nn.Module hook framework provides a way to monitor, trace and debug model training, validation and evaluation.
Here are APIs.
```python
   # The hook will be called before each forward() invocation. prepend=False lets the hook be filred
   # after all existing forward_pre hooks in the Module. Otherwise before all existing hooks.
   # The hook signature: hook(module, args, kwargs) -> None or a tuple of modified input and kwargs
   Module.register_forward_pre_hook(hook, *, prepend=False, with_kwargs=False)

   # The hook will be called after forward() is invocated.
   # The hook signature: hook(module, args, kwargs, output) -> None or modified output
   Module.register_forward_hook(hook, *, prepend=False, with_kwargs=False, always_call=False)

   # The hook will be invoked before the gradients of the module are computed.If the gradients
   # of the module are not computed, the hook will not be invoked.
   # The hook signature: hook(module, grad_output) -> tuple[Tensor] or None
   Module.register_full_backward_pre_hook(hook, prepend=False)

   # The hook will be invoked after the gradients of the module are computed.
   # The hook signature: hook(module, grad_input, grad_output) -> tuple(Tensor) or None
   Module.register_full_backward_hook(hook, prepend=False)
```

##### Modes

A torch.nn.Module can be in one of folllowing modes,
```
   Evaluation,
   Training
```

The APIs to change a Module's mode are,
```python
   Module.eval()           # set the module in evaluation mode. it is equivalent with Module.train(False)
   Module.train(mode=True) # set the module in training mode=True otherwise set it in evaluation mode
```
   
### Build Large Model

To build a large and complicated model, pytorch provides a set of Module containers and pre-defined Neural Network layers.
In this section, we will review the APIs related LLMs in the both categories. For the details of both containers and
layers, one may refer to the pytorch [torch.nn](https://pytorch.org/docs/stable/nn.html#) link.

##### Module Containers
Since a Module can be nested within another Module, a generalization of such nesting behaviors can reuse the code development.
Three Module container Modules are provided in pytorch.
```pytorch
   torch.nn.Sequential(*args: Module)
   torch.nn.Sequential(arg: OrderedDict[str, Module])
   torch.nn.ModuleList(modules=None)
   torch.nn.ModuleDict(modules=None)
```
The difference between Sequential and ModuleList is that Sequential.forward() shall cascadely invoke 
the submodules' forward() method but ModuleList only provide access to the ordered list of submodules.

ModuleDict is an ordered dictionary data structure of submodules with respect to the order of insertion.
##### Neural Network Layters

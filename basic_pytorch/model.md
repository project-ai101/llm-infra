# Model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

In pytorch, a model is a user defined torch.nn.Module which contains a set of torch.Module objects. And 
the set of torch.nn.Module objects form a tree structure with the model Module as the root. In this 
tutorial section, a close look of the torch.Module has been paid attention.

### torch.nn.Module
The pytorch class Module is defined in [torch.nn package](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).  


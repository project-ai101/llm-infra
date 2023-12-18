# Tensor and Model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

In this tutorial, the basic concepts of tensor and model are introduced. They are
explained from pytorch programming perspectives, not from rigorous mathematical
ways. 

### Tensor
A tensor is an object in pytorch, which has an array of elements with a single 
data type and is viewed as a multi-dimensional matrix. A pytorch tensor can be
located in the CPU memory or a device, such as, GPU, or TPU, or a storage device.

To deep dive into the pytorch tensor, the following properties are discussed,
- creation
- shape and reshaping
- location and movement
- mathematic methods

##### Creation
Let's create a simple tensor from a Python list as following and name the python
script file as tensor_creation.py.
```python
import torch

def main():
    t1 : torch.Tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    print(t1)

if __name__ == "__main__":
     main()
```

The output of the execution of tensor_creation.py is
```
tensor([[1., 0.],
        [0., 1.]])
```


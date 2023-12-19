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
From the output, pytorch does show that a tensor is maintained as a multi-dimentional
matrix internally. 

If one wants to create a tensor with all zero value elements and a large dimension, torch.tensor may
not be the desirable way to do it. Is there any other
APIs which can create a tensor and initialize its elements into zeros? Yes. There is one, torch.zeros. 
Here is how to use it to create and initialize a tensor of dimension
(3, 3, 3) will all zeros.
```python
    t2 : torch.Tensor = torch.zeros(3, 3, 3)
    print(t2)
```

The output is
```
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]])
```
The extension of the torch.zeros() API is the torch.full() API which create a tensor
with element values initialized by fill_value.

```python
    t3 : torch.Tensor = torch.full(3, 3, 3, fill_value=2.0)
    print(t3)
```
The output is
```
tensor([[[2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.]],

        [[2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.]],

        [[2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.]]])
```
In many situations, we want to create a tensor with random values. The following example
shows how to use the torch.rand() API to create such a tensor,
```python
    t4: torch.Tensor = torch.rand([3, 3, 3])
    print(t4)
```
The output is
```
tensor([[[0.8335, 0.3090, 0.7877],
         [0.9333, 0.9194, 0.3154],
         [0.7494, 0.6031, 0.9997]],

        [[0.7832, 0.7962, 0.4333],
         [0.7171, 0.2824, 0.4686],
         [0.8430, 0.9686, 0.4810]],

        [[0.6353, 0.2938, 0.6488],
         [0.2266, 0.0630, 0.7867],
         [0.8079, 0.2836, 0.6959]]])
```

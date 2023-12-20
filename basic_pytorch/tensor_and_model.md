# Tensor
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

In this tutorial, the basic concepts of tensor are introduced. They are
explained from pytorch programming perspectives, not from rigorous mathematical
ways. 

A tensor is an object in pytorch, which has an array of elements with a single 
data type and is viewed as a multi-dimensional matrix. A pytorch tensor can be
located in the CPU memory or a device, such as, GPU, or TPU, or a storage device.

To deep dive into the pytorch tensor, the following properties are discussed,
- creation
- shape and reshaping
- location and movement
- mathematic methods

### creation
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
    t2 : torch.Tensor = torch.zeros(2, 3, 4)
    print(t2)
```

The output is
```
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
```
The extension of the torch.zeros() API is the torch.full() API which create a tensor
with element values initialized by fill_value.

```python
    t3 : torch.Tensor = torch.full(2, 3, 4, fill_value=2.0)
    print(t3)
```
The output is
```
tensor([[[2., 2., 2., 2.],
         [2., 2., 2., 2.],
         [2., 2., 2., 2.]],

        [[2., 2., 2., 2.],
         [2., 2., 2., 2.],
         [2., 2., 2., 2.]]])
```
In many situations, we want to create a tensor with random values. The following example
shows how to use the torch.rand() API to create such a tensor,
```python
    t4: torch.Tensor = torch.rand([2, 3, 4])
    print(t4)
```
The output is
```
tensor([[[0.5924, 0.1014, 0.3705, 0.3951],
         [0.4991, 0.4058, 0.6068, 0.9218],
         [0.1601, 0.4973, 0.3497, 0.6603]],

        [[0.4302, 0.8504, 0.9999, 0.9311],
         [0.8682, 0.7242, 0.2806, 0.1216],
         [0.0695, 0.9797, 0.6955, 0.6626]]])
```

### shape, slice, stide, view and reshape
When the tensors t2, t3 and t4 were created in the above section, the dimensions were passed 
in three formats. The dimensions describe the shapes of the tensors. Pytorch use torch.Size 
( a subclass of tuple) data type to define the dimensions of a tensor. Here is an example 
to retrieve a tensor's shape
(aka size).
```python
    print(t4.shape)
    s : torch.Size = t4.size();
    print(s)
```
The output is 
```
torch.Size([2, 3, 4])
torch.Size([2, 3, 4])
```

Pytorch treats a tensor like a multi-dimensional array (or matrix). A tensor can be access by indexes
like a normal python multi-dimensional array. For example,
```python
    tt4 : torch.Tensor = t4[0]
    print(tt4)
    print(tt4.shape)

    ttt4 : torch.Tensor = t4[:,:,0];
    print(ttt4)
    print(ttt4.shape)
```
The output is
```
---------- t4 slice tt4 = t4[0]:
tensor([[0.5924, 0.1014, 0.3705, 0.3951],
        [0.4991, 0.4058, 0.6068, 0.9218],
        [0.1601, 0.4973, 0.3497, 0.6603]])
torch.Size([3, 4])
---------- t4 slice ttt4 = t4[:,:,0]:
tensor([[0.5924, 0.4991, 0.1601],
        [0.4302, 0.8682, 0.0695]])
torch.Size([2, 3])
```
Further, though tt4 and ttt4 are different tensor objects with respect to t4, they share the same
data elements. To prove this, let's change the value of element [0, 0] in tt4 to 10.0f and print
out the value of element [0, 0, 0] in t4.
```python
    tt4[0,0] = 10.0
    print(t4[0,0,0])
```
The output is
```
tensor(10.)
```
How the elements of a tensor are layouted in the memory has big impact to the overall 
tensor calculation performance. The memory layout typcially is described by stride. A stride
decribes how far the next element along the dimension is apart in the actually memory. For example,
the stride of dimension 0 of the tensor t4 is defined by the memory address distance of between
t4[0,0,0] and t4[1,0,0]. It could be 12 or 1. The API torch.Tensor.stride() returns the strides in
a tuple.
```python
    print(t4.stride())
```
The output is
```
(12, 4, 1)
```
This means the stride in dimension 0 is 12, the stride in dimension 1 is 4 and the stride
in dimension 2 is 1. Therefore, the default layout of a tensor in pytorch is a row-major layout.

View is a tensor with different shape but sharing the same underlying data with its base tensor. Let's
take a close look the following code.
```python
    t5 : torch.Tensor = t4.view(2, 12)
    print(t5.shape)
    print(t5.stride())
    print("Do t4 and t5 share the same ungerlying data? ",
          t4.storage().data_ptr() == t5.storage().data_ptr())
```
The output is
```
torch.Size([2, 12])
(12, 1)
Do t4 and t5 share the same underlying data?  True
```

There are two additional ways to change the shape of a tensor. They are transpose and reshape APIs? What are
their difference? Let's take a close look by running through following code snippet,
```python
    t6 : torch.Tensor = t4.view(2,4,3)
    print("---------- t6 = t4.view(2,4,3):")
    print(t6.shape)
    print(t6.stride())
    print("Is t6 contiguous? ", t6.is_contiguous())
    print("Do t4 and t6 share the same underlying data? ",
          t4.storage().data_ptr() == t6.storage().data_ptr())
    print(t6)

    t7 : torch.Tensor = t4.transpose(1,2)
    print("---------- t7 = t4.transpose(1,2):")
    print(t7.shape)
    print(t7.stride())
    print("Is t7 contiguous? ", t7.is_contiguous())
    print("Do t4 and t7 share the same underlying data? ",
          t4.storage().data_ptr() == t7.storage().data_ptr())
    print(t7)

    t8 : torch.Tensor = t4.reshape((2,4,3))
    print("---------- t8 = t4.reshape((2,4,3)):")
    print(t8.shape)
    print(t8.stride())
    print("Is t8 contiguous? ", t8.is_contiguous())
    print("Do t4 and t8 share the same underlying data? ",
          t4.storage().data_ptr() == t8.storage().data_ptr())
    print(t8)
```
The output is
```
---------- t6 = t4.view(2,4,3):
torch.Size([2, 4, 3])
(12, 3, 1)
Is t6 contiguous?  True
Do t4 and t6 share the same underlying data?  True
tensor([[[10.0000,  0.1014,  0.3705],
         [ 0.3951,  0.4991,  0.4058],
         [ 0.6068,  0.9218,  0.1601],
         [ 0.4973,  0.3497,  0.6603]],

        [[ 0.4302,  0.8504,  0.9999],
         [ 0.9311,  0.8682,  0.7242],
         [ 0.2806,  0.1216,  0.0695],
         [ 0.9797,  0.6955,  0.6626]]])
---------- t7 = t4.transpose(1,2):
torch.Size([2, 4, 3])
(12, 1, 4)
Is t7 contiguous?  False
Do t4 and t7 share the same underlying data?  True
tensor([[[10.0000,  0.4991,  0.1601],
         [ 0.1014,  0.4058,  0.4973],
         [ 0.3705,  0.6068,  0.3497],
         [ 0.3951,  0.9218,  0.6603]],

        [[ 0.4302,  0.8682,  0.0695],
         [ 0.8504,  0.7242,  0.9797],
         [ 0.9999,  0.2806,  0.6955],
         [ 0.9311,  0.1216,  0.6626]]])
---------- t8 = t4.reshape((2,4,3)):
torch.Size([2, 4, 3])
(12, 3, 1)
Is t8 contiguous?  True
Do t4 and t8 share the same underlying data?  True
tensor([[[10.0000,  0.1014,  0.3705],
         [ 0.3951,  0.4991,  0.4058],
         [ 0.6068,  0.9218,  0.1601],
         [ 0.4973,  0.3497,  0.6603]],

        [[ 0.4302,  0.8504,  0.9999],
         [ 0.9311,  0.8682,  0.7242],
         [ 0.2806,  0.1216,  0.0695],
         [ 0.9797,  0.6955,  0.6626]]])

```

From above output, when the new shape (2,4,3) is compatible with the existing shape (2,4,3), the 
Tensor.view() is equivalent to Tensor.reshape() which just reinterpret the underline data in
contiguous way. However, Tensor.transpose() does not keep the contiguous of strides.

The example code for the discussion is [here](./tensor_creation.py). Note that re-running the example 
shall have different values of tensors due to the randomness of t4.

# Pytorch Augograd
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

Understanding Autograd is quite straight forward through computation graph. First, Autograd is designed for nothing else but computing gradients. 
Lets start from a simple model,
```math
\begin {split}
  y &= f(x) = A\cdot x + b, \quad where \quad x\in R^3 \quad and \quad y\in R^2 \\
  z &= g(y) = y_1^2 + y_2^2.
\end {split}
```
The python implementation with pytorch is following
```python
import torch
from torch.nn import Module as Module
from torch.nn import Linear as Linear


class ExampleModel(Module):
    def __init__(self):
        super().__init__()
        self.f = Linear(3, 2)


    def forward(self, x):
        y = self.f(x)
        z = y.square().sum()

        return z

def main():
    m = ExampleModel()
    x = torch.randn(3)
    z = m(x)

    print(m.f.weight.grad)
    z.backward()
    print(m.f.weight.grad)

if __name__ == "__main__":
    main()
```

The one output is
```
None
tensor([[-0.5090,  1.0023,  0.0259],
        [ 0.5329, -1.0493, -0.0271]])

```

# Gradient and Backward Propogation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Author: Bin Tan

### Gradient
Gradient is one of the most important concepts in AI training. The pytorch reference is 
[here](https://pytorch.org/docs/stable/generated/torch.gradient.html#torch.gradient). In this 
tutorial, I will explain the concept in a different way to make it easy to be understood.

Let's first look into the simplest gradient, the derivative of one dimensional function (aka,
scalar function), $` f: R \to R `$,
```math
   f^{'}(x) = \frac{df(x)}{dx}=\lim\limits_{\Delta \to 0} \frac{f(x + \Delta) - f(x)}{\Delta}
```
When the $`\Delta x`$ is small enough and the function $` f(x) `$ is differentiable, its derivative
(aka gradient) along $` x `$ can be approximated as
```math
  (1) \hspace {2cm} f^{'}(x) = \frac{f(x + \Delta) - f(x)}{\Delta}
```
Now, since above formular is the approximation of the derivative, one may want to ask what the
error rate should be. If $` f(x) `$ is smooth enough (aka, the high order derivatives do exist),
based on Tylor expansion theorem, we have
```math
   f(x + \Delta) = f(x) + f^{'}(x){\Delta} + \frac{1}{2!} f^{''}(x) {\Delta^{2}} + \frac{1}{3!} f^{'''}(x) {\Delta^{3}} + ...
```
Therefore, the error rate of the derivative approximation is $` o(\Delta x^{2}) `$ with two samples. 
Now, one may ask if the error rate can be lower where there are three samples. The answer is yes providing that
the smoothness of the function is in the order of 3 or higher. 

Lets consider three sample points, $` x - \Delta_{l}, x, x + \Delta_{r} `$. Then, following the Tylor expansion 
theorem again, we have
```math
f(x + \Delta_{r}) = f(x) + f^{'}(x){\Delta_{r}} + \frac{1}{2!} f^{''}(x) {\Delta_{r}^{2}} + \frac{1}{3!} f^{'''}(x) {\Delta_{r}^{3}} + ...
```
```math
f(x - \Delta_{l}) = f(x) - f^{'}(x){\Delta_{l}} + \frac{1}{2!} f^{''}(x) {\Delta_{l}^{2}} - \frac{1}{3!} f^{'''}(x) {\Delta_{l}^{3}} + ...
```
By subtracting two equations, one has
```math
f(x + \Delta_{r}) \Delta_{l}^{2} - f(x - \Delta_{l}) \Delta_{r}^{2} =
f(x)(\Delta_{l}^{2} - \Delta_{r}^{2}) + f^{'}(x)(\Delta_{r}\Delta_{l}^{2} + \Delta_{l}\Delta_{r}^{2}) + o(\Delta^{5})
```
Then, the gradient (aka derivative for the scalar function $` f: R \to R`$ is given by
```math
   (2) \hspace {2cm} f^{'}(x) = \frac{f(x + \Delta_{r}) \Delta_{l}^{2} - f(x - \Delta_{l}) \Delta_{r}^{2} - f(x)(\Delta_{l}^{2} - \Delta_{r}^{2})}
                    {\Delta_{r}\Delta_{l}^{2} + \Delta_{l}\Delta_{r}^{2}}
               + o(\Delta^{2})
```
Therefore, if there are three samples, the derivative of the function at middle point can be estimated with error rate of order 2.

How do we interpret or describe above calculation in tensor format? 

For any function $` f : R^{n} \to R `$, we can use two n-order tensors to represent it. The first n-order tensor
describes the coordinates of all points (aka lattice). The second n-order tensor represents the function values
at all points in the lattice. 

For example, let's first look into a simple scalar function, $` f(x) = x^{2} `$. Consider the following points,
```python
   tensor(X) = [0, 1, 2, 3].
```
It represents 4 points along the dimension 0. Further, the spacing between adjacent points is 1. The values of 
the function $` f(x) = x^{2} `$ can be easily calculated and represented by the following tensor
```python
   tensor(Y) = [0, 2, 4, 6].
```

# Linear Regression Model Training
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan


The purpose of this example is to demonstrate the basic flow of machine learning training. The mathematical simplicit of linear regression 
makes it a good candidate to present the basic flow of machine learning clearly. The complicated examples, such as LLM training, follow
the same flow though the details are more challenging and frontier. 

### Model

Assume that we have a datasets $X = \{ X_1, X_2, ..., X_n\} = \{X_i \in R^M\}$ and $Y=\{ y_1, y_2, ..., y_n\} = \{ y_i \in R\}$ where $X$ is the input dateset and $Y$ 
is the output dateset. We want to predict the value y given an input data x. 

The prediction is normally based on the correlation behaviors between existing input and output dateset. Function approximation is one of 
powerful methods to find the correlation behaviors. The base model is

```math
\hspace {6cm}  \tilde y_i = f(X_i, \theta) - \epsilon_i \hspace {3cm} (1)
``` 

where $f: R^M \times R^K \to R$ is a function, $\theta$ is a K-dimensional parameter and $\epsilon_i$ is a noise term. To make the example simple and clear,
let's GUESS that the datasets could follow the linear model,

```math
\hspace {6cm} \tilde y_i = b + X_i \cdot \theta^T \hspace {3cm} (2)
```

where $\theta_i \in R$ and $\theta =\{ \theta_1, \theta_2, ..., \theta_M \} \in R^M$. Further more, for the demonstration purpose, we set M = 5.

Therefore, the model can be programmed as

```python
input_features = 5
output_features = 1
model = torch.nn.Linear(input_dim, output_dim, bias = True, dtype = torch.float)
```

### Loss Function
So far, the linear model (2) is just a conjesture that it may approximate the datasets X and Y. How do we know how good the approximation is?

This is the role which loss functions play. The one good measurement about the approximation is the distance between the observed $Y$ and the
estimated (or approximated) $\tilde Y = \{ \tilde y_1, ..., \tilde y_n \}$. Often, the MSE (Mean Sqaure Error) is used,
```math
\hspace {2cm} MSE = \frac {1}{n} \sum ^{n}_{i=1} (y_i - \tilde y_i)^2 = \sum ^n_{i=1} (y_i - \theta_0 - X_i \cdot \theta)^2 \hspace {1cm} (3)
```

Now, our goal is to try to find good parameter values $b$ and $\theta$ so that MSE has a minimum. 

The Pytorch function for MSE is

```python
loss = torch.nn.MSELoss()
```

### Calculate Gradients

Though we can have an analytic minimum solution for (3) due to the simplicity of the linear regression, we will try to 
use Pytorch Autograd to calculate the gradient numerically instead to show how a machine learning task is carried out in a 
practical way.

```python
yt = torch.tensor(patch_size, dtype=torch.float)
for i in range(n):
  yt[i] = model(x[i])

output = loss(yt, y)
output.backward()
```

After output.backward() is called, the model parameter gradients are stored at model.parameters().

### Optimizer
Now,we can use an optimizer to update the model.parameters(). One very common optimizer is SGD (Stochastic Gradient Descend).

```pyhton
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

#update the parameters
optimizer.step()
```

### Data Preparation

In the real world, normally the data exist already. For tutorial purpose, we will generate data so that it is easy to demonstrate
concepts and processes. Also, in the real world, there may not be any mathematical models existing behind the oberservation 
data. Again, for tutorial purpose, we assume the "UNKNOWN" linear model is

```math
\hspace {6cm} y = 3.0 + 2.0 x_1 -1.5 x_2 + 3.0x_3 -0.5x_4 + 0.3x_5 + \epsilon \hspace {2cm} (5)
```
where $\epsilon$ represents the noise term.

As a typical machine learning task, we need both a training dataset and a test dataset. Further, we use cross-validation to
improve the quality of machine learning. We will sample the datasets to form many pairs of training datasets and test dataset. 
At last, for each training dataset, we partitiion it into many small mini-patches. Here is the steps to prepare the data 
for the linear regression machine learning task,

- generate 360 raw data points in format of $x_{i, 1}, x_{i, 2}, x_{i, 3}, x_{i, 4}, x_{i, 5}, y_i$
- partition the raw data points into 6 datasets and dataset has 60 data points. Denote datasets as $DS_1, DS_2, DS_3, DS_4, DS_5, DS_6$.
- choose one dataset as a test dataset and remaining as training datasets. We have 6 pairs of test and training datasets.
- For each training dataset, parition it into 10 mini-patch. Each mini-patch has 6 datapoints.

One remaining important data is to measure if training is approaching to an optimal solution. This can be done by checking
the progress of MSEs. We set it to 0.01. This means when the MSE progress is less than 0.01, the training is already near an 
optimal solution.

### Storage
Since the size of the observation data is relative small, stored them into a file called raw_data.txt

### Main

### Execution
The training data log is saved in train_log.txt. The column definitions in the train_log.txt is

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mini_batch_sequence number,  training mse (mean square error), bias, $\theta_1, \theta_2, \theta_3, \theta_4, \theta_5$

The linear regression learning does converge with Training MSE = 0.001, Bias = 3.056 and 
$\theta_1 = 2.003$,$\theta_2 = -1.475$, $\theta_3 = 2.959$, $\theta_4 = -0.477$, $\theta_5 = 0.29$
       
The evaluation test data is saved in eval_log.txt. The column definitions in the eval_log.txt is

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; teset_batch_sequence number, test mse (mean square error), bias,  $\theta_1, \theta_2, \theta_3, \theta_4, \theta_5$

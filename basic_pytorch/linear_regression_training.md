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
\hspace {6cm} \tilde y_i = \theta_0 + X_i \cdot \theta^T \hspace {3cm} (2)
```

where $\theta_i \in R$ and $\theta =$ {$ \theta_1, \theta_2, ..., \theta_M$} $\in R^M$.

### Loss Function
So far, the linear model (2) is just a conjesture that it may approximate the datasets X and Y. How do we know how good the approximation is?

This is the role which loss functions play. The one good measurement about the approximation is the distance between the observed $Y$ and the
estimated (or approximated) $\tilde Y = \{ \tilde y_1, ..., \tilde y_n \}$. Formally, the TSSE (total sum of square error),
```math
\hspace {2cm} TSSE = \sum ^{n}_{i=1} (y_i - \tilde y_i)^2 = \sum ^n_{i=1} (y_i - \theta_0 - X_i \cdot \theta)^2 \hspace {1cm} (3)
```

Now, our goal is to try to find good parameter values {$\theta_0, \theta_1, ..., \theta_M$} so that TSSE has a minimum. 

### Calculate Gradients

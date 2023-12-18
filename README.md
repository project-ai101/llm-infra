# llm-infra
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

In-depth tutorials and examples on LLM training and inference infrastructure. They are Pytorch, Fairscale, Nvidia AI Modules (cuDNN, tensorRT, Megatron-LM) and HuggingFace.
The main LLMs are open source models, LLAMA 2 from Meta, Mistral and Mistral MoE.

### LLMs

##### LLM Comparison

##### Llama 

##### Mistral


### Basic Pytorch
In this section, the basic concepts and APIs associated with them are discussed.
And many examples are given to help on how to use them. After this tutorial, 
- [Tensor and Model](./basic_pytorch/tensor_model.md)
- [Auto Gradient and Backward Propagation](./basic_pytorch/autogradient_backwordpropagation.md)
- [Model Training](./basic_pytorch/model_training.md)
- [Graph_Model Format](./basic_pytorch/graph_model_format.md)
- [DataSets](./basic_pytorch/datasets.md): a list of links for popular data sets used for LLM training and validation
are provided.

### Distributed Pytorch - Fairscale
For LLMs, the memory and computation capacity demands require many GPUs. Fairscale is 
framework to support Pytorch run over distributed GPUs. The framework library is an
open source project from Meta.

### Advanced Pytorch

- [Torch Graph](./advanced_pytorch/group.md)
- [Torch Script](./advanced_pytorch/script.md)
- [PyTorch C++](./advanced_pytorch/cpp.md)


### Nvidia AI Modules

##### cuDNN

##### tensorRT

##### tensorRT-LLM
[tensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)


##### Megatron and Megatron-LM
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)


### HuggingFace

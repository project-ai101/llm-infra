# llm-infra
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

LLM-Infra respository provides in-depth tutorials and examples on LLM training and inference infrastructure in one place
for people to learn LLM based AI. The areas which I mainly focus on are Pytorch, Fairscale, Nvidia AI packages 
(cuDNN, tensorRT, Megatron-LM) and HuggingFace.
Also, only open source LLMs, such as, LLAMA 2 from Meta and Mistral LLMs from Mistral, are extensively used 
in the discussion. Open source LLMs allow us to look into the details of LLMs.

In LLMs, tensor computation is the fundamental tool to turn LLMs into reality. For tensor computation,
there are two important aspects, one is to make complex tensor computation easy to use and another is to
make massive tensor computation fast. The Basic Pytorch section is mainly designed to explore how pytorch
makes tensor computation easy. Further, the Advanced Pytorch section looks into ways to extend pytorch with
format exchangability, the LLM Automation section reviews Pytorch Lighting as a platform to automate the
deep learning development flow, the LLM Community section uses Huggingface a focal point for various LLM
related resources, such as, LLMs, Dataset, LLM based applications.

Both Distributed Pytorch and LLM HPC - Nvidia AI Tools sections are mainly for high performance tensor
computation.

### LLMs Overview

##### LLM Comparison

##### Llama 

##### Mistral

### Basic Pytorch


##### Pytorch as a tensor library
- [Tensor](./basic_pytorch/tensor.md)
- [Model](./basic_pytorch/model.md)

##### Pytorch with Autograd
- [Gradient](./basic_pytorch/gradient.md)
- [Autograd](./basic_pytorch/autograd.md)

##### Model Training
- [Model Training](./basic_pytorch/model_training.md)

##### Data Sets
- [DataSets](./basic_pytorch/datasets.md): a list of links for popular data sets used for LLM training and validation
are provided.

### Distributed Pytorch

##### torch.distributed

##### torch Pipeline

##### Fairscale
For LLMs, the memory and computation capacity demands require many GPUs. Fairscale is 
framework to support Pytorch run over distributed GPUs. The framework library is an
open source project from Meta. The github link is 
[here](https://github.com/facebookresearch/fairscale)

### Advanced Pytorch Topics

- [Model Format and Exchange](./advanced_pytorch/group.md)
- [Torch Script](./advanced_pytorch/script.md)
- [PyTorch C++](./advanced_pytorch/cpp.md)
- [PyTorch Quantization](./advanced_pytorch/quantization.md)

### LLM Automation - Pytorch Lightning
Pytorch Lightning is an open source project to develop a deep learning framework 
to pretrain, finetune and deploy AI models. The github link is 
[here](https://github.com/Lightning-AI/pytorch-lightning). 


### LLM HPC - Nvidia AI Tools
##### cuda Graph

##### cuDNN

##### tensorRT

##### tensorRT-LLM
[tensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)


##### Megatron and Megatron-LM
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)


### LLM Community - HuggingFace

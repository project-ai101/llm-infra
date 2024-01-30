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

##### Technical Articles
- The foundation for LLM is the transformer: [Attention is all your need](https://arxiv.org/abs/1706.03762)
- Transformer optimizations: [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf), [PageAttention](https://arxiv.org/abs/2309.06180)
- Quantization: [bitsandbytes](https://arxiv.org/abs/2208.07339), [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SqueezeLLM](https://arxiv.org/abs/2306.07629)
- The technical specification of LLAMA 2: [Llama 2: The Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
- One very important article for MoE (Mixture of Experts): [Misture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
##### LLM Comparison

##### Llama 
To download Llama 2, one may send out the request to access the next version llama to meta via this [meta llama download link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
Then, an email from meta will have detail on how to get the final download links. In the request, it comes three sets of models to be chosen. One is the base model set. They are Llama 2 & Llama 2 Chat.
- Llama 2 7B, 13B and 70B and Llama 2 Chat 7B, 13B and 70B models.

The second set are Code Llama which have 12 models. The introduction can be found in this [link](https://ai.meta.com/blog/code-llama-large-language-model-coding/). The 12 models are
- Code Llama 7B, 13B, 34B and 70B
- Code Llama - Python 7B, 13B, 34B and 70B
- Code Llama - Instruct 7B, 13B, 34B and 70B

The third set is Llama Guard which a Llama 2 - 7B based instruction-tuned input-output safguard for human-AI converstaions. For more information about Llama Guard,
one may refer to this [introduction link](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/).
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

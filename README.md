# llm-infra
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan, [my linkedin](https://www.linkedin.com/in/bin-tan-3145721/)

LLM-Infra respository provides in-depth tutorials and examples on LLM training and inference infrastructure in one place
for people to learn LLM based AI. The areas which I mainly focus on are Pytorch, Fairscale, Nvidia AI packages 
(cuDNN, tensorRT, Megatron-LM) and HuggingFace.
Also, only open source LLMs, such as, LLAMA 2 and Llama 3 from Meta, Mistral LLMs, Gemma from Google 
and Phi from Microsoft from Mistral, are extensively used 
in the discussion. Two new open source LLMs, QWen 2.5 and DeepSeek 2.5 are also in watching. Open source LLMs allow us to look into the details of LLMs.

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
- Word Embedding: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Transformer optimizations: [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf), [FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf), [PageAttention](https://arxiv.org/abs/2309.06180), [vAttention](https://arxiv.org/pdf/2405.04437v1)
- Position Encoding: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Quantization: [bitsandbytes](https://arxiv.org/abs/2208.07339), [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SqueezeLLM](https://arxiv.org/abs/2306.07629)
- The technical specification of LLAMA 2: [Llama 2: The Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
- MoE (Mixture of Experts): [Misture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)

##### Tutorial Links
- Andrej Karpathy's Youtbue Link: [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- Andrej Karpathy's Youtube Link: [Let's Build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Andrej Karpathy's Reproduce GPT-2(124M) tutorial video
    - Youtube Link: [Let's reproduce GPT-2(124M)](https://www.youtube.com/watch?v=l8pRSuU81PU).
    - The source code for the Youtube video in github is [here](https://github.com/karpathy/build-nanogpt).
    - The source code of OpenAI GPT-2 in github is [here](https://github.com/openai/gpt-2)
- Andrej Karpathy's llm.c github Link: [~2000 lines of C code for training GPT-2](https://github.com/karpathy/llm.c)

### Open Source Models

##### Meta Llama 
<ins>Llama 2</ins>: To download Llama 2, one may send out the request to access the next version llama to meta via this [meta llama download link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
Then, an email from meta will have detail on how to get the final download links. In the request, it comes three sets of models to be chosen. One is the base model set. They are Llama 2 & Llama 2 Chat.
- Llama 2 7B, 13B and 70B and Llama 2 Chat 7B, 13B and 70B models.

<ins>Code Llama</ins>: The second set are Code Llama which have 12 models. The introduction can be found in this [link](https://ai.meta.com/blog/code-llama-large-language-model-coding/). The 12 models are
- Code Llama 7B, 13B, 34B and 70B
- Code Llama - Python 7B, 13B, 34B and 70B
- Code Llama - Instruct 7B, 13B, 34B and 70B

<ins>LLama Guard</ins>: The third set is Llama Guard which a Llama 2 - 7B based instruction-tuned input-output safguard for human-AI converstaions. For more information about Llama Guard,
one may refer to this [introduction link](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/).

<ins> Llama 3 </ins>: Llama 3 has two models, 8B and 70B models. To download the models (aka model weights), you may follow the same steps as for downloading Llama2 above. The Llama 3
inference model is hosted at the [github site] (https://github.com/meta-llama/llama3). The LLama 3 Meta page is [here](https://llama.meta.com/llama3/).

<ins> Llama 3.1 </ins>: Llama 3.1 is state of the art multilingual open source LLMs which has three models, 8B, 70B and 405B. The LLM with larger size parameters typically has stronger reasoning capabilities and more accurate responses. To download the models, follow the [link](https://www.llama.com/llama-downloads/)

<ins> Llama 3.2 </ins>: Llama 3.2 provides two categories of LLMs. 
- 11B and 90B models for flexible and can reasoning on high resolution images and output text.
- 1B and 3B models for low cost and also for embedded and edge devices.

To download the models, follow the [link](https://www.llama.com/llama-downloads/)


##### Mistral
Mistral comes with two model Mistral 2B and Mistral 8X7B. Following links have brief introduction of them and links to download and other resources.

- [Mistral 7B in short](https://mistral.ai/news/announcing-mistral-7b/)
- [Mistral 8X7B: Mixtral of Experts, A high quality Sparse Mixture\-of\-Experts](https://mistral.ai/news/mixtral-of-experts/)

##### Google Gemma
Gemma Inference Github Repository
- [Gemma Pytorch](https://github.com/google/gemma_pytorch)
- [Gemma C++](https://github.com/google/gemma.cpp)

##### Microsoft Phi
Microsoft released following Phi 3 open models
- Phi 3 Mini (4K Instruct and 12K Instruct, 3.8B)
- Phi 3 Small (8K Instruct and 128K Instruct, 7B)
- Phi 3 Medium (4K Instruct and 128K Instruct, 14B)
- Phi 3 Vision (128K Instruct, 4.15B)

[Microsoft Phi-Cookbook](https://github.com/microsoft/Phi-3CookBook) is the Macrosoft Phi github repository. To download models, we can get them from Huggingface with huggingface-cli commands.
If you need information on how to install huggingface-cli and how to use the command, please follow the [link](https://huggingface.co/docs/huggingface_hub/guides/cli)

The latest releases from Microsoft are following Phi 3.5 open models
- Phi 3.5 Mini (128K Instruct, 3.82B)
- Phi 3.5 Vision (128K Instruct, 4.15B)
- Phi 3.5 MoE (128K Instruct, 16 x 3.8B parameters with 6.6 active parameters)

##### LLM Comparison


### Basic Pytorch

##### Pytorch as a tensor library
- [Tensor](./basic_pytorch/tensor.md) goes through basic concepts, APIs and usage of torch.Tensor.
- [Model](./basic_pytorch/model.md) introduces how to use torch.Module to build a model and explores the properties of a model in PyTorch.

##### Pytorch with Autograd
- [Gradient](./basic_pytorch/gradient.md) introduces the basic tensor gradient calculation.
- [Autograd](./basic_pytorch/autograd.md) gives a tutorial on PyTorch AutoGrad usages

##### Model Training
Training a LLM is a non-trivial task. From 1000 feet, training models shares some common steps. 

> 1. Model Build
> 2. Loss Function Definition
> 3. Gradient Calculation
> 4. Data Preparation
>>   - Training Data
>>   - Validation Data
> 5. Storage
> 6. Main Function
> 7. Execution


- [Linear Regression Model Training](./basic_pytorch/linear_regression_training.md) A simple example to demonstrate the whole flow of a model training.
- [BERT LLM Training](./basic_pytorch/bert_training.md) A complete BERT model training implementation with Python and Pytorch from scratch. 

##### Data Sets
- [DataSets within PyTorch](./basic_pytorch/pytorch_datasets.md): A brief description about datasets in PyTorch package.
- [Common Open Source DataSets](./basic_pytorch/common_datasets.md): A list of links for popular open source data sets used for LLM training and validation
are provided.

### Distributed Pytorch

##### DataParallel and DistributedDataParallel
torch.nn.DataParallel parallelizes the application of the given torch.nn.Module by splitting the input accross the specified devices by chunking in the batch dimension. This means that th e Module is "replicated" on each devices. This containers has single process and multiple threads.

torch.nn.parallel.DistributedDataParallel (DDP) provides data parallelism by synchronizing gradients across each model replica. It is users' respobilities to chunk or shard the input across participating GPUs. DDP is built based torch.distributed. So, torch.distributed must be initialized by torch.distributed.init_process_group() before create DDP.

In general, DDP is faster than DataParallel. Even for a single node, DDP is recommended.

##### Fully Sharded Data-Parallel Training (FSDP)

##### Tensor Parallel

##### Pipeline Parallel

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

HuggingFace is a LLM community which hosts many models, datasets and others. To use the resources,
you may need to join the HuggingFace community first. The website of Huggingface is [huggingface.co](https://huggingface.co).

##### Download models from HuggingFace
If you want to run your models locally, you need to download them from huggingface website. One way is to use huggingface-cli
commnad. For the detail information about huggingface-cli, one may look into this [webpage](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
Here, the command usage over Ubuntu 22.04 is instroduced briefly.

The command to install huggingface-cli command utility is
```
    pip install -U "huggingface_hub[cli]"
```

After the command util is installed, you may use the help command to list the supported huggingface-cli commands.
```
    huggingface-cli --help
```

To login into the huggingface via huggingface-cli, one may need to have the access token which is stored at
https://huggingface.co/settings/token/

With the token generated to you, you can login into the huggingface site by
```
    huggingface-cli login
```
After login into the huggingface, one can download any supported models, for example,
```
    huggingface-cli download google/codegemma-7b --repo-type=model
```
##### Main concepts in HuggingFace
In HuggingFace, one of the most important concept is transformers. HuggingFace transformers is a library providing APIs for people to use models and datasets in HuggingFace website to perform various tasks. In this library, some of the most important concepts are pipeline, model, tokenizer, dataset and trainer.

<b>pipeline</b> is an action object class which encapsulating model, tokenizer and data

<b>model</b>

<b>tokenizer</b>



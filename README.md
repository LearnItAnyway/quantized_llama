# quantized_llama

This repository contains quantized Llama models, llama model that quantize first few layers using [`gptq`](https://github.com/qwopqwop200/GPTQ-for-LLaMa). 
See [`LearnItAnyway/llama-7b-hf-28q_4bit-128g_WVU`](https://huggingface.co/LearnItAnyway/llama-7b-hf-28q_4bit-128g_WVU) for the detailed descrition of quantized llama.

## Using the Model

```python
# Import the necessary libraries
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM

# First, clone the model repository using git-lfs
# Note: ensure git-lfs is installed before proceeding
# !git clone https://huggingface.co/LearnItAnyway/llama-7b-hf-28q_4bit-128g_WVU

# Specify the model name
model_name = './llama-7b-hf-28q_4bit-128g_WVU'

# Other available models are:
# LearnItAnyway/llama-13b-hf-35q_4bit-128g_WVU
# LearnItAnyway/llama-30b-hf-53q_4bit-128g_WVU

# Load the model and tokenizer
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

```

You can see the test code in `test.ipynb`

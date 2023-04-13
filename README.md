<h1 align="center">Awesome LLMOps</h1>
<p align="center"><a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome" /></a></p>
<p align="center"><img src="./cover.png" height="240" alt="Awesome LLMOps - Awesome list of LLMOps" /></p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [What is LLMOps?](#what-is-llmops)
- [Prompt Engineering](#prompt-engineering)
- [Models](#models)
- [Optimization](#optimization)
- [Tools](#tools)
- [RLHF](#rlhf)
- [Awesome](#awesome)

## What is LLMOps?

LLMOps is a part of MLOps practices, specialized form of MLOps that focuses on managing the entire lifecycle of large language models(LLM).

Starting in 2021, as LLMs evolved rapidly and the technology matured, we began to focus on practices for managing LLMs efficiently, and LLMOps, which are adaptations of traditional MLOps practices to LLMs, began to be talked about.

### LLMOps vs MLOps

| | LLMOps | MLOps |
|-|--------|-------|
| Definition | Tools and infrastructure specifically for the development and deployment of large language models | Tools and infrastructure for general machine learning workflows |
| Focus | Unique requirements and challenges of large language models | General machine learning workflows | Examples of offerings	Foundation model fine-tuning, no-code LLM deployment, GPU access and optimization, prompt experimentation, prompt chaining, data synthesis and augmentation	Model versioning, automated testing, model monitoring, deployment automation, data pipeline management |
| Key technologies | Language model, Transformers library, human-in-the-loop annotation platforms | Kubeflow, MLflow, TensorFlow Extended |
| Key skills | NLP expertise, knowledge of large language models, data management for text data | Data engineering, DevOps, Software engineering, Machine learning expertise |
| Key challenges | Managing and labeling large amounts of text data, fine-tuning foundation models for specific tasks, ensuring fairness and ethics in language models | Managing complex data pipelines, ensuring model interpretability and explainability, addressing model bias and fairness |
| Industry adoption | Emerging, with a growing number of startups and companies focusing on LLMOps | Established, with a large ecosystem of tools and frameworks available
| Future outlook | LLMOps is expected to become an increasingly important area of study as large language models become more prevalent and powerful | MLOps will continue to be a critical component of the machine learning industry, with a focus on improving efficiency, scalability, and model reliability |


## Prompt Engineering

- [PromptBase](https://promptbase.com/) - Marketplace of the prompt engineering
- [PromptHero](https://prompthero.com/) - The website for prompt engineering
- [Prompt Search](https://www.ptsearch.info/tags/list/) - The search engine for the prompt engineering
- [Prompt Perfect](https://promptperfect.jina.ai/) - Auto Prompt Optimizer
- [Learn Prompting](https://learnprompting.org/) - The tutorial website for the prompt engineering
- [Blog: Exploring Prompt Injection Attacks](https://research.nccgroup.com/2022/12/05/exploring-prompt-injection-attacks/)
- [Blog: Prompt Leaking](https://learnprompting.org/docs/prompt_hacking/leaking)
- [Paper: Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353.pdf)

## Models

| Name                       | Parameter size    | Announcement date |
|----------------------------|-------------------|-------------------|
| BERT-Large (336M)          | 336 million       | 2018              |
| T5 (11B)                   | 11 billion        | 2020              |
| Gopher (280B)              | 280 billion       | 2021              |
| GPT-J (6B)                 | 6 billion         | 2021              |
| LaMDA (137B)               | 137 billion       | 2021              |
| Megatron-Turing NLG (530B) | 530 billion       | 2021              |
| T0 (11B)                   | 11 billion        | 2021              |
| Macaw (11B)                | 11 billion        | 2021              |
| GLaM (1.2T)                | 1.2 trillion      | 2021              |
| T5 FLAN (540B)             | 540 billion       | 2022              |
| OPT-175B (175B)            | 175 billion       | 2022              |
| ChatGPT (175B)             | 175 billion       | 2022              |
| GPT 3.5 (175B)             | 175 billion       | 2022              |
| AlexaTM (20B)              | 20 billion        | 2022              |
| Bloom (176B)               | 176 billion       | 2022              |
| Bard                       | Not yet announced | 2023              |
| GPT 4                      | Not yet announced | 2023              |
| AlphaCode (41.4B)          | 41.4 billion      | 2022              |
| Chinchilla (70B)           | 70 billion        | 2022              |
| Sparrow (70B)              | 70 billion        | 2022              |
| PaLM (540B)                | 540 billion       | 2022              |
| NLLB (54.5B)               | 54.5 billion      | 2022              |
| UL2 (20B)                  | 20 billion        | 2022              |
| LLaMA (65B)                | 65 billion        | 2023              |
| Stanford Alpaca (7B)       | 7 billion         | 2023              |
| GPT-NeoX 2.0 (20B)         | 20 billion        | 2023              |
| BloombergGPT               | 50 billion        | 2023              |

## Optimization

- [Blog: A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [Blog: Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)
- [Blog: Handling big models for inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [Blog: How To Fine-Tune the Alpaca Model For Any Language | ChatGPT Alternative](https://medium.com/@martin-thissen/how-to-fine-tune-the-alpaca-model-for-any-language-chatgpt-alternative-370f63753f94)
- [Paper: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [Gist: Script to decompose/recompose LLAMA LLM models with different number of shards](https://gist.github.com/benob/4850a0210b01672175942203aa36d300)

## Tools

- [Promptify](https://github.com/promptslab/Promptify) - ![Repo stars of promptslab/Promptify](https://img.shields.io/github/stars/promptslab/Promptify?style=social) - An utility / tookit for Prompt engineering.
- [trlx](https://github.com/CarperAI/trlx) - ![Repo stars of promptslab/Promptify](https://img.shields.io/github/stars/CarperAI/trlx?style=social) - A repo for distributed training of language models with Reinforcement Learning via Human Feedback. (RLHF)
- [dalai](https://github.com/cocktailpeanut/dalai) - ![Repo stars of cocktailpeanut/dalai](https://img.shields.io/github/stars/cocktailpeanut/dalai?style=social) - The cli tool to run LLaMA on the local machine.
- [haystack](https://github.com/deepset-ai/haystack) - ![Repo stars of deepset-ai/haystack](https://img.shields.io/github/stars/deepset-ai/haystack?style=social) -an open source NLP framework to interact with the data using Transformer models and LLMs.
- [langchain](https://github.com/hwchase17/langchain) - ![Repo stars of hwchase17/langchain](https://img.shields.io/github/stars/hwchase17/langchain?style=social) - The library which assists in the development of applications with LLM.
- [deeplake](https://github.com/activeloopai/deeplake) - ![Repo stars of activeloopai/deeplake](https://img.shields.io/github/stars/activeloopai/deeplake?style=social) - Data Lake for Deep Learning. Build, manage, query, version, & visualize datasets.
- [alpaca-lora](https://github.com/tloen/alpaca-lora) - ![Repo stars of tloen/alpaca-lora](https://img.shields.io/github/stars/tloen/alpaca-lora?style=social) - Instruct-tune LLaMA on consumer hardware.
- [bosquet](https://github.com/BrewLLM/bosquet) - ![Repo stars of BrewLLM/bosquet](https://img.shields.io/github/stars/BrewLLM/bosquet?style=social) - LLMOps for Large Language Model based applications.

## RLHF

- [openai/evals](https://github.com/openai/evals) - ![Repo stars of BrewLLM/bosquet](https://img.shields.io/github/stars/openai/evals?style=social) - A curated list of reinforcement learning with human feedback resources.

## Awesome

- [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)
- [KennethanCeyer/awesome-llm](https://github.com/KennethanCeyer/awesome-llm)
- [f/awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts)
- [promptslab/Awesome-Prompt-Engineering](https://github.com/promptslab/Awesome-Prompt-Engineering)
- [tensorchord/awesome-open-source-llmops](https://github.com/tensorchord/awesome-open-source-llmops)

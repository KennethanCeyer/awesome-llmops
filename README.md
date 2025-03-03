<h1 id="top" align="center">Awesome LLMOps</h1>
<p align="center"><a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome" /></a></p>
<p align="center"><img src="./cover.png" height="240" alt="Awesome LLMOps - Awesome list of LLMOps" /></p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [What is LLMOps?](#what-is-llmops)
- [Prompt Engineering](#prompt-engineering)
- [Models](#models)
- [Optimization](#optimization)
- [Tools (GitHub)](#tools-github)
- [Tools (Other)](#tools-other)
- [RLHF](#rlhf)
- [Awesome](#awesome)
- [Contributing](#contributing)

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

[:arrow_up: Go to top](#top)

## Prompt Engineering

- [PromptBase](https://promptbase.com/) - Marketplace of the prompt engineering
- [PromptHero](https://prompthero.com/) - The website for prompt engineering
- [Prompt Search](https://www.ptsearch.info/tags/list/) - The search engine for the prompt engineering
- [Prompt Perfect](https://promptperfect.jina.ai/) - Auto Prompt Optimizer
- [Learn Prompting](https://learnprompting.org/) - The tutorial website for the prompt engineering
- [Blog: Exploring Prompt Injection Attacks](https://research.nccgroup.com/2022/12/05/exploring-prompt-injection-attacks/)
- [Blog: Prompt Leaking](https://learnprompting.org/docs/prompt_hacking/leaking)
- [Paper: Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353.pdf)

[:arrow_up: Go to top](#top)

## Models

| Name                                      | Parameter Size                     | Announcement Date   | Provider                                      |
|-------------------------------------------|------------------------------------|---------------------|-----------------------------------------------|
| Grok‑3                                    | Undisclosed                        | Feburary 2025       | xAI                                           |
| DeepSeek R1                               | 671 billion                        | January 2025        | DeepSeek                                      |
| DeepSeek V3                               | 671 billion                        | December 2024       | DeepSeek                                      |
| Qwen 2.5 Max                              | Undisclosed                        | June 2024           | Alibaba Cloud                                 |
| Gemma 2                                   | 27 billion                         | June 2024           | Google                                        |
| DeepSeek VL2                              | 4.5 billion                        | June 2024           | DeepSeek                                      |
| Qwen 1.5‑MoE‑A2.7B                        | 14.9B (2.7B active)                | May 2024            | Alibaba Cloud                                 |
| Yi‑1.5‑6B‑Chat                            | 6 billion                          | May 2024            | 01.AI                                         |
| Yi‑1.5‑9B                                 | 9 billion                          | May 2024            | 01.AI                                         |
| DeepSeek V2                               | 238 billion (21B active)           | May 2024            | DeepSeek                                      |
| DeepSeek VL                               | 7 billion                          | May 2024            | DeepSeek                                      |
| Microsoft Phi‑2                           | 2.7 billion                        | April 2024          | Microsoft                                     |
| Llama 3 (8B, 70B)                         | 8B, 70B                            | April 2024          | Meta                                          |
| Gemma 1.1                                 | 7 billion                          | April 2024          | Google                                        |
| Claude 3 (Opus, Sonnet, Haiku)            | Undisclosed                        | March 2024          | Anthropic                                     |
| DBRX (132B total, 36B active)             | 132 billion (36B active)           | March 2024          | Databricks                                    |
| Grok‑1                                    | 314 billion                        | March 2024          | xAI                                           |
| Gemma (2B, 7B)                            | 2B, 7B                             | February 2024       | Google                                        |
| Qwen 1.5 (0.5B, 1.8B, 4B, 7B, 14B, 72B)   | 0.5B, 1.8B, 4B, 7B, 14B, 72B       | February 2024       | Alibaba Cloud                                 |
| Solar                                     | 10.7 billion                       | December 2023       | Upstage                                       |
| Microsoft Phi‑2                           | 2.7 billion                        | December 2023       | Microsoft                                     |
| Google Gemini (Pro, Ultra, Nano)          | Undisclosed                        | December 2023       | Google                                        |
| Mixtral 8x7B                              | 46.7 billion (active)              | December 2023       | Mistral AI                                    |
| Grok‑0                                    | 33 billion                         | November 2023       | xAI                                           |
| Yi‑34B‑200K                               | 34 billion                         | November 2023       | 01.AI                                         |
| Qwen‑VL / Qwen‑VL‑Chat                    | Undisclosed                        | Oct/Nov 2023        | Alibaba Cloud                                 |
| Zephyr‑7b‑beta                            | 7 billion                          | October 2023        | HuggingFace H4                                |
| Mistral 7B                                | 7.3 billion                        | September 2023      | Mistral AI                                    |
| Qwen‑14B‑Chat                             | 14 billion (est.)                  | September 2023      | Alibaba Cloud                                 |
| Qwen‑14B                                  | 14 billion (est.)                  | September 2023      | Alibaba Cloud                                 |
| Qwen‑7B‑Chat                              | 7 billion (est.)                   | August 2023         | Alibaba Cloud                                 |
| Qwen‑7B                                   | 7 billion (est.)                   | August 2023         | Alibaba Cloud                                 |
| Llama 2 (7B, 13B, 70B)                    | 7B, 13B, 70B                       | July 2023           | Meta                                          |
| XGen (7B)                                 | 7 billion                          | July 2023           | Salesforce                                    |
| Falcon (7B, 40B, 180B)                    | 7B, 40B, 180B                      | June/Sept 2023      | Technology Innovation Institute (UAE)         |
| MPT (7B, 30B)                             | 7B, 30B                            | May/June 2023       | MosaicML                                      |
| LIMA (65B)                                | 65 billion                         | May 2023            | Meta AI                                       |
| PaLM 2                                    | Undisclosed                        | May 2023            | Google                                        |
| Vicuna (7B, 13B, 33B)                     | 7B, 13B, 33B                       | March 2023          | LMSYS ORG                                     |
| Koala (13B)                               | 13 billion                         | April 2023          | UC Berkeley                                   |
| OpenAssistant (LLaMA 30B)                 | 30 billion                         | April 2023          | LAION                                         |
| Jurassic‑2                                | Undisclosed                        | April 2023          | AI21 Labs                                     |
| Dolly (Databricks) (6B and 12B)           | 6 & 12 billion                     | March/April 2023    | Databricks                                    |
| BloombergGPT                              | 50 billion                         | March 2023          | Bloomberg                                     |
| GPT‑4                                     | Undisclosed                        | March 2023          | OpenAI                                        |
| Bard                                      | Undisclosed                        | March 2023          | Google                                        |
| Stanford Alpaca (7B)                      | 7 billion                          | March 2023          | Stanford University                           |
| LLaMA (7B, 13B, 33B, 65B)                 | 7B, 13B, 33B, 65B                  | February 2023       | Meta                                          |
| ChatGPT                                   | Undisclosed                        | November 2022       | OpenAI                                        |
| GPT‑3.5 (series)                          | 175 billion (for largest models)   | November 2022       | OpenAI                                        |
| Jurassic‑1 (178B)                         | 178 billion                        | November 2022       | AI21                                          |
| Galactica (120B)                          | 120 billion                        | November 2022       | Meta                                          |
| Sparrow (70B)                             | 70 billion                         | September 2022      | DeepMind                                      |
| NLLB (54.5B)                              | 54.5 billion                       | July 2022           | Meta                                          |
| BLOOM (176B)                              | 176 billion                        | July 2022           | BigScience (Hugging Face)                     |
| AlexaTM (20B)                             | 20 billion                         | August 2022         | Amazon                                        |
| UL2 (20B)                                 | 20 billion                         | May 2022            | Google                                        |
| OPT‑175B                                  | 175 billion                        | May 2022            | Meta (Facebook)                               |
| PaLM (540B)                               | 540 billion                        | April 2022          | Google                                        |
| AlphaCode (41.4B)                         | 41.4 billion                       | February 2022       | DeepMind                                      |
| Chinchilla (70B)                          | 70 billion                         | March 2022          | DeepMind                                      |
| GLaM (1.2T)                               | 1.2 trillion                       | December 2021       | Google                                        |
| Macaw (11B)                               | 11 billion                         | October 2021        | Allen Institute for AI                        |
| T0 (11B)                                  | 11 billion                         | October 2021        | Hugging Face                                  |
| Megatron‑Turing NLG (530B)                | 530 billion                        | January 2022        | Microsoft & NVIDIA                            |
| LaMDA (137B)                              | 137 billion                        | January 2022        | Google                                        |
| Gopher (280B)                             | 280 billion                        | December 2021       | DeepMind                                      |
| GPT‑J (6B)                                | 6 billion                          | June 2021           | EleutherAI                                    |
| GPT‑NeoX 2.0 (20B)                        | 20 billion                         | February 2022       | EleutherAI                                    |
| T5 (11B)                                  | 11 billion                         | October 2019        | Google                                        |
| BERT‑Large                                | 336 million                        | October 2018        | Google                                        |

[:arrow_up: Go to top](#top)

## Optimization

- [Blog: A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [Blog: Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)
- [Blog: Handling big models for inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [Blog: How To Fine-Tune the Alpaca Model For Any Language | ChatGPT Alternative](https://medium.com/@martin-thissen/how-to-fine-tune-the-alpaca-model-for-any-language-chatgpt-alternative-370f63753f94)
- [Paper: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [Gist: Script to decompose/recompose LLAMA LLM models with different number of shards](https://gist.github.com/benob/4850a0210b01672175942203aa36d300)

[:arrow_up: Go to top](#top)

## Tools (GitHub)

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) - ![Repo stars of tatsu-lab/stanford_alpaca](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=social) - A repository of Stanford Alpaca project,  a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations.
- [LoRA](https://github.com/microsoft/LoRA) - ![Repo stars of microsoft/LoRA](https://img.shields.io/github/stars/microsoft/LoRA?style=social) - An implementation of "LoRA: Low-Rank Adaptation of Large Language Models".
- [Dolly](https://github.com/databrickslabs/dolly) - ![Repo stars of databrickslabs/dolly](https://img.shields.io/github/stars/databrickslabs/dolly?style=social) - A large language model trained on the Databricks Machine Learning Platform.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - ![Repo stars of microsoft/DeepSpeed](https://img.shields.io/github/stars/microsoft/DeepSpeed?style=social) - A deep learning optimization library that makes distributed training and inference easy, efficient, and effective.
- [LMFlow](https://github.com/OptimalScale/LMFlow) - ![Repo stars of OptimalScale/LMFlow](https://img.shields.io/github/stars/OptimalScale/LMFlow?style=social) - An Extensible Toolkit for Finetuning and Inference of Large Foundation Models. Large Model for All.
- [Promptify](https://github.com/promptslab/Promptify) - ![Repo stars of promptslab/Promptify](https://img.shields.io/github/stars/promptslab/Promptify?style=social) - An utility / tookit for Prompt engineering.
- [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) - ![Repo stars of Significant-Gravitas/Auto-GPT](https://img.shields.io/github/stars/Significant-Gravitas/Auto-GPT?style=social) - An experimental open-source attempt to make GPT-4 fully autonomous.
- [Jarvis](https://github.com/microsoft/JARVIS) - ![Repo stars of microsoft/JARVIS](https://img.shields.io/github/stars/microsoft/JARVIS?style=social) - A system to connect LLMs with ML community, a composite model connector via the LLM interface.
- [dalai](https://github.com/cocktailpeanut/dalai) - ![Repo stars of cocktailpeanut/dalai](https://img.shields.io/github/stars/cocktailpeanut/dalai?style=social) - The cli tool to run LLaMA on the local machine.
- [haystack](https://github.com/deepset-ai/haystack) - ![Repo stars of deepset-ai/haystack](https://img.shields.io/github/stars/deepset-ai/haystack?style=social) -an open source NLP framework to interact with the data using Transformer models and LLMs.
- [langchain](https://github.com/hwchase17/langchain) - ![Repo stars of hwchase17/langchain](https://img.shields.io/github/stars/hwchase17/langchain?style=social) - The library which assists in the development of applications with LLM.
- [langflow](https://github.com/logspace-ai/langflow) - ![Repo stars of logspace-ai/langflow](https://img.shields.io/github/stars/logspace-ai/langflow?style=social) - An UI for LangChain, designed with react-flow to provide an effortless way to experiment and prototype flows.
- [deeplake](https://github.com/activeloopai/deeplake) - ![Repo stars of activeloopai/deeplake](https://img.shields.io/github/stars/activeloopai/deeplake?style=social) - Data Lake for Deep Learning. Build, manage, query, version, & visualize datasets.
- [alpaca-lora](https://github.com/tloen/alpaca-lora) - ![Repo stars of tloen/alpaca-lora](https://img.shields.io/github/stars/tloen/alpaca-lora?style=social) - Instruct-tune LLaMA on consumer hardware.
- [bosquet](https://github.com/BrewLLM/bosquet) - ![Repo stars of BrewLLM/bosquet](https://img.shields.io/github/stars/BrewLLM/bosquet?style=social) - LLMOps for Large Language Model based applications.
- [llama_index](https://github.com/jerryjliu/llama_index) - ![Repo stars of jerryjliu/llama_index](https://img.shields.io/github/stars/jerryjliu/llama_index?style=social) - A project that provides a central interface to connect your LLM's with external data.
- [gradio](https://github.com/gradio-app/gradio) - ![Repo stars of gradio-app/gradio](https://img.shields.io/github/stars/gradio-app/gradio?style=social) - An UI helper for the machine learning model.
- [sharegpt](https://github.com/domeccleston/sharegpt) - ![Repo stars of domeccleston/sharegpt](https://img.shields.io/github/stars/domeccleston/sharegpt?style=social) - An open-source Chrome Extension for you to share your wildest ChatGPT conversations with one click.
- [Starwhale](https://github.com/star-whale/starwhale) - ![Repo stars of star-whale/starwhale](https://img.shields.io/github/stars/star-whale/starwhale?style=social) - An MLOps/LLMOps platform for model building, evaluation, and fine-tuning.
- [keras-nlp](https://github.com/keras-team/keras-nlp) - ![Repo stars of keras-team/keras-nlp](https://img.shields.io/github/stars/keras-team/keras-nlp?style=social) - A natural language processing library that supports users through their entire development cycle.
- [Snowkel AI](https://github.com/snorkel-team/snorkel) - ![Repo stars of snorkel-team/snorkel](https://img.shields.io/github/stars/snorkel-team/snorkel?style=social) - The data platform for foundation models.
- [promptflow](https://github.com/microsoft/promptflow) - ![Repo stars of microsoft/promptflow](https://img.shields.io/github/stars/microsoft/promptflow?style=social) - A toolkit that simplifies the development of LLM-based AI applications, from ideation to deployment.

[:arrow_up: Go to top](#top)

## Tools (Other)

- [PaLM2 API](https://developers.generativeai.google/) - An API service that makes PaLM2, Large Language Models (LLMs), available to Google Cloud Vertex AI.
- [Perspective API](https://perspectiveapi.com/) - A tool that can help mitigate toxicity and ensure healthy dialogue online.
- [LangSmith](https://langsmith.io/) - A monitoring and debugging platform by the LangChain team that provides systematic performance tracking, error analysis, and logging for LLM-based applications.
- [OpenLLM (by BentoML)](https://docs.bentoml.org/en/latest/openllm/) - A deployment tool from BentoML that simplifies serving various large language models in production environments.
- [PromptLayer](https://promptlayer.com/) - A tool for tracking and analyzing prompt engineering experiments, helping optimize prompt performance and outcomes.

[:arrow_up: Go to top](#top)

## RLHF

- [evals](https://github.com/openai/evals) - ![Repo stars of openai/evals](https://img.shields.io/github/stars/openai/evals?style=social) - A curated list of reinforcement learning with human feedback resources.
- [trlx](https://github.com/CarperAI/trlx) - ![Repo stars of promptslab/Promptify](https://img.shields.io/github/stars/CarperAI/trlx?style=social) - A repo for distributed training of language models with Reinforcement Learning via Human Feedback. (RLHF)
- [PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch) - ![Repo stars of lucidrains/PaLM-rlhf-pytorch](https://img.shields.io/github/stars/lucidrains/PaLM-rlhf-pytorch?style=social) - Implementation of RLHF (Reinforcement Learning with Human Feedback) on top of the PaLM architecture.

[:arrow_up: Go to top](#top)

## Awesome

- [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)
- [KennethanCeyer/awesome-llm](https://github.com/KennethanCeyer/awesome-llm)
- [f/awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts)
- [promptslab/Awesome-Prompt-Engineering](https://github.com/promptslab/Awesome-Prompt-Engineering)
- [tensorchord/awesome-open-source-llmops](https://github.com/tensorchord/awesome-open-source-llmops)
- [opendilab/awesome-RLHF](https://github.com/opendilab/awesome-RLHF)

[:arrow_up: Go to top](#top)

## Contributing

We welcome contributions to the Awesome LLMOps list! If you'd like to suggest an addition or make a correction, please follow these guidelines:

1. Fork the repository and create a new branch for your contribution.
2. Make your changes to the README.md file.
3. Ensure that your contribution is relevant to the topic of LLMOps.
4. Use the following format to add your contribution:
  ```markdown
  [Name of Resource](Link to Resource) - Description of resource
  ```
5. Add your contribution in alphabetical order within its category.
6. Make sure that your contribution is not already listed.
7. Provide a brief description of the resource and explain why it is relevant to LLMOps.
8. Create a pull request with a clear title and description of your changes.

We appreciate your contributions and thank you for helping to make the Awesome LLMOps list even more awesome!

[:arrow_up: Go to top](#top)

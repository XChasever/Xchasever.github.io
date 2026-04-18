---
title: 技术介绍：GPT的原理
date: 2026-04-17 20:10:38
tags:
mathjax: true
---
# GPT-1 详解：生成式预训练Transformer的开创之作
![图片](/images/gpt.jpg)
## 1. 模型背景与核心思想

GPT-1（Generative Pre-trained Transformer）由OpenAI于2018年提出，是**生成式预训练**范式的开创性工作。其核心思想分为两个阶段：

- **无监督预训练**：在海量无标签文本上，通过语言建模任务学习通用的语义和语法知识。
- **有监督微调**：在少量标注数据上对模型参数进行微调，以适应具体下游任务（如分类、问答等）。

这种“预训练+微调”的范式显著降低了对标注数据的依赖，并开启了大规模语言模型时代。

## 2. 模型架构

GPT-1基于**Transformer解码器**结构，移除了原始Transformer中的编码器-注意力交叉模块，仅保留掩码自注意力与位置前馈网络。具体参数如下：

- **层数**：12层
- **隐藏层维度**：768
- **注意力头数**：12
- **参数量**：约1.17亿
- **上下文窗口**：512个token

### 2.1 掩码自注意力（Masked Self-Attention）

为了保持自回归特性（即预测下一个词时只能看到左侧的词），GPT-1在注意力计算中引入上三角掩码矩阵，将未来位置的注意力分数置为 `$-\infty$`，使得softmax后的权重为0。

### 2.2 位置编码

采用可学习的位置嵌入（Learned Positional Embedding），与词嵌入相加后输入模型。与原始Transformer的正弦编码不同，可学习编码能让模型自适应地捕捉位置信息。

### 2.3 激活函数与正则化

- **激活函数**：GELU（Gaussian Error Linear Unit），相比ReLU更平滑，常用于Transformer模型。
- **正则化**：每层后使用残差连接与层归一化（LayerNorm），并采用Dropout（比例0.1）防止过拟合。

## 3. 无监督预训练阶段

### 3.1 训练任务：因果语言建模（Causal Language Modeling）

给定一段文本序列 $U = (u_1, u_2, ..., u_n)$，模型最大化以下似然：

$$
L_1(U) = \sum_{i} \log P(u_i \mid u_{i-1}, ..., u_1; \Theta)
$$

其中 $\Theta$ 为模型参数。即：根据前 $i-1$ 个 token，预测第 $i$ 个 token 的概率。

### 3.2 训练数据

- **数据集**：BooksCorpus（约7000本未出版的书籍，涵盖冒险、浪漫、科幻等多种类型）。
- **数据特点**：长文本连续性较好，有利于模型学习长距离依赖。
- **预处理**：使用Byte Pair Encoding（BPE）将词切分为子词，词汇表大小为40000。

### 3.3 训练细节

- **优化器**：Adam（学习率2.5e-4，权重衰减0.01）。
- **学习率调度**：线性预热（前2000步）后余弦衰减。
- **批次大小**：64
- **训练步数**：100万步
- **硬件**：8张NVIDIA V100 GPU，训练时间约1个月。

## 4. 有监督微调阶段

### 4.1 任务适配

为了将预训练模型应用到不同下游任务，GPT-1设计了统一的输入变换方式，无需改动模型结构：

| 任务类型 | 输入构造方式 | 输出处理 |
|---------|-------------|----------|
| 文本分类 | 在文本前加`[Start]`，后加`[Extract]` | 取最后一层`[Extract]`位置的表征送入线性分类层 |
| 自然语言推理 | 将前提和假设拼接：`[Start]`前提`[Delim]`假设`[Extract]` | 同上 |
| 相似度判断 | 将两个句子按两种顺序拼接，分别输入模型，将两个`[Extract]`表征相加 | 同上 |
| 多项选择 | 将每个选项与上下文拼接，分别输入模型，各自通过`[Extract]`后取softmax | 选择概率最高的选项 |

### 4.2 微调目标函数

联合优化语言建模损失与分类损失：

$$
L_2 = L_{\text{classify}} + \lambda \cdot L_{\text{LM}}
$$

其中 $\lambda = 0.5$。加入语言建模损失可提高微调时的泛化能力。

### 4.3 微调参数

- **学习率**：6.25e-5
- **批次大小**：32
- **训练轮数**：3轮

## 5. 主要实验结果

GPT-1在12个自然语言处理任务中的9个上取得了当时最优结果（相比BERT之前的主流模型）：

| 任务集 | 代表任务 | GPT-1性能 | 对比基线 |
|-------|---------|----------|----------|
| 自然语言推理 | SNLI | 89.9% | 89.4%（Liu et al. 2018） |
| 问答 | RACE | 59.0% | 47.1%（Match-LSTM） |
| 文本分类 | Stanford Sentiment Treebank | 91.8% | 90.4%（LSTM+Attn） |
| 常识推理 | Story Cloze | 86.5% | 84.7（IA-MNLI） |

## 6. GPT-1 的历史意义与局限性

### 6.1 意义

- **范式开创**：首次证明了大规模预训练+微调的有效性，直接启发了BERT、GPT-2等模型。
- **跨任务通用**：统一的输入构造方式避免为每个任务设计专门网络结构。
- **生成与理解统一**：同一个模型既可用于生成（预训练）也可用于判别（微调）。

### 6.2 局限性

- **上下文窗口短**：仅512个token，无法处理书籍等超长文本。
- **单向注意力**：只能看到左侧上下文，在完形填空、双向理解任务上弱于BERT。
- **数据规模**：BooksCorpus（5GB）远小于后来模型的训练数据（如GPT-3的45TB）。
- **无指令微调**：模型不会遵循人类指令，需为每个任务单独微调。

## 7. 快速复现参考

```python
# 使用HuggingFace Transformers加载GPT-1（openai-gpt）
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")

input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(input_ids, max_length=50, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
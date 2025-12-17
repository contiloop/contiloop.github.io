---
title: "DeepSeek-V3-Paper Review"
date: 2025-12-15
categories: [LLM, Paper Review]
tags: [deepseek, moe, mla, transformer, llm]
author: <author_id> 
assets_path: /assets/img/posts/2025-12-15
math: true
mermaid: true
---
# DeepSeek-V3 Technical Report 

## 0. Abstract

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 

To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance.

We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities.

## 1. Introduction

Beyond closed-source models, open-source models, including DeepSeek series, LLaMA series, Qwen series, and Mistral series, are also making significant strides, endeavoring to close the gap with their closed-source counterparts. To further push the boundaries of open-source model capabilities, we scale up our models and introduce DeepSeek-V3, a large Mixture-of-Experts (MoE) model with 671B parameters, of which 37B are activated for each token.

With a forward-looking perspective, we consistently strive for strong model performance and economical costs. Therefore, in terms of architecture, DeepSeek-V3 still adopts Multi-head Latent Attention (MLA) for efficient inference and DeepSeekMoE  for cost-effective training.

Beyond the basic architecture, we implement two additional strategies to further enhance the model capabilities. Firstly,DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing, with the aim of minimizing the adverse impact on model performance that arises from the effort to encourage load balancing. Secondly, DeepSeek-V3 employs a multi-token prediction training objective, which we have observed to enhance the overall performance on evaluation benchmarks.

In order to achieve efficient training, we support the FP8 mixed precision training and implement comprehensive optimizations for the training framework. Low-precision training has emerged as a promising solution for efficient training  its evolution being closely tied to advancements in hardware capabilities.

In this work, we introduce an FP8 mixed precision training framework and, for the first time, validate its effectiveness on an extremely large-scale model. Through the support for FP8 computation and storage, we achieve both accelerated training and reduced GPU memory usage. As for the training framework, we design the DualPipe algorithm for efficient pipeline parallelism, which has fewer pipeline bubbles and hides most of the communication during training through computation-communication overlap. This overlap ensures that, as the model further scales up, as long as we maintain a constant computation-to-communication ratio, we can still employ fine-grained experts across nodes while achieving a near-zero all-to-all communication overhead. In addition, we also develop efficient cross-node all-to-all communication kernels to fully utilize InfiniBand (IB) and NVLink bandwidths. Furthermore, we meticulously optimize the memory footprint, making it possible to train DeepSeek-V3 without using costly tensor parallelism. Combining these efforts, we achieve high training efficiency.

During pre-training, we train DeepSeek-V3 on 14.8T high-quality and diverse tokens. The pre-training process is remarkably stable. Throughout the entire training process, we did not encounter any irrecoverable loss spikes or have to roll back. Next, we conduct a two-stage context length extension for DeepSeek-V3. In the first stage, the maximum context length is extended to 32K, and in the second stage, it is further extended to 128K. Following this, we conduct post-training, including Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) on the base model of DeepSeek-V3, to align it with human preferences and further unlock its potential. During the post-training stage, we distill the reasoning capability from the DeepSeek- R1 series of models, and meanwhile carefully maintain the balance between model accuracy and generation length.

| 항목 | H800 GPU 시간 | 비용(USD) |
|------|---------------|-----------|
| 사전 학습 | 2,664K | $5.328M |
| 컨텍스트 확장 | 119K | $0.238M |
| 후속 학습 | 5K | $0.01M |
| **전체** | **2,788K** | **$5.576M** |

During the pre-training stage, training DeepSeek-V3 on each trillion tokens requires only 180K H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs. Consequently, our pretraining stage is completed in less than two months and costs 2664K GPU hours. Combined with 119K GPU hours for the context length extension and 5K GPU hours for post-training, DeepSeek-V3 costs only 2.788M GPU hours for its full training. Assuming the rental price of the H800 GPU is <span>$</span>2 per GPU hour, our total training costs amount to only $5.576M. Note that the aforementioned costs include only the official training of DeepSeek-V3, excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data.

## 2. Architecture

We first introduce the basic architecture of DeepSeek-V3, featured by Multi-head Latent Attention (MLA) for efficient inference and DeepSeekMoE for economical training. Then, we present a Multi-Token Prediction (MTP) training objective, which we have observed to enhance the overall performance on evaluation benchmarks. For other minor details not explicitly mentioned, DeepSeek-V3 adheres to the settings of DeepSeek-V2.

### 2.1 Basic Architecture

![DeepSeek-V3 Architecture]({{ page.assets_path }}/deepseek_V3.png)

For efficient inference and economical training, DeepSeek-V3 also adopts MLA and DeepSeekMoE, which have been thoroughly validated by DeepSeek-V2. Compared with DeepSeek-V2, an exception is that we additionally introduce an auxiliary-loss-free load balancing strategy for DeepSeekMoE to mitigate the performance degradation induced by the effort to ensure load balance. Figure 2 illustrates the basic architecture of DeepSeek-V3, and we will briefly review the details of MLA and DeepSeekMoE in this section.

#### 2.1.1. Mulit-Head Latent Attention
#### 2.1.2. DeepSeekMoE with Auxiliary-Loss-Free Load Balancing
[How DeepSeek rewrote Mixture of Experts (MoE)? [YouTube]](https://www.youtube.com/watch?v=KnSIZ83iPKs)

**Basic Architecture of DeepSeekMoE.** For Feed-Forward Networks (FFNs), DeepSeek-V3 employs the DeepSeekMoE architecture. Compared with traditional MoE architectures like GShard, DeepSeekMoE uses finer-grained experts and isolates some experts as shared ones. Let $u_𝑡$ denote the FFN input of the 𝑡-th token, we compute the FFN output $h'_t$ as follows:

![DeepSeekMoe]({{ page.assets_path}}/deepseek_moe.png)

Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a normalization among all selected affinity scores to produce the gating values.

기존 MOE의 문제점
- Traning Loss에 lb_loss term이 더해져서, ce_loss와 lb_loss의 gradient tradeoff issue가 발생함.
- experts들이 서로 비슷한 가중치 벡터 방향을 가지게 됨.

해결 방법:
- total loss에서 lb_loss를 빼고, bias를 도입하여 load balancing을 구현함.
- shared experts와 fine-grained experts를 분리함.

**Auxiliary-Loss-Free Load Balancing.** Conventional solutions usually rely on the auxiliary loss to avoid unbalanced load. However, too large an auxiliary loss will impair the model performance. To achieve a better trade-off between load balance and model performance, we pioneer an auxiliary-loss-free load balancing strategy to ensure load balance. To be specific, we introduce a bias term 𝑏𝑖 for each expert and add it to the corresponding affinity scores 𝑠𝑖,𝑡 to determine the top-K routing. ... DeepSeek-V3 keeps balanced expert load during training, and achieves better performance than models that encourage load balance through pure auxiliary losses.

**Complementary Sequence-Wise Auxiliary Loss.** Although DeepSeek-V3 mainly relies on the auxiliary-loss-free strategy for load balance, to prevent extreme imbalance within any single sequence, we also employ a complementary sequence-wise balance loss. The sequence-wise balance loss encourages the expert load on each sequence to be balanced.

**Node-Limited Routing.** Like the device-limited routing used by DeepSeek-V2, DeepSeek-V3 also uses a restricted routing mechanism to limit communication costs during training. In short, we ensure that each token will be sent to at most 𝑀 nodes, which are selected according to the sum of the highest $𝐾_𝑟 /𝑀$ affinity scores of the experts distributed on each node. Under this constraint, our MoE training framework can nearly achieve full computation-communication overlap.

**No Token-Dropping.** Due to the effective load balancing strategy, DeepSeek-V3 keeps a good load balance during its full training. Therefore, DeepSeek-V3 does not drop any tokens during training. In addition, we also implement specific deployment strategies to ensure inference load balance, so DeepSeek-V3 also does not drop tokens during inference.

### 2.2 Multi-Token Prediction

[How DeepSeek rewrote Multi-Token Prediction (MTP)?](https://www.youtube.com/watch?v=4GmwJLvwaXE)

we investigate and set a Multi-Token Prediction (MTP) objective for DeepSeek-V3, which extends the prediction scope to multiple future tokens at each position. On the one hand, an MTP objective densifies the training signals and may improve data efficiency. On the other hand, MTP may enable the model to pre-plan its representations for better prediction of future tokens. Figure 3 illustrates our implementation of MTP. Different from Gloeckle et al. (2024), which parallelly predicts 𝐷 additional tokens using independent output heads, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth. We introduce the details of our MTP implementation in this section.

**MTP Modules.** To be specific, our MTP implementation uses 𝐷 sequential modules to predict 𝐷 additional tokens. The 𝑘-th MTP module consists of a shared embedding layer $Emb(·)$, a shared output head $OutHead(·)$, a Transformer block $TRM
_𝑘(·)$, and a projection matrix $𝑀_𝑘 ∈ R^{𝑑×2𝑑}$. 
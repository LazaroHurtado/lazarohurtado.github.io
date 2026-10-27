---
title: "Model parallelism techniques for large batches"
date: 2026-10-25
description: "How to efficiently train on millions of tokens per batch"
---

## Introduction
If you've read any recent open-source LLM papers you might have noticed how they train on billions to trillions of tokens during the pre-training stage, with the batch size itself also on the order of millions of tokens. Kaplan's scaling laws[^kaplan] not only show optimal dataset and model parameter size but also capture a trend in critical batch size as a function of loss. Unless otherwise specified, we will talk about batch size in terms of tokens.

Kaplan's $B_\text{crit}(\mathcal{L})$ is the optimal batch size that trades off compute and time to achieve a loss $\mathcal{L}$, independent of model size! For a CE loss of $3$ they found $B_\text{crit}\approx10^6$, with a $13\%$ decrease in loss for every doubling of $B_\text{crit}$. Meaning, early into pre-training the critical batch size is relatively smaller as opposed to the final stages of pre-training. We can see this exact relationship on a bigger scale with DeepSeek-V3[^deepseek-v3], where their global batch size gradually increases from 12.6M tokens to 62.9M tokens within the first $469$B tokens. In normal batch size terms (not in tokens), this leads to $3072$ and $15360$ samples in the batch, calculated by using DeepSeek-V3's $4096$ pre-training sequence length.

Let's think about what this means. If a single batch element has $s=4096$ tokens and we are training a model with $L$ layers and hidden dimension $h$, what would our memory requirement be just for the activations of this single batch element? With flash attention this would lead to $\sim16sh$ element activations needed to be stored, per layer. Thus, across all layers we get $16Lsh$ activations, and if we train using bf16 it's $2(16Lsh)$ bytes total. A small $7$B language model with $L=32, \ h=4096$ would lead to $2*16*32*4096*4096\approx17.2$GB of VRAM, for just one batch element and not taking into account other memory requirements! It's worth mentioning that there are techniques for reducing activation memory, such as activation checkpointing[^activation-checkpointing], which is a trade-off between memory and time. For this blog, we focus on handling large batch sizes as is.

This raises the question we aim at answering
> **how do language models efficiently train on millions of tokens per batch?**

## Gradient Accumulation
The key word is "efficiently", and we stress that because there is one common technique that is typically used, gradient accumulation[^gradient-accumulation]. Gradient accumulation breaks a batch size $B$ into a micro-batch $m_b$ and before doing an optimizer step it accumulates each micro-batch's gradient. This is valid due to the linearity of differentiation, $\nabla_\theta(f+g)=\nabla_\theta f+\nabla_\theta g$.
Expressed formally, the loss for batch size $B$ is
$$
\mathcal{L}(\theta)=\frac{1}{B}\sum_i^B \ell(x_i;\theta)
$$while the loss for micro-batches $m_b$ is
$$
\mathcal{L}(\theta)=\frac{1}{Km_b}\sum_k^K\sum_{x_i \in m_b^k}\ell(x_{i};\theta)=
\frac{1}{K}\sum_k^K\frac{1}{m_b}\sum_{x_i \in m_b^k}\ell(x_{i};\theta)=
\frac{1}{K}\sum_k^K\mathcal{L}_k(\theta)
$$
with equivalent gradients w.r.t $\theta$ since $Km_b=B$. So this gives us a technique to break a large batch size down into smaller $m_b$ sizes:
```python
for batch in data: # len(batch) % K = 0
    optimizer.zero_grad()
    
    for micro_batch in batch: # sequential
		x, y_true = micro_batch
		y_pred = model(x)
		
		mb_l = loss(y_pred, y_true) / K
		mb_l.backward()
    
    optimizer.step()
```
but this introduces an inefficiency, most notably the fact that a single optimizer step takes $K$ times longer now.

## DDP
Model training is all about parallelization: prefetching data, overlapping host and device work, overlapping communication primitives, etc. So a natural question is, can we speed up this micro-batch model training strategy? The answer is yes, and the key is to compute batches separately and in parallel.

Before we continue, let's align with the common notation. Since we are breaking batches up into micro-batches and will now parallelize them, we will use the official term global batch size, denoted by $gbs$. The micro-batches we use will be denoted as $mbs$, and the gradient accumulation hyper-parameter $K$ will change into $\text{grad\_acc}$.

Distributed data parallel (DDP)[^ddp] does exactly what we are asking for. DDP creates a replica of our model on every accelerator it sees and distributes batches across each replica. In other words, we have a copy of our model across our accelerators and each handles a unique batch from our dataset. Now, when we call `.backward()` on our loss, we sum and average the gradients across all replicas (recall this is valid due to linearity). The total number of accelerators we have is given by the world size $W$, and the number of replicas we have is given by $dp$ ($W$ does not always equal $dp$!). With this, we can reduce the number of sequential micro-batches in favor of parallelized $dp$ ranks, $gbs=mbs\times\frac{\text{grad\_acc}}{dp}\times dp$.

Great, so we found a way of speeding up micro-batch size training by using more GPUs, but if we were to ask ourselves if there are any inefficiencies we would realize that there is 1 new inefficiency introduced by DDP.

## FSDP
DDP served us well in parallelizing micro-batches but its downside is the need to replicate models across accelerators. The issue is not that we are replicating, but what we replicate. Every single parameter, gradient, and optimizer state is replicated across all accelerators, how unfortunately redundant. This leads to the next question, can we have one copy of all these values and share them across accelerators when they need them? As you expect, the answer is also yes here!

Fully Sharded Data Parallel (FSDP)[^fsdp] takes the parameters, gradients, and optimizer states and shards them across all $dp$ ranks so each accelerator holds a slice of each rather than an exact copy. Since no single rank holds a full layer's weight matrix, extra communication is introduced for each rank to gather a layer's parameters. This doesn't have to be one layer at a time, with PyTorch's `fully_shard` we can shard subsections of our model. FSDP will fetch each sharded unit and every rank will compute the forward pass on that unit (`forward` will be slowed down by the slowest accelerator) while FSDP is fetching the next unit in the background. The same will be done during the backward pass to compute the gradients. We should note that FSDP is PyTorch's version of DeepSpeed ZeRO[^zero], which was the paper and library that first introduced sharding of these parameters across replicas. DeepSpeed supports three sharding levels:
1. ZeRO-1: Sharding just the optimizer states
2. ZeRO-2: Sharding the optimizer states + gradients
3. ZeRO-3: Sharding the optimizer states + gradients + parameters (what FSDP offers)

So how much VRAM do we require when using FSDP? For a model with $N$ parameters being trained as bf16 and using Adam with a master copy of the weights in fp32, we require
$$
\text{FSDP}_\text{base}(N)\approx\frac{1}{dp}(\underbrace{2N}_\text{params}+\underbrace{2N}_\text{gradients}+\underbrace{12N}_\text{optimizer states})=\frac{16N}{dp}
$$
During training we have to materialize the current unit, $u_l$ and however many we are prefetching, $k$. During the backward pass we need the unsharded weights, gradient buffer, and prefetch buffers as well. Of these two, the backward pass requires more memory than the forward pass since it needs both the unsharded weights and gradients. This leads to an active memory requirement of
$$
\text{FSDP}_\text{active}(N)\approx\frac{16N}{dp}+4k\max_l(u_l)
$$where $k$ is the prefetch factor and $\max_l(u_l)$ is the largest FSDP sharded unit by parameters in our model, and the $4$ multiplier comes from the bf16 unsharded weights and gradients. Putting this together, we will need a peak VRAM of
$$
\text{FSDP}_\text{peak}(N)\approx\text{ActivationMem}+\text{FSDP}_\text{active}(N)
$$If we were able to support DDP we can definitely handle FSDP, else we must at least accommodate $\text{FSDP}_\text{peak}$.

The global batch size formula stays the same, $gbs=mbs\times\frac{\text{grad\_acc}}{dp}\times dp$, but FSDP grants us extra VRAM headroom, potentially enabling us to use a larger $mbs$ and thus further speed up compute.

## Bottleneck at Scale
As you probably expect by now, we aren't done. Although FSDP resolved the inefficiency of DDP there is a new bottleneck that only comes into play at scale for both FSDP and DDP. The bottleneck is a side-effect of distributed communication primitives and ring-based networking topology that dominates beyond $16-32$ nodes[^tp], with 8 GPUs per node.

DDP has a single communication primitive, `all-reduce`, on the gradients. FSDP, on the other hand, has two `all-gather` and one `reduce-scatter`, but this is equivalent to one `all-gather` and one `all-reduce` since `all-reduce = reduce-scatter + all-gather`. So the difference between DDP and FSDP is 1 communication primitive call. Within a single GPU node there is usually high-speed GPU-to-GPU interconnect, an example being NVLink, but beyond that the networking bandwidth starts to drastically slow down having to do transfers over InfiniBand/Spectrum Ethernet switches.

> **Note:** NVL72 expands NVLink to all nodes in the rack, allowing for NVLink speeds on 72 GPUs rather than just 8 GPUs on a single node.

Let's consider just the `all-gather` primitive since it's easiest to visualize and consistent with how other primitives work. The `all-gather` primitive, under Nvidia's communication collective library (NCCL), works by sending its slice to a neighboring GPU while receiving another slice from another neighbor. This topology is called ring-based, and requires at most $N-1$ hops for all $N$ GPUs to have every slice, hence the `all-gather` name. Every communication primitive over the network is limited by bandwidth and startup latency. For example, a single hop takes as much as
$$
T_\text{single hop}=\alpha+\frac{S}{B}
$$
where $\alpha$ is the startup latency, $S$ is the slice size in bytes, and $B$ is the networking bandwidth. In our FSDP scenario, $S=\frac{M}{N}$ since each unit's size $M$ must be shared across all $N$ ranks. Extending this to the full $N-1$ hops we get the total time
$$
T=\underbrace{(N-1)\alpha}_\text{ring latency} + \underbrace{\frac{N-1}{N}\frac{M}{B}}_\text{bandwidth}
$$The bandwidth term is constant but unfortunately the ring latency term grows linearly with $N$. Initially the bandwidth is the main bottleneck but as we increase from $128$ nodes ($1024$ GPUs) to $512$ nodes ($4096$ GPUs) the ring latency dominates. This leads to communication taking up majority of the forward pass, while the actual compute time itself remains manageable.

## Forward looking
A new problem leads to room for new innovation, which is what led to tensor parallelism and hybrid sharded data parallel, two techniques to curb the effects of communication overhead at scale. We leave discussion of those techniques to future writing. For now, we have come to an answer to our original question of how modern LLMs handle millions of tokens efficiently and at scale.

## References

[^kaplan]: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
[^deepseek-v3]: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).
[^activation-checkpointing]: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174).
[^gradient-accumulation]: [Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187).
[^ddp]: [PyTorch distributed data parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).
[^fsdp]: [PyTorch fully sharded data parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).
[^zero]: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
[^tp]: [PyTorch tensor parallelism tutorial: when and why to apply tensor parallelism](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html#when-and-why-you-should-apply-tensor-parallel).
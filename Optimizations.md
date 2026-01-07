## Emformer Model Architecture
<img width="597" height="514" alt="image" src="https://github.com/user-attachments/assets/1ebe15d3-ef03-40e1-99e5-6b4b34eaac94" />

The specific model instantiated by the code consists of an input projection, 20 emformer layers, followed by 3 LSTM layers, and finally 2 linear layers. The total size of the model with FP32 parameters is 292MiB.
For RTF requirements, we need one pass through this model every 0.16s, for a parameter bandwidth requirement of 1.78GiB/s

Running `lscpu` in the Colab instance, the Intel Xeon processor being used has the following memory hierarchy:

| Cache    | Agg. Size |
| -------- | ------- |
| L1d  | 32KiB    |
| L1i | 32KiB     |
| L2    | 256KiB    |
| L3    | 55MiB    |

This means the model weights will not fit in Cache and will be read from DDR for each forward pass of the model. Still, 1.78GiB/s is not a significant BW constraint on a modern datacenter CPU. It seems that this model on this hardware is primarily compute bound on the large matrix multiplication (linear, addmm) layers based on the operator breakdown in `Analysis.md`.

## Optimization Proposals

### Low hanging fruit
Some quick options to try in different optimization scenarios include batching across the chunks, however since this model uses overlapping chunks in an autoregressive manner (`T_i` depends on `T_i-1`), this is not applicable here. True batching oly applies in the multi-stream scenario (e.g. multi-user), and then will only serve to improve compute efficiency/aggregate system throughput, but not RTF or TTFT latency.

Compiling the model with torch.compile() can optimize the model via operator fusion, optimized kernel selection etc. I tried this by adding `decoder = torch.compile(decoder)`, and saw no meaningful difference (i.e. average chunk latency difference in the noise), and most optional arguments for torch.compile() are for GPU inference.

### Mixed/Reduced Precision Inference

The provided model has FP32 parameters and computes on fp32 intermediate activations. On hardware that supports increased throughput with smaller datatypes (e.g. bfloat16, fp16, fp8), mixed or reduced precision inference can provide significant performance improvements over an fp32 baseline. Naive casting, using torch.amp (below), incurred an overhead (~2x performance loss) as the casting of the weights itself seems to incur significant latency in this case.
```python
with torch.inference_mode(), torch.amp.autocast(
                'cpu',
                dtype=torch.float16,
                enabled=True
            ):
```

To fully benefit from this optimization, we could either re-train the full network in fp16/bf16 (or, more effectively, cast and then fine-tune the weights), or try direct casting offline of the weights to produce a new frozen .pt of the model.

### Mixture-of-Experts

Recalling the operator breakdown table in `Analysis.md`, a significant proportion of the time and compute goes to the linear layers of the feed-forward network in each Emformer block (as much as 34% of CPU time in the run profiled). One technique to reduce inference compute/bandwidth requirements that has become widespread in laguage modelling is to use Mixture-of-Expert (MoE) Feed Forward Networks (FFNs) in the Transformer blocks of LLMs.

<img width="1148" height="665" alt="image" src="https://github.com/user-attachments/assets/43046f8f-23b0-4431-978c-502329fb2c20" />

The key concept is shown in the diagram above, where each FFN is replaced with a number of parallel "Expert" FFNs, each with unique weigths. At inference time, each token is dynamically routed to some small subset of the experts, and the outputs of the experts are combined with a weighted sum based on a score that is output by the router/gating function. The key saving comes from the fact that the dimensions of each expert FFN can be smaller than the dimensions that would be required to achieve the same accuracy using a monolithic FFN, as each smaller expert FFN can "specialize" in processing certain types of input tokens.

The MoE implementation applied to Transformers is described in detail in the [GShard paper](https://arxiv.org/pdf/2006.16668).

A basic PyTorch implementation of the MoE router is included below:

```python
class MoERouter(nn.Module):
    def __init__(self, dim_in, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(dim_in, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states):
        # Project through linear gating layer
        router_logits = self.gate(hidden_states) # Shape: (tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        router_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1) # shape: (tokens, top_k)
        # Normalize expert weights
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts) # Shape: (tokens, top_k, num_experts)
        return router_logits, router_weights, selected_experts, expert_mask

```

The output of the experts to be combined/aggregated as a weighted sum per-token using the `router_weights`.
Besides the router and expert aggregation, the implementation of each Expert FFN is identical to the standard FFN already used in the provided Emformer RNNT model, with the exception that the dimensions may change.
The exact dimensions to use, and indeed the general parameters of the MoE block (including the `k` in top-k) would need emperical analysis to ensure there isn't an accuracy issue. As a starting point, let's assume k=2 and a reduced FFN intermediate dimension of 768 (as opposed to the current 2048), and 8 experts.

During the forward pass, the current flops per FFN per token is approx. `2*(512*2048 + 2048*512) = 4,194,304`. With the MoE change, this would reduce to `2*(512*768 + 768*512)*2 = 3,145,728`, a 25% flops saving. The cost of this is that the memory footprint of each FFN has increased from `4Bytes * (512*2048 + 2048*512) = 8MiB` to `4Bytes * (512*768 + 768*512)*8 = 24MiB`, a 3x increase. However, if the FFN is truly compute-bound on CPU rather than bandwidth-bound, this will still result in a performance improvement. Based on the profiling, this could be a ~9% latency saving for the model forward pass, enough to improve the RTF of some tail-latency samples in the experiments run to below 1.

There is a risk that, in introducing MoE, we have converted a compute-bound problem into a bandwidth-bound one, as MoE routing of tokens to separate experts decreases the arithmetic intensity of the FFN. This would require further profiling of an initial implementation to determine.

Of course, any modifications to the model architecture like this will require re-training the model. This is a cost consideration, as training runs can be costly.

### Further considerations
There are multiple further options to consider at the algorithmic level that may improve some measures of performance at the cost of accuracy or other trade-offs.

Increasing the frame width/window that the Mel Spectrogram-based feature extractor considers to produce a single vector of inputs to the network would mean less inference compute per unit of audio time, but may incur accuracy challenges - this is likely a parameter that has been fine tuned for this trade-off already.

Decreasing the right-context size used by the Emformer would similarly result in reduced inference compute (better RTF), but may again impact accuracy. This has the additional benefit of potentially reducing the TTFT, as we can capture/process less audio input at the start of the stream before making an inference.






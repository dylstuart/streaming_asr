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

The provided model has FP32 parameters and computes on fp32 intermediate activations. On hardware that supports increased throughput with smaller datatypes (e.g. bfloat16, fp16, fp8), mixed or reduced precision inference can provide significant performance improvements over an fp32 baseline. Naive casting, using torch.amp (below), incurred an overhead (~2x performance loss) as the casting itself seems to incur significant latency in this case.
```python
with torch.inference_mode(), torch.amp.autocast(
                'cpu',
                dtype=torch.float16,
                enabled=True
            ):
```

However, 

### Mixture-of-Experts

Recalling the operator breakdown table in `Analysis.md`, a significant proportion of the time and compute goes to the linear layers of the feed-forward network in each Emformer block (as much as 34% of CPU time in the run profiled). One technique to reduce inference compute/bandwidth requirements that has become widespread in laguage modelling is to use Mixture-of-Expert (MoE) Feed Forward Networks (FFNs) in the Transformer blocks of LLMs.

<img width="1148" height="665" alt="image" src="https://github.com/user-attachments/assets/43046f8f-23b0-4431-978c-502329fb2c20" />

The key concept is shown in the diagram above, where each FFN is replaced with a number of parallel "Expert" FFNs, each with unique weigths. At inference time, each token is dynamically routed to some small subset of the experts, and the outputs of the experts are combined with a weighted sum based on a score that is output by the router/gating function. The key saving comes from the fact that the dimensions of each expert FFN can be smaller than the dimensions that would be required to achieve the same accuracy using a monolithic FFN, as each smaller expert FFN can "specialize" in processing certain types of input tokens.







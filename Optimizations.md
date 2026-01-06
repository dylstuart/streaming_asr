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

This means the model weights will not fit in Cache and will be read from DDR for each forward pass of the model. Still, 1.78GiB/s is not a significant BW constraint on a modern datacenter CPU.

The model hidden dimension is 512, with a 4x expansion in the feedforward block. For 20 layers and 640 frames of context, this is 12.5MiB of K,V cache. However, the beam search width is a multiplier on the K,V cache size, as we (naively) must keep the cache live for each path in the tree of the beam search. Nevertheless, even with a `beam_width` of 10 used in the notebook, the additional overall bandwidth requirements are still unlikely to be a significant bottleneck (Additional 0.76GiB/s).

## Mixed/Reduced Precision Inference

Naive casting, using torch.amp (below), incurred an overhead (~2x performance loss) as the casting itself incurs significant latency.
```python
with torch.inference_mode(), torch.amp.autocast(
                'cpu',
                dtype=torch.float16,
                enabled=True
            ):
```

## MLA
<img width="1547" height="472" alt="image" src="https://github.com/user-attachments/assets/9e837bd7-81b8-45f6-ab92-a9a65e3774c2" />
DeepSeek’s innovation is to introduce a weight matrix WDKV∈Rdc×d to compress the input X∈Rd×n to a lower rank matrix CKV∈Rdc×n. This CKV matrix is then stored in the cache. Then two other weight matrices WUK and WUV∈RdhH×dc uncompress the same CKV matrix to the key K and value V respectively. The above figure shows this visually.



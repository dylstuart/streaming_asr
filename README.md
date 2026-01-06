# Streaming ASR Model Benchmarking and Optimization
This repository contains python notebooks providing profiling and analysis of an Emforming ASR model running on CPU.

The primary notebook is `asr_analysis.ipynb`, which includes code to run and analyse the latency of the full ASR pipeline.

The `asr_analysis_torch_profile.ipynb` is used solely to extract a per-operater CPU latency breakdown (the act of which affects overall pipeline latency).

## Setup

pip3 install -r requirements.txt
Note: torchaudio.io.StreamReader used in this repo requires torchaudio==2.8.0

Additional on Windows for ffmpeg installation:
`winget install ffmpeg -v 6.1`

### Dataset:
Samples in the `/samples` directory are a subset randomly selected from the dataset at the following link:
https://datacollective.mozillafoundation.org/datasets/cminc35no007no707hql26lzk

## Relevant metrics
Time to first token

Real-Time Factor

Bottleneck analysis:
- Feature extraction latency
- RNNT model latency
- Decoding latency

## Optimization opportunities
Amortize long-latency operations over larger chunks
- Trades off TTFT (responsiveness) vs RTF/Throughput

## Further reading
RNNTBundle details: https://docs.pytorch.org/audio/2.2.0/generated/torchaudio.pipelines.RNNTBundle.html#torchaudio.pipelines.RNNTBundle

Emformer paper: https://arxiv.org/pdf/2010.10759

Multihead Latent Attention: https://liorsinai.github.io/machine-learning/2025/02/22/mla.html

https://github.com/pytorch/audio/tree/main/examples/asr/emformer_rnnt

PyTorch Base Emformer Model:
https://download.pytorch.org/torchaudio/models/emformer_rnnt_base_librispeech.pt

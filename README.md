# streaming_asr
Streaming ASR Model Benchmarking and Optimization

## Setup
winget install ffmpeg -v 6.1

pip3 install -r requirements.txt
Note: torchaudio.io.StreamReader used in this repo requires torchaudio==2.8.0

Dataset to download:
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

# Streaming ASR Model Benchmarking and Optimization
This repository contains Jupyter notebooks providing profiling and analysis of an Emformer ASR model running on CPU.

The primary notebook is `asr_analysis.ipynb`, which includes code to run and analyse the latency of the full ASR pipeline.

The `asr_analysis_torch_profile.ipynb` is a simplified notebook used solely to extract a per-operater CPU latency breakdown (the act of which affects overall pipeline latency) - see `Analysis.md`.

## Setup
This repo is tested with `python 3.12.12` on `Ubuntu 22.04.5 LTS` in a Google Colab environment.

The notebooks provided include installation of additional dependencies, but many dependencies are pre-installed in the standard Colab environment.

If running in a local environment, you can install dependencies with:
```
pip3 install -r requirements.txt
```
Note: torchaudio.io.StreamReader used in this repo requires torchaudio==2.8.0

The notebooks (.ipynb files) can be run by opening them in Colab (links included in files), starting a Jupyter server, or using an IDE or IDE plugin (VSCode can run .ipynb notebooks).

If your local environment is Windows, you can install ffmpeg with:
`winget install ffmpeg -v 6.1`

### Dataset:
Audio samples in the `/samples` directory are a subset randomly selected from the Mozilla Common Voice dataset at the following link:
[Mozilla Common Voice Spontaneous Speech ASR Shared Task Test Data](https://datacollective.mozillafoundation.org/datasets/cminc35no007no707hql26lzk)

If running and Colab, you can either [clone the repo directly in Colab](https://gist.github.com/odewahn/2c0d515a838e65b0e9dcdd0c6e177c29), or upload the samples from this repo to a directory named `/samples` created in that session, recreating the structure of this repo.

## Further reading
RNNTBundle details: https://docs.pytorch.org/audio/2.2.0/generated/torchaudio.pipelines.RNNTBundle.html#torchaudio.pipelines.RNNTBundle

Emformer paper: https://arxiv.org/pdf/2010.10759

GShard paper: https://arxiv.org/pdf/2010.10759

PyTorch Base Emformer Model:
https://download.pytorch.org/torchaudio/models/emformer_rnnt_base_librispeech.pt

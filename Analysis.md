# Model Description
The pipeline primarily consists of 3 components:
Feature extractor:

```python
return _ModuleFeatureExtractor(
            torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length
                ),
                _FunctionalModule(lambda x: x.transpose(1, 0)),
                _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
                _GlobalStatsNormalization(local_path),
            )
        )
  ```
  Decoder, which consists of the actual neural network, and the beam search algorithm:
  
```python
   model = self._get_model()
   return RNNTBeamSearch(model, self._blank)
```
   Token Processor:
   ```python
   return _SentencePieceTokenProcessor(local_path)
```

# Latency Analysis

## CDF of chunk latencies by pipeline component
Let's confirm that the neural network forward pass is the primary bottleneck by plotting the latencies of each pipeline component.

<img width="789" height="590" alt="image" src="https://github.com/user-attachments/assets/ef4d8ffb-fbb4-4d4c-b677-89fc41c9a71f" />

The chunk processing latency is dominated by the Emformer forward pass (which includes Beam Search), so latency optimizations should be focused there.

## Time To First Token
The time to first token (TTFT) gives a measure of responsiveness of the application. In a streaming scenario, it will represent how soon after the request for output the user will have to wait to begin receiving output.

<img width="687" height="547" alt="image" src="https://github.com/user-attachments/assets/425903d3-de83-4731-85db-2d30b2145c70" />


## Real Time Factor
Real time factor (RTF) is a ratio of the processing time for a given sample of input to the actual duration of the sample. An RTF > 1 means that it takes longer to process an input than the duration of the input, meaning the output will lag further and further behind the real-time input (unless audio frames are "dropped" to allow catch-up), leading to eventual failure of the application from either a practical perspecitve (user experience is too poor to be useful) or a system perspective (input buffers run out of memory). An RTF <= 1 is therefore required for continuous streaming ASR without loss of quality.

Note that, as measured in the code provided, the RTF includes any padding needed for the last chunk in the audio sample, introducing some error if we use a strict definition of RTF. However, in the streaming case, if we consider a very long audio stream, this final padding diminishes to insignificance, and so measuring this way gives an indication of RTF in the final application context. Even as is, with a segment/chunk duration of 0.16s and audio samples of duration in the order of ~10s, this error will be at most 1%-2%.

<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/99daa727-1466-4866-a39e-cb1e5a75bec4" />


There is a relatively tight spread of RTF around 1.0, with some above 1.0 [in this particular run, 4 out of 25 audio files had an RTF > 1]. This requires performance optimization to consistently meet real time streaming requirements.




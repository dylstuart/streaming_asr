# Model Description
The pipeline primarily consists of 3 components (code from installed torchaudio package below):

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

The underlying pretrained RNNT model instantiated seems is downloaded from: https://download.pytorch.org/torchaudio/models/emformer_rnnt_base_librispeech.pt

# Latency Analysis

## CDF of chunk latencies by pipeline component
Let's confirm that the neural network forward pass is the primary bottleneck by plotting the latencies of each pipeline component.

<img width="788" height="590" alt="image" src="https://github.com/user-attachments/assets/d199e5b3-452f-4030-92eb-3d3daf9d5cae" />

The chunk processing latency is dominated by the Emformer forward pass (which includes Beam Search), so latency optimizations should be focused there.

## Time To First Token
The time to first token (TTFT) gives a measure of responsiveness of the application. In a streaming scenario, it will represent how soon after the request for output the user will have to wait to begin receiving output.

<img width="687" height="547" alt="image" src="https://github.com/user-attachments/assets/425903d3-de83-4731-85db-2d30b2145c70" />


## Real Time Factor
Real time factor (RTF) is a ratio of the processing time for a given sample of input to the actual duration of the sample. An RTF > 1 means that it takes longer to process an input than the duration of the input, meaning the output will lag further and further behind the real-time input (unless audio frames are "dropped" to allow catch-up), leading to eventual failure of the application from either a practical perspecitve (user experience is too poor to be useful) or a system perspective (input buffers run out of memory). An RTF <= 1 is therefore required for continuous streaming ASR without loss of quality.

Note that, as measured in the code provided, the RTF includes any padding needed for the last chunk in the audio sample, introducing some error if we use a strict definition of RTF. However, in the streaming case, if we consider a very long audio stream, this final padding diminishes to insignificance, and so measuring this way gives an indication of RTF in the final application context. Even as is, with a segment/chunk duration of 0.16s and audio samples of duration in the order of ~10s, this error will be at most 1%-2%.

<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/99daa727-1466-4866-a39e-cb1e5a75bec4" />


There is a relatively tight spread of RTF around 1.0, with some above 1.0 [in this particular run, 4 out of 25 audio files had an RTF > 1]. This requires performance optimization to consistently meet real time streaming requirements.

## Per-chunk latency over time

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/ea57bba1-787e-4465-b935-d61282085080" />

We can see the TTFT is typically larger than steady-state processing latency, and that the processing latency doesn't appear to significantly increase over time.

## Network Latency Breakdown

Using `torch.profile`, we can extract the per-operator times of the RNNT Emformer forward pass (This is provided as a separate notebook, as enabling profiling, especially with shape capture, affects latency, but we can still get an idea of operator dominance)

-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ----------------------------------------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls                                    Input Shapes  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ----------------------------------------------  
                 aten::linear         0.38%     480.413us        21.85%      27.611ms       1.381ms            20              [[5, 1, 512], [2048, 512], [2048]]  
                  aten::addmm        21.04%      26.585ms        21.27%      26.872ms       1.344ms            20         [[2048], [5, 512], [512, 2048], [], []]  
                 aten::linear         0.28%     348.930us        20.73%      26.193ms       6.548ms             4        [[10, 1, 1, 1024], [4097, 1024], [4097]]  
                  aten::addmm        20.19%      25.508ms        20.35%      25.711ms       6.428ms             4      [[4097], [10, 1024], [1024, 4097], [], []]  
                 aten::linear         0.40%     507.451us        16.34%      20.644ms       1.032ms            20              [[5, 1, 2048], [512, 2048], [512]]  
                  aten::addmm        15.03%      18.995ms        15.50%      19.584ms     979.202us            20         [[512], [5, 2048], [2048, 512], [], []]  
                 aten::linear         0.62%     785.439us        10.72%      13.552ms     338.795us            40                [[5, 1, 512], [512, 512], [512]]  
                 aten::linear         0.14%     179.531us         9.86%      12.455ms     622.732us            20              [[5, 1, 512], [1024, 512], [1024]]  
                  aten::addmm         9.21%      11.644ms         9.61%      12.139ms     303.468us            40           [[512], [5, 512], [512, 512], [], []]  
                  aten::addmm         9.42%      11.901ms         9.57%      12.092ms     604.607us            20         [[1024], [5, 512], [512, 1024], [], []]  
                    aten::cat         2.99%       3.775ms         3.65%       4.610ms      25.056us           184                                        [[], []]  
                   aten::gelu         2.13%       2.688ms         2.13%       2.688ms     134.396us            20                              [[5, 1, 2048], []]  
                     aten::to         0.22%     283.860us         1.56%       1.975ms       8.815us           224                            [[], [], [], [], []]  
             aten::layer_norm         0.25%     321.178us         1.45%       1.838ms      30.630us            60         [[5, 1, 512], [], [512], [512], [], []]  
               aten::_to_copy         0.62%     779.968us         1.34%       1.691ms       7.548us           224                    [[], [], [], [], [], [], []]  
      aten::native_layer_norm         0.82%       1.036ms         1.20%       1.517ms      25.277us            60             [[5, 1, 512], [], [512], [512], []]  
                    aten::add         0.35%     447.407us         0.89%       1.131ms       9.422us           120                                    [[], [], []]  
                  aten::slice         0.49%     620.802us         0.74%     938.436us       3.878us           242                   [[5, 1, 512], [], [], [], []]  
                    aten::bmm         0.69%     872.206us         0.70%     878.542us      43.927us            20                       [[8, 5, 64], [8, 64, 35]]  
            aten::masked_fill         0.22%     273.476us         0.64%     810.685us      40.534us            20                    [[8, 5, 35], [1, 5, 35], []]  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ----------------------------------------------  

We can focus on the CPU total column, where it's clear that most of inference time is spent in the `linear` and `addmm` layers - both matmul-based operators. The data is also split by different input shapes. We can clearly see the 20 Emformer block layers in the # of Calls counts.






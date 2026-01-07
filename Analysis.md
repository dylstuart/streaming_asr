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

-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ----------------------------------------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls                                    Input Shapes   Total FLOPs   
                 aten::linear         0.57%     734.831us        17.49%      22.617ms       1.131ms            20              [[5, 1, 2048], [512, 2048], [512]]            --  
                 aten::linear         0.49%     635.094us        16.78%      21.693ms       1.085ms            20              [[5, 1, 512], [2048, 512], [2048]]            --  
                 aten::linear         0.18%     234.473us        16.55%      21.397ms       5.349ms             4        [[10, 1, 1, 1024], [4097, 1024], [4097]]            --  
                  aten::addmm        16.19%      20.939ms        16.50%      21.334ms       1.067ms            20         [[512], [5, 2048], [2048, 512], [], []]  209715200.000  
                  aten::addmm        16.13%      20.855ms        16.27%      21.044ms       5.261ms             4      [[4097], [10, 1024], [1024, 4097], [], []]  335626240.000  
                  aten::addmm        15.64%      20.229ms        15.97%      20.652ms       1.033ms            20         [[2048], [5, 512], [512, 2048], [], []]  209715200.000  
                 aten::linear         0.93%       1.207ms        11.76%      15.211ms     380.278us            40                [[5, 1, 512], [512, 512], [512]]            --  
                 aten::linear         0.46%     600.811us        10.40%      13.452ms     672.591us            20              [[5, 1, 512], [1024, 512], [1024]]            --  
                  aten::addmm         9.32%      12.050ms         9.61%      12.425ms     621.256us            20         [[1024], [5, 512], [512, 1024], [], []]  104857600.000  
                  aten::addmm         8.82%      11.404ms         9.30%      12.025ms     300.631us            40           [[512], [5, 512], [512, 512], [], []]  104857600.000  
                    aten::cat         3.18%       4.111ms         4.22%       5.462ms      29.684us           184                                        [[], []]            --  
                   aten::gelu         2.83%       3.656ms         2.83%       3.656ms     182.819us            20                              [[5, 1, 2048], []]            --  
                     aten::to         0.41%     533.672us         2.69%       3.485ms      15.557us           224                            [[], [], [], [], []]            --  
               aten::_to_copy         1.19%       1.540ms         2.28%       2.951ms      13.175us           224                    [[], [], [], [], [], [], []]            --  
             aten::layer_norm         0.29%     375.633us         1.92%       2.487ms      41.445us            60         [[5, 1, 512], [], [512], [512], [], []]            --  
                    aten::add         0.64%     828.128us         1.86%       2.404ms      20.031us           120                                    [[], [], []]       120.000  
      aten::native_layer_norm         1.19%       1.538ms         1.63%       2.111ms      35.185us            60             [[5, 1, 512], [], [512], [512], []]            --  
                      aten::t         1.07%       1.378ms         1.29%       1.672ms      41.805us            40                                    [[512, 512]]            --  
                   aten::topk         1.09%       1.415ms         1.09%       1.415ms     353.843us             4                       [[40960], [], [], [], []]            --  
                  aten::slice         0.85%       1.094ms         1.07%       1.387ms       5.732us           242                   [[5, 1, 512], [], [], [], []]            --  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ----------------------------------------------  ------------  

We can focus on the CPU total column, where it's clear that most of inference time is spent in the `linear` and `addmm` layers - both matmul-based operators. The data is also split by different input shapes. We can clearly see the 20 Emformer block layers in the # of Calls counts.

The Total FLOPs column also correlates with the CPU total time for `addmm` layers, indicating that these matmul layers may be compute bound (`torch.profile` only reported flops for `addmm` layers, but we can calculate the `linear` layer flops equivalently ourselves, or assume the flops for a `linear` layer is equivalent to the flops for an `addmm` of the same shape)








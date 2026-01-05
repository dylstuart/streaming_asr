# Latency Analysis

## CDF of chunk latencies by pipeline component
<img width="789" height="590" alt="image" src="https://github.com/user-attachments/assets/36b1a089-f570-44b4-a6fc-b3205eb14b0b" />

The chunk processing latency is dominated by the Emformer forward pass, so latency optimizations should be focused there.

## Time To First Token
The time to first token (TTFT) gives a measure of responsiveness of the application. In a streaming scenario, it will represent how soon after the request for output the user will have to wait to begin receiving output.

## Real Time Factor
Real time factor (RTF) is a ratio of the processing time for a given sample of input to the actual duration of the sample. An RTF > 1 means that it takes longer to process an input than the duration of the input, meaning the output will lag further and further behind the real-time input (unless audio frames are "dropped" to allow catch-up), leading to eventual failure of the application from either a practical perspecitve (user experience is too poor to be useful) or a system perspective (input buffers run out of memory). An RTF <= 1 is therefore required for continuous streaming ASR without loss of quality.

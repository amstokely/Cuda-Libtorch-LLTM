# Cuda Libtorch LLTM
Implementation of a custom torch CUDA kernel based off of this example (https://pytorch.org/tutorials/advanced/cpp_extension.html).
The only significant difference is I use raw pointers in the CUDA kernels vs. Tensor accessors. With this modification, the forward pass kernel is roughly 30X faster than the PyTorch version (both using CUDA). Interestingly, the custom backwards pass kernel is only 5% faster. All benchmarks were run on an Nvidia RTX5000 gpu.

## Installation
As long as CUDA and cuDNN are discoverable, and you have torch installed, you should be able to install via 
```
python setup.py
```

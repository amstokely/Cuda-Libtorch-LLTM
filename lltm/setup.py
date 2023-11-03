from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm',
      ext_modules=[cpp_extension.CUDAExtension('lltm_cuda',
                                               ['lltm/lltm_cuda.cpp', 'lltm/lltm_kernel.cu'],
                                              extra_compile_args=['-O3'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      )


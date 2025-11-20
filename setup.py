# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


supported_archs = ["80"]
cc_flag = []
for arch in supported_archs:
    cc_flag.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    
# setup(
#     name='vecadd_extension',
#     ext_modules=[
#         CUDAExtension(
#             'vecadd_extension', # 模块名称
#             [
#                 'vecadd_kernel.cu',  # CUDA Kernel 文件
#                 'binding.cc', # C++ 绑定文件
#             ],
#             extra_compile_args={'nvcc': ['-O3'] + cc_flag} # 优化级别
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )

setup(
    name='sparse_kernel_extension',
    ext_modules=[
        CUDAExtension(
            'sparse_kernel_extension',
            sources=['get_table_kernel.cu'],
            extra_compile_args={'nvcc': ['-O3'] + cc_flag}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
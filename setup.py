import os
import sys

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

use_openmp = sys.platform != 'darwin'
extra_args = ['-std=c++17', '-march=native', '-O3', '-fopenmp']
extra_link_args = []

if use_openmp:
    extra_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
if sys.platform == 'darwin':
    extra_args += ['-mmacosx-version-min=10.9', '-stdlib=libc++']
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9'

ext_modules = [
    Pybind11Extension(
        "FalconnPP",
        ["python/python_wrapper.cpp", "src/main.cpp", "src/InputParser.cpp",
         "src/FalconnPP.cpp", "src/Utilities.cpp",
         "src/fht.c", "src/fast_copy.c", "src/BF.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
            extra_compile_args=extra_args,
            extra_link_args=extra_link_args,
        include_dirs=['src', '/usr/include/eigen3','/usr/local/include/eigen3', '/usr/include/pybind11','/usr/local/include/pybind11', 'libs']
    ),
]


setup(
    name='FalconnPP',
    version='0.0.1',
    author='Ninh Pham',
    author_email='ninh.pham@auckland.ac.nz',
    url='https://github.com/NinhPham/FalconnLSF',
    description=
    'LSF-based similarity search',
    license='MIT',
    keywords=
    'nearest neighbor search, similarity, locality-sensitive filtering, locality-sensitive hashing, inner product',
    # include_package_data=True,
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)

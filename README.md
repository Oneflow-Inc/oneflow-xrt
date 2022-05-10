## OneFlow-XRT

OneFlow-XRT is an OneFlow extension that provides an easy to use, flexible and unified way to integrate third-party computing engines in OneFlow.

OneFlow-XRT has support multiple third-party computing engines, such as XLA. Different engines support different backend hardware. For the same computing graph, XRT allows multiple computing engines to be used in combination to obtain better acceleration effects.

| engine       | device         | inference | training                    |
| ------------ | -------------- | --------- | --------------------------- |
| XRT-XLA      | X86 CPU + CUDA | &#10004;  | &#10004;                    |
| XRT-TensorRT | CUDA           | &#10004;  | &#10004; only no weights op |
| XRT-OpenVINO | X86 CPU        | &#10004;  | &#10004; only no weights op |
| XRT-TVM      | -              | -         | -                           |



## Installation

### pip

To install OneFlow-XRT via pip, use the following command:

```shell
# TODO
# pip3 install oneflow-xrt
```

## Building From Source

#### Prerequisites

- install cmake
- install oneflow
- install CUDA if building with TensorRT or oneflow is CUDA support

### building

Create an build directory and inside it, then run the following command:

```shell
export ONEFLOW_XRT_ROOT=/home/oneflow-xrt
cmake -S ${ONEFLOW_XRT_ROOT} && make -j10
```

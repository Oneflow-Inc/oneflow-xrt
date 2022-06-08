## OneFlow-XRT

OneFlow-XRT is an OneFlow extension that provides an easy to use, flexible and unified way to integrate third-party computing engines in OneFlow.

OneFlow-XRT has support multiple third-party computing engines, such as XLA and TensorRT. Different engines support different backend hardware. For the same computing graph, XRT allows multiple computing engines to be used in combination to obtain better acceleration effects.

| engine       | device         | inference | training                    |
| ------------ | -------------- | --------- | --------------------------- |
| XRT-XLA      | X86 CPU + CUDA | &#10004;  | &#10004;                    |
| XRT-TensorRT | CUDA           | &#10004;  | &#10004; only no weights op |
| XRT-OpenVINO | X86 CPU        | &#10004;  | &#10004; only no weights op |
| XRT-TVM      | -              | -         | -                           |


## Architecture

<img width="1046" alt="截屏2022-05-17 下午3 54 32" src="https://user-images.githubusercontent.com/13991173/168759503-a3ebda4b-af4c-4415-883a-8eb1c7814359.png">

## Installation

### pip

To install OneFlow-XRT via pip, use the following command:

```shell
# TODO
# pip3 install oneflow_xrt

# run the following commands according to your needs
# pip3 install oneflow_xrt_xla
# pip3 install oneflow_xrt_tensorrt
# pip3 install oneflow_xrt_openvino
```

### Building From Source

#### Prerequisites

- install cmake
- install oneflow
- install CUDA if oneflow supports CUDA device or building TensorRT
- install bazel if building XLA
- download and unzip TensorRT if building TensorRT
- download and unzip OpenVINO runtime if building OpenVINO

#### Get the OneFlow-XRT Source

```shell
git clone https://github.com/Oneflow-Inc/oneflow-xrt
```

#### building

Inside OneFlow-XRT source directory, then run the following command to install `oneflow_xrt`:

```shell
python3 setup.py install
```

The following components are optional, run the command to install it according to your needs,

- `oneflow_xrt_xla`

```shell
BUILD_XLA=ON python3 setup.py install
```

- `oneflow_xrt_tensorrt`

```shell
BUILD_TENSORRT=ON TENSORRT_ROOT=/home/TensorRT-8.4.0.6 python3 setup.py install
```

- `oneflow_xrt_openvino`

```shell
BUILD_OPENVINO=ON OPENVINO_ROOT=/home/intel/openvino_2022.1.0.643/runtime python3 setup.py install
```

## Run A Toy Program

```python
# python3

>>> import oneflow as flow
>>> import oneflow_xrt as ofrt
>>> m = flow.nn.Linear(3, 4).to("cuda")
>>> m = ofrt.XRTModule(m, engine=["tensorrt"])
>>> x = flow.randn(4, 3, device="cuda")
>>> y = m(x)
>>> print(y)
tensor([[ 0.2404,  0.7121,  0.4473,  0.4782],
        [-0.8697,  1.5353,  0.2829,  0.4772],
        [-0.3865, -1.2719,  1.0911,  0.1179],
        [ 0.3779,  0.7363,  0.5319,  0.3167]], device='cuda:0', dtype=oneflow.float32)
```



## Documentation

- [OneFlow XRT overall architecture and introduction](https://github.com/Oneflow-Inc/oneflow-xrt/wiki/OneFlow-XRT整体架构及简介)
- [Summary of operators in OneFlow XRT](https://github.com/Oneflow-Inc/oneflow-xrt/wiki/Summary-of-operators-in-OneFlow-XRT)
- [How to use low precision in OneFlow XRT](https://github.com/Oneflow-Inc/oneflow-xrt/wiki/OneFlow-XRT如何使用低精度计算)
- [How to extend custom engine components](https://github.com/Oneflow-Inc/oneflow-xrt/wiki/如何在XRT框架下添加自定义的后端引擎)


## Roadmap

TODO

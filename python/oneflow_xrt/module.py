"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Dict, Optional, Union, List, Callable
import numpy as np
import oneflow as flow
import oneflow_xrt as ofrt
from .import_engine import try_import_engine


class XRTModule(flow.nn.Module):
    """Compile and accelerate OneFlow model training or inference using XRT.

    Args:
        - module:
            Initial oneflow module (nn.Module) or graph (nn.Graph).
        - engine:
            The desired engine used to accelerate the module. Can be a str or a list of str.
        - use_fp16:
            If use fp16 precision. Default: False
        - use_int8:
            If use int8 precision. Default: False
        - int8_calibration:
            The directory of TensorRT style int8 calibration table. Default: None
        - max_batch_size:
            The maximum batch size for training or inference. Default: 1
        - max_workspace_size:
            The maximum available workspace for XRT. Default: None
        - strict_types:
            It does not guarantee to use low precision if just set use_int8 or use_fp16, but you can set strict_types to enforce the engine to use low precision. Default: False
        - force_precision_constraints:
            In order to reduce the computation precision loss, some ops specify a precision constraint, but this constraint is not mandatory, and the engine may still choose an appropriate precision based on it's tuning result.
            This option will make the constraint to be mandatory. Default: True
        - force_compile:
            Force compile on every execution without using the cached results. Default: False
        - cluster_minimum_nodes:
            The minimum subgraph size.
            XRT ensure that the size of the clustered subgraph will not be less than it. Default: 1
        - cluster_maximum_nodes:
            The maximum subgraph size.
            XRT ensure that the size of the clustered subgraph will not be larger than it. Usually used in debugging. Default: None
        - cluster_ignore_pipeline:
            XRT will not strictly take execution dependencies into consideration when cluster subgraph. Default: True
        - cluster_max_iteration:
            The maximum iteration when cluster subgraph. Default: 20
        - unsafe_transfer_unused_params_to_host:
            Transfer unused parameter from device to host to save device memory. Default: False
        - dump_subgraph_dir:
            The subgraph clustered will be dumped in this directory. Default: None
        - verbose:
            If output some details. Default: False

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow_xrt as ofrt

        >>> m = flow.nn.Linear(3, 4).to("cuda")
        >>> m = ofrt.XRTModule(m, engine=["tensorrt"])
        >>> x = flow.randn(4, 3, device="cuda")
        >>> y = m(x)

    """

    def __init__(
        self,
        module,
        engine,
        use_fp16=False,
        use_int8=False,
        int8_calibration=None,
        max_batch_size=1,
        max_workspace_size=None,
        strict_types=False,
        force_precision_constraints=True,
        force_compile=False,
        cluster_minimum_nodes=1,
        cluster_maximum_nodes=None,
        cluster_ignore_pipeline=True,
        cluster_max_iteration=20,
        unsafe_transfer_unused_params_to_host=False,
        dump_subgraph_dir=None,
        verbose=False,
    ):
        super().__init__()
        assert not isinstance(module, XRTModule)

        if isinstance(module, flow.nn.Module):
            self.module = self.make_naive_graph(module)
        else:
            assert isinstance(
                module, flow.nn.Graph
            ), "the module should be flow.nn.Module or flow.nn.Graph"
            self.module = module
        self.is_compiled = False
        self.engine = self.make_engine(engine)
        self.clustering_options = self.make_clustering_options(
            cluster_minimum_nodes,
            cluster_maximum_nodes,
            cluster_ignore_pipeline,
            cluster_max_iteration,
            dump_subgraph_dir,
        )
        self.execution_options = self.make_execution_options(
            use_fp16,
            use_int8,
            int8_calibration,
            max_batch_size,
            max_workspace_size,
            strict_types,
            force_precision_constraints,
            force_compile,
        )
        self.unsafe_transfer_unused_params_to_host = unsafe_transfer_unused_params_to_host
        self.verbose = verbose

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            try:
                return self.module.__getattr__(name)
            except:
                raise AttributeError(
                    "'{}' object has no attribute '{}'".format(
                        type(self).__name__, name
                    )
                )

    def make_naive_graph(self, module):
        """Transform a normal nn.Module to nn.Graph
        """

        class InternalNaiveGraph(flow.nn.Graph):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def build(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        return InternalNaiveGraph(module)

    def make_engine(self, engine):
        supported_engine = ["XLA", "TENSORRT", "OPENVINO"]
        if not isinstance(engine, (list, tuple)):
            engine = [
                engine,
            ]
        result = []
        for item in engine:
            assert isinstance(item, str), "engine should be str or a list of str"
            item_up = item.upper()
            assert item_up in supported_engine, (
                "engine %s is not supported, and these engines are %s supported"
                % (item, ", ".join(supported_engine))
            )
            result.append(item_up)
            try_import_engine(item_up)
        return result

    def make_clustering_options(
        self,
        minimum_nodes=1,
        maximum_nodes=None,
        ignore_pipeline=True,
        max_iteration=20,
        dump_subgraph_dir=None,
    ):
        options = ofrt.ClusteringOptions()
        options.minimum_nodes = minimum_nodes
        if maximum_nodes is not None:
            options.maximum_nodes = maximum_nodes
        options.ignore_pipeline = ignore_pipeline
        options.max_iteration = max_iteration
        if dump_subgraph_dir is not None:
            options.dump_subgraph_dir = dump_subgraph_dir
        return options

    def make_execution_options(
        self,
        use_fp16=False,
        use_int8=False,
        int8_calibration=None,
        max_batch_size=1,
        max_workspace_size=None,
        strict_types=False,
        force_precision_constraints=True,
        force_compile=False,
    ):
        options = ofrt.ReBuildJobOptions()
        options.use_fp16 = use_fp16
        options.use_int8 = use_int8
        if int8_calibration is not None:
            options.int8_calibration = int8_calibration
        options.max_batch_size = max_batch_size
        if max_workspace_size is not None:
            options.max_workspace_size = max_workspace_size
        options.strict_types = strict_types
        options.force_precision_constraints = force_precision_constraints
        options.force_compile = force_compile
        return options

    def register_subgraph_params(self, job):
        self.registered_params = []
        alive_ops = set()
        for op in job.net.op:
            alive_ops.add(op.name)

        for state_block in self.module._state():
            state_tensor = state_block.origin
            op_name = state_block.name_prefix + state_block.name
            if op_name not in alive_ops and isinstance(state_tensor, flow.nn.Parameter) and state_tensor.is_local:
                param = np.copy(state_block.origin.detach().cpu().numpy())
                self.registered_params.append(param)
                if self.unsafe_transfer_unused_params_to_host:
                    with flow.no_grad():
                        state_block.origin.data = state_block.origin.to("cpu")
                ofrt.register_buffer(op_name, param)

    def forward(self, *args, **kwargs):
        if self.is_compiled:
            return self.module(*args, **kwargs)

        origin_job, _ = self.module.build_graph(*args, **kwargs)
        graph = ofrt.Graph(origin_job)

        for engine in self.engine:
            self.clustering_options.engine = engine
            graph = ofrt.cluster_subgraph(graph, self.clustering_options)

        job = ofrt.rebuild_job(graph, origin_job, self.execution_options)

        if self.verbose:
            print("job after XRT compilation: ", job)

        if not self.module.training:
            self.register_subgraph_params(job)

        self.module._full_graph_proto = job
        self.module.finish_complie_and_init_runtime()
        self.is_compiled = True
        return self.module(*args, **kwargs)

import oneflow_xrt._oneflow_xrt_internal

class Graph(oneflow_xrt._oneflow_xrt_internal.Graph):
    def __init__(self, job):
        serialized_job = job.SerializeToString()
        super().__init__(serialized_job)

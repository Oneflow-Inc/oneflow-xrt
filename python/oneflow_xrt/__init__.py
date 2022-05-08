from oneflow_xrt._oneflow_xrt_internal import cluster_subgraph
from oneflow_xrt._oneflow_xrt_internal import ClusteringOptions, ReBuildJobOptions
from .graph import *

import oneflow.core.job.job_pb2 as job_pb

def rebuild_job(graph, origin_job, options):
    serialized_origin_job = origin_job.SerializeToString()
    serialized_job = oneflow_xrt._oneflow_xrt_internal.rebuild_job(graph, serialized_origin_job, options)
    new_job = job_pb.Job()
    new_job.ParseFromString(serialized_job)
    return new_job


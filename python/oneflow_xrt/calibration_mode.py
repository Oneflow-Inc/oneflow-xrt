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

from oneflow_xrt._oneflow_xrt_internal import PTQCalibrationMode


class ptq_calibration_mode:
    def __init__(self, cache_path=None):
        self.cache_path = "" if cache_path is None else cache_path

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with PTQCalibrationMode(self.cache_path):
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        self.calibration_mode = PTQCalibrationMode(self.cache_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

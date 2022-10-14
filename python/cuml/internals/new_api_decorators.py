#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import rmm
from cupy.cuda import using_allocator as cupy_using_allocator
from functool import wraps

from cuml.internals.global_settings import global_settings


def cuml_api_decorator(func):
    """Simply marks a function as already having been wrapped by a CUML API
    decorator"""
    func.__cuml_is_wrapped = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        with cupy_using_allocator(rmm.rmm_cupy_allocator):
            return func(*args, **kwargs)

    return wrapper

def api_return_array(func):
    func = cuml_api_decorator(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        output_type = global_settings.pop_output_stack()
        # TODO(wphicks): Perform conversion
        return arr

# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================


function(add_module_gpu_default FILENAME)
    set (extra_args ${ARGN})
    list(LENGTH extra_args extra_count)
    if (${extra_count} GREATER 0 OR
        ${CUML_UNIVERSAL} OR
        ${CUML_GPU})
      list(APPEND cython_sources
           ${FILENAME})
      set (cython_sources ${cython_sources} PARENT_SCOPE)
    endif()
endfunction()

function(add_module_universal_default FILENAME)
    set (extra_args ${ARGN})
    list(LENGTH extra_args extra_count)
    if (${extra_count} GREATER 0 OR
        ${CUML_UNIVERSAL})
      list(APPEND cython_sources
           ${FILENAME})
      set (cython_sources ${cython_sources} PARENT_SCOPE)
    endif()
endfunction()

set(cython_sources
    fil.pyx
)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_sg_libraries}"
  MODULE_PREFIX experimental_fil_
)

foreach(cython_module IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../library")
endforeach()

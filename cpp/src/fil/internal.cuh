/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file internal.h cuML-internal interface to Forest Inference Library. */

#pragma once

#include <cuml/cuml.hpp>
#include <cuml/fil/fil.h>

namespace ML {
namespace fil {

/** val_t is the payload within a FIL leaf */
union val_t {
  /** threshold value for branch node or output value (e.g. class
      probability or regression summand) for leaf node */
  float f;
  /** class label */
  int idx;
};

/** base_node contains common implementation details for dense and sparse nodes */
struct base_node {
  /** val is either the threshold (for inner nodes, always float)
      or the tree prediction (for leaf nodes) */
  val_t val;
  /** bits encode various information about the node, with the exact nature of
      this information depending on the node type; it includes e.g. whether the
      node is a leaf or inner node, and for inner nodes, additional information,
      e.g. the default direction, feature id or child index */
  int bits;
  static const int FID_MASK = (1 << 30) - 1;
  static const int DEF_LEFT_MASK = 1 << 30;
  static const int IS_LEAF_MASK = 1 << 31;
  template <class o_t>
  __host__ __device__ o_t output() const {
    return val;
  }
  __host__ __device__ float thresh() const { return val.f; }
  __host__ __device__ int fid() const { return bits & FID_MASK; }
  __host__ __device__ bool def_left() const { return bits & DEF_LEFT_MASK; }
  __host__ __device__ bool is_leaf() const { return bits & IS_LEAF_MASK; }
  __host__ __device__ base_node() : val({.f = 0}), bits(0){};
  base_node(val_t output, float thresh, int fid, bool def_left, bool is_leaf) {
    bits = (fid & FID_MASK) | (def_left ? DEF_LEFT_MASK : 0) |
           (is_leaf ? IS_LEAF_MASK : 0);
    if (is_leaf)
      val = output;
    else
      val.f = thresh;
  }
};

/** dense_node is a single node of a dense forest */
struct alignas(8) dense_node : base_node {
  dense_node() = default;
  dense_node(val_t output, float thresh, int fid, bool def_left, bool is_leaf)
    : base_node(output, thresh, fid, def_left, is_leaf) {}
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return 2 * curr + 1; }
};

/** sparse_node16 is a 16-byte node in a sparse forest */
struct alignas(16) sparse_node16 : base_node {
  int left_idx;
  int dummy;  // make alignment explicit and reserve for future use
  __host__ __device__ sparse_node16() : left_idx(0), dummy(0) {}
  sparse_node16(val_t output, float thresh, int fid, bool def_left,
                bool is_leaf, int left_index)
    : base_node(output, thresh, fid, def_left, is_leaf),
      left_idx(left_index),
      dummy(0) {}
  __host__ __device__ int left_index() const { return left_idx; }
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return left_idx; }
};

/** sparse_node8 is a node of reduced size (8 bytes) in a sparse forest */
struct alignas(8) sparse_node8 : base_node {
  static const int FID_NUM_BITS = 14;
  static const int FID_MASK = (1 << FID_NUM_BITS) - 1;
  static const int LEFT_OFFSET = FID_NUM_BITS;
  static const int LEFT_NUM_BITS = 16;
  static const int LEFT_MASK = ((1 << LEFT_NUM_BITS) - 1) << LEFT_OFFSET;
  static const int DEF_LEFT_OFFSET = LEFT_OFFSET + LEFT_NUM_BITS;
  static const int DEF_LEFT_MASK = 1 << DEF_LEFT_OFFSET;
  static const int IS_LEAF_OFFSET = 31;
  static const int IS_LEAF_MASK = 1 << IS_LEAF_OFFSET;
  __host__ __device__ int fid() const { return bits & FID_MASK; }
  __host__ __device__ bool def_left() const { return bits & DEF_LEFT_MASK; }
  __host__ __device__ bool is_leaf() const { return bits & IS_LEAF_MASK; }
  __host__ __device__ int left_index() const {
    return (bits & LEFT_MASK) >> LEFT_OFFSET;
  }
  sparse_node8() = default;
  sparse_node8(val_t output, float thresh, int fid, bool def_left, bool is_leaf,
               int left_index) {
    if (is_leaf)
      val = output;
    else
      val.f = thresh;
    bits = fid | left_index << LEFT_OFFSET |
           (def_left ? 1 : 0) << DEF_LEFT_OFFSET |
           (is_leaf ? 1 : 0) << IS_LEAF_OFFSET;
  }
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return left_index(); }
};

template <leaf_algo_t leaf_algo>
struct leaf_output_t {};
template <>
struct leaf_output_t<leaf_algo_t::FLOAT_UNARY_BINARY> {
  typedef float T;
};
template <>
struct leaf_output_t<leaf_algo_t::CATEGORICAL_LEAF> {
  typedef int T;
};
template <>
struct leaf_output_t<leaf_algo_t::GROVE_PER_CLASS_FEW_CLASSES> {
  typedef float T;
};
template <>
struct leaf_output_t<leaf_algo_t::GROVE_PER_CLASS_MANY_CLASSES> {
  typedef float T;
};

/** init_dense uses params and nodes to initialize the dense forest stored in pf
 *  @param h cuML handle used by this function
 *  @param pf pointer to where to store the newly created forest
 *  @param nodes nodes for the forest, of length
      (2**(params->depth + 1) - 1) * params->ntrees
 *  @param params pointer to parameters used to initialize the forest
 */
void init_dense(const raft::handle_t& h, forest_t* pf, const dense_node* nodes,
                const forest_params_t* params);

/** init_sparse uses params, trees and nodes to initialize the sparse forest
 *  with sparse nodes stored in pf
 *  @tparam fil_node_t node type to use with the sparse forest;
 *    must be sparse_node16 or sparse_node8
 *  @param h cuML handle used by this function
 *  @param pf pointer to where to store the newly created forest
 *  @param trees indices of tree roots in the nodes arrray, of length params->ntrees
 *  @param nodes nodes for the forest, of length params->num_nodes
 *  @param params pointer to parameters used to initialize the forest
 */
template <typename fil_node_t>
void init_sparse(const raft::handle_t& h, forest_t* pf, const int* trees,
                 const fil_node_t* nodes, const forest_params_t* params);

}  // namespace fil
}  // namespace ML

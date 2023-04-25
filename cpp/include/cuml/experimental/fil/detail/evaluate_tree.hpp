/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#pragma once
#include <stdint.h>
#ifndef __CUDACC__
#include <math.h>
#endif
#include <cuml/experimental/fil/detail/bitset.hpp>
#include <cuml/experimental/fil/detail/subtree.hpp>
#include <cuml/experimental/fil/detail/raft_proto/gpu_support.hpp>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {

/*
 * Evaluate a single tree on a single row
 *
 * @tparam has_vector_leaves Whether or not this tree has vector leaves
 * @tparam has_categorical nodes Whether or not this tree has any nodes with
 * categorical splits
 * @tparam node_t The type of nodes in this tree
 * @tparam io_t The type used for input to and output from this tree (typically
 * either floats or doubles)
 * @param node Pointer to the root node of this tree
 * @param row Pointer to the input data for this row
 */
template<
  bool has_vector_leaves,
  bool has_categorical_nodes,
  typename node_t,
  typename io_t
>
HOST DEVICE auto evaluate_tree(
    node_t const* __restrict__ node,
    io_t const* __restrict__ row
) {
  using categorical_set_type = bitset<uint32_t, typename node_t::index_type const>;
  auto cur_node = *node;
  do {
    auto input_val = row[cur_node.feature_index()];
    auto condition = true;
    if constexpr (has_categorical_nodes) {
      if (cur_node.is_categorical()) {
        auto valid_categories = categorical_set_type{
          &cur_node.index(),
          uint32_t(sizeof(typename node_t::index_type) * 8)
        };
        condition = valid_categories.test(input_val);
      } else {
        condition = (input_val < cur_node.threshold());
      }
    } else {
      condition = (input_val < cur_node.threshold());
    }
    if (!condition && cur_node.default_distant()) {
      condition = isnan(input_val);
    }
    node += cur_node.child_offset(condition);
    cur_node = *node;
  } while (!cur_node.is_leaf());
  return cur_node.template output<has_vector_leaves>();
}

/*
 * Evaluate a single tree which requires external categorical storage on a
 * single node
 *
 * For non-categorical models and models with a relatively small number of
 * categories for any feature, all information necessary for model evaluation
 * can be stored on a single node. If the number of categories for any
 * feature exceeds the available space on a node, however, the
 * categorical split data must be stored external to the node. We pass a
 * pointer to this external data and reconstruct bitsets from it indicating
 * the positive and negative categories for each categorical node.
 *
 * @tparam has_vector_leaves Whether or not this tree has vector leaves
 * @tparam node_t The type of nodes in this tree
 * @tparam io_t The type used for input to and output from this tree (typically
 * either floats or doubles)
 * @tparam categorical_storage_t The underlying type used for storing
 * categorical data (typically char)
 * @param node Pointer to the root node of this tree
 * @param row Pointer to the input data for this row
 * @param categorical_storage Pointer to where categorical split data is
 * stored.
 */
template<
  bool has_vector_leaves,
  typename node_t,
  typename io_t,
  typename categorical_storage_t
>
HOST DEVICE auto evaluate_tree(
    node_t const* __restrict__ node,
    io_t const* __restrict__ row,
    categorical_storage_t const* __restrict__ categorical_storage
) {
  using categorical_set_type = bitset<uint32_t, categorical_storage_t const>;
  auto cur_node = *node;
  do {
    auto input_val = row[cur_node.feature_index()];
    auto condition = cur_node.default_distant();
    if (!isnan(input_val)) {
      if (cur_node.is_categorical()) {
        auto valid_categories = categorical_set_type{
          categorical_storage + cur_node.index() + 1,
          uint32_t(categorical_storage[cur_node.index()])
        };
        condition = valid_categories.test(input_val);
      } else {
        condition = (input_val < cur_node.threshold());
      }
    }
    node += cur_node.child_offset(condition);
    cur_node = *node;
  } while (!cur_node.is_leaf());
  return cur_node.template output<has_vector_leaves>();
}

/*
 * Evaluate a single tree on a single row
 *
 * @tparam has_vector_leaves Whether or not this tree has vector leaves
 * @tparam has_categorical nodes Whether or not this tree has any nodes with
 * categorical splits
 * @tparam node_t The type of nodes in this tree
 * @tparam io_t The type used for input to and output from this tree (typically
 * either floats or doubles)
 * @param node Pointer to the root node of this tree
 * @param row Pointer to the input data for this row
 */
template<
  bool has_vector_leaves,
  bool has_categorical_nodes,
  typename node_t,
  typename io_t
>
HOST DEVICE auto evaluate_tree(
    subtree<node_t> const* __restrict__ subtree,
    io_t const* __restrict__ row
) {
  using categorical_set_type = bitset<uint32_t, typename node_t::index_type const>;
  auto cur_subtree = *subtree;
  bool conditions[3] = {true, true, true};
  auto terminal_subtree = false;
  do {
    node_t const& nodes[3] = {
      cur_subtree.parent(),
      cur_subtree.near_child(),
      cur_subtree.distant_child(),
    };
    io_t input_vals[3] = {
      row[nodes[0].feature_index()],
      row[nodes[1].feature_index()],
      row[nodes[2].feature_index()],
    };
    bool is_leaf[3] = {
      nodes[0].is_leaf(),
      nodes[1].is_leaf(),
      nodes[2].is_leaf()
    };

    for (auto i = 0; i < 3; ++i) {
      if constexpr (has_categorical_nodes) {
        if (nodes[i].is_categorical()) {
          auto valid_categories = categorical_set_type{
            &(nodes[i].index()),
            uint32_t(sizeof(typename node_t::index_type) * 8)
          };
          conditions[i] = valid_categories.test(input_vals[i]);
        } else {
          conditions[i] = (input_vals[i] < nodes[i].threshold());
        }
      } else {
        conditions[i] = (input_vals[i] < nodes[i].threshold());
      }
      if (!conditions[i] && nodes[i].default_distant()) {
        conditions[i] = isnan(input_vals[i]);
      }
      terminal_subtree |= is_leaf[i];
    }

    auto subtree_child_index = (
      2 * int{conditions[0]} +
      int{!conditions[0]} * int{conditions[1]} +
      int{conditions[0]} * int{conditions[2]}
    );
    subtree += int{!terminal_subtree} * cur_subtree.child_offset(subtree_child_index);
    cur_subtree = *subtree;
  } while (!terminal_subtree);

  return cur_subtree.template output<has_vector_leaves>(conditions[0]);
}

}
}
}
}

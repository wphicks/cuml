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
#include <type_traits>
#include <cuml/experimental/fil/detail/raft_proto/gpu_support.hpp>

namespace ML {
namespace experimental {
namespace fil {

template <typename node_t>
struct subtree {
  using node_type = node_t;
  subtree(node_t parent, node_t child0=node_t{}, node_t child1=node_t{})
    : parent_{parent}, child0_{child0}, child1_{child1} { }
  auto const& parent() const {
    return parent_;
  }
  auto const& near_child() const {
    return child0_;
  }
  auto const& distant_child() const {
    return child1_;
  }

  auto child_offset(int subtree_child_index) {
    auto result = typename node_type::offset_type{};
    switch (subtree_child_index){
      case 0:
        result = child0_.child_offset(false);
        break;
      case 1:
        result = child0_.child_offset(true);
        break;
      case 2:
        result = child1_.child_offset(false);
        break;
      case 3:
        result = child1_.child_offset(true);
        break;
    }
    return result;
  }

  template <bool has_vector_leaves>
  auto constexpr output(bool parent_result) {
    using result_type = std::conditional_t<
      has_vector_leaves,
      typename node_type::index_type,
      typename node_type::value_type
    >;

    auto parent_is_leaf = parent_.is_leaf();
    return (
      result_type{parent_is_leaf} * parent_.template output<has_vector_leaves>() +
      result_type{!parent_result && !parent_is_leaf} * child0_.template output<has_vector_leaves>() +
      result_type{parent_result && !parent_is_leaf} * child1_.template output<has_vector_leaves>()
    );
  }
  
 private:
  node_type parent_;
  node_type child0_;
  node_type child1_;
};

}
}
}

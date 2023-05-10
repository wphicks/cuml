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
namespace ML {
namespace experimental {
namespace fil {
enum class tree_layout : unsigned char {
  depth_first           = 0,
  breadth_first         = 1,
  subtree_depth_first   = 2,
  subtree_breadth_first = 3
};

auto constexpr is_subtree_layout(tree_layout layout)
{
  return (layout == tree_layout::subtree_depth_first ||
          layout == tree_layout::subtree_breadth_first);
}
auto constexpr is_depth_first_layout(tree_layout layout)
{
  return (layout == tree_layout::depth_first || layout == tree_layout::subtree_depth_first);
}
auto constexpr is_breadth_first_layout(tree_layout layout)
{
  return (layout == tree_layout::breadth_first || layout == tree_layout::subtree_breadth_first);
}
auto constexpr subtree_size_for_layout(tree_layout layout)
{
  return int{!is_subtree_layout(layout)} + int{is_subtree_layout(layout)} * 3;
}

}  // namespace fil
}  // namespace experimental
}  // namespace ML

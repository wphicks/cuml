/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <decisiontree/quantile/quantile.h>
#include <raft/cudart_utils.h>
#include <common/iota.cuh>
#include <cuml/common/logger.hpp>
#include <iomanip>
#include <locale>
#include <queue>
#include <random>
#include <type_traits>
#include <treelite/tree.h>
#include "batched-levelalgo/builder.cuh"
#include "decisiontree_impl.h"
#include "levelalgo/levelfunc_classifier.cuh"
#include "levelalgo/levelfunc_regressor.cuh"
#include "levelalgo/metric.cuh"
#include "memory.cuh"
#include "quantile/quantile.cuh"
#include "treelite_util.h"

namespace ML {

bool is_dev_ptr(const void *p) {
  cudaPointerAttributes pointer_attr;
  cudaError_t err = cudaPointerGetAttributes(&pointer_attr, p);
  if (err == cudaSuccess) {
    return pointer_attr.devicePointer;
  } else {
    err = cudaGetLastError();
    return false;
  }
}

namespace DecisionTree {

template <class T, class L>
void print(const SparseTreeNode<T, L> &node, std::ostream &os) {
  if (node.colid == -1) {
    os << "(leaf, " << node.prediction << ", " << node.best_metric_val << ")";
  } else {
    os << "(" << node.colid << ", " << node.quesval << ", "
       << node.best_metric_val << ")";
  }
  return;
}

template <class T, class L>
std::string get_node_text(const std::string &prefix,
                          const std::vector<SparseTreeNode<T, L>> &sparsetree,
                          int idx, bool isLeft) {
  const SparseTreeNode<T, L> &node = sparsetree[idx];

  std::ostringstream oss;

  // print the value of the node
  std::stringstream ss;
  ss << prefix.c_str();
  ss << (isLeft ? "├" : "└");
  ss << node;

  oss << ss.str();

  if ((node.colid != -1)) {
    // enter the next tree level - left and right branch
    oss << "\n"
        << get_node_text(prefix + (isLeft ? "│   " : "    "), sparsetree,
                         node.left_child_id, true)
        << "\n"
        << get_node_text(prefix + (isLeft ? "│   " : "    "), sparsetree,
                         node.left_child_id + 1, false);
  }
  return oss.str();
}

template <typename T>
std::string to_string_high_precision(T x) {
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "T must be float, double, or integer");
  std::ostringstream oss;
  oss.imbue(std::locale::classic());  // use C locale
  if (std::is_floating_point<T>::value) {
    oss << std::setprecision(std::numeric_limits<T>::max_digits10) << x;
  } else {
    oss << x;
  }
  return oss.str();
}

template <class T, class L>
std::string get_node_json(const std::string &prefix,
                          const std::vector<SparseTreeNode<T, L>> &sparsetree,
                          int idx) {
  const SparseTreeNode<T, L> &node = sparsetree[idx];

  std::ostringstream oss;
  if ((node.colid != -1)) {
    oss << prefix << "{\"nodeid\": " << idx
        << ", \"split_feature\": " << node.colid
        << ", \"split_threshold\": " << to_string_high_precision(node.quesval)
        << ", \"gain\": " << to_string_high_precision(node.best_metric_val)
        << ", \"yes\": " << node.left_child_id
        << ", \"no\": " << (node.left_child_id + 1) << ", \"children\": [\n";
    // enter the next tree level - left and right branch
    oss << get_node_json(prefix + "  ", sparsetree, node.left_child_id) << ",\n"
        << get_node_json(prefix + "  ", sparsetree, node.left_child_id + 1)
        << "\n"
        << prefix << "]}";
  } else {
    oss << prefix << "{\"nodeid\": " << idx
        << ", \"leaf_value\": " << to_string_high_precision(node.prediction)
        << "}";
  }
  return oss.str();
}

template <typename T, typename L>
std::ostream &operator<<(std::ostream &os, const SparseTreeNode<T, L> &node) {
  DecisionTree::print(node, os);
  return os;
}

template <typename T, typename L>
struct Node_ID_info {
  const SparseTreeNode<T, L> &node;
  int unique_node_id;

  Node_ID_info(const SparseTreeNode<T, L> &cfg_node, int cfg_unique_node_id)
    : node(cfg_node), unique_node_id(cfg_unique_node_id) {}
};

template <class T, class L>
tl::Tree<T, T> build_treelite_tree(
    const DecisionTree::TreeMetaDataNode<T, L>& rf_tree, unsigned int num_class) {
  tl::Tree<T, T> tl_tree;
  tl_tree.Init();


  std::queue<Node_ID_info<T, L>> cur_level_queue;
  std::queue<Node_ID_info<T, L>> next_level_queue;

  cur_level_queue.push(Node_ID_info<T, L>(rf_tree.sparsetree[0], 0));

  while (!cur_level_queue.empty()) {
    int cur_level_size = cur_level_queue.size();

    for (int i = 0; i < cur_level_size; i++) {
      Node_ID_info<T, L> q_node = cur_level_queue.front();
      cur_level_queue.pop();

      bool is_leaf_node = q_node.node.colid == -1;
      int node_id = q_node.unique_node_id;

      if (!is_leaf_node) {
        tl_tree.AddChilds(node_id);

        // Push left child to next_level queue.
        next_level_queue.push(Node_ID_info<T, L>(
          rf_tree.sparsetree[q_node.node.left_child_id],
          tl_tree.LeftChild(node_id)
        ));

        // Push right child to next_level deque.
        next_level_queue.push(Node_ID_info<T, L>(
          rf_tree.sparsetree[q_node.node.left_child_id + 1],
          tl_tree.RightChild(node_id)
        ));

        // Set node from current level as numerical node. Children IDs known.
        tl_tree.SetNumericalSplit(node_id, q_node.node.colid,
                                  q_node.node.quesval, true,
                                  tl::Operator::kLE);

      } else {
        if (num_class == 1) {
          tl_tree.SetLeaf(node_id, static_cast<T>(q_node.node.prediction));
        } else {
          std::vector<T> leaf_vector(num_class, 0);
          leaf_vector[q_node.node.prediction] = 1;
          tl_tree.SetLeafVector(node_id, leaf_vector);
        }
      }
    }

    // The cur_level_queue is empty here, as all the elements are already poped out.
    cur_level_queue.swap(next_level_queue);
  }
  return tl_tree;
}

/**
 * @brief Print high-level tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 */
template <typename T, typename L>
void DecisionTreeBase<T, L>::print_tree_summary() const {
  PatternSetter _("%v");
  CUML_LOG_DEBUG(" Decision Tree depth --> %d and n_leaves --> %d",
                 depth_counter, leaf_counter);
  CUML_LOG_DEBUG(" Total temporary memory usage--> %lf MB",
                 ((double)total_temp_mem / (1024 * 1024)));
  CUML_LOG_DEBUG(" Shared memory used --> %d B", shmem_used);
  CUML_LOG_DEBUG(" Tree Fitting - Overall time --> %lf s",
                 prepare_time + train_time);
  CUML_LOG_DEBUG("   - preparing for fit time: %lf s", prepare_time);
  CUML_LOG_DEBUG("   - tree growing time: %lf s", train_time);
}

/**
 * @brief Print detailed tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] sparsetree: Sparse tree strcut
 */
template <typename T, typename L>
void DecisionTreeBase<T, L>::print(
  const std::vector<SparseTreeNode<T, L>> &sparsetree) const {
  DecisionTreeBase<T, L>::print_tree_summary();
  get_node_text<T, L>("", sparsetree, 0, false);
}

/**
 * @brief This function calls the relevant regression oir classification with input parameters.
 * @tparam T: datatype of input data (float ot double)
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[out] sparsetree: This will be the generated Decision Tree
 * @param[in] data: Input data
 * @param[in] ncols: Original number of columns in the dataset
 * @param[in] nrows: Original number of rows in dataset
 * @param[in] labels: Labels of input dataset
 * @param[in] rowids: List of selected rows for the tree building
 * @param[in] n_sampled_rows: Number of rows after subsampling
 * @param[in] unique_labels: Number of unique classes for calssification. Its set to 1 for regression
 * @param[in] treeid: Tree id in case of building multiple tree from RF.
 */
template <typename T, typename L>
void DecisionTreeBase<T, L>::plant(
  std::vector<SparseTreeNode<T, L>> &sparsetree, const T *data, const int ncols,
  const int nrows, const L *labels, unsigned int *rowids,
  const int n_sampled_rows, int unique_labels, const int treeid) {
  dinfo.NLocalrows = nrows;
  dinfo.NGlobalrows = nrows;
  dinfo.Ncols = ncols;
  n_unique_labels = unique_labels;

  if (tree_params.split_algo == SPLIT_ALGO::GLOBAL_QUANTILE &&
      tree_params.quantile_per_tree) {
    preprocess_quantile(data, rowids, n_sampled_rows, ncols, dinfo.NLocalrows,
                        tree_params.n_bins, tempmem);
  }
  CUDA_CHECK(cudaStreamSynchronize(
    tempmem->stream));  // added to ensure accurate measurement

  //Bootstrap features
  unsigned int *h_colids = tempmem->h_colids->data();
  if (tree_params.bootstrap_features) {
    srand(treeid * 1000);
    for (int i = 0; i < dinfo.Ncols; i++) {
      h_colids[i] = rand() % dinfo.Ncols;
    }
  } else {
    std::iota(h_colids, h_colids + dinfo.Ncols, 0);
  }
  prepare_time = prepare_fit_timer.getElapsedSeconds();

  total_temp_mem = tempmem->totalmem;
  MLCommon::TimerCPU timer;
  if (tree_params.use_experimental_backend) {
    if (treeid == 0) {
      CUML_LOG_WARN("Using experimental backend for growing trees\n");
    }
    T *quantiles = tempmem->d_quantile->data();
    int *colids = (int *)tempmem->device_allocator->allocate(
      sizeof(int) * ncols, tempmem->stream);
    MLCommon::iota(colids, 0, 1, ncols, tempmem->stream);
    grow_tree(tempmem->device_allocator, tempmem->host_allocator, data, ncols,
              nrows, labels, quantiles, (int *)rowids, (int *)colids,
              n_sampled_rows, unique_labels, tree_params, tempmem->stream,
              sparsetree, this->leaf_counter, this->depth_counter);
  } else {
    grow_deep_tree(data, labels, rowids, n_sampled_rows, ncols,
                   tree_params.max_features, dinfo.NLocalrows, sparsetree,
                   treeid, tempmem);
  }
  train_time = timer.getElapsedSeconds();
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::predict(const raft::handle_t &handle,
                                     const TreeMetaDataNode<T, L> *tree,
                                     const T *rows, const int n_rows,
                                     const int n_cols, L *predictions,
                                     int verbosity) const {
  if (verbosity >= 0) {
    ML::Logger::get().setLevel(verbosity);
  }
  ASSERT(!is_dev_ptr(rows) && !is_dev_ptr(predictions),
         "DT Error: Current impl. expects both input and predictions to be CPU "
         "pointers.\n");

  ASSERT(tree && (tree->sparsetree.size() != 0),
         "Cannot predict w/ empty tree, tree size %zu",
         tree->sparsetree.size());
  ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
  ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

  predict_all(tree, rows, n_rows, n_cols, predictions);
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::predict_all(const TreeMetaDataNode<T, L> *tree,
                                         const T *rows, const int n_rows,
                                         const int n_cols, L *preds) const {
  for (int row_id = 0; row_id < n_rows; row_id++) {
    preds[row_id] = predict_one(&rows[row_id * n_cols], tree->sparsetree, 0);
  }
}

template <typename T, typename L>
L DecisionTreeBase<T, L>::predict_one(
  const T *row, const std::vector<SparseTreeNode<T, L>> sparsetree,
  int idx) const {
  int colid = sparsetree[idx].colid;
  T quesval = sparsetree[idx].quesval;
  int leftchild = sparsetree[idx].left_child_id;
  if (colid == -1) {
    CUML_LOG_DEBUG("Leaf node. Predicting %f",
                   (float)sparsetree[idx].prediction);
    return sparsetree[idx].prediction;
  } else if (row[colid] <= quesval) {
    CUML_LOG_DEBUG("Classifying Left @ node w/ column %d and value %f", colid,
                   (float)quesval);
    return predict_one(row, sparsetree, leftchild);
  } else {
    CUML_LOG_DEBUG("Classifying Right @ node w/ column %d and value %f", colid,
                   (float)quesval);
    return predict_one(row, sparsetree, leftchild + 1);
  }
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::set_metadata(TreeMetaDataNode<T, L> *&tree) {
  tree->depth_counter = depth_counter;
  tree->leaf_counter = leaf_counter;
  tree->train_time = train_time;
  tree->prepare_time = prepare_time;
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::base_fit(
  const std::shared_ptr<MLCommon::deviceAllocator> device_allocator_in,
  const std::shared_ptr<MLCommon::hostAllocator> host_allocator_in,
  const cudaStream_t stream_in, const T *data, const int ncols, const int nrows,
  const L *labels, unsigned int *rowids, const int n_sampled_rows,
  int unique_labels, std::vector<SparseTreeNode<T, L>> &sparsetree,
  const int treeid, bool is_classifier,
  std::shared_ptr<TemporaryMemory<T, L>> in_tempmem) {
  prepare_fit_timer.reset();
  const char *CRITERION_NAME[] = {"GINI", "ENTROPY", "MSE", "MAE", "END"};
  CRITERION default_criterion =
    (is_classifier) ? CRITERION::GINI : CRITERION::MSE;
  CRITERION last_criterion =
    (is_classifier) ? CRITERION::ENTROPY : CRITERION::MAE;

  validity_check(tree_params);
  if (tree_params.n_bins > n_sampled_rows) {
    CUML_LOG_WARN("Calling with number of bins > number of rows!");
    CUML_LOG_WARN("Resetting n_bins to %d.", n_sampled_rows);
    tree_params.n_bins = n_sampled_rows;
  }

  if (
    tree_params.split_criterion ==
    CRITERION::
      CRITERION_END) {  // Set default to GINI (classification) or MSE (regression)
    tree_params.split_criterion = default_criterion;
  }
  ASSERT((tree_params.split_criterion >= default_criterion) &&
           (tree_params.split_criterion <= last_criterion),
         "Unsupported criterion %s\n",
         CRITERION_NAME[tree_params.split_criterion]);

  if (in_tempmem != nullptr) {
    tempmem = in_tempmem;
  } else {
    tempmem = std::make_shared<TemporaryMemory<T, L>>(
      device_allocator_in, host_allocator_in, stream_in, nrows, ncols,
      unique_labels, tree_params);
    tree_params.quantile_per_tree = true;
  }

  plant(sparsetree, data, ncols, nrows, labels, rowids, n_sampled_rows,
        unique_labels, treeid);
  if (in_tempmem == nullptr) {
    tempmem.reset();
  }
}

template <typename T>
void DecisionTreeClassifier<T>::fit(
  const raft::handle_t &handle, const T *data, const int ncols, const int nrows,
  const int *labels, unsigned int *rowids, const int n_sampled_rows,
  const int unique_labels, TreeMetaDataNode<T, int> *&tree,
  DecisionTreeParams tree_parameters,
  std::shared_ptr<TemporaryMemory<T, int>> in_tempmem) {
  this->tree_params = tree_parameters;
  this->base_fit(handle.get_device_allocator(), handle.get_host_allocator(),
                 handle.get_stream(), data, ncols, nrows, labels, rowids,
                 n_sampled_rows, unique_labels, tree->sparsetree, tree->treeid,
                 true, in_tempmem);
  this->set_metadata(tree);
}

template <typename T>

void DecisionTreeClassifier<T>::fit(
  const std::shared_ptr<MLCommon::deviceAllocator> device_allocator_in,
  const std::shared_ptr<MLCommon::hostAllocator> host_allocator_in,
  const cudaStream_t stream_in, const T *data, const int ncols, const int nrows,
  const int *labels, unsigned int *rowids, const int n_sampled_rows,
  const int unique_labels, TreeMetaDataNode<T, int> *&tree,
  DecisionTreeParams tree_parameters,
  std::shared_ptr<TemporaryMemory<T, int>> in_tempmem) {
  this->tree_params = tree_parameters;
  this->base_fit(device_allocator_in, host_allocator_in, stream_in, data, ncols,
                 nrows, labels, rowids, n_sampled_rows, unique_labels,
                 tree->sparsetree, tree->treeid, true, in_tempmem);
  this->set_metadata(tree);
}

template <typename T>
void DecisionTreeRegressor<T>::fit(
  const raft::handle_t &handle, const T *data, const int ncols, const int nrows,
  const T *labels, unsigned int *rowids, const int n_sampled_rows,
  TreeMetaDataNode<T, T> *&tree, DecisionTreeParams tree_parameters,
  std::shared_ptr<TemporaryMemory<T, T>> in_tempmem) {
  this->tree_params = tree_parameters;
  this->base_fit(handle.get_device_allocator(), handle.get_host_allocator(),
                 handle.get_stream(), data, ncols, nrows, labels, rowids,
                 n_sampled_rows, 1, tree->sparsetree, tree->treeid, false,
                 in_tempmem);
  this->set_metadata(tree);
}

template <typename T>
void DecisionTreeRegressor<T>::fit(
  const std::shared_ptr<MLCommon::deviceAllocator> device_allocator_in,
  const std::shared_ptr<MLCommon::hostAllocator> host_allocator_in,
  const cudaStream_t stream_in, const T *data, const int ncols, const int nrows,
  const T *labels, unsigned int *rowids, const int n_sampled_rows,
  TreeMetaDataNode<T, T> *&tree, DecisionTreeParams tree_parameters,
  std::shared_ptr<TemporaryMemory<T, T>> in_tempmem) {
  this->tree_params = tree_parameters;
  this->base_fit(device_allocator_in, host_allocator_in, stream_in, data, ncols,
                 nrows, labels, rowids, n_sampled_rows, 1, tree->sparsetree,
                 tree->treeid, false, in_tempmem);
  this->set_metadata(tree);
}

template <typename T>
void DecisionTreeClassifier<T>::grow_deep_tree(
  const T *data, const int *labels, unsigned int *rowids,
  const int n_sampled_rows, const int ncols, const float colper,
  const int nrows, std::vector<SparseTreeNode<T, int>> &sparsetree,
  const int treeid, std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  int leaf_cnt = 0;
  int depth_cnt = 0;
  grow_deep_tree_classification(data, labels, rowids, ncols, colper,
                                n_sampled_rows, nrows, this->n_unique_labels,
                                this->tree_params, depth_cnt, leaf_cnt,
                                sparsetree, treeid, tempmem);
  this->depth_counter = depth_cnt;
  this->leaf_counter = leaf_cnt;
}

template <typename T>
void DecisionTreeRegressor<T>::grow_deep_tree(
  const T *data, const T *labels, unsigned int *rowids,
  const int n_sampled_rows, const int ncols, const float colper,
  const int nrows, std::vector<SparseTreeNode<T, T>> &sparsetree,
  const int treeid, std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  int leaf_cnt = 0;
  int depth_cnt = 0;
  grow_deep_tree_regression(data, labels, rowids, ncols, colper, n_sampled_rows,
                            nrows, this->tree_params, depth_cnt, leaf_cnt,
                            sparsetree, treeid, tempmem);
  this->depth_counter = depth_cnt;
  this->leaf_counter = leaf_cnt;
}

//Class specializations
template class DecisionTreeBase<float, int>;
template class DecisionTreeBase<float, float>;
template class DecisionTreeBase<double, int>;
template class DecisionTreeBase<double, double>;

template class DecisionTreeClassifier<float>;
template class DecisionTreeClassifier<double>;

template class DecisionTreeRegressor<float>;
template class DecisionTreeRegressor<double>;

template tl::Tree<float, float> build_treelite_tree<float, int>(
  const DecisionTree::TreeMetaDataNode<float, int>& rf_tree, unsigned int num_class);
template tl::Tree<double, double> build_treelite_tree<double, int>(
  const DecisionTree::TreeMetaDataNode<double, int>& rf_tree, unsigned int num_class);
template tl::Tree<float, float> build_treelite_tree<float, float>(
  const DecisionTree::TreeMetaDataNode<float, float>& rf_tree, unsigned int num_class);
template tl::Tree<double, double> build_treelite_tree<double, double>(
  const DecisionTree::TreeMetaDataNode<double, double>& rf_tree, unsigned int
  num_class);
}  //End namespace DecisionTree

}  //End namespace ML

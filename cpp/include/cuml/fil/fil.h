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

/** @file fil.h Interface to the forest inference library. */

#pragma once

#include <cuml/cuml.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <cuml/ensemble/treelite_defs.hpp>

namespace ML {
namespace fil {

/** @note FIL only supports inference with single precision.
 *  TODO(canonizer): parameterize the functions and structures by the data type
 *  and the threshold/weight type.
 */

/** Inference algorithm to use. */
enum algo_t {
  /** choose the algorithm automatically; currently chooses NAIVE for sparse forests 
      and BATCH_TREE_REORG for dense ones */
  ALGO_AUTO,
  /** naive algorithm: 1 thread block predicts 1 row; the row is cached in
      shared memory, and the trees are distributed cyclically between threads */
  NAIVE,
  /** tree reorg algorithm: same as naive, but the tree nodes are rearranged
      into a more coalescing-friendly layout: for every node position,
      nodes of all trees at that position are stored next to each other */
  TREE_REORG,
  /** batch tree reorg algorithm: same as tree reorg, but predictions multiple rows (up to 4)
      in a single thread block */
  BATCH_TREE_REORG
};

/** storage_type_t defines whether to import the forests as dense or sparse */
enum storage_type_t {
  /** decide automatically; currently always builds dense forests */
  AUTO,
  /** import the forest as dense */
  DENSE,
  /** import the forest as sparse (currently always with 16-byte nodes) */
  SPARSE,
  /** (experimental) import the forest as sparse with 8-byte nodes; can fail if
      8-byte nodes are not enough to store the forest, e.g. there are too many
      nodes in a tree or too many features; note that the number of bits used to
      store the child or feature index can change in the future; this can affect
      whether a particular forest can be imported as SPARSE8 */
  SPARSE8,
};

struct forest;

/** forest_t is the predictor handle */
typedef forest* forest_t;

/** treelite_params_t are parameters for importing treelite models */
struct treelite_params_t {
  // algo is the inference algorithm
  algo_t algo;
  // output_class indicates whether thresholding will be applied
  // to the model output
  bool output_class;
  // threshold may be used for thresholding if output_class == true,
  // and is ignored otherwise. threshold is ignored if leaves store
  // vectorized class labels. in that case, a class with most votes
  // is returned regardless of the absolute vote count
  float threshold;
  // storage_type indicates whether the forest should be imported as dense or sparse
  storage_type_t storage_type;
  // blocks_per_sm, if nonzero, works as a limit to improve cache hit rate for larger forests
  // suggested values (if nonzero) are from 2 to 7
  // if zero, launches ceildiv(num_rows, NITEMS) blocks
  int blocks_per_sm;
};

/** leaf_algo_t describes what the leaves in a FIL forest store (predict)
    and how FIL aggregates them into class margins/regression result/best class
**/
enum leaf_algo_t {
  /** storing a class probability or regression summand. We add all margins
      together and determine regression result or use threshold to determine
      one of the two classes. **/
  FLOAT_UNARY_BINARY = 0,
  /** storing a class label. Trees vote on the resulting class.
      Probabilities are just normalized votes. */
  CATEGORICAL_LEAF = 1,
  /** 1-vs-rest, or tree-per-class, where trees are assigned round-robin to
      consecutive categories and predict a floating-point margin. Used in
      Gradient Boosted Decision Trees. We sum margins for each group separately
      **/
  GROVE_PER_CLASS = 2,
  /** 1-vs-rest, or tree-per-class, where trees are assigned round-robin to
      consecutive categories and predict a floating-point margin. Used in
      Gradient Boosted Decision Trees. We sum margins for each group separately
      This is a more specific version of GROVE_PER_CLASS.
      _FEW_CLASSES means fewer (or as many) classes than threads. **/
  GROVE_PER_CLASS_FEW_CLASSES = 3,
  /** 1-vs-rest, or tree-per-class, where trees are assigned round-robin to
      consecutive categories and predict a floating-point margin. Used in
      Gradient Boosted Decision Trees. We sum margins for each group separately
      This is a more specific version of GROVE_PER_CLASS.
      _MANY_CLASSES means more classes than threads. **/
  GROVE_PER_CLASS_MANY_CLASSES = 4,
  // to be extended
};

/**
 * output_t are flags that define the output produced by the FIL predictor; a
 * valid output_t values consists of the following, combined using '|' (bitwise
 * or), which define stages, which operation in the next stage applied to the
 * output of the previous stage:
 * - one of RAW or AVG, indicating how to combine individual tree outputs into the forest output
 * - optional SIGMOID for applying the sigmoid transform
 * - optional CLASS, to output the class label
 */
enum output_t {
  /** raw output: the sum of the tree outputs; use for GBM models for
      regression, or for binary classification for the value before the
      transformation; note that this value is 0, and may be omitted
      when combined with other flags */
  RAW = 0x0,
  /** average output: divide the sum of the tree outputs by the number of trees
      before further transformations; use for random forests for regression
      and binary classification for the probability */
  AVG = 0x1,
  /** sigmoid transformation: apply 1/(1+exp(-x)) to the sum or average of tree
      outputs; use for GBM binary classification models for probability */
  SIGMOID = 0x10,
  /** output class label: either apply threshold to the output of the previous stage (for binary classification),
      or select the class with the most votes to get the class label (for multi-class classification).  */
  CLASS = 0x100,
  SIGMOID_CLASS = SIGMOID | CLASS,
  AVG_CLASS = AVG | CLASS,
  AVG_SIGMOID_CLASS = AVG | SIGMOID | CLASS,
};

/** forest_params_t are the trees to initialize the predictor */
struct forest_params_t {
  // total number of nodes; ignored for dense forests
  int num_nodes;
  // maximum depth
  int depth;
  // ntrees is the number of trees
  int num_trees;
  // num_cols is the number of columns in the data
  int num_cols;
  // leaf_algo determines what the leaves store (predict)
  leaf_algo_t leaf_algo;
  // algo is the inference algorithm;
  // sparse forests do not distinguish between NAIVE and TREE_REORG
  algo_t algo;
  // output is the desired output type
  output_t output;
  // threshold is used to for classification if leaf_algo == FLOAT_UNARY_BINARY && (output & OUTPUT_CLASS) != 0 && !predict_proba,
  // and is ignored otherwise
  float threshold;
  // global_bias is added to the sum of tree predictions
  // (after averaging, if it is used, but before any further transformations)
  float global_bias;
  // only used for CATEGORICAL_LEAF inference. since we're storing the
  // labels in leaves instead of the whole vector, this keeps track
  // of the number of classes
  int num_classes;
  // blocks_per_sm, if nonzero, works as a limit to improve cache hit rate for larger forests
  // suggested values (if nonzero) are from 2 to 7
  // if zero, launches ceildiv(num_rows, NITEMS) blocks
  int blocks_per_sm;
};

/** from_treelite uses a treelite model to initialize the forest
 * @param handle cuML handle used by this function
 * @param pforest pointer to where to store the newly created forest
 * @param model treelite model used to initialize the forest
 * @param tl_params additional parameters for the forest
 */
void from_treelite(const raft::handle_t& handle, forest_t* pforest,
                   ModelHandle model, const treelite_params_t* tl_params);

void from_cuml_rf(const raft::handle_t& handle, forest_t* pforest,
                  const RandomForestMetaData<double, double>* forest,
                  forest_params_t* params, storage_type_t storage_type,
                  bool output_class);
void from_cuml_rf(const raft::handle_t& handle, forest_t* pforest,
                  const RandomForestMetaData<double, float>* forest,
                  forest_params_t* params, storage_type_t storage_type,
                  bool output_class);
void from_cuml_rf(const raft::handle_t& handle, forest_t* pforest,
                  const RandomForestMetaData<float, double>* forest,
                  forest_params_t* params, storage_type_t storage_type,
                  bool output_class);
void from_cuml_rf(const raft::handle_t& handle, forest_t* pforest,
                  const RandomForestMetaData<float, float>* forest,
                  forest_params_t* params, storage_type_t storage_type,
                  bool output_class);
void from_cuml_rf(const raft::handle_t& handle, forest_t* pforest,
                  const RandomForestMetaData<double, int>* forest,
                  forest_params_t* params, storage_type_t storage_type,
                  bool output_class);
void from_cuml_rf(const raft::handle_t& handle, forest_t* pforest,
                  const RandomForestMetaData<float, int>* forest,
                  forest_params_t* params, storage_type_t storage_type,
                  bool output_class);

/** free deletes forest and all resources held by it; after this, forest is no longer usable
 *  @param h cuML handle used by this function
 *  @param f the forest to free; not usable after the call to this function
 */
void free(const raft::handle_t& h, forest_t f);

/** predict predicts on data (with n rows) using forest and writes results into preds;
 *  the number of columns is stored in forest, and both preds and data point to GPU memory
 *  @param h cuML handle used by this function
 *  @param f forest used for predictions
 *  @param preds array in GPU memory to store predictions into
        size == predict_proba ? (2*num_rows) : num_rows
 *  @param data array of size n * cols (cols is the number of columns
 *      for the forest f) from which to predict
 *  @param num_rows number of data rows
 *  @param predict_proba for classifier models, this forces to output both class probabilities
 *      instead of binary class prediction. format matches scikit-learn API
 */
void predict(const raft::handle_t& h, forest_t f, float* preds,
             const float* data, size_t num_rows, bool predict_proba = false);

}  // namespace fil
}  // namespace ML

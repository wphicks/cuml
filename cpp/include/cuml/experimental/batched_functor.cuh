#include <array>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>
#include <cuda_runtime.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdbuffer.cuh>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/memory_type.hpp>

namespace ML {
namespace experimental {

namespace detail {

template <template<typename...> typename Template, typename T>
struct tuple_type_forwarder;

template <template<typename...> typename Template, typename... Ts>
struct tuple_type_forwarder<Template, std::tuple<Ts...>> {
  using type = Template<Ts...>;
};

/* Infer the return type and arg types of a callable */
template <typename lambda_t>
struct lambda_traits;

template <typename return_t, typename... args>
struct lambda_traits<return_t(args...)> {
  using return_type = return_t;
  using args_tuple_type = std::tuple<args...>;
};

template <typename lambda_t>
struct lambda_traits : lambda_traits<decltype(&lambda_t::operator())> {};

template <typename lambda_t, typename return_t, typename... args>
struct lambda_traits<return_t(lambda_t::*)(args...) const> : lambda_traits<return_t(args...)>{};

template <typename T>
struct only_first_type;

template <typename T, typename... Ts>
struct only_first_type<std::tuple<T, Ts...>> {
  using type = T;
};

template <
  template<typename...> typename Condition1,
  template<typename...> typename Condition2,
  typename... Ts
>
struct either : std::disjunction<Condition1<Ts...>, Condition2<Ts...>> {};

template <
  template<typename...> typename Condition1,
  template<typename...> typename Condition2,
  typename... Ts
>
using all_of_either = std::conjunction<either<Condition1, Condition2, Ts>...>;

template <
  template<typename...> typename Condition1,
  template<typename...> typename Condition2,
  typename T
>
struct tuple_all_of_either;

template <
  template<typename...> typename Condition1,
  template<typename...> typename Condition2,
  typename... Ts
>
struct tuple_all_of_either<Condition1, Condition2, std::tuple<Ts...>>
  : all_of_either<Condition1, Condition2, Ts...> {};

template <typename T>
struct all_but_first_type;

template <typename T, typename... Ts>
struct all_but_first_type<std::tuple<T, Ts...>> {
  using type = std::tuple<Ts...>;
};

template <typename> struct is_tuple: std::false_type {};
template <typename... Ts> struct is_tuple<std::tuple<Ts...>>: std::true_type {};

template <typename T>
auto static constexpr const is_tuple_v = is_tuple<T>::value;

template <typename... Ts>
struct largest_integer;

template <>
struct largest_integer<> {
  using type = std::size_t;
};

template <typename T>
struct largest_integer<T> {
  using type = T;
};

template <typename T, typename U>
struct largest_integer<T, U> {
  using type = std::conditional_t<
    std::numeric_limits<T>::max() >= std::numeric_limits<U>::max(),
    T,
    U
  >;
};

template <typename T, typename U, typename... Ts>
struct largest_integer<T, U, Ts...> {
  using type = std::conditional_t<
    std::numeric_limits<T>::max() >= std::numeric_limits<U>::max(),
    largest_integer<T, Ts...>,
    largest_integer<U, Ts...>
  >;
};

template<typename lambda_t>
using is_batchable = std::conjunction<
  // Must return only mdarrays
  tuple_type_forwarder<
    raft::is_array_interface,
    typename lambda_traits<lambda_t>::return_type
  >,
  // First argument must be raft::resources
  std::is_same<
    std::remove_cv_t<std::remove_reference_t<only_first_type<
      typename lambda_traits<lambda_t>::args_tuple_type
    >>>,
    raft::resources
  >,
  // All arguments must be mdspans or mdbuffers with const elements
  tuple_all_of_either<
    raft::is_input_mdspan,
    raft::is_input_mdbuffer,
    typename all_but_first_type<
      typename lambda_traits<lambda_t>::args_tuple_type
    >::type
  >
>;

template<typename lambda_t>
auto static constexpr const is_batchable_v = is_batchable<lambda_t>::value;

template <
  typename lambda_t,
  std::enable_if_t<is_batchable_v<lambda_t>> * = nullptr
>
struct batchable_lambda_traits {
  // Drop first argument; it is the raft::resources object
  using arg_types = all_but_first_type<
    typename lambda_traits<lambda_t>::args_tuple_type
  >;
  using input_types = typename all_but_first_type<
    typename lambda_traits<lambda_t>::args_tuple_type
  >::type;
  using return_type = typename lambda_traits<lambda_t>::return_type;

  using output_types = std::conditional_t<
    is_tuple_v<return_type>,
    return_type,
    std::tuple<return_type>
  >;
};

void synchronize_all_resource_streams(raft::resources const& res) {
  raft::resource::sync_stream(res);
  if (raft::resource::is_stream_pool_initialized(res)) {
    raft::resource::sync_stream_pool(res);
  }
}

}  // namespace detail

template <
  raft::memory_type MemType,
  typename lambda_t,
  std::enable_if_t<detail::is_batchable_v<lambda_t>>* = nullptr
>
struct batched_functor :
  private std::enable_shared_from_this<batched_functor<MemType, lambda_t>>
{
  using input_types = typename
    detail::batchable_lambda_traits<lambda_t>::input_types;
  using output_types = typename
    detail::batchable_lambda_traits<lambda_t>::output_types;
  using return_type = typename
    detail::batchable_lambda_traits<lambda_t>::return_type;

  auto constexpr static const memory_type = MemType;

  /*-----------------------Construct size_type-----------------------*/
 private:

  template <typename T, typename U>
  struct size_type_constructor;

  template <typename... Ts, typename... Us>
  struct size_type_constructor<std::tuple<Ts...>, std::tuple<Us...>> {
    using type = typename detail::tuple_type_forwarder<
      detail::largest_integer,
      std::tuple<typename Ts::size_type..., typename Us::size_type...>
    >::type;
  };

 public:
  using size_type = typename size_type_constructor<
    input_types,
    output_types
  >::type;

  /*-----------------------Construct batch types --------------------*/
 private:
  template<typename T>
  struct batch_type_constructor;

  template<typename... Ts>
  struct batch_type_constructor<std::tuple<Ts...>> {
    using type = std::tuple<
      raft::mdarray<
        typename Ts::value_type,
        typename Ts::extents_type,
        // Layout right is guaranteed to return an mdspan of the same layout
        // when sliced, but the same is not true for other layouts. For any
        // layout without this property, we may need to copy the data in order
        // to get a view of *only* the data in the current batch with the
        // mdspan type expected by the callable. By using layout_right here, we
        // at least ensure that a copy is not required to construct the
        // original sliced view of the input batch and then copy if necessary
        // to match the expected mdspan type.
        raft::layout_right,
        typename raft::default_buffer_container_policy<
          typename Ts::value_type
        >::template container_policy<memory_type>
      >...
    >;
  };

  template<typename T>
  struct input_buffer_type_constructor;

  template<typename... Ts>
  struct input_buffer_type_constructor<std::tuple<Ts...>> {
    using type = std::tuple<
      std::conditional_t<
        raft::is_input_mdbuffer_v<Ts>,
        Ts,
        raft::mdbuffer<
          typename Ts::element_type,
          typename Ts::extents_type,
          typename Ts::layout_type,
          raft::default_buffer_container_policy<
            typename Ts::element_type
          >
        >
      >...
    >;
  };
  using input_buffer_type = typename input_buffer_type_constructor<
    input_types
  >::type;

  template <std::size_t... I>
  auto constexpr get_input_memory_types(std::index_sequence<I...>) {
    return std::array<std::optional<raft::memory_type>, std::tuple_size_v<input_types>> {
      []() {
        auto result = std::optional<raft::memory_type>{};
        if constexpr (raft::is_mdbuffer_v<std::tuple_element_t<I, input_types>>) {
          result = std::nullopt;
        } else {
          result = std::tuple_element_t<I, input_types>::accessor_policy::mem_type;
        }
        return result;
      }()...
    };
  }

  auto constexpr get_input_memory_types() {
    return get_input_memory_types(
      std::make_index_sequence<std::tuple_size_v<input_types>>()
    );
  }


 public:
  using input_batch_type = typename batch_type_constructor<input_types>::type;
  using output_batch_type = typename batch_type_constructor<output_types>::type;

  /*-----------------------API implementation------------------------*/
 private:
  batched_functor(lambda_t&& lambda) :
    lambda_{std::forward<lambda_t>(lambda)},
    input_batch_{},
    output_batch_{},
    cur_batch_size_{} {}

 public:
  friend std::shared_ptr<batched_functor<memory_type, lambda_t>> as_batched_functor(lambda_t&& lambda);

  struct batched_output_proxy {
    batched_output_proxy(
      size_type begin_index,
      size_type end_index,
      std::shared_ptr<batched_functor<MemType, lambda_t>> functor
    ) :
      batch_dim_range_{begin_index, end_index},
      functor_{functor},
      output_batch_{nullptr} {}

    auto get(raft::resources const& res) {
      if (!output_batch_) {
        functor_->process_batch(res, this);
      }
      if constexpr (detail::is_tuple_v<return_type>) {
        return create_final_results(
          res,
          std::make_index_sequence<std::tuple_size_v<return_type>>()
        );
      } else {
        return std::get<0>(create_final_results(
          res,
          std::make_index_sequence<std::tuple_size_v<return_type>>()
        ));
      }
    }

    void set_batch(std::shared_ptr<output_batch_type> batch) {
      output_batch_ = batch;
    }

   private:
    std::array<size_type, 2> batch_dim_range_ = std::array{size_type{}, size_type{}};
    std::shared_ptr<batched_functor<MemType, lambda_t>> functor_ = nullptr;
    std::shared_ptr<output_batch_type> output_batch_ = nullptr;

    template<std::size_t... I>
    auto create_final_results(
      raft::resources const& res,
      std::index_sequence<I...>
    ) {
      return std::make_tuple(
        [&res, this](auto&& mda) {
          using mda_out_t = std::tuple_element_t<I, output_types>;
          auto result_batch_size = batch_dim_range_[1] - batch_dim_range_[0];
          auto new_extents = slice_extents(
            mda.extents(),
            result_batch_size
          );
          auto result = mda_out_t{
            res,
            typename mda_out_t::mapping_type{new_extents},
            typename mda_out_t::container_policy_type{}
          };
          auto batch_dim_offset = std::size_t{1};
          for (auto i = std::size_t{}; i < mda.rank(); ++i) {
            batch_dim_offset *= mda.extent(i);
          }
          // NOTE: This uses the fact that we enforce use of layout_right for
          // output batches. This can be relaxed with the use of submdspan to
          // include a broader range of layouts.
          auto batch_slice = raft::make_mdspan(
            mda.data_handle() + batch_dim_range_[0] * batch_dim_offset,
            new_extents
          );
          raft::copy(res, result.view(), batch_slice);
          return result;
        }(std::get<I>(*output_batch_))...
      );
    }
  };

  auto reserve(size_type reserved_size) {
    // TODO: Allow for reservation of input batch
  }

  template<
    typename FirstArg,
    typename... Args,
    std::enable_if_t<raft::is_mdspan_v<Args...>>* = nullptr
  >
  auto operator()(raft::resources const& res, FirstArg&& first_arg, Args&&... args) {
    auto input_batch_size = size_type{first_arg.extent(0)};

    // Must always synchronize initialization before input copy
    // Must always synchronize input copy before processing
    // Must always synchronize processing before replacing input batch

    // Create input batch workspace if it does not exist
    if (!input_batch_.has_value()) {
      auto lock = std::lock_guard<std::recursive_mutex>{mtx_};
      if (!input_batch_.has_value()) {
        input_batch_ = create_input_batch(res, first_arg, args..., size_type{});
      }
    }

    // Grow input batch workspace if it is not large enough
    if (
      cur_batch_size_.load() + input_batch_size >
      size_type{std::get<0>(input_batch_).extent(0)}
    ) {
      auto lock = std::lock_guard<std::recursive_mutex>{mtx_};
      if (
        cur_batch_size_.load() + input_batch_size >
        size_type{std::get<0>(input_batch_).extent(0)}
      ) {
        // Process the current batch so that we can clear the input
        process_batch(res);
        // We must synchronize here because deallocation of input_batch_ may
        // occur on a stream provided by another thread (i.e. the stream used
        // for allocation). Therefore, before we replace the current input
        // batch with a new one, we need to make sure that our own stream is
        // done with the previous input batch. Then it is safe to trigger
        // deallocation, which *may* occur on some other stream, by replacing
        // input_batch_.
        detail::synchronize_all_resource_streams(res);
        input_batch_ = create_input_batch(res, first_arg, args..., size_type{});
      }
    }

    // Copy inputs into batch
    {
      auto lock = std::lock_guard<std::recursive_mutex>{mtx_};

      if (
        cur_batch_size_.load() + input_batch_size >
        size_type{std::get<0>(input_batch_).extent(0)}
      ) {
        process_batch(res);
        // No need to synchronize here because we will be copying in the input
        // on the same strem as we used to initiate processing
      }
      // Input batch workspace must be allocated before copying to it
      synchronize_input_initializer_streams(res);
      copy_inputs_to_batch(
        res,
        std::forward_as_tuple(first_arg, args...),
        std::make_index_sequence<sizeof...(args) + 1>()
      );
      return batched_output_proxy{
        cur_batch_size_.fetch_add(input_batch_size),
        cur_batch_size_.load(),
        this->shared_from_this()
      };
    }
  }

 private:

  template <typename out_extents_t, typename in_extents_t, typename rank_t, rank_t batch_rank, rank_t... I>
  static auto grow_extents(
    in_extents_t&& extents,
    std::integer_sequence<rank_t, batch_rank, I...>,
    size_type target_batch_dim = size_type{}
  ) {
    auto batch_dim = typename out_extents_t::size_type{extents.extent(batch_rank)};
    batch_dim = target_batch_dim > batch_dim ? target_batch_dim : batch_dim;
    batch_dim += std::max(
      batch_dim + batch_dim / typename out_extents_t::size_type{2}, 
      batch_dim + typename out_extents_t::size_type{1}
    );
    return out_extents_t{
      batch_dim,
      extents.extent(I)...
    };
  }

  /* Compute new extent values  with the first (batch) dimension increased by
   * a factor of 1.5. If target_batch_dim is greater than the current batch
   * dimension, grow to 1.5 times target_batch_dim instead */
  template <typename out_extents_t, typename in_extents_t>
  static auto grow_extents(
    in_extents_t&& extents,
    size_type target_batch_dim = size_type{}
  ) {
    return grow_extents<out_extents_t>(
      extents,
      std::integer_sequence<typename in_extents_t::rank_type, extents.rank()>{},
      target_batch_dim
    );
  }

  template<typename args_tuple_t, std::size_t... I>
  auto create_input_batch(
    raft::resources const& res,
    args_tuple_t&& args_tuple,
    size_type target_batch_dim,
    std::index_sequence<I...>
  ) {
    record_input_initializer_streams(res);
    return std::make_tuple(
      [&res, target_batch_dim](auto&& arg) {
        using array_type = std::tuple_element_t<I, input_batch_type>;
        return array_type{
          res,
          typename array_type::mapping_type{
            grow_extents<typename array_type::extents_type>(
              arg.extents(),
              target_batch_dim
            )
          },
          typename array_type::container_policy_type{}
        };
      }()...
    );
  }

  template<typename args_tuple_t, std::size_t... I>
  auto copy_inputs_to_batch(
    raft::resources const& res,
    args_tuple_t&& args,
    std::index_sequence<I...>
  ) {
    record_input_streams(res);
    (
      raft::copy(
        res,
        [this](auto&& mds) {
          return raft::make_mdspan(
            mds.data_handle(),
            slice_extents(mds.extents(), cur_batch_size_.load())
          );
        }(std::get<I>(*input_batch_).view()),
        std::get<I>(args)
      ),
      ...
    );
  }

  template<typename... Args>
  auto create_input_batch(
      raft::resources const& res,
      Args&&... args,
      size_type target_batch_dim
    ) {
    return create_input_batch(
      res,
      std::forward_as_tuple(args...),
      target_batch_dim,
      std::make_index_sequence<sizeof...(Args)>()
    );
  }

  void record_input_initializer_streams(raft::resources const& res) {
    record_streams(res, input_initializer_streams_);
  }
  void synchronize_input_initializer_streams(raft::resources const& res) {
    synchronize_if_required(res, input_initializer_streams_);
  }
  void record_input_streams(raft::resources const& res) {
    record_streams(res, input_streams_);
  }
  void synchronize_input_streams(raft::resources const& res) {
    synchronize_if_required(res, input_streams_);
  }
  void record_processing_streams(raft::resources const& res) {
    record_streams(res, processing_streams_);
  }
  void synchronize_processing_streams(raft::resources const& res) {
    synchronize_if_required(res, processing_streams_);
  }

  static void record_streams(raft::resources const& res, std::set<cudaStream_t>& stream_set) {
    stream_set.insert(raft::resource::get_cuda_stream(res).value());
    if (raft::resource::is_stream_pool_initialized(res)) {
      for (
        auto stream_idx = std::size_t{};
        stream_idx < raft::resource::get_stream_pool_size(res);
        ++stream_idx
      ) {
        stream_set.insert(
          raft::resource::get_stream_from_stream_pool(res, stream_idx).value());
      }
    }
  }

  static void synchronize(std::set<cudaStream_t>& stream_set) {
    while (stream_set.size() != std::size_t{}) {
      for (auto stream : stream_set) {
        auto status = cudaStreamQuery(stream);
        if (status != cudaErrorNotReady) {
          if (stream_set.erase(stream) != std::size_t{}) {
            if (status != cudaErrorInvalidResourceHandle) { RAFT_CUDA_TRY(status); }
            break;  // Do not continue to iterate on modified set
          }
        }
      }
    }
  }

  static void synchronize_if_required(raft::resources const& res, std::set<cudaStream_t>& stream_set) {
    if(
      raft::resource::is_stream_pool_initialized(res) ||
      stream_set.size() != 1 || (
        stream_set.size() == 1 &&
        *std::begin(stream_set) != raft::resource::get_cuda_stream(res).value()
      )
    ) {
      synchronize(stream_set);
    }
  }

  /* Process current batch only if requested output proxy is part of this
   * batch. Ensure that processing is not still ongoing on other streams.
   */
  void process_batch(
    raft::resources const& res,
    batched_output_proxy const* requested_output
  ) {
    auto lock = std::lock_guard<std::recursive_mutex>{mtx_};
    if (std::find(std::begin(results_), std::end(results_), requested_output)) {
      process_batch(res);
    }
    synchronize_processing_streams(res);
  }

  /* Apply callable to current batch. Output batch will be assigned (as a
   * shared pointer) to all output proxies associated with current batch.
   * batched_functor will not hold ownership of this batch after it has been
   * distributed to output proxies */
  void process_batch(raft::resources const& res) {
    auto lock = std::lock_guard<std::recursive_mutex>{mtx_};
    synchronize_input_streams(res);
    record_processing_streams(res);
    auto input_buffers = get_input_buffers(res);
    auto batch_result = result_to_output_batch(res, [this, input_buffers]() {
      if constexpr(detail::is_tuple_v<return_type>) {
        return lambda_(
          std::apply(
            [](auto&&... buffer) {
              return std::make_tuple(
                buffer.view()...
              );
            },
            input_buffers
          )
        );
      } else {
        return std::make_tuple(
          lambda_(
            std::apply(
              [](auto&&... buffer) {
                return std::make_tuple(
                  buffer.view()...
                );
              },
              input_buffers
            )
          )
        );
      }
    }());
    std::for_each(
      std::begin(results_),
      std::end(results_),
      [batch_result](auto&& out) {
        out.set_batch(batch_result);
      }
    );
    results_.clear();
    cur_batch_size_.store(size_type{});
  }

  template<typename results_t, std::size_t... I>
  auto result_to_output_batch(
    raft::resources const& res,
    results_t&& results,
    std::index_sequence<I...>
  ) {
    return std::make_shared(std::make_tuple(
      [&res](auto&& mda) {
        if constexpr (
          std::is_same_v<
            std::tuple_element_t<I, results_t>,
            std::tuple_element_t<I, output_batch_type>
          >
        ) {
          return std::move(mda);
        } else {
          using mda_out_t = std::tuple_element_t<I, output_batch_type>;
          auto view = mda.view();
          auto mda_out = mda_out_t{
            res,
            typename mda_out_t::mapping_type{
              convert_extents_type<typename mda_out_t::extents_type>(
                view.extents()
              )
            },
            typename mda_out_t::container_policy_type{}
          };
          raft::copy(res, mda_out.view(), view);
        }
      }(std::get<I>(results))...
    ));
  }

  template<typename results_t>
  auto result_to_output_batch(raft::resources const& res, results_t&& results) {
    return result_to_output_batch(
      res,
      std::forward<results_t>(results),
      std::make_index_sequence<std::tuple_size_v<results_t>>()
    );
  }

  template <std::size_t... I>
  auto get_input_buffers(raft::resources const& res, std::index_sequence<I...>) {
    return std::make_tuple(
      std::tuple_element_t<I, input_buffer_type>{
        res,
        raft::mdbuffer{
          slice_to_batch(std::get<I>(*input_batch_).view(), cur_batch_size_.load())
        },
        std::get<I>(get_input_memory_types())
      }...
    );
  }

  auto get_input_buffers(raft::resources const& res) {
    return get_input_buffers(
      res,
      std::index_sequence<std::tuple_size_v<input_batch_type>>{}
    );
  }

  template<typename out_extents_t, typename extents_t, typename rank_t, rank_t... I>
  static auto convert_extents_type(
    extents_t&& extents, std::integer_sequence<rank_t, I...>
  ) {
    return out_extents_t{
      typename out_extents_t::size_type{extents.extent(I)}...
    };
  }

  template<typename out_extents_t, typename extents_t>
  static auto convert_extents_type(extents_t&& extents) {
    return convert_extents_type(
      extents,
      std::integer_sequence<typename extents_t::rank_type, extents.rank()>{}
    );
  }

  template<typename extents_t, typename rank_t, rank_t batch_rank, rank_t... I>
  static auto slice_extents(
    extents_t&& extents,
    size_type batch_dim,
    std::integer_sequence<rank_t, batch_rank, I...>
  ) {
    return extents_t{
      typename extents_t::size_type{batch_dim},
      extents.extent(I)...
    };
  }

  template <typename extents_t>
  static auto slice_extents(
    extents_t&& extents,
    size_type batch_dim
  ) {
    return slice_extents<extents_t>(
      extents,
      batch_dim,
      std::integer_sequence<typename extents_t::rank_type, extents.rank()>{}
    );
  }

  template <
    typename mdspan_t,
    std::enable_if_t<
      std::is_same_v<typename mdspan_t::layout_type, raft::layout_right>
    >* = nullptr
  >
  static auto slice_to_batch(mdspan_t mds, size_type batch_size) {
    // TODO(wphicks): Replace this when submdspan is available
    auto new_extents = mds.extents();
    return raft::make_mdspan(
      mds.data_handle(),
      slice_extents(mds.extents(), batch_size)
    );
  }

  lambda_t lambda_;
  std::optional<input_batch_type> input_batch_;
  std::optional<output_batch_type> output_batch_;
  std::atomic<size_type> cur_batch_size_;
  std::recursive_mutex mtx_;
  std::set<cudaStream_t> input_initializer_streams_;
  std::set<cudaStream_t> input_streams_;
  std::set<cudaStream_t> processing_streams_;
};

template<raft::memory_type MemType, typename lambda_t>
auto as_batched_functor(lambda_t&& lambda) {
  return std::make_shared<batched_functor<
    MemType,
    lambda_t
  >>(std::forward<lambda_t>(lambda));
};


}  // experimental
}  // ML

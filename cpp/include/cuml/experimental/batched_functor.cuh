#include <array>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <cuda_runtime.h>
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

template <template<typename...> typename Template, typename T>
struct tuple_value_forwarder;

template <template<typename...> typename Template, typename... Ts>
struct tuple_value_forwarder<Template, std::tuple<Ts...>> {
  auto static constexpr const value = Template<Ts...>::value;
};

/* Split types based on a condition, constructing tuples of each */
template <
  template<typename...> typename Condition,
  typename false_index_t,
  typename false_tuple_t,
  typename true_index_t,
  typename true_tuple_t,
  typename remaining_index_t,
  typename... Ts
>
struct type_filter_builder;

template <
  template<typename...> typename Condition,
  typename false_index_t,
  typename false_tuple_t,
  typename true_index_t,
  typename true_tuple_t
>
struct type_filter_builder<
  Condition,
  false_index_t,
  false_tuple_t,
  true_index_t,
  true_tuple_t,
  std::index_sequence<>
> {
  using type = true_tuple_t;
  using sequence = true_index_t;
  using false_type = false_tuple_t;
  using false_sequence = false_index_t;
};

template <
  template<typename...> typename Condition,
  typename... false_ts,
  std::size_t... false_indexes,
  typename... true_ts,
  std::size_t... true_indexes,
  std::size_t current_index,
  std::size_t... remaining_indexes,
  typename T,
  typename... Ts
>
struct type_filter_builder<
  Condition,
  std::index_sequence<false_indexes...>,
  std::tuple<false_ts...>,
  std::index_sequence<true_indexes...>,
  std::tuple<true_ts...>,
  std::index_sequence<current_index, remaining_indexes...>,
  T, Ts...
>{
  using type = std::conditional_t<
    Condition<T>::value,
    type_filter_builder<
      Condition,
      std::index_sequence<false_indexes...>,
      std::tuple<false_ts...>,
      std::index_sequence<true_indexes..., current_index>,
      std::index_sequence<remaining_indexes...>,
      std::tuple<true_ts..., T>,
      Ts...
    >,
    type_filter_builder<
      Condition,
      std::index_sequence<false_indexes..., current_index>,
      std::tuple<false_ts..., T>,
      std::index_sequence<true_indexes...>,
      std::index_sequence<remaining_indexes...>,
      std::tuple<true_ts...>,
      Ts...
    >
  >;
};

template <template<typename...> typename Condition, typename... Ts>
using type_filter = type_filter_builder<
  Condition,
  std::index_sequence<>,
  std::tuple<>,
  std::index_sequence<>,
  std::tuple<>,
  std::make_index_sequence<sizeof...(Ts)>,
  Ts...
>;

/* Given a template which, when applied to a single type, produces a type with a
 * boolean constexpr value attribute, construct a tuple containing only types
 * for which that value is true. The order of types in the returned tuple will
 * match the order in which they appear in the original pack of types. */
template <template<typename...> typename Condition, typename... Ts>
using type_filter_t = typename type_filter<Condition, Ts...>::type;

/* Given a template which, when applied to a single type, produces a type with a
 * boolean constexpr value attribute, constructs an index sequence containing
 * the indexes for which that value is true in the original pack of types */
template <template<typename...> typename Condition, typename... Ts>
using type_filter_sequence = typename type_filter<Condition, Ts...>::sequence;

template <typename tuple_t, std::size_t... I>
auto constexpr tuple_from_sequence(tuple_t tuple, std::index_sequence<I...>) {
  return std::forward_as_tuple(std::get<I>(tuple)...);
}

template <
  template<typename...> typename Condition,
  typename... Ts
>
auto constexpr filter_tuple(std::tuple<Ts...>&& tuple) {
  return tuple_from_sequence(
    std::forward<std::tuple<Ts...>&&>(tuple),
    type_filter_sequence<Condition, Ts...>{}
  );
}

template <
  template<typename...> typename Condition,
  typename lambda_t,
  typename... Args
>
auto apply_if(lambda_t&& lambda, Args&&... args) {
  std::apply(
    std::forward<lambda_t>(lambda),
    filter_tuple<Condition>(std::forward_as_tuple(args...))
  );
}

template<
  typename mdspan_t,
  std::size_t... I,
  std::enable_if_t<raft::is_mdspan_v<mdspan_t>>* = nullptr
>
auto extents_as_tuple(mdspan_t mds, std::index_sequence<I...>) {
  return std::make_tuple(mds.extent(I)...);
}

template<
  typename mdspan_t,
  std::enable_if_t<raft::is_mdspan_v<mdspan_t>>* = nullptr
>
auto extents_as_tuple(mdspan_t&& mds) {
  auto constexpr rank = mdspan_t::rank();
  return extents_as_tuple(mds, std::make_index_sequence<rank>());
}

template<
  typename... Args,
  std::enable_if_t<raft::is_mdspan_v<Args...>>* = nullptr
>
auto get_extents_as_tuples(Args&&... args) {
  return std::make_tuple(
    extents_as_tuple(args)...
  );
}

template <template<typename...> typename Condition, typename T>
struct tuple_if_types;

template <template<typename...> typename Condition, typename... Ts>
struct tuple_if_types<Condition, std::tuple<Ts...>>{
  using type = type_filter_t<Condition, Ts...>;
};

template <template<typename...> typename Condition, typename T>
using tuple_if_types_t = typename tuple_if_types<Condition, T>::type;

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

template <typename T>
struct all_but_first_type;

template <typename T, typename... Ts>
struct all_but_first_type<std::tuple<T, Ts...>> {
  using type = std::tuple<Ts...>;
};

template <typename T>
struct only_first_type;

template <typename T, typename... Ts>
struct only_first_type<std::tuple<T, Ts...>> {
  using type = T;
};

/* Given a tuple of types, construct a tuple of mdbuffers containing elements of
 * those types */
template<typename Extents, typename T>
struct mdbuffers_tuple;

template<typename Extents, typename... Ts>
struct mdbuffers_tuple<Extents, std::tuple<Ts...>> {
  using type = std::tuple<raft::mdbuffer<Ts, Extents>...>;
};
template<typename Extents, typename tuple_of_element_ts>
using mdbuffers_tuple_t = typename mdbuffers_tuple<Extents, tuple_of_element_ts>::type;

/* Given a tuple of types, construct a tuple of mdarrays containing elements of
 * those types with CV qualifiers removed*/
template<raft::memory_type MemType, typename Extents, typename T>
struct non_cv_mdarrays_tuple;

template<raft::memory_type MemType, typename Extents, typename... Ts>
struct non_cv_mdarrays_tuple<MemType, Extents, std::tuple<Ts...>> {
  using type = std::tuple<typename raft::mdbuffer<Ts, Extents>::owning_type...>;
};
template<raft::memory_type MemType, typename Extents, typename tuple_of_element_ts>
using non_cv_mdarrays_tuple_t = typename non_cv_mdarrays_tuple<MemType, Extents, tuple_of_element_ts>::type;

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

template <typename> struct is_tuple: std::false_type {};
template <typename... Ts> struct is_tuple<std::tuple<Ts...>>: std::true_type {};

template <typename T>
auto static constexpr const is_tuple_v = is_tuple<T>::value;

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

template <typename... Ts>
using is_input_mdspan_or_mdbuffer = all_of_either<
  raft::is_input_mdspan,
  raft::is_input_mdbuffer, 
  Ts...
>;

template <typename... Ts>
using is_output_mdspan_or_mdbuffer = all_of_either<
  raft::is_output_mdspan,
  raft::is_output_mdbuffer, 
  Ts...
>;

template<typename lambda_t>
using is_batchable = std::conjunction<
  // Must not return anything
  std::is_same<typename lambda_traits<lambda_t>::return_type, void>,
  // First argument must be raft::resources
  std::is_same<
    std::remove_cv_t<std::remove_reference_t<only_first_type<
      typename lambda_traits<lambda_t>::args_tuple_type
    >>>,
    raft::resources
  >,
  // All arguments must be mdspan or mdbuffer types
  tuple_all_of_either<
    raft::is_mdspan,
    raft::is_mdbuffer,
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
  using input_types = tuple_if_types_t<
    is_input_mdspan_or_mdbuffer,
    arg_types
  >;
  using output_types = tuple_if_types_t<
    is_output_mdspan_or_mdbuffer,
    arg_types
  >;
};

template <
  typename lambda_t,
  std::enable_if_t<
    std::is_same_v<
      std::remove_cv_t<std::remove_reference_t<only_first_type<
        typename lambda_traits<lambda_t>::args_tuple_type
      >>>,
      raft::resources
    >
  >* = nullptr
>
struct raft_lambda_inputs {
  // Drop first argument; it is the raft::resources object
  using type = all_but_first_type<
    typename lambda_traits<lambda_t>::args_tuple_type
  >;
};

}

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
        // If layout is contiguous, use it. Otherwise use C-contiguous
        std::conditional_t<
          std::disjunction_v<
            std::is_same<typename Ts::layout_type, raft::layout_c_contiguous>,
            std::is_same<typename Ts::layout_type, raft::layout_f_contiguous>
          >,
          typename Ts::layout_type,
          raft::layout_c_contiguous
        >,
        typename raft::default_buffer_container_policy<
          typename Ts::value_type
        >::template container_policy<memory_type>
      >...
    >;
  };


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
    batched_output_proxy(output_types&& output) : output_{std::move(output)} {}

    private:
     output_types output_;
  };

  template<
    typename FirstArg,
    typename... Args,
    std::enable_if_t<raft::is_mdspan_v<Args...>>* = nullptr
  >
  auto operator()(raft::resources const& res, FirstArg&& first_arg, Args&&... args) {
    // Create input and output batch spaces if they do not yet exist
    if (!input_batch_.has_value()) {
      auto lock = std::unique_lock<std::mutex>{mtx_};
      if (!input_batch_.has_value()) {
        input_batch_ = create_batch<input_batch_type>(
          res, 
          detail::filter_tuple<raft::is_input_mdspan>(std::forward_as_tuple(first_arg, args...))
        );
        output_batch_ = create_batch<output_batch_type>(
          res, 
          detail::filter_tuple<raft::is_output_mdspan>(std::forward_as_tuple(first_arg, args...))
        );
      }
    }

    // Grow input and output batch spaces if they are not large enough
    auto input_batch_size = size_type{first_arg.extent(0)};
    if (
      cur_batch_size_ + input_batch_size >
      size_type{std::get<0>(input_batch_).extent(0)}
    ) {
      auto lock = std::unique_lock<std::mutex>{mtx_};
      if (
        cur_batch_size_ + input_batch_size >
        size_type{std::get<0>(input_batch_).extent(0)}
      ) {
        // TODO: Process current batch
        input_batch_ = create_batch<input_batch_type>(
          res, 
          detail::filter_tuple<raft::is_input_mdspan>(
            std::forward_as_tuple(first_arg, args...)
          ),
          cur_batch_size_ + input_batch_size
        );
        output_batch_ = create_batch<output_batch_type>(
          res, 
          detail::filter_tuple<raft::is_output_mdspan>(
            std::forward_as_tuple(first_arg, args...)
          ),
          cur_batch_size_ + input_batch_size
        );
      }
    }

    // Copy inputs into batch
    {
      auto lock = std::unique_lock<std::mutex>{mtx_};
      if (
        cur_batch_size_ + input_batch_size >
        size_type{std::get<0>(input_batch_).extent(0)}
      ) {
        // TODO: Process current batch
      }
      // TODO: Copy inputs
      cur_batch_size_ += input_batch_size;

      // TODO: Generate and return output proxies
    }
  }

 private:

  template <typename extents_t, typename rank_t, rank_t batch_rank, rank_t... I>
  auto static grow_extents(
    extents_t const& extents,
    std::integer_sequence<rank_t, batch_rank, I...>,
    size_type target_batch_dim = size_type{}
  ) {
    auto batch_dim = extents.extent(batch_rank);
    batch_dim = target_batch_dim > batch_dim ? target_batch_dim : batch_dim;
    batch_dim += std::max(
      batch_dim + batch_dim / typename extents_t::size_type{2}, 
      batch_dim + typename extents_t::size_type{1}
    );
    return raft::make_extents(
      batch_dim,
      extents.extent(I)...
    );
  }

  /* Compute new extent values  with the first (batch) dimension increased by
   * a factor of 1.5. If target_batch_dim is greater than the current batch
   * dimension, grow to 1.5 times target_batch_dim instead */
  template <typename extents_t>
  auto static grow_extents(
    extents_t const& extents,
    size_type target_batch_dim = size_type{}
  ) {
    return grow_extents(
      extents,
      std::integer_sequence<typename extents_t::rank_type, extents.rank()>{},
      target_batch_dim
    );
  }

  template <typename batch_t, typename... mdspan_ts, std::size_t... I>
  auto static create_batch(
    raft::resources const& res,
    std::tuple<mdspan_ts...> args,
    std::index_sequence<I...>,
    size_type target_batch_dim = size_type{}
  ) {
    return std::make_tuple(
      typename std::tuple_element<I, batch_t>::type{
        res,
        typename std::tuple_element<I, batch_t>::type::mapping_type{
          grow_extents(std::get<I>(args).extents(), target_batch_dim)
        },
        typename std::tuple_element<
          I, batch_t
        >::type::container_policy_type{}
      }...
    );
  }

  template <typename batch_t, typename... mdspan_ts>
  auto static create_batch(
    raft::resources const& res,
    std::tuple<mdspan_ts...> mds_tuple,
    size_type target_batch_dim = size_type{}
  ) {
    return create_batch(
      res,
      mds_tuple,
      std::make_index_sequence<sizeof...(mdspan_ts)>{},
      target_batch_dim
    );
  }

  lambda_t lambda_;
  std::optional<input_batch_type> input_batch_;
  std::optional<output_batch_type> output_batch_;
  size_type cur_batch_size_;
  std::mutex mtx_;
  std::set<cudaStream_t> input_streams_;
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

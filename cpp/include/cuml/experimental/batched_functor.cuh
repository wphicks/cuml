#include <memory>
#include <mutex>
#include <type_traits>
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
  typename lambda_t,
  std::enable_if_t<
    std::is_same_v<
      std::remove_cv_t<std::remove_reference_t<only_first_type<
        typename lambda_traits<lambda_t>::args_tuple_type
      >>>,
      raft::device_resources
    >
  >* = nullptr
>
struct raft_lambda_inputs {
  // Drop first argument; it is the raft::device_resources object
  using type = all_but_first_type<
    typename lambda_traits<lambda_t>::args_tuple_type
  >;
};

template <typename lambda_t>
struct lambda_outputs {
  using original_return_type = typename lambda_traits<lambda_t>::return_type;
  using type = std::conditional_t<
    is_tuple_v<original_return_type>,
    original_return_type,
    std::tuple<original_return_type>
  >;
};

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
struct tuple_all_of_either<Condition1, Condition2, std::tuple<Ts...>> {
  using value = std::conjunction<std::disjunction<Condition1<Ts>, Condition2<Ts>>...>;
};

}

template <
  raft::memory_type MemType,
  typename lambda_t,
  std::enable_if_t<
    std::conjunction_v<
      detail::tuple_all_of_either<
        raft::is_input_mdspan,
        raft::is_input_mdbuffer,
        typename detail::raft_lambda_inputs<lambda_t>::type
      >,
      detail::tuple_type_forwarder<
        raft::is_array_interface,
        typename detail::lambda_outputs<lambda_t>::type
      >
    >
  >* = nullptr
>
struct batched_functor :
  private std::enable_shared_from_this<batched_functor<MemType, lambda_t>>
{
  using input_types = typename detail::raft_lambda_inputs<lambda_t>::type;
  using output_type = typename detail::lambda_outputs<
    lambda_t
  >::original_return_type;
  auto constexpr static const memory_type = MemType;

 private:
  using output_types_tuple = typename detail::lambda_outputs<lambda_t>::type;

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
    output_types_tuple
  >::type;

 private:
  template<typename T>
  struct batch_type_constructor;

  template<typename... Ts>
  struct batch_type_constructor<std::tuple<Ts...>> {
    using type = std::tuple<
      raft::mdarray<
        typename Ts::element_type,
        typename Ts::extents_type,
        // If input layout is contiguous, use it. Otherwise use C-contiguous
        std::conditional_t<
          std::disjunction_v<
            std::is_same<typename Ts::layout_type, raft::layout_c_contiguous>,
            std::is_same<typename Ts::layout_type, raft::layout_f_contiguous>
          >,
          typename Ts::layout_type,
          raft::layout_c_contiguous
        >,
        std::conditional_t<
          raft::is_mdbuffer_v<Ts>,
          typename Ts::template container_policy<memory_type>,
          raft::alternate_from_mem_type<
            memory_type,
            raft::default_container_policy_variant<typename Ts::element_type>
          >
      >...
    >;
  };


 public:
  using input_batch_type = typename batch_type_constructor<input_types>::type;
  using output_batch_type = typename batch_type_constructor<output_types_tuple>::type;

 private:
  batched_functor(lambda_t&& lambda) :
    lambda_{std::forward<lambda_t>(lambda)},
    input_batch_{},
    output_batch_{},
    cur_batch_size_{} {}

 public:
  friend std::shared_ptr<batched_functor<memory_type, lambda_t>> as_batched_functor(lambda_t&& lambda);

  struct batched_output_proxy {
    batched_output_proxy(output_type&& output) : output_{std::move(output)} {}

    private:
     output_type output_;
  };

  template<
    typename... Args,
    std::enable_if_t<
      std::is_same_v<
        std::tuple<Args...>,
        input_types
      >
    >* = nullptr
  >
  auto operator()(raft::resources const& res, Args&&... args) {
    if (!input_batch_.has_value()) {
      auto lock = std::unique_lock<std::mutex>{mtx_};
      if (!input_batch_.has_value()) {
        input_batch_.emplace(std::forward<Args>(args)...)
        auto first_result = lambda_(res, std::forward<Args>(args)...);
      }
    }
  }

 private:
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

#include <type_traits>
#include <raft/core/mdbuffer.cuh>
#include <raft/core/mdspan.hpp>
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

template <typename T>
struct all_but_first_type;

template <typename T, typename... Ts>
struct all_but_first_type<std::tuple<T, Ts...>> {
  using type = std::tuple<Ts...>;
};

/* Split types into non-const and const, constructing a std::tuple of each */
template <typename non_const_tuple_t, typename const_tuple_t, typename... Ts>
struct const_splitter;

template <typename non_const_tuple_t, typename const_tuple_t>
struct const_splitter<non_const_tuple_t, const_tuple_t> {
  using non_const_tuple_type = non_const_tuple_t;
  using const_tuple_type = const_tuple_t;
};

template <typename... non_const_ts, typename... const_ts, typename T, typename... Ts>
struct const_splitter<std::tuple<non_const_ts...>, std::tuple<const_ts...>, T, Ts...>{
  using type = std::conditional_t<
    std::is_const_v<std::remove_reference_t<T>>,
    const_splitter<std::tuple<non_const_ts...>, std::tuple<const_ts..., T>, Ts...>,
    const_splitter<std::tuple<non_const_ts..., T>, std::tuple<const_ts...>, Ts...>
  >;
};

/* Given a set of types, construct a std::tuple of only the non-const types from
 * that set in the same order they appear in the original parameter pack */
template <typename... Ts>
using only_non_const_tuple = typename const_splitter<
  std::tuple<>, std::tuple<>, Ts...
>::type::non_const_tuple_type;

/* Given a set of types, construct a std::tuple of only the const types from
 * that set in the same order they appear in the original parameter pack */
template <typename... Ts>
using only_const_tuple = typename const_splitter<
  std::tuple<>, std::tuple<>, Ts...
>::type::const_tuple_type;

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

template <typename lambda_t>
struct raft_lambda_inputs {
  // Drop first argument; it should be the raft::device_resources object
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

}

template <raft::memory_type MemType, typename lambda_t>
struct batched_functor {
  using input_types = typename detail::raft_lambda_inputs<lambda_t>::type;
  using output_type = typename detail::lambda_outputs<
    lambda_t
  >::original_return_type;

 private:
  using internal_output_types = typename detail::lambda_outputs<lambda_t>::type;

 public:
  batched_functor(lambda_t&& lambda) :
    lambda_{std::forward<lambda_t>(lambda)} {}

  auto operator()() {
  }

 private:
  lambda_t lambda_;
  inputs_batch_type input_batch_;
  outputs_batch_type output_batch_;
};

template<raft::memory_type MemType, typename InputExtents, typename OutputExtents, typename... io_ts, typename lambda_t>
auto to_batched_functor(lambda_t&& lambda) {
  return batched_functor<
    MemType,
	InputExtents,
    OutputExtents,
    lambda_t,
    io_ts...
  >{std::forward<lambda_t>(lambda)};
};


}  // experimental
}  // ML

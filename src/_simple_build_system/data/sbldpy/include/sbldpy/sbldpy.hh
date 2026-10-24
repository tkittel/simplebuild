#ifndef sbldpy_sbldpy_hh
#define sbldpy_sbldpy_hh

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// sbldpy: Simple Python bindings for C++ code, based on ctypes.              //
//                                                                            //
// This header-only C++17 library is used together with the pure-Python      //
// runtime module "sbldpy". Code using it is NOT compiled with Python headers //
// and NOT linked with libpython. Each module exports exactly four plain C    //
// functions (sbldpy_abi_version, sbldpy_module_info, sbldpy_call and         //
// sbldpy_last_error), and exceptions never cross those (or the callbacks    //
// into Python). See the README.md file accompanying this file for details.  //
//                                                                            //
// Example:                                                                   //
//                                                                            //
//   #include "sbldpy/sbldpy.hh"                                              //
//   double add( double a, double b ) { return a + b; }                       //
//   SBLDPY_MODULE( mymod )                                                   //
//   {                                                                        //
//     m.doc("My module");                                                    //
//     m.def( "add", &add, "Add two numbers", { "a", sbldpy::arg("b",1.0) } );//
//   }                                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#define SBLDPY_VERSION_MAJOR 0
#define SBLDPY_VERSION_MINOR 1
#define SBLDPY_VERSION_PATCH 0
#define SBLDPY_VERSION_STR "0.1.0"
//Ever increasing integer (major*1000000+minor*1000+patch):
#define SBLDPY_VERSION 1000
//Version of the binary interface between modules and the Python runtime:
#define SBLDPY_ABI_VERSION 1

#if __cplusplus < 201703L && !( defined(_MSVC_LANG) && _MSVC_LANG >= 201703L )
#  error "sbldpy requires C++17 or later"
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define SBLDPY_EXPORT __declspec(dllexport)
#else
#  define SBLDPY_EXPORT __attribute__((visibility("default")))
#endif

namespace sbldpy {

  //Exceptions which can be thrown to raise specific Python exceptions (other
  //exceptions are mapped as follows: std::invalid_argument, std::domain_error
  //and std::length_error -> ValueError, std::out_of_range -> IndexError,
  //std::overflow_error -> OverflowError, std::bad_alloc -> MemoryError, and
  //everything else -> RuntimeError):
  class TypeError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
  };

  //Thrown from Callback::operator() when the Python callback raised an
  //exception. If not caught, the original Python exception is raised (with
  //its traceback) when control returns to Python:
  class CallbackError : public std::runtime_error {
  public:
    CallbackError() : std::runtime_error("exception raised in Python callback") {}
    using std::runtime_error::runtime_error;
  };

  namespace detail {
    template<class T> struct dependent_false : std::false_type {};
    template<class T> using rmcvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    template<class T>
    constexpr bool is_charlike_v = ( std::is_same_v<T,char> || std::is_same_v<T,wchar_t>
                                     || std::is_same_v<T,char16_t> || std::is_same_v<T,char32_t>
#if defined(__cpp_char8_t)
                                     || std::is_same_v<T,char8_t>
#endif
                                     );

    //Integers which can be returned to Python (any integer up to 64 bits):
    template<class T>
    constexpr bool is_ret_int_v = ( std::is_integral_v<T> && !std::is_same_v<T,bool>
                                    && !is_charlike_v<T> && sizeof(T) <= 8 );
    //Integers which can be received from Python (only 64 bit signed integers,
    //i.e. std::int64_t, to avoid integer width issues):
    template<class T>
    constexpr bool is_param_int_v = is_ret_int_v<T> && std::is_signed_v<T> && sizeof(T) == 8;

    //Element types of numpy arrays:
    template<class T>
    constexpr bool is_dtype_v = ( std::is_same_v<T,double> || std::is_same_v<T,float>
                                  || is_ret_int_v<T> );
    //Element types of arrays received from Python:
    template<class T>
    constexpr bool is_input_dtype_v = ( std::is_same_v<T,double> || is_param_int_v<T>
                                        || std::is_same_v<T,std::uint8_t> );

    template<class T>
    std::string dtype_name()
    {
      if constexpr ( std::is_same_v<T,double> ) {
        return "f64";
      } else if constexpr ( std::is_same_v<T,float> ) {
        return "f32";
      } else {
        static_assert( is_ret_int_v<T> );
        return ( std::is_signed_v<T> ? "i" : "u" ) + std::to_string( 8 * sizeof(T) );
      }
    }

    //Binary layout of data exchanged with the Python runtime:
    struct StrData { const char * data; std::int64_t size; };
    struct ArrArg { const void * data; std::int64_t size; };
    struct ArrRes { const void * data; std::int64_t size; std::int64_t ndim; const std::int64_t * shape; };
    using AssignFct = int(*)( void * dest, const void * data, std::int64_t n );
    struct AssignRes { void * dest; AssignFct assign; };
    using TrampolineFct = int(*)( std::uint64_t token, void ** args, void * result );
    using ReleaseFct = void(*)( std::uint64_t token );

    //Pointers to functions in the Python runtime (one instance per module):
    struct RuntimeCtx {
      std::atomic<TrampolineFct> trampoline{nullptr};
      std::atomic<ReleaseFct> release{nullptr};
      std::atomic<bool> shutdown{false};
    };

    //Each callback object passed from Python is identified by a token, which
    //is released when the last C++ copy of the Callback object dies:
    struct CallbackCore {
      std::uint64_t token;
      RuntimeCtx * ctx;
      CallbackCore( std::uint64_t t, RuntimeCtx * c ) : token(t), ctx(c) {}
      CallbackCore( const CallbackCore& ) = delete;
      CallbackCore& operator=( const CallbackCore& ) = delete;
      ~CallbackCore()
      {
        if ( ctx->shutdown.load() )
          return;
        auto fct = ctx->release.load();
        if ( fct )
          fct( token );
      }
    };

    template<class T> struct CbArg;
    template<class T> struct CbRet;
  }

  //Copy of array data, which becomes a numpy array on the Python side. Must
  //be returned by value:
  template<class T>
  class NumpyCopy {
    static_assert( detail::is_dtype_v<T>,
                   "sbldpy: NumpyCopy<T> requires T to be a fixed width integer type, float or double" );
  public:
    using value_type = T;
    NumpyCopy() = default;
    NumpyCopy( std::vector<T> data )
      : m_data(std::move(data)), m_shape{ static_cast<std::int64_t>(m_data.size()) } {}
    NumpyCopy( std::vector<T> data, std::vector<std::int64_t> shape )
      : m_data(std::move(data)), m_shape(std::move(shape))
    {
      std::int64_t n = 1;
      for ( auto e : m_shape ) {
        if ( e < 0 )
          throw std::invalid_argument("sbldpy::NumpyCopy: negative dimension in shape");
        n *= e;
      }
      if ( n != static_cast<std::int64_t>(m_data.size()) )
        throw std::invalid_argument("sbldpy::NumpyCopy: shape does not match size of data");
    }
    const std::vector<T>& data() const noexcept { return m_data; }
    std::vector<T>& data() noexcept { return m_data; }
    const std::vector<std::int64_t>& shape() const noexcept { return m_shape; }
    std::size_t size() const noexcept { return m_data.size(); }
  private:
    std::vector<T> m_data;
    std::vector<std::int64_t> m_shape;
  };

  //View of array data without copying.
  //
  //As a parameter (only as NumpyView<const double>, NumpyView<const
  //std::int64_t> or NumpyView<const std::uint8_t>), it provides access to
  //data passed from Python (numpy arrays, other objects with the buffer
  //protocol of the correct type, or lists). The data is only valid during
  //the call.
  //
  //As a return value (only from methods of classes), it becomes a numpy
  //array referring directly to the data (read-only if T is const), which
  //keeps the Python object alive. The data must be owned by the C++ object,
  //and the numpy array becomes invalid if the data is reallocated or
  //destroyed (e.g. when a std::vector is resized). Use .copy() on the Python
  //side when in doubt.
  template<class T>
  class NumpyView {
    using value_type_ = std::remove_const_t<T>;
    static_assert( detail::is_dtype_v<value_type_>,
                   "sbldpy: NumpyView<T> requires T to be a (possibly const) fixed width integer type, float or double" );
    using vector_ref = std::conditional_t<std::is_const_v<T>,
                                          const std::vector<value_type_>&,
                                          std::vector<value_type_>&>;
  public:
    using value_type = value_type_;
    NumpyView( T * data, std::size_t size )
      : m_data(data), m_size(size), m_shape{ static_cast<std::int64_t>(size) } {}
    NumpyView( T * data, std::vector<std::int64_t> shape )
      : m_data(data), m_size(0), m_shape(std::move(shape))
    {
      std::int64_t n = 1;
      for ( auto e : m_shape ) {
        if ( e < 0 )
          throw std::invalid_argument("sbldpy::NumpyView: negative dimension in shape");
        n *= e;
      }
      m_size = static_cast<std::size_t>(n);
    }
    NumpyView( vector_ref v ) : NumpyView( v.data(), v.size() ) {}
    NumpyView( vector_ref v, std::vector<std::int64_t> shape )
      : NumpyView( v.data(), std::move(shape) )
    {
      if ( m_size != v.size() )
        throw std::invalid_argument("sbldpy::NumpyView: shape does not match size of data");
    }
    T * data() const noexcept { return m_data; }
    std::size_t size() const noexcept { return m_size; }
    bool empty() const noexcept { return m_size == 0; }
    T * begin() const noexcept { return m_data; }
    T * end() const noexcept { return m_data + m_size; }
    T& operator[]( std::size_t i ) const noexcept { return m_data[i]; }
    const std::vector<std::int64_t>& shape() const noexcept { return m_shape; }
  private:
    T * m_data;
    std::size_t m_size;
    std::vector<std::int64_t> m_shape;
  };

  //Python callable received as a parameter. It can be copied and stored, and
  //the Python object is kept alive until the last copy dies. Only plain data
  //is passed (copied): Parameters can be integers, floating point numbers,
  //bools, strings, tuples and NumpyCopy objects. Return values can be
  //std::int64_t, double, bool, std::string, tuples, std::vector<double>,
  //std::vector<std::int64_t> and std::vector<std::uint8_t>. If the Python
  //callable raises an exception, a sbldpy::CallbackError is thrown.
  template<class Signature> class Callback;

  template<class R, class... Args>
  class Callback<R(Args...)> {
  public:
    using result_type = R;
    explicit Callback( std::shared_ptr<detail::CallbackCore> core ) : m_core(std::move(core)) {}
    R operator()( Args... args ) const
    {
      return invoke_impl( std::index_sequence_for<Args...>{}, args... );
    }
  private:
    std::shared_ptr<detail::CallbackCore> m_core;
    template<std::size_t... I>
    R invoke_impl( std::index_sequence<I...>, Args&... args ) const;
  };

  //Argument name, optionally with a default value (which must be an integer,
  //floating point number, bool, string, or tuple of those):
  struct Arg {
    std::string name;
    std::string default_json;//empty if no default
    Arg( const char * n ) : name(n) {}
    Arg( std::string n ) : name(std::move(n)) {}
    template<class V>
    Arg( std::string n, const V& defval );
  };
  template<class V>
  Arg arg( std::string name, const V& defval ) { return Arg( std::move(name), defval ); }
  inline Arg arg( std::string name ) { return Arg( std::move(name) ); }

  //Optional documentation string and/or argument names given when defining
  //functions, methods, etc.:
  struct Extra {
    const char * doc = nullptr;
    std::vector<Arg> args;
    bool has_args = false;
    Extra() = default;
    Extra( const char * d ) : doc(d) {}
    Extra( std::initializer_list<Arg> a ) : args(a), has_args(true) {}
  };

  namespace detail {

    ////////////////////////////////////////////////////////////////////////////
    // JSON helpers                                                           //
    ////////////////////////////////////////////////////////////////////////////

    inline std::string json_str( std::string_view s )
    {
      std::string res;
      res.reserve( s.size() + 2 );
      res += '"';
      for ( char c : s ) {
        switch ( c ) {
        case '"': res += "\\\""; break;
        case '\\': res += "\\\\"; break;
        case '\n': res += "\\n"; break;
        case '\r': res += "\\r"; break;
        case '\t': res += "\\t"; break;
        default:
          if ( static_cast<unsigned char>(c) < 0x20 ) {
            char buf[8];
            std::snprintf( buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)) );
            res += buf;
          } else {
            res += c;
          }
        }
      }
      res += '"';
      return res;
    }

    inline std::string json_double( double v )
    {
      //Exact encoding as hexadecimal floating point (decoded with
      //float.fromhex on the Python side):
      char buf[64];
      std::snprintf( buf, sizeof(buf), "%a", v );
      return std::string("{\"f\":\"") + buf + "\"}";
    }

    template<class T> struct is_tuple_like : std::false_type {};
    template<class... T> struct is_tuple_like<std::tuple<T...>> : std::true_type {};
    template<class A, class B> struct is_tuple_like<std::pair<A,B>> : std::true_type {};
    template<class T, std::size_t N> struct is_tuple_like<std::array<T,N>> : std::true_type {};
    template<class T> constexpr bool is_tuple_like_v = is_tuple_like<T>::value;

    template<class T> struct is_numpy_copy : std::false_type {};
    template<class T> struct is_numpy_copy<NumpyCopy<T>> : std::true_type {};
    template<class T> struct is_numpy_view : std::false_type {};
    template<class T> struct is_numpy_view<NumpyView<T>> : std::true_type {};
    template<class T> struct is_callback : std::false_type {};
    template<class S> struct is_callback<Callback<S>> : std::true_type {};
    template<class T> struct is_vector : std::false_type {};
    template<class T, class A> struct is_vector<std::vector<T,A>> : std::true_type {};

    template<class T>
    constexpr bool is_string_v = ( std::is_same_v<T,std::string> || std::is_same_v<T,std::string_view>
                                   || std::is_same_v<T,const char*> );

    //Applies f to each element type of a tuple-like type (passing a null
    //pointer of that type):
    template<class Tup, class F, std::size_t... I>
    constexpr void for_each_type_impl( F&& f, std::index_sequence<I...> )
    {
      ( f( static_cast<std::tuple_element_t<I,Tup>*>(nullptr) ), ... );
    }
    template<class Tup, class F>
    constexpr void for_each_type( F&& f )
    {
      for_each_type_impl<Tup>( std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<Tup>>{} );
    }

    template<class T>
    std::string encode_value( const T& v )
    {
      using U = rmcvref_t<T>;
      if constexpr ( std::is_same_v<U,bool> ) {
        return v ? "true" : "false";
      } else if constexpr ( is_ret_int_v<U> ) {
        return std::to_string( v );
      } else if constexpr ( std::is_same_v<U,double> || std::is_same_v<U,float> ) {
        return json_double( static_cast<double>(v) );
      } else if constexpr ( std::is_same_v<U,std::string> || std::is_same_v<U,std::string_view> ) {
        return json_str( v );
      } else if constexpr ( std::is_same_v<std::decay_t<U>,const char*>
                            || std::is_same_v<std::decay_t<U>,char*> ) {
        return json_str( v );
      } else if constexpr ( is_tuple_like_v<U> ) {
        std::string res = "[";
        std::apply( [&res]( const auto&... e )
        {
          bool first = true;
          ( ( res += ( first ? "" : "," ), res += encode_value( e ), first = false ), ... );
        }, v );
        return res + "]";
      } else {
        static_assert( dependent_false<T>::value,
                       "sbldpy: default values and constants must be integers, floating point numbers,"
                       " bools, strings or tuples of those" );
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Per-thread state                                                       //
    ////////////////////////////////////////////////////////////////////////////

    //Error message of the last failed call on this thread:
    inline std::string& last_error()
    {
      static thread_local std::string s;
      return s;
    }

    //Data referred to by results (strings and arrays), which is kept alive
    //until the next call returning such data on this thread (the Python
    //runtime copies it immediately):
    inline std::vector<std::shared_ptr<void>>& result_slot()
    {
      static thread_local std::vector<std::shared_ptr<void>> s;
      return s;
    }

    template<class T>
    T& keep_in_slot( T&& v )
    {
      auto p = std::make_shared<rmcvref_t<T>>( std::forward<T>(v) );
      T& ref = *p;
      result_slot().push_back( std::move(p) );
      return ref;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Registry (class names for type descriptors)                            //
    ////////////////////////////////////////////////////////////////////////////

    struct Registry {
      std::map<std::type_index,std::string> classnames;
      std::string classname( std::type_index ti, const char * cppname ) const
      {
        auto it = classnames.find( ti );
        if ( it == classnames.end() )
          throw std::logic_error( std::string("type is not supported, or a class which was not"
                                              " (yet) added to the module with add_class"
                                              " (C++ type id: ") + cppname + ")" );
        return it->second;
      }
    };

    template<class T>
    constexpr bool is_object_v = ( std::is_class_v<T> && !is_string_v<T> && !is_tuple_like_v<T>
                                   && !is_numpy_copy<T>::value && !is_numpy_view<T>::value
                                   && !is_callback<T>::value && !is_vector<T>::value );

    ////////////////////////////////////////////////////////////////////////////
    // Compile time checks of parameter and return types                      //
    ////////////////////////////////////////////////////////////////////////////

    template<class T, bool in_tuple = false>
    constexpr bool check_param()
    {
      using U = rmcvref_t<T>;
      constexpr bool is_nonconst_lref = ( std::is_lvalue_reference_v<T>
                                          && !std::is_const_v<std::remove_reference_t<T>> );
      if constexpr ( is_object_v<U> ) {
        static_assert( !in_tuple, "sbldpy: tuples can not contain objects" );
        static_assert( !std::is_rvalue_reference_v<T>,
                       "sbldpy: objects can not be passed as rvalue references" );
        static_assert( std::is_reference_v<T> || std::is_copy_constructible_v<U>,
                       "sbldpy: objects passed by value must be copy constructible" );
      } else if constexpr ( std::is_pointer_v<U> ) {
        static_assert( std::is_same_v<U,const char*>,
                       "sbldpy: pointers are not supported as parameters (except const char* for strings)" );
        static_assert( !in_tuple, "sbldpy: tuples can contain strings only as std::string" );
      } else {
        static_assert( !is_nonconst_lref,
                       "sbldpy: non-const references are only supported for objects of classes" );
        static_assert( !std::is_same_v<U,char>, "sbldpy: char is not supported (use std::int64_t)" );
        static_assert( !std::is_enum_v<U>, "sbldpy: enums are not supported (use std::int64_t)" );
        static_assert( !std::is_same_v<U,float>, "sbldpy: float is not supported as parameter type (use double)" );
        static_assert( !std::is_same_v<U,long double>, "sbldpy: long double is not supported (use double)" );
        static_assert( !std::is_integral_v<U> || std::is_same_v<U,bool> || is_param_int_v<U>,
                       "sbldpy: integers can only be received as std::int64_t" );
        static_assert( !is_vector<U>::value,
                       "sbldpy: std::vector is not supported as parameter type (use sbldpy::NumpyView<const T>)" );
        static_assert( !is_numpy_copy<U>::value,
                       "sbldpy: NumpyCopy is not supported as parameter type (use sbldpy::NumpyView<const T>)" );
        if constexpr ( is_numpy_view<U>::value ) {
          static_assert( !in_tuple, "sbldpy: tuples can not contain NumpyView objects" );
          static_assert( std::is_const_v<typename std::remove_reference_t<decltype(*std::declval<U>().data())>>
                         && is_input_dtype_v<typename U::value_type>,
                         "sbldpy: arrays can only be received as NumpyView<const double>,"
                         " NumpyView<const std::int64_t> or NumpyView<const std::uint8_t>" );
        }
        if constexpr ( is_callback<U>::value )
          static_assert( !in_tuple, "sbldpy: tuples can not contain callbacks" );
        if constexpr ( std::is_same_v<U,std::string_view> )
          static_assert( !in_tuple, "sbldpy: tuples can contain strings only as std::string" );
        if constexpr ( is_tuple_like_v<U> ) {
          for_each_type<U>( []( auto * p ) { check_param<std::remove_pointer_t<decltype(p)>,true>(); } );
        }
        static_assert( std::is_arithmetic_v<U> || std::is_same_v<U,std::string>
                       || std::is_same_v<U,std::string_view> || is_tuple_like_v<U>
                       || is_numpy_view<U>::value || is_callback<U>::value,
                       "sbldpy: unsupported parameter type" );
      }
      return true;
    }

    template<class T, bool is_method = false, bool in_tuple = false>
    constexpr bool check_return()
    {
      static_assert( !std::is_reference_v<T>,
                     "sbldpy: references can not be returned (return by value, or use sbldpy::NumpyView)" );
      static_assert( !std::is_pointer_v<T>,
                     "sbldpy: pointers can not be returned (return by value, or use sbldpy::NumpyView)" );
      using U = std::remove_cv_t<T>;
      if constexpr ( std::is_void_v<U> ) {
        static_assert( !in_tuple );
      } else if constexpr ( is_object_v<U> ) {
        static_assert( !in_tuple, "sbldpy: tuples can not contain objects" );
      } else {
        static_assert( !std::is_same_v<U,char>, "sbldpy: char is not supported as return type" );
        static_assert( !std::is_enum_v<U>, "sbldpy: enums are not supported as return type" );
        static_assert( !std::is_same_v<U,long double>, "sbldpy: long double is not supported as return type" );
        static_assert( !std::is_integral_v<U> || std::is_same_v<U,bool> || is_ret_int_v<U>,
                       "sbldpy: unsupported integer return type" );
        static_assert( !is_vector<U>::value,
                       "sbldpy: std::vector can not be returned directly (return sbldpy::NumpyCopy"
                       " or sbldpy::NumpyView)" );
        static_assert( !std::is_same_v<U,std::string_view>,
                       "sbldpy: std::string_view can not be returned (return std::string)" );
        static_assert( !is_callback<U>::value, "sbldpy: callbacks can not be returned" );
        if constexpr ( is_numpy_view<U>::value ) {
          static_assert( !in_tuple, "sbldpy: tuples can not contain NumpyView objects" );
          static_assert( is_method, "sbldpy: NumpyView can only be returned from methods of classes"
                         " (and keeps the Python object alive)" );
        }
        if constexpr ( is_tuple_like_v<U> ) {
          for_each_type<U>( []( auto * p ) { check_return<std::remove_pointer_t<decltype(p)>,false,true>(); } );
        }
        static_assert( std::is_arithmetic_v<U> || std::is_same_v<U,std::string>
                       || is_tuple_like_v<U> || is_numpy_view<U>::value || is_numpy_copy<U>::value,
                       "sbldpy: unsupported return type" );
      }
      return true;
    }

    template<class T, bool in_tuple = false>
    constexpr bool check_cbarg()
    {
      using U = rmcvref_t<T>;
      static_assert( !std::is_lvalue_reference_v<T> || std::is_const_v<std::remove_reference_t<T>>,
                     "sbldpy: callback parameters can not be non-const references" );
      static_assert( !std::is_same_v<U,char>, "sbldpy: char is not supported in callbacks" );
      static_assert( !std::is_enum_v<U>, "sbldpy: enums are not supported in callbacks" );
      static_assert( !std::is_integral_v<U> || std::is_same_v<U,bool> || is_ret_int_v<U>,
                     "sbldpy: unsupported integer type in callback parameters" );
      if constexpr ( std::is_pointer_v<U> )
        static_assert( std::is_same_v<U,const char*>,
                       "sbldpy: pointers are not supported as callback parameters (except const char*)" );
      if constexpr ( is_tuple_like_v<U> )
        for_each_type<U>( []( auto * p ) { check_cbarg<std::remove_pointer_t<decltype(p)>,true>(); } );
      static_assert( ( std::is_arithmetic_v<U> && !std::is_same_v<U,long double> )
                     || is_string_v<U> || is_tuple_like_v<U> || is_numpy_copy<U>::value,
                     "sbldpy: unsupported callback parameter type (only integers, floating point"
                     " numbers, bools, strings, tuples and NumpyCopy objects are supported)" );
      return true;
    }

    template<class T>
    constexpr bool check_cbret()
    {
      using U = T;
      static_assert( !std::is_reference_v<T> && !std::is_const_v<T>,
                     "sbldpy: callback return types must be plain values" );
      if constexpr ( !std::is_void_v<U> ) {
        static_assert( !std::is_integral_v<U> || std::is_same_v<U,bool> || is_param_int_v<U>,
                       "sbldpy: integers can only be returned from callbacks as std::int64_t" );
        if constexpr ( is_vector<U>::value )
          static_assert( is_input_dtype_v<typename U::value_type>,
                         "sbldpy: only std::vector<double>, std::vector<std::int64_t> and"
                         " std::vector<std::uint8_t> can be returned from callbacks" );
        if constexpr ( is_tuple_like_v<U> )
          for_each_type<U>( []( auto * p ) { check_cbret<std::remove_pointer_t<decltype(p)>>(); } );
        static_assert( std::is_same_v<U,double> || std::is_same_v<U,bool> || is_param_int_v<U>
                       || std::is_same_v<U,std::string> || is_vector<U>::value || is_tuple_like_v<U>,
                       "sbldpy: unsupported callback return type (only std::int64_t, double, bool,"
                       " std::string, tuples and std::vector<double|std::int64_t|std::uint8_t>"
                       " are supported)" );
      }
      return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Type descriptors (for the Python runtime)                              //
    ////////////////////////////////////////////////////////////////////////////

    template<class Tup, class F>
    std::string desc_items( const Registry& reg, F descfct )
    {
      std::string res = "{\"k\":\"tuple\",\"items\":[";
      bool first = true;
      for_each_type<Tup>( [&]( auto * p )
      {
        if ( !first )
          res += ',';
        first = false;
        res += descfct( reg, p );
      } );
      return res + "]}";
    }

    template<class T> std::string desc_cbarg( const Registry& );
    template<class T> std::string desc_cbret( const Registry& );
    template<class Sig> struct cb_describer;
    template<class T> struct callback_sig;
    template<class S> struct callback_sig<Callback<S>> { using type = S; };

    template<class T>
    std::string desc_param( const Registry& reg )
    {
      using U = rmcvref_t<T>;
      if constexpr ( std::is_same_v<U,bool> ) {
        return "\"bool\"";
      } else if constexpr ( std::is_integral_v<U> ) {
        return "\"i64\"";
      } else if constexpr ( std::is_floating_point_v<U> ) {
        return "\"f64\"";
      } else if constexpr ( std::is_same_v<U,const char*> ) {
        return "\"cstr\"";
      } else if constexpr ( is_string_v<U> ) {
        return "\"str\"";
      } else if constexpr ( is_numpy_view<U>::value ) {
        return "{\"k\":\"arr\",\"dt\":\"" + dtype_name<typename U::value_type>() + "\"}";
      } else if constexpr ( is_tuple_like_v<U> ) {
        return desc_items<U>( reg, []( const Registry& r, auto * p )
        { return desc_param<std::remove_pointer_t<decltype(p)>>( r ); } );
      } else if constexpr ( is_callback<U>::value ) {
        return cb_describer<typename callback_sig<U>::type>::describe( reg );
      } else {
        static_assert( is_object_v<U> );
        return "{\"k\":\"obj\",\"cls\":" + json_str( reg.classname( typeid(U), typeid(U).name() ) ) + "}";
      }
    }

    template<class T>
    std::string desc_return( const Registry& reg )
    {
      using U = std::remove_cv_t<T>;
      if constexpr ( std::is_void_v<U> ) {
        return "\"none\"";
      } else if constexpr ( std::is_same_v<U,bool> ) {
        return "\"bool\"";
      } else if constexpr ( std::is_integral_v<U> ) {
        return std::is_signed_v<U> ? "\"int\"" : "\"uint\"";
      } else if constexpr ( std::is_floating_point_v<U> ) {
        return "\"f64\"";
      } else if constexpr ( is_string_v<U> ) {
        return "\"str\"";
      } else if constexpr ( is_numpy_copy<U>::value ) {
        return "{\"k\":\"copy\",\"dt\":\"" + dtype_name<typename U::value_type>() + "\"}";
      } else if constexpr ( is_numpy_view<U>::value ) {
        constexpr bool ro = std::is_const_v<std::remove_reference_t<decltype(*std::declval<U>().data())>>;
        return ( "{\"k\":\"view\",\"dt\":\"" + dtype_name<typename U::value_type>()
                 + "\",\"ro\":" + ( ro ? "true" : "false" ) + "}" );
      } else if constexpr ( is_tuple_like_v<U> ) {
        return desc_items<U>( reg, []( const Registry& r, auto * p )
        { return desc_return<std::remove_pointer_t<decltype(p)>>( r ); } );
      } else {
        static_assert( is_object_v<U> );
        return "{\"k\":\"obj\",\"cls\":" + json_str( reg.classname( typeid(U), typeid(U).name() ) ) + "}";
      }
    }

    template<class T>
    std::string desc_cbarg( const Registry& reg )
    {
      //Same representation as return values:
      return desc_return<std::decay_t<rmcvref_t<T>>>( reg );
    }

    template<class T>
    std::string desc_cbret( const Registry& reg )
    {
      using U = T;
      if constexpr ( std::is_void_v<U> ) {
        return "\"none\"";
      } else if constexpr ( std::is_same_v<U,bool> ) {
        return "\"bool\"";
      } else if constexpr ( std::is_integral_v<U> ) {
        return "\"i64\"";
      } else if constexpr ( std::is_floating_point_v<U> ) {
        return "\"f64\"";
      } else if constexpr ( std::is_same_v<U,std::string> ) {
        return "\"str\"";
      } else if constexpr ( is_vector<U>::value ) {
        return "{\"k\":\"vec\",\"dt\":\"" + dtype_name<typename U::value_type>() + "\"}";
      } else {
        static_assert( is_tuple_like_v<U> );
        return desc_items<U>( reg, []( const Registry& r, auto * p )
        { return desc_cbret<std::remove_pointer_t<decltype(p)>>( r ); } );
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Conversion of parameters (Python -> C++)                               //
    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    decltype(auto) get_param( void * p, RuntimeCtx& ctx );

    template<class Tup, std::size_t... I>
    Tup get_tuple( void ** elems, RuntimeCtx& ctx, std::index_sequence<I...> )
    {
      return Tup{ get_param<std::tuple_element_t<I,Tup>>( elems[I], ctx )... };
    }

    template<class T>
    decltype(auto) get_param( void * p, RuntimeCtx& ctx )
    {
      using U = rmcvref_t<T>;
      if constexpr ( std::is_same_v<U,bool> ) {
        return static_cast<bool>( *static_cast<const std::uint8_t*>(p) != 0 );
      } else if constexpr ( std::is_integral_v<U> ) {
        return static_cast<U>( *static_cast<const std::int64_t*>(p) );
      } else if constexpr ( std::is_floating_point_v<U> ) {
        return *static_cast<const double*>(p);
      } else if constexpr ( std::is_same_v<U,const char*> ) {
        return static_cast<const char*>( static_cast<const StrData*>(p)->data );
      } else if constexpr ( std::is_same_v<U,std::string_view> ) {
        auto s = static_cast<const StrData*>(p);
        return std::string_view( s->data, static_cast<std::size_t>(s->size) );
      } else if constexpr ( std::is_same_v<U,std::string> ) {
        auto s = static_cast<const StrData*>(p);
        return std::string( s->data, static_cast<std::size_t>(s->size) );
      } else if constexpr ( is_numpy_view<U>::value ) {
        auto a = static_cast<const ArrArg*>(p);
        using E = typename U::value_type;
        return U( static_cast<const E*>(a->data), static_cast<std::size_t>(a->size) );
      } else if constexpr ( is_tuple_like_v<U> ) {
        return get_tuple<U>( static_cast<void**>(p), ctx,
                             std::make_index_sequence<std::tuple_size_v<U>>{} );
      } else if constexpr ( is_callback<U>::value ) {
        return U( std::make_shared<CallbackCore>( *static_cast<const std::uint64_t*>(p), &ctx ) );
      } else {
        static_assert( is_object_v<U> );
        //Return a reference to the object:
        return *static_cast<U*>( *static_cast<void**>(p) );
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Conversion of return values (C++ -> Python)                            //
    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    constexpr bool uses_slot()
    {
      using U = std::remove_cv_t<T>;
      if constexpr ( is_tuple_like_v<U> ) {
        bool res = false;
        for_each_type<U>( [&res]( auto * p ) { res = res || uses_slot<std::remove_pointer_t<decltype(p)>>(); } );
        return res;
      } else {
        return is_string_v<U> || is_numpy_copy<U>::value || is_numpy_view<U>::value;
      }
    }

    template<class T, class V>
    void put_value( void * res, V&& v )
    {
      using U = std::remove_cv_t<T>;
      if constexpr ( std::is_same_v<U,bool> ) {
        *static_cast<std::uint8_t*>(res) = ( v ? 1 : 0 );
      } else if constexpr ( std::is_integral_v<U> ) {
        if constexpr ( std::is_signed_v<U> )
          *static_cast<std::int64_t*>(res) = static_cast<std::int64_t>(v);
        else
          *static_cast<std::uint64_t*>(res) = static_cast<std::uint64_t>(v);
      } else if constexpr ( std::is_floating_point_v<U> ) {
        *static_cast<double*>(res) = static_cast<double>(v);
      } else if constexpr ( is_string_v<U> ) {
        //(const char* and std::string_view only for callback arguments, which
        //are not kept in the slot):
        if constexpr ( std::is_same_v<U,std::string> ) {
          auto& s = keep_in_slot( std::string( std::forward<V>(v) ) );
          *static_cast<StrData*>(res) = StrData{ s.data(), static_cast<std::int64_t>(s.size()) };
        } else {
          std::string_view sv( v );
          *static_cast<StrData*>(res) = StrData{ sv.data(), static_cast<std::int64_t>(sv.size()) };
        }
      } else if constexpr ( is_numpy_copy<U>::value || is_numpy_view<U>::value ) {
        auto& a = keep_in_slot( U( std::forward<V>(v) ) );
        if constexpr ( is_numpy_copy<U>::value ) {
          *static_cast<ArrRes*>(res) = ArrRes{ static_cast<const void*>( a.data().data() ),
                                               static_cast<std::int64_t>( a.size() ),
                                               static_cast<std::int64_t>( a.shape().size() ),
                                               a.shape().data() };
        } else {
          *static_cast<ArrRes*>(res) = ArrRes{ static_cast<const void*>( a.data() ),
                                               static_cast<std::int64_t>( a.size() ),
                                               static_cast<std::int64_t>( a.shape().size() ),
                                               a.shape().data() };
        }
      } else if constexpr ( is_tuple_like_v<U> ) {
        void ** elems = static_cast<void**>(res);
        std::apply( [elems]( auto&&... e )
        {
          std::size_t i = 0;
          ( put_value<rmcvref_t<decltype(e)>>( elems[i++], std::forward<decltype(e)>(e) ), ... );
        }, std::forward<V>(v) );
      } else {
        static_assert( dependent_false<T>::value, "sbldpy: unexpected return type" );
      }
    }

    //Calls producer() and puts the returned value into the result:
    template<class R, class F>
    void put_result( void * res, F&& producer )
    {
      using U = std::remove_cv_t<R>;
      if constexpr ( std::is_void_v<U> ) {
        producer();
      } else if constexpr ( is_object_v<U> ) {
        //New object owned by Python (without needing a move or copy):
        *static_cast<void**>(res) = static_cast<void*>( new U( producer() ) );
      } else {
        if constexpr ( uses_slot<U>() )
          result_slot().clear();
        put_value<U>( res, producer() );
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Callback arguments and return values                                   //
    ////////////////////////////////////////////////////////////////////////////

    //Storage of a callback argument in the representation of return values:
    template<class T>
    struct CbArg {
      using U = std::decay_t<rmcvref_t<T>>;
      using storage_t = std::conditional_t<std::is_same_v<U,bool>, std::uint8_t,
                        std::conditional_t<std::is_integral_v<U>,
                                           std::conditional_t<std::is_signed_v<U>,std::int64_t,std::uint64_t>,
                        std::conditional_t<std::is_floating_point_v<U>, double,
                        std::conditional_t<is_string_v<U>, StrData, ArrRes>>>>;
      storage_t storage;
      void * ptr() { return &storage; }
      template<class V>
      explicit CbArg( const V& v )
      {
        if constexpr ( is_string_v<U> ) {
          std::string_view sv( v );
          storage = StrData{ sv.data(), static_cast<std::int64_t>( sv.size() ) };
        } else if constexpr ( is_numpy_copy<U>::value ) {
          storage = ArrRes{ static_cast<const void*>( v.data().data() ),
                            static_cast<std::int64_t>( v.size() ),
                            static_cast<std::int64_t>( v.shape().size() ),
                            v.shape().data() };
        } else {
          put_value<U>( &storage, v );
        }
      }
    };

    template<class... T>
    struct CbArg<std::tuple<T...>> {
      std::tuple<CbArg<T>...> elems;
      std::array<void*,sizeof...(T)> ptrs;
      template<class V>
      explicit CbArg( const V& v ) : CbArg( v, std::index_sequence_for<T...>{} ) {}
      template<class V, std::size_t... I>
      CbArg( const V& v, std::index_sequence<I...> ) : elems( std::get<I>(v)... )
      {
        std::apply( [this]( auto&... e ) { std::size_t i = 0; ( ( ptrs[i++] = e.ptr() ), ... ); }, elems );
      }
      CbArg( const CbArg& ) = delete;
      void * ptr() { return ptrs.data(); }
    };
    template<class A, class B>
    struct CbArg<std::pair<A,B>> : CbArg<std::tuple<A,B>> {
      using CbArg<std::tuple<A,B>>::CbArg;
    };
    template<class T, std::size_t N>
    struct CbArg<std::array<T,N>> {
      std::array<std::unique_ptr<CbArg<T>>,N> elems;
      std::array<void*,N> ptrs;
      explicit CbArg( const std::array<T,N>& v )
      {
        for ( std::size_t i = 0; i < N; ++i ) {
          elems[i] = std::make_unique<CbArg<T>>( v[i] );
          ptrs[i] = elems[i]->ptr();
        }
      }
      void * ptr() { return ptrs.data(); }
    };
    template<class T> struct CbArg<const T> : CbArg<T> { using CbArg<T>::CbArg; };
    template<class T> struct CbArg<T&> : CbArg<std::remove_const_t<T>> { using CbArg<std::remove_const_t<T>>::CbArg; };

    template<class T>
    int assign_string( void * dest, const void * data, std::int64_t n ) noexcept
    {
      try {
        static_cast<std::string*>(dest)->assign( static_cast<const char*>(data), static_cast<std::size_t>(n) );
        return 0;
      } catch ( ... ) {
        return 1;
      }
    }
    template<class V>
    int assign_vector( void * dest, const void * data, std::int64_t n ) noexcept
    {
      try {
        using E = typename V::value_type;
        auto d = static_cast<const E*>(data);
        static_cast<V*>(dest)->assign( d, d + n );
        return 0;
      } catch ( ... ) {
        return 1;
      }
    }

    //Storage of a callback return value:
    template<class T>
    struct CbRet {
      T value{};
      std::conditional_t<std::is_same_v<T,bool>, std::uint8_t,
      std::conditional_t<std::is_integral_v<T>, std::int64_t,
      std::conditional_t<std::is_floating_point_v<T>, double, AssignRes>>> storage{};
      CbRet()
      {
        if constexpr ( std::is_same_v<T,std::string> )
          storage = AssignRes{ &value, &assign_string<T> };
        else if constexpr ( is_vector<T>::value )
          storage = AssignRes{ &value, &assign_vector<T> };
      }
      CbRet( const CbRet& ) = delete;
      void * ptr() { return &storage; }
      T get()
      {
        if constexpr ( std::is_same_v<T,bool> )
          return storage != 0;
        else if constexpr ( std::is_arithmetic_v<T> )
          return static_cast<T>( storage );
        else
          return std::move( value );
      }
    };
    template<>
    struct CbRet<void> {
      void * ptr() { return nullptr; }
      void get() {}
    };
    template<class... T>
    struct CbRet<std::tuple<T...>> {
      std::tuple<CbRet<T>...> elems;
      std::array<void*,sizeof...(T)> ptrs;
      CbRet()
      {
        std::apply( [this]( auto&... e ) { std::size_t i = 0; ( ( ptrs[i++] = e.ptr() ), ... ); }, elems );
      }
      CbRet( const CbRet& ) = delete;
      void * ptr() { return ptrs.data(); }
      std::tuple<T...> get()
      {
        return std::apply( []( auto&... e ) { return std::tuple<T...>( e.get()... ); }, elems );
      }
    };
    template<class A, class B>
    struct CbRet<std::pair<A,B>> {
      CbRet<std::tuple<A,B>> t;
      void * ptr() { return t.ptr(); }
      std::pair<A,B> get() { auto v = t.get(); return { std::move(std::get<0>(v)), std::move(std::get<1>(v)) }; }
    };
    template<class T, std::size_t N>
    struct CbRet<std::array<T,N>> {
      std::array<CbRet<T>,N> elems;
      std::array<void*,N> ptrs;
      CbRet() { for ( std::size_t i = 0; i < N; ++i ) ptrs[i] = elems[i].ptr(); }
      CbRet( const CbRet& ) = delete;
      void * ptr() { return ptrs.data(); }
      std::array<T,N> get()
      {
        std::array<T,N> res;
        for ( std::size_t i = 0; i < N; ++i )
          res[i] = elems[i].get();
        return res;
      }
    };

  }//detail

  template<class R, class... Args>
  template<std::size_t... I>
  inline R Callback<R(Args...)>::invoke_impl( std::index_sequence<I...>, Args&... args ) const
  {
    auto& ctx = *m_core->ctx;
    auto tramp = ctx.trampoline.load();
    if ( !tramp || ctx.shutdown.load() )
      throw CallbackError("Python callback invoked after shutdown of the Python runtime");
    std::tuple<detail::CbArg<Args>...> cbargs( args... );
    (void)cbargs;
    std::array<void*,sizeof...(Args)+1> argptrs{ std::get<I>(cbargs).ptr()..., nullptr };
    detail::CbRet<R> res;
    if ( tramp( m_core->token, argptrs.data(), res.ptr() ) != 0 )
      throw CallbackError();
    return res.get();
  }

  template<class V>
  inline Arg::Arg( std::string n, const V& defval )
    : name(std::move(n)), default_json( detail::encode_value( defval ) )
  {
  }

  namespace detail {

    ////////////////////////////////////////////////////////////////////////////
    // Signatures of callables                                                //
    ////////////////////////////////////////////////////////////////////////////

    template<class R, class... A>
    struct sig {
      using ret = R;
      using args = std::tuple<A...>;
      static constexpr std::size_t nargs = sizeof...(A);
    };

    template<class F, class = void> struct fn_traits {
      static constexpr bool ok = false;
    };
    template<class R, class... A> struct fn_traits<R(*)(A...)> : sig<R,A...> {
      static constexpr bool ok = true; static constexpr bool is_member = false;
    };
    template<class R, class... A> struct fn_traits<R(*)(A...) noexcept> : fn_traits<R(*)(A...)> {};
#define SBLDPY_MEMFN_TRAITS(QUAL,SELFQUAL)                                    \
    template<class R, class C, class... A> struct fn_traits<R(C::*)(A...) QUAL> : sig<R,C SELFQUAL,A...> { \
      static constexpr bool ok = true; static constexpr bool is_member = true; using cls = C; \
    };                                                                        \
    template<class R, class C, class... A> struct fn_traits<R(C::*)(A...) QUAL noexcept> \
      : fn_traits<R(C::*)(A...) QUAL> {};
    SBLDPY_MEMFN_TRAITS(,&)
    SBLDPY_MEMFN_TRAITS(const,const &)
    SBLDPY_MEMFN_TRAITS(&,&)
    SBLDPY_MEMFN_TRAITS(const &,const &)
#undef SBLDPY_MEMFN_TRAITS

    //Lambdas and other functors (signature of operator(), without the object):
    template<class M> struct functor_sig;
#define SBLDPY_FUNCTOR_SIG(QUAL)                                              \
    template<class R, class C, class... A> struct functor_sig<R(C::*)(A...) QUAL> : sig<R,A...> {}; \
    template<class R, class C, class... A> struct functor_sig<R(C::*)(A...) QUAL noexcept> : sig<R,A...> {};
    SBLDPY_FUNCTOR_SIG()
    SBLDPY_FUNCTOR_SIG(const)
    SBLDPY_FUNCTOR_SIG(&)
    SBLDPY_FUNCTOR_SIG(const &)
#undef SBLDPY_FUNCTOR_SIG
    template<class F> struct fn_traits<F,std::void_t<decltype(&F::operator())>>
      : functor_sig<decltype(&F::operator())> {
      static constexpr bool ok = true; static constexpr bool is_member = false;
    };

    template<class F>
    constexpr void check_callable()
    {
      static_assert( fn_traits<F>::ok,
                     "sbldpy: could not deduce the signature of the callable (note that overloaded"
                     " functions must be selected with a cast, and generic lambdas are not supported)" );
    }

  }//detail

  namespace detail {
    template<class R, class... A>
    struct cb_describer<R(A...)> {
      static std::string describe( const Registry& reg )
      {
        static_assert( check_cbret<R>() );
        static_assert( ( check_cbarg<A>() && ... && true ) );
        std::string res = "{\"k\":\"cb\",\"args\":[";
        bool first = true;
        ( ( res += ( first ? "" : "," ), res += desc_cbarg<A>( reg ), first = false ), ... );
        return res + "],\"ret\":" + desc_cbret<R>( reg ) + "}";
      }
    };
  }

  ////////////////////////////////////////////////////////////////////////////
  // Module definition                                                      //
  ////////////////////////////////////////////////////////////////////////////

  namespace detail {

    using Entry = std::function<void(void**,void*)>;

    struct FuncDef {
      std::string name;
      std::string doc;
      std::vector<Arg> args;
      bool has_argnames = false;
      int id = -1;
      std::size_t nparams = 0;//excluding self
      std::function<std::string(const Registry&)> describe;//"[params],ret"
    };

    struct PropDef {
      std::string name;
      std::string doc;
      int get_id = -1;
      int set_id = -1;
      std::function<std::string(const Registry&)> describe;
    };

    struct ConstDef {
      std::string name;
      std::string value_json;
    };

    struct ClassDef {
      std::string name;
      std::string doc;
      int dtor_id = -1;
      std::vector<FuncDef> inits;
      std::vector<FuncDef> methods;
      std::vector<FuncDef> statics;
      std::vector<PropDef> props;
      std::vector<ConstDef> consts;
    };

    struct ModuleData {
      std::string name;
      std::string doc;
      Registry reg;
      std::vector<Entry> table;
      std::vector<FuncDef> functions;
      std::vector<std::unique_ptr<ClassDef>> classes;
      std::vector<ConstDef> consts;
      RuntimeCtx rt;
      int add_entry( Entry e )
      {
        table.push_back( std::move(e) );
        return static_cast<int>( table.size() - 1 );
      }
    };

    inline void apply_extras( FuncDef& fd, std::initializer_list<const Extra*> extras )
    {
      bool has_doc = false;
      for ( auto e : extras ) {
        if ( e->doc ) {
          if ( has_doc )
            throw std::logic_error( "documentation given twice for " + fd.name );
          has_doc = true;
          fd.doc = e->doc;
        }
        if ( e->has_args ) {
          if ( fd.has_argnames )
            throw std::logic_error( "argument names given twice for " + fd.name );
          fd.has_argnames = true;
          fd.args = e->args;
        }
      }
      if ( fd.has_argnames && fd.args.size() != fd.nparams )
        throw std::logic_error( "wrong number of argument names given for " + fd.name
                                + " (expected " + std::to_string( fd.nparams ) + ")" );
      bool seen_default = false;
      for ( auto& a : fd.args ) {
        if ( !a.default_json.empty() )
          seen_default = true;
        else if ( seen_default )
          throw std::logic_error( "arguments without default values can not follow arguments"
                                  " with default values in " + fd.name );
      }
    }

    template<class... A>
    std::string desc_param_list( const Registry& reg )
    {
      static_assert( ( check_param<A>() && ... && true ) );
      std::string res = "[";
      bool first = true;
      ( ( res += ( first ? "" : "," ), res += desc_param<A>( reg ), first = false ), ... );
      return res + "]";
    }

    template<class Tup> struct desc_param_list_tuple;
    template<class... A> struct desc_param_list_tuple<std::tuple<A...>> {
      static std::string get( const Registry& reg ) { return desc_param_list<A...>( reg ); }
    };
    template<class Tup> struct drop_first;
    template<> struct drop_first<std::tuple<>> { using type = std::tuple<>; };
    template<class A0, class... A> struct drop_first<std::tuple<A0,A...>> { using type = std::tuple<A...>; };

    //Parameter I of a callable (for methods, the object passed as the first
    //parameter is of class SelfT, while the callable might be a member
    //function of a base class):
    template<std::size_t I, class SelfT, class A>
    decltype(auto) get_arg( void * p, RuntimeCtx& ctx )
    {
      if constexpr ( I == 0 && !std::is_void_v<SelfT> )
        return static_cast<A>( *static_cast<SelfT*>( *static_cast<void**>(p) ) );
      else
        return get_param<A>( p, ctx );
    }

    //Creates table entry calling f with parameters of the types in Tup:
    template<class R, class Tup, class SelfT = void> struct Caller;
    template<class R, class... A, class SelfT>
    struct Caller<R,std::tuple<A...>,SelfT> {
      template<class F>
      static Entry make( F f, RuntimeCtx& ctx )
      {
        return [f = std::move(f), &ctx]( void ** args, void * res ) mutable
        {
          call( f, args, res, ctx, std::index_sequence_for<A...>{} );
        };
      }
      template<class F, std::size_t... I>
      static void call( F& f, void ** args, void * res, RuntimeCtx& ctx, std::index_sequence<I...> )
      {
        (void)args; (void)ctx;
        put_result<R>( res, [&]() -> R
        {
          return std::invoke( f, get_arg<I,SelfT,A>( args[I], ctx )... );
        } );
      }
    };

    template<bool is_method, class Traits>
    std::function<std::string(const Registry&)> describer()
    {
      using R = typename Traits::ret;
      static_assert( check_return<R,is_method>() );
      using Args = std::conditional_t<is_method,typename drop_first<typename Traits::args>::type,
                                      typename Traits::args>;
      return []( const Registry& reg )
      {
        return desc_param_list_tuple<Args>::get( reg ) + ",\"ret\":" + desc_return<R>( reg );
      };
    }

  }//detail

  template<class T> class Class;

  class Module {
  public:
    explicit Module( detail::ModuleData& d ) : m_d(d) {}

    Module& doc( const char * d ) { m_d.doc = d; return *this; }

    //Define function (any function pointer or lambda with supported parameter
    //and return types). Optionally followed by doc string and/or argument
    //names (e.g. { "a", sbldpy::arg("b",1.0) }):
    template<class F>
    Module& def( const char * name, F&& f, const Extra& e1 = Extra(), const Extra& e2 = Extra() )
    {
      using FD = std::decay_t<F>;
      detail::check_callable<FD>();
      using Tr = detail::fn_traits<FD>;
      static_assert( !Tr::is_member, "sbldpy: member functions must be defined on classes" );
      m_d.functions.push_back( make_funcdef<Tr,void>( name, FD( std::forward<F>(f) ), e1, e2 ) );
      return *this;
    }

    template<class T>
    Class<T> add_class( const char * name, const char * doc = nullptr );

    template<class V>
    Module& add_constant( const char * name, const V& value )
    {
      m_d.consts.push_back( detail::ConstDef{ name, detail::encode_value( value ) } );
      return *this;
    }

    //Implementation details:
    template<class Tr, class SelfT, class FD>
    detail::FuncDef make_funcdef( const char * name, FD f, const Extra& e1, const Extra& e2 )
    {
      constexpr bool is_method = !std::is_void_v<SelfT>;
      detail::FuncDef fd;
      fd.name = name;
      fd.nparams = Tr::nargs - ( is_method ? 1 : 0 );
      detail::apply_extras( fd, { &e1, &e2 } );
      fd.describe = detail::describer<is_method,Tr>();
      fd.id = m_d.add_entry( detail::Caller<typename Tr::ret,typename Tr::args,SelfT>::make( std::move(f), m_d.rt ) );
      return fd;
    }
    detail::ModuleData& data() { return m_d; }
  private:
    detail::ModuleData& m_d;
  };

  template<class T>
  class Class {
    static_assert( std::is_class_v<T> && detail::is_object_v<T>, "sbldpy: unsupported class type" );
  public:
    Class( Module& m, detail::ClassDef& d ) : m_m(m), m_d(d) {}

    //Constructor, calling T(A...):
    template<class... A>
    Class& def_init( const Extra& e1 = Extra(), const Extra& e2 = Extra() )
    {
      return def_init_impl<detail::sig<T,A...>>( []( A... a ) { return new T( static_cast<A>(a)... ); }, e1, e2 );
    }

    //Constructor, calling a factory function returning T by value:
    template<class F>
    Class& def_init( F&& f, const Extra& e1 = Extra(), const Extra& e2 = Extra() )
    {
      using FD = std::decay_t<F>;
      detail::check_callable<FD>();
      using Tr = detail::fn_traits<FD>;
      static_assert( !Tr::is_member && std::is_same_v<std::remove_cv_t<typename Tr::ret>,T>,
                     "sbldpy: factory functions must return the class by value" );
      return def_init_impl<Tr>( [f = FD(std::forward<F>(f))]( auto&&... a )
      {
        return new T( std::invoke( f, std::forward<decltype(a)>(a)... ) );
      }, e1, e2 );
    }

    //Method (member function pointer, or function/lambda taking T& or const
    //T& as first parameter):
    template<class F>
    Class& def( const char * name, F&& f, const Extra& e1 = Extra(), const Extra& e2 = Extra() )
    {
      using FD = std::decay_t<F>;
      check_method<FD>();
      m_d.methods.push_back( m_m.template make_funcdef<detail::fn_traits<FD>,T>( name, FD( std::forward<F>(f) ), e1, e2 ) );
      return *this;
    }

    //Static method:
    template<class F>
    Class& def_static( const char * name, F&& f, const Extra& e1 = Extra(), const Extra& e2 = Extra() )
    {
      using FD = std::decay_t<F>;
      detail::check_callable<FD>();
      using Tr = detail::fn_traits<FD>;
      static_assert( !Tr::is_member, "sbldpy: static methods can not be member functions" );
      m_d.statics.push_back( m_m.template make_funcdef<Tr,void>( name, FD( std::forward<F>(f) ), e1, e2 ) );
      return *this;
    }

    //Read-only property (getter is a method without parameters, or a
    //pointer to a data member):
    template<class G>
    Class& def_property( const char * name, G&& getter, const char * doc = nullptr )
    {
      add_property( name, std::forward<G>(getter), nullptr, doc );
      return *this;
    }

    //Read-write property (setter is a method with one parameter, or a pointer
    //to a data member):
    template<class G, class S>
    Class& def_property( const char * name, G&& getter, S&& setter, const char * doc = nullptr )
    {
      static_assert( !std::is_same_v<std::decay_t<S>,const char*> && !std::is_same_v<std::decay_t<S>,char*>,
                     "sbldpy: invalid setter" );
      add_property( name, std::forward<G>(getter), std::forward<S>(setter), doc );
      return *this;
    }

    template<class V>
    Class& add_constant( const char * name, const V& value )
    {
      m_d.consts.push_back( detail::ConstDef{ name, detail::encode_value( value ) } );
      return *this;
    }

  private:
    Module& m_m;
    detail::ClassDef& m_d;

    template<class Tr, class F>
    Class& def_init_impl( F f, const Extra& e1, const Extra& e2 )
    {
      using Args = typename Tr::args;
      detail::FuncDef fd;
      fd.name = m_d.name + ".__init__";
      fd.nparams = Tr::nargs;
      detail::apply_extras( fd, { &e1, &e2 } );
      fd.name = "__init__";
      fd.describe = []( const detail::Registry& reg )
      {
        return detail::desc_param_list_tuple<Args>::get( reg ) + ",\"ret\":" + detail::desc_return<T>( reg );
      };
      fd.id = m_m.data().add_entry( make_ctor_entry<Args>( std::move(f) ) );
      m_d.inits.push_back( std::move(fd) );
      return *this;
    }

    template<class Args, class F>
    detail::Entry make_ctor_entry( F f )
    {
      auto& ctx = m_m.data().rt;
      return [f = std::move(f), &ctx]( void ** args, void * res ) mutable
      {
        ctor_call<Args>( f, args, res, ctx, std::make_index_sequence<std::tuple_size_v<Args>>{} );
      };
    }
    template<class Args, class F, std::size_t... I>
    static void ctor_call( F& f, void ** args, void * res, detail::RuntimeCtx& ctx, std::index_sequence<I...> )
    {
      (void)args; (void)ctx;
      T * obj = f( detail::get_param<std::tuple_element_t<I,Args>>( args[I], ctx )... );
      *static_cast<void**>(res) = static_cast<void*>( obj );
    }

    template<class FD>
    static constexpr void check_method()
    {
      detail::check_callable<FD>();
      using Tr = detail::fn_traits<FD>;
      static_assert( Tr::nargs >= 1, "sbldpy: methods must take the object as first parameter" );
      using Self = std::tuple_element_t<0,typename Tr::args>;
      static_assert( std::is_lvalue_reference_v<Self>
                     && std::is_base_of_v<detail::rmcvref_t<Self>,T>,
                     "sbldpy: methods must be member functions of the class, or take the object as"
                     " first parameter (as T& or const T&)" );
    }

    template<class G, class S>
    void add_property( const char * name, G&& getter, S&& setter, const char * doc )
    {
      detail::PropDef pd;
      pd.name = name;
      pd.doc = doc ? doc : "";
      using GD = std::decay_t<G>;
      if constexpr ( std::is_member_object_pointer_v<GD> ) {
        add_property( name, [g = GD(getter)]( const T& o ) { return o.*g; }, std::forward<S>(setter), doc );
        return;
      } else if constexpr ( std::is_member_object_pointer_v<std::decay_t<S>> ) {
        using SD = std::decay_t<S>;
        using V = detail::rmcvref_t<decltype(std::declval<T&>().*std::declval<SD>())>;
        add_property( name, std::forward<G>(getter),
                      [s = SD(setter)]( T& o, const V& v ) { o.*s = v; }, doc );
        return;
      } else {
        check_method<GD>();
        using TrG = detail::fn_traits<GD>;
        static_assert( TrG::nargs == 1, "sbldpy: property getters can not take parameters" );
        using Ret = typename TrG::ret;
        static_assert( !std::is_void_v<Ret>, "sbldpy: property getters must return a value" );
        pd.get_id = m_m.data().add_entry( detail::Caller<Ret,typename TrG::args,T>::make( GD( std::forward<G>(getter) ),
                                                                                         m_m.data().rt ) );
        if constexpr ( !std::is_same_v<std::decay_t<S>,std::nullptr_t> ) {
          using SD = std::decay_t<S>;
          check_method<SD>();
          using TrS = detail::fn_traits<SD>;
          static_assert( TrS::nargs == 2, "sbldpy: property setters must take exactly one parameter" );
          pd.set_id = m_m.data().add_entry( detail::Caller<typename TrS::ret,typename TrS::args,T>::make( SD( std::forward<S>(setter) ),
                                                                                                      m_m.data().rt ) );
          using VS = std::tuple_element_t<1,typename TrS::args>;
          static_assert( detail::check_param<VS>() );
          static_assert( detail::check_return<Ret,true>() );
          pd.describe = []( const detail::Registry& reg )
          {
            return "\"type\":" + detail::desc_return<Ret>( reg ) + ",\"settype\":" + detail::desc_param<VS>( reg );
          };
        } else {
          static_assert( detail::check_return<Ret,true>() );
          pd.describe = []( const detail::Registry& reg )
          {
            return "\"type\":" + detail::desc_return<Ret>( reg );
          };
        }
        m_d.props.push_back( std::move(pd) );
      }
    }
  };

  template<class T>
  inline Class<T> Module::add_class( const char * name, const char * doc )
  {
    auto& reg = m_d.reg;
    if ( reg.classnames.count( typeid(T) ) )
      throw std::logic_error( std::string("class added twice: ") + name );
    reg.classnames[typeid(T)] = name;
    m_d.classes.push_back( std::make_unique<detail::ClassDef>() );
    auto& cd = *m_d.classes.back();
    cd.name = name;
    cd.doc = doc ? doc : "";
    cd.dtor_id = m_d.add_entry( []( void ** args, void * )
    {
      delete static_cast<T*>( *static_cast<void**>( args[0] ) );
    } );
    return Class<T>( *this, cd );
  }

  namespace detail {

    inline std::string json_args( const FuncDef& fd )
    {
      std::string res = "[";
      for ( std::size_t i = 0; i < fd.args.size(); ++i ) {
        if ( i )
          res += ',';
        res += "{\"name\":" + json_str( fd.args[i].name );
        if ( !fd.args[i].default_json.empty() )
          res += ",\"default\":" + fd.args[i].default_json;
        res += '}';
      }
      return res + "]";
    }

    inline std::string json_func( const FuncDef& fd, const Registry& reg )
    {
      return ( "{\"name\":" + json_str( fd.name ) + ",\"id\":" + std::to_string( fd.id )
               + ",\"doc\":" + json_str( fd.doc )
               + ( fd.has_argnames ? ",\"argnames\":" + json_args( fd ) : std::string() )
               + ",\"params\":" + fd.describe( reg ) + "}" );
    }

    template<class V, class F>
    std::string json_list( const V& v, F f )
    {
      std::string res = "[";
      bool first = true;
      for ( auto& e : v ) {
        if ( !first )
          res += ',';
        first = false;
        res += f( e );
      }
      return res + "]";
    }

    inline std::string json_consts( const std::vector<ConstDef>& v )
    {
      return json_list( v, []( const ConstDef& c )
      {
        return "{\"name\":" + json_str( c.name ) + ",\"value\":" + c.value_json + "}";
      } );
    }

    //Hidden table entries used by the Python runtime:
    inline void add_hidden_entries( ModuleData& d, int& id_init, int& id_shutdown )
    {
      RuntimeCtx * ctx = &d.rt;
      id_init = d.add_entry( [ctx]( void ** args, void * )
      {
        ctx->trampoline.store( reinterpret_cast<TrampolineFct>( *static_cast<void**>( args[0] ) ) );
        ctx->release.store( reinterpret_cast<ReleaseFct>( *static_cast<void**>( args[1] ) ) );
        ctx->shutdown.store( false );
      } );
      id_shutdown = d.add_entry( [ctx]( void **, void * ) { ctx->shutdown.store( true ); } );
    }

    struct ModuleState {
      ModuleData data;
      std::string info;
      std::string init_error;
      ModuleState( const char * name, void(*deffct)(Module&) )
      {
        try {
          data.name = name;
          Module m( data );
          deffct( m );
          int id_init, id_shutdown;
          add_hidden_entries( data, id_init, id_shutdown );
          const Registry& reg = data.reg;
          auto jf = [&reg]( const FuncDef& fd ) { return json_func( fd, reg ); };
          std::string classes = json_list( data.classes, [&reg,&jf]( const std::unique_ptr<ClassDef>& c )
          {
            std::string props = json_list( c->props, [&reg]( const PropDef& p )
            {
              return ( "{\"name\":" + json_str( p.name ) + ",\"doc\":" + json_str( p.doc )
                       + ",\"get\":" + std::to_string( p.get_id ) + ",\"set\":" + std::to_string( p.set_id )
                       + "," + p.describe( reg ) + "}" );
            } );
            return ( "{\"name\":" + json_str( c->name ) + ",\"doc\":" + json_str( c->doc )
                     + ",\"dtor\":" + std::to_string( c->dtor_id )
                     + ",\"inits\":" + json_list( c->inits, jf )
                     + ",\"methods\":" + json_list( c->methods, jf )
                     + ",\"staticmethods\":" + json_list( c->statics, jf )
                     + ",\"properties\":" + props
                     + ",\"constants\":" + json_consts( c->consts ) + "}" );
          } );
          info = ( "{\"sbldpy_version\":" + std::to_string( SBLDPY_VERSION )
                   + ",\"sbldpy_version_str\":" + json_str( SBLDPY_VERSION_STR )
                   + ",\"abi_version\":" + std::to_string( SBLDPY_ABI_VERSION )
                   + ",\"module\":" + json_str( data.name )
                   + ",\"doc\":" + json_str( data.doc )
                   + ",\"functions\":" + json_list( data.functions, jf )
                   + ",\"classes\":" + classes
                   + ",\"constants\":" + json_consts( data.consts )
                   + ",\"hidden\":{\"init_runtime\":" + std::to_string( id_init )
                   + ",\"shutdown\":" + std::to_string( id_shutdown ) + "}}" );
        } catch ( std::exception& e ) {
          init_error = std::string("Error in definition of sbldpy module ") + name + ": " + e.what();
        } catch ( ... ) {
          init_error = std::string("Unknown error in definition of sbldpy module ") + name;
        }
      }
    };

    //Error codes returned by sbldpy_call:
    enum ErrorCode : int { EC_OK = 0, EC_RUNTIME = 1, EC_VALUE = 2, EC_INDEX = 3, EC_TYPE = 4,
                           EC_OVERFLOW = 5, EC_MEMORY = 6, EC_CALLBACK = 7 };

    inline const char * module_info( ModuleState& s ) noexcept
    {
      if ( !s.init_error.empty() ) {
        last_error() = s.init_error;
        return nullptr;
      }
      return s.info.c_str();
    }

    inline int module_call( ModuleState& s, int id, void ** args, void * result ) noexcept
    {
      try {
        if ( !s.init_error.empty() )
          throw std::runtime_error( s.init_error );
        if ( id < 0 || static_cast<std::size_t>(id) >= s.data.table.size() )
          throw std::runtime_error( "sbldpy: invalid function id" );
        s.data.table[static_cast<std::size_t>(id)]( args, result );
        return EC_OK;
      } catch ( CallbackError& e ) {
        last_error() = e.what();
        return EC_CALLBACK;
      } catch ( TypeError& e ) {
        last_error() = e.what();
        return EC_TYPE;
      } catch ( std::invalid_argument& e ) {
        last_error() = e.what();
        return EC_VALUE;
      } catch ( std::domain_error& e ) {
        last_error() = e.what();
        return EC_VALUE;
      } catch ( std::length_error& e ) {
        last_error() = e.what();
        return EC_VALUE;
      } catch ( std::out_of_range& e ) {
        last_error() = e.what();
        return EC_INDEX;
      } catch ( std::overflow_error& e ) {
        last_error() = e.what();
        return EC_OVERFLOW;
      } catch ( std::bad_alloc& ) {
        return EC_MEMORY;
      } catch ( std::exception& e ) {
        try { last_error() = e.what(); } catch ( ... ) {}
        return EC_RUNTIME;
      } catch ( ... ) {
        try { last_error() = "unknown C++ exception"; } catch ( ... ) {}
        return EC_RUNTIME;
      }
    }

    inline const char * get_last_error() noexcept
    {
      return last_error().c_str();
    }

    constexpr bool streq( const char * a, const char * b )
    {
      return *a == *b && ( *a == '\0' || streq( a + 1, b + 1 ) );
    }

  }//detail

}//sbldpy

#define SBLDPY_STRINGIFY_( x ) #x
#define SBLDPY_STRINGIFY( x ) SBLDPY_STRINGIFY_( x )

//When building with simplebuild, the module name is checked against the
//directory name (pymod_<name>):
#ifdef SBLDPY_MODNAME
#  define SBLDPY_CHECK_MODNAME( modname )                                     \
  static_assert( ::sbldpy::detail::streq( #modname, SBLDPY_STRINGIFY(SBLDPY_MODNAME) ), \
                 "sbldpy: name in SBLDPY_MODULE(..) must match the name of the module" );
#else
#  define SBLDPY_CHECK_MODNAME( modname )
#endif

//Defines the module. Must be used exactly once per module, followed by a
//block of code defining the contents of the module (available as "m"):
#define SBLDPY_MODULE( modname )                                              \
  SBLDPY_CHECK_MODNAME( modname )                                             \
  static void sbldpy_define_module_impl( ::sbldpy::Module& );                 \
  static ::sbldpy::detail::ModuleState& sbldpy_module_state()                 \
  {                                                                           \
    static ::sbldpy::detail::ModuleState s( #modname, &sbldpy_define_module_impl ); \
    return s;                                                                 \
  }                                                                           \
  extern "C" SBLDPY_EXPORT int sbldpy_abi_version()                           \
  {                                                                           \
    return SBLDPY_ABI_VERSION;                                                \
  }                                                                           \
  extern "C" SBLDPY_EXPORT const char * sbldpy_module_info()                  \
  {                                                                           \
    return ::sbldpy::detail::module_info( sbldpy_module_state() );            \
  }                                                                           \
  extern "C" SBLDPY_EXPORT int sbldpy_call( int id, void ** args, void * result ) \
  {                                                                           \
    return ::sbldpy::detail::module_call( sbldpy_module_state(), id, args, result ); \
  }                                                                           \
  extern "C" SBLDPY_EXPORT const char * sbldpy_last_error()                   \
  {                                                                           \
    return ::sbldpy::detail::get_last_error();                                \
  }                                                                           \
  static void sbldpy_define_module_impl( ::sbldpy::Module& m )

#endif

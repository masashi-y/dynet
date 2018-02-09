%module dynet_swig

%{
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "param-init.h"
#include "model.h"
#include "tensor.h"
#include "dynet.h"
#include "training.h"
#include "expr.h"
#include "rnn.h"
#include "lstm.h"
#include "gru.h"
#include "fast-lstm.h"
#include "io.h"
%}

/* %include std_vector.i */
/* %include std_string.i */
/* %include std_pair.i */
%include cpointer.i
%include stl.i
%include exception.i
%include std_except.i

// Convert C++ exceptions into Java exceptions. This provides
// nice error messages for each listed exception, and a default
// "unknown error" message for all others.
/* %catches(std::invalid_argument, ...); */

namespace std {
  %template(IntVector)                    vector<int>;
  %template(UnsignedVector)               vector<unsigned>;
  %template(DoubleVector)                 vector<double>;
  %template(FloatVector)                  vector<float>;
  %template(LongVector)                   vector<long>;
  %template(StringVector)                 vector<std::string>;
  %template(ExpressionVector)             vector<dynet::Expression>;
  %template(ParameterStorageVector)       vector<dynet::ParameterStorage*>;
  %template(LookupParameterStorageVector) vector<dynet::LookupParameterStorage*>;
  %template(ExpressionVectorVector)       vector<vector<dynet::Expression>>;
  %template(ParameterVector)              vector<dynet::Parameter>;
  %template(ParameterVectorVector)        vector<vector<dynet::Parameter>>;
}

namespace dynet {

// Some declarations etc to keep swig happy
typedef float real;
typedef int RNNPointer;
struct VariableIndex;
/*{
  unsigned t;
  explicit VariableIndex(const unsigned t_): t(t_) {};
};*/
struct Tensor;
struct Node;
struct ParameterStorage;
struct LookupParameterStorage;

///////////////////////////////////
// declarations from dynet/dim.h //
///////////////////////////////////

%rename(get) Dim::operator[];

struct Dim {
    Dim() : nd(0), bd(1) {}
    Dim(const std::vector<long> & x);
    Dim(const std::vector<long> & x, unsigned int b);

    unsigned int size();
    unsigned int batch_size();
    unsigned int sum_dims();

    Dim truncate();
    Dim single_batch();

    void resize(unsigned int i);
    unsigned int ndims();
    unsigned int rows();
    unsigned int cols();
    unsigned int batch_elems();
    void set(unsigned int i, unsigned int s);
    unsigned int operator[](unsigned int i);
    unsigned int size(unsigned int i);

    void delete_dim(unsigned int i);

    Dim transpose();
};

////////////////////////////////////
// declarations from dynet/init.h //
////////////////////////////////////

struct DynetParams {
  DynetParams();
  ~DynetParams();
  unsigned random_seed; /**< The seed for random number generation */
  std::string mem_descriptor; /**< Total memory to be allocated for Dynet */
  float weight_decay; /**< Weight decay rate for L2 regularization */
  int autobatch; /**< Whether to autobatch or not */
  int profiling; /**< Whether to show autobatch debug info or not */
  bool shared_parameters; /**< TO DOCUMENT */
  bool ngpus_requested; /**< GPUs requested by number */
  bool ids_requested; /**< GPUs requested by ids */
  bool cpu_requested; /**< CPU requested in multi-device case */
  int requested_gpus; /**< Number of requested GPUs */
  std::vector<int> gpu_mask; /**< List of required GPUs by ids */
};


/* %ignore initialize; */
/* %rename ("dynet_initialize") initialize(int& argc, char**& argv, bool shared_parameters = false); */
void initialize(DynetParams& params);
void initialize(int& argc, char**& argv, bool shared_parameters = false);
void cleanup();

}

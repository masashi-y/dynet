%module dynet_swig

%{
#include <sstream>
#include <iostream>
#include <sstream>
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

struct Dim {
    /* Dim() : nd(0), bd(1) {} */
    /* Dim(const std::vector<long> & x); */
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

/////////////////////////////////////
// declarations from dynet/model.h //
/////////////////////////////////////

class ParameterCollection;
struct Parameter {
  Parameter();
  void zero();

  Dim dim();
  Tensor* values();

  void set_updated(bool b);
  bool is_updated();

};

struct LookupParameter {
  LookupParameter();
  void initialize(unsigned index, const std::vector<float>& val) const;
  void zero();
  Dim dim();
  std::vector<Tensor>* values();
  void set_updated(bool b);
  bool is_updated();
};

struct ParameterInit {
  ParameterInit() {}
  virtual ~ParameterInit() {}
  virtual void initialize_params(Tensor & values) const = 0;
};

struct ParameterInitNormal : public ParameterInit {
  ParameterInitNormal(float m = 0.0f, float v = 1.0f) : mean(m), var(v) {}
  virtual void initialize_params(Tensor& values) const override;
 private:
  float mean, var;
};

struct ParameterInitUniform : public ParameterInit {
  ParameterInitUniform(float scale) :
    left(-scale), right(scale) { assert(scale != 0.0f); }
  ParameterInitUniform(float l, float r) : left(l), right(r) { assert(l != r); }
  virtual void initialize_params(Tensor & values) const override;
 private:
  float left, right;
};

struct ParameterInitConst : public ParameterInit {
  ParameterInitConst(float c) : cnst(c) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float cnst;
};

struct ParameterInitIdentity : public ParameterInit {
  ParameterInitIdentity() {}
  virtual void initialize_params(Tensor & values) const override;
};

struct ParameterInitGlorot : public ParameterInit {
  ParameterInitGlorot(bool is_lookup = false) : lookup(is_lookup) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  bool lookup;
};

/* I AM NOT ACTUALLY IMPLEMENTED IN THE DYNET CODE
struct ParameterInitSaxe : public ParameterInit {
  ParameterInitSaxe() {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float cnst;
};
*/

struct ParameterInitFromFile : public ParameterInit {
  ParameterInitFromFile(std::string f) : filename(f) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::string filename;
};

struct ParameterInitFromVector : public ParameterInit {
  ParameterInitFromVector(std::vector<float> v) : vec(v) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::vector<float> vec;
};


struct ParameterStorageBase {
  virtual void scale_parameters(float a) = 0;
  virtual void zero() = 0;
  virtual void squared_l2norm(float* sqnorm) const = 0;
  virtual void g_squared_l2norm(float* sqnorm) const = 0;
  virtual size_t size() const = 0;
  virtual ~ParameterStorageBase();
};

%nodefaultctor ParameterStorage;
struct ParameterStorage : public ParameterStorageBase {
  void scale_parameters(float a) override;
  void zero() override;
  void squared_l2norm(float* sqnorm) const override;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;

  void copy(const ParameterStorage & val);
  void accumulate_grad(const Tensor& g);
  void clear();

  Dim dim;
  Tensor values;
  Tensor g;
};

%nodefaultctor LookupParameterStorage;
struct LookupParameterStorage : public ParameterStorageBase {
  void scale_parameters(float a) override;
  void zero() override;
  void squared_l2norm(float* sqnorm) const override;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;
  void initialize(unsigned index, const std::vector<float>& val);

  void copy(const LookupParameterStorage & val);
  void accumulate_grad(unsigned index, const Tensor& g);
  void accumulate_grads(unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
  void clear();

  // Initialize each individual lookup from the overall tensors
  void initialize_lookups();
};

// extra code for parameter lookup
%extend ParameterCollection {
   // SWIG can't get the types right for `parameters_list`, so here are replacement methods
   // for which it can. (You might worry that these would cause infinite recursion, but
   // apparently they don't.
   std::vector<std::shared_ptr<ParameterStorage>> parameters_list() const {
     return $self->parameters_list();
   }

   std::vector<std::shared_ptr<LookupParameterStorage>> lookup_parameters_list() const {
     return $self->lookup_parameters_list();
   }
};

class ParameterCollection {
 public:
  ParameterCollection();
  ~ParameterCollection();
  float gradient_l2_norm() const;
  void reset_gradient();

  Parameter add_parameters(const Dim& d, float scale = 0.0f);
  Parameter add_parameters(const Dim& d, const ParameterInit & init);
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d);
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d, const ParameterInit & init);

  void project_weights(float radius = 1.0f);
  void set_weight_decay_lambda(float lambda);

  size_t parameter_count() const;
  size_t updated_parameter_count() const;
};

//////////////////////////////////////
// declarations from dynet/tensor.h //
//////////////////////////////////////

struct Tensor {
  Dim d;
  float* v;
};

real as_scalar(const Tensor& t);
std::vector<real> as_vector(const Tensor& v);

struct TensorTools {
  static float access_element(const Tensor& v, const Dim& index);
};

////////////////////////////////////////
// declarations from dynet/training.h //
////////////////////////////////////////

// Need to disable constructor as SWIG gets confused otherwise
%nodefaultctor Trainer;
struct Trainer {
  virtual void update();
  //void update(const std::vector<unsigned> & updated_params, const std::vector<unsigned> & updated_lookup_params);
  void update_epoch(real r = 1.0);

  virtual void restart() = 0;
  void restart(real lr);

  float clip_gradients();
  void rescale_and_reset_weight_decay();

  real learning_rate;

  bool clipping_enabled;
  real clip_threshold;
  real clips;
  real updates;

  real clips_since_status;
  real updates_since_status;

  bool sparse_updates_enabled;
  unsigned aux_allocated;
  unsigned aux_allocated_lookup;

  void status();

  ParameterCollection* model;
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(ParameterCollection& m, real learning_rate = 0.1);
  void restart() override;
};

struct CyclicalSGDTrainer : public Trainer {
  explicit CyclicalSGDTrainer(ParameterCollection& m, float learning_rate_min = 0.01, float learning_rate_max = 0.1, float step_size = 2000, float gamma = 0.0, float edecay = 0.0);
  void restart() override;
  void update() override;
};

struct MomentumSGDTrainer : public Trainer {
  explicit MomentumSGDTrainer(ParameterCollection& m, real learning_rate = 0.01, real mom = 0.9);
  void restart() override;
};

struct AdagradTrainer : public Trainer {
  explicit AdagradTrainer(ParameterCollection& m, real learning_rate = 0.1, real eps = 1e-20);
  void restart() override;
};

struct AdadeltaTrainer : public Trainer {
  explicit AdadeltaTrainer(ParameterCollection& m, real eps = 1e-6, real rho = 0.95);
  void restart() override;
};

struct RMSPropTrainer : public Trainer {
   explicit RMSPropTrainer(ParameterCollection& m, real learning_rate = 0.1, real eps = 1e-20, real rho = 0.95);
   void restart() override;
};

struct AdamTrainer : public Trainer {
  explicit AdamTrainer(ParameterCollection& m, float learning_rate = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8);
  void restart() override;
};

struct AmsgradTrainer : public Trainer {
  explicit AmsgradTrainer(ParameterCollection& m, float learning_rate = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8);
  void restart() override;
};

//struct EGTrainer : public Trainer {
//  explicit EGTrainer(ParameterCollection& mod, real learning_rate = 0.1, real mom = 0.9, real ne = 0.0);
//  void enableCyclicalLR(float _learning_rate_min = 0.01, float _learning_rate_max = 0.1, float _step_size = 2000, float _gamma = 0.0);
//  void update() override;
//  void restart() override;
//};

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

%{
std::string dim_show(dynet::Dim& d) {
    std::stringstream os;
    os << d;
    return os.str();
}
%}
std::string dim_show(dynet::Dim& d);

%{
std::string tensor_show(dynet::Tensor& d) {
    std::stringstream os;
    os << d;
    return os.str();
}
%}
std::string tensor_show(dynet::Tensor& d);


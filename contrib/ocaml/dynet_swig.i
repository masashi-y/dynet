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
%catches(std::invalid_argument, ...);   /* does not work !!!!!!! why? */


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

/////////////////////////////////////
// declarations from dynet/nodes.h //
/////////////////////////////////////

struct Sum;
struct LogSumExp;
struct AffineTransform;
struct Concatenate;
struct Average;

////////////////////////////////////
// declarations from dynet/expr.h //
////////////////////////////////////


struct ComputationGraph;

struct Expression {
  ComputationGraph *pg;
  VariableIndex i;
  Expression(ComputationGraph *pg, VariableIndex i) : pg(pg), i(i) { };
  const Tensor& value();
  const Dim& dim() const { return pg->get_dimension(i); }
};

// These templates get used to instantiate operations on vector<Expression>
namespace detail {
template <typename F, typename T> Expression f(const T& xs);

template <typename F, typename T, typename T1>
Expression f(const T& xs, const T1& arg1);
}

/* INPUT OPERATIONS */

%rename(parameter_Parameter) parameter(ComputationGraph& g, Parameter p);
%rename(parameter_LookupParameter) parameter(ComputationGraph& g, LookupParameter lp);

%rename(const_parameter_Parameter) const_parameter(ComputationGraph& g, Parameter p);
%rename(const_parameter_LookupParameter) const_parameter(ComputationGraph& g, LookupParameter lp);

%rename(lookup_pindex) lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);
%rename(const_lookup_pindex) const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);
%rename(lookup_vector) lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);
%rename(const_lookup_vector) const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);

%rename(input_scalar) input(ComputationGraph& g, real s);
%rename(input_sparse) input(ComputationGraph& g, const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, float defdata = 0.f);

Expression input(ComputationGraph& g, real s);
/* Expression input(ComputationGraph& g, const real *ps); */
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>* pdata);
Expression input(ComputationGraph& g, const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, float defdata = 0.f);

Expression parameter(ComputationGraph& g, Parameter p);
Expression parameter(ComputationGraph& g, LookupParameter lp);
Expression const_parameter(ComputationGraph& g, Parameter p);
Expression const_parameter(ComputationGraph& g, LookupParameter lp);
Expression lookup(ComputationGraph& g, LookupParameter p, unsigned index);
Expression lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);
Expression const_lookup(ComputationGraph& g, LookupParameter p, unsigned index);
Expression const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);
Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);
//Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);
Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);
//Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);

Expression zeros(ComputationGraph& g, const Dim& d);
Expression zeroes(ComputationGraph& g, const Dim& d);
Expression ones(ComputationGraph& g, const Dim& d);
Expression constant(ComputationGraph& g, const Dim& d, float val);
Expression random_normal(ComputationGraph& g, const Dim& d);
Expression random_bernoulli(ComputationGraph& g, const Dim& d, real p, real scale = 1.0f);
Expression random_uniform(ComputationGraph& g, const Dim& d, real left, real right);
Expression random_gumbel(ComputationGraph& g, const Dim& d, real mu = 0.0, real beta = 1.0);

/* ARITHMETIC OPERATIONS */

// Rename operators to valid Java function names
%rename(exprNeg) operator-(const Expression& x);
%rename(exprPlusExEx) operator+(const Expression& x, const Expression& y);
%rename(exprPlusExRe) operator+(const Expression& x, real y);
%rename(exprPlusReEx) operator+(real x, const Expression& y);
%rename(exprMinusExEx) operator-(const Expression& x, const Expression& y);
%rename(exprMinusReEx) operator-(real x, const Expression& y);
%rename(exprMinusExRe) operator-(const Expression& x, real y);
%rename(exprTimesExEx) operator*(const Expression& x, const Expression& y);
%rename(exprTimesReEx) operator*(const Expression& x, float y);
%rename(exprTimesExRe) operator*(float y, const Expression& x); // { return x * y; }
%rename(exprDivide) operator/(const Expression& x, float y); // { return x * (1.f / y); }

Expression operator-(const Expression& x);
Expression operator+(const Expression& x, const Expression& y);
Expression operator+(const Expression& x, real y);
Expression operator+(real x, const Expression& y);
Expression operator-(const Expression& x, const Expression& y);
Expression operator-(real x, const Expression& y);
Expression operator-(const Expression& x, real y);
Expression operator*(const Expression& x, const Expression& y);
Expression operator*(const Expression& x, float y);
Expression operator*(float y, const Expression& x); // { return x * y; }
Expression operator/(const Expression& x, float y); // { return x * (1.f / y); }

%template(affine_transform) detail::f<AffineTransform, std::vector<Expression>>;
%template(sum) detail::f<Sum, std::vector<Expression>>;

Expression sum_elems(const Expression& x);
Expression moment_elems(const Expression& x, unsigned r);
Expression mean_elems(const Expression& x);
Expression std_elems(const Expression& x);

Expression sum_batches(const Expression& x);
Expression moment_batches(const Expression& x, unsigned r);
Expression mean_batches(const Expression& x);
Expression std_batches(const Expression& x);

Expression sum_dim(const Expression& x, const std::vector<unsigned>& dims, bool b=false);

// These are deprecated but kept for backward compatibility
Expression sum_rows(const Expression& x);
Expression sum_cols(const Expression& x);

Expression moment_dim(const Expression& x, const std::vector<unsigned>& dims, unsigned r, bool b=false, unsigned n=0);
Expression mean_dim(const Expression& x, const std::vector<unsigned>& dims, bool b=false, unsigned n=0);
Expression std_dim(const Expression& x, const std::vector<unsigned>& dims, bool b=false, unsigned n=0);

%template(average) detail::f<Average, std::vector<Expression>>;

Expression sqrt(const Expression& x);
Expression abs(const Expression& x);
Expression erf(const Expression& x);
Expression tanh(const Expression& x);
Expression exp(const Expression& x);
Expression square(const Expression& x);
Expression cube(const Expression& x);
Expression lgamma(const Expression& x);
Expression log(const Expression& x);
Expression logistic(const Expression& x);
Expression rectify(const Expression& x);
Expression elu(const Expression& x, float alpha=1.f);
Expression selu(const Expression& x);
Expression silu(const Expression& x, float beta=1.f);
Expression softsign(const Expression& x);
Expression pow(const Expression& x, const Expression& y);

Expression min(const Expression& x, const Expression& y);

// We need two overloaded versions of `max`, but apparently %template
// gets unhappy when you use it to overload a function, so we have to define
// the `ExpressionVector` version of `max` explicitly.
%{
namespace dynet {
Expression max(const std::vector<Expression>& xs) {
  return detail::f<Max, std::vector<Expression>>(xs);
};
}
%}

Expression max(const Expression& x, const Expression& y);
Expression max(const std::vector<Expression>& xs);
Expression dot_product(const Expression& x, const Expression& y);
Expression cmult(const Expression& x, const Expression& y);
Expression cdiv(const Expression& x, const Expression& y);
Expression colwise_add(const Expression& x, const Expression& bias);

/* PROBABILITY / LOSS OPERATIONS */

Expression softmax(const Expression& x);
Expression log_softmax(const Expression& x);
Expression log_softmax(const Expression& x, const std::vector<unsigned>& restriction);
Expression logsumexp_dim(const Expression& x, unsigned d);

%template(logsumexp) detail::f<LogSumExp, std::vector<Expression>>;

Expression pickneglogsoftmax(const Expression& x, unsigned v);
/* Expression pickneglogsoftmax(const Expression& x, const unsigned* pv); */
/* Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned>& v); */

Expression hinge(const Expression& x, unsigned index, float m = 1.0);
Expression hinge(const Expression& x, unsigned* pindex, float m = 1.0);
Expression hinge(const Expression& x, const std::vector<unsigned>& indices, float m = 1.0);

Expression hinge_dim(const Expression& x, const std::vector<unsigned>& indices, unsigned d = 0, float m = 1.0);
// Expression hinge_dim(const Expression& x, const std::vector<unsigned>* pindex, unsigned d = 0, float m = 1.0);
Expression hinge_dim(const Expression& x, const std::vector<std::vector<unsigned> >& indices, unsigned d = 0, float m = 1.0);
// Expression hinge_dim(const Expression& x, const std::vector<std::vector<unsigned> >* pindices, unsigned d = 0, float m = 1.0);

Expression sparsemax(const Expression& x);
Expression sparsemax_loss(const Expression& x, const std::vector<unsigned>& target_support);

Expression squared_norm(const Expression& x);
Expression l2_norm(const Expression& x);
Expression squared_distance(const Expression& x, const Expression& y);
Expression l1_distance(const Expression& x, const Expression& y);
Expression huber_distance(const Expression& x, const Expression& y, float c = 1.345f);
Expression binary_log_loss(const Expression& x, const Expression& y);
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m = 1.0);
Expression poisson_loss(const Expression& x, unsigned y);
Expression poisson_loss(const Expression& x, const unsigned* py);

/* FLOW / SHAPING OPERATIONS */

Expression nobackprop(const Expression& x);
Expression flip_gradient(const Expression& x);
Expression reshape(const Expression& x, const Dim& d);
Expression transpose(const Expression& x);
Expression select_rows(const Expression& x, const std::vector<unsigned> &rows);
Expression select_cols(const Expression& x, const std::vector<unsigned> &cols);

Expression pick(const Expression& x, unsigned v, unsigned d = 0);
Expression pick(const Expression& x, const std::vector<unsigned>& v, unsigned d = 0);
Expression pick(const Expression& x, const unsigned* v, unsigned d = 0);
Expression pick_range(const Expression& x, unsigned s, unsigned e, unsigned d = 0);
Expression pick_batch_elem(const Expression& x, unsigned v);
Expression pick_batch_elems(const Expression& x, const std::vector<unsigned> & v);

// Concatenate and ConcatenateCols got changed around, need to implement
// explicitly now.
%{
namespace dynet {
inline Expression concatenate_to_batch(const std::vector<Expression>& xs) {
  return detail::f<ConcatenateToBatch>(xs);
};

inline Expression concatenate_cols(const std::vector<Expression>& xs) {
  return detail::f<Concatenate>(xs, 1);
};

inline Expression concatenate(const std::vector<Expression>& xs, unsigned d = 0) {
  return detail::f<Concatenate>(xs, d);
};
}
%}

Expression concatenate_to_batch(const std::vector<Expression>& xs);
Expression concatenate_cols(const std::vector<Expression>& xs);
Expression concatenate(const std::vector<Expression>& xs, unsigned d);

Expression max_dim(const Expression& x, unsigned d = 0);
Expression min_dim(const Expression& x, unsigned d = 0);

/* NOISE OPERATIONS */

Expression noise(const Expression& x, real stddev);
Expression dropout(const Expression& x, real p);
Expression dropout_dim(const Expression& x, unsigned d, real p);
Expression dropout_batch(const Expression& x, real p);
Expression block_dropout(const Expression& x, real p);

/* CONVOLUTION OPERATIONS */

// These two were commented out in the C++ source code.
//Expression conv1d_narrow(const Expression& x, const Expression& f);
//Expression conv1d_wide(const Expression& x, const Expression& f);
Expression filter1d_narrow(const Expression& x, const Expression& f);
Expression kmax_pooling(const Expression& x, unsigned k);
Expression fold_rows(const Expression& x, unsigned nrows=2);
Expression average_cols(const Expression& x);
Expression kmh_ngram(const Expression& x, unsigned n);

Expression conv2d(const Expression& x, const Expression& f, const std::vector<unsigned>& stride, bool is_valid = true);
Expression conv2d(const Expression& x, const Expression& f, const Expression& b, const std::vector<unsigned>& stride, bool is_valid = true);
Expression maxpooling2d(const Expression& x, const std::vector<unsigned>& ksize, const std::vector<unsigned>& stride, bool is_valid = true);

/* TENSOR OPERATIONS */

Expression contract3d_1d(const Expression& x, const Expression& y);
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z);
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const Expression& b);
Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);

/* LINEAR ALGEBRA OPERATIONS */

Expression inverse(const Expression& x);
Expression logdet(const Expression& x);
Expression trace_of_product(const Expression& x, const Expression& y);

/* NORMALIZATION OPERATIONS */

Expression layer_norm(const Expression& x, const Expression& g, const Expression& b);
Expression weight_norm(const Expression& w, const Expression& g);

/////////////////////////////////////
// declarations from dynet/dynet.h //
/////////////////////////////////////

struct Device;

%rename(add_lookup_pindex) ComputationGraph::add_lookup(LookupParameter p, const unsigned* pindex);
%rename(add_lookup_pindces) ComputationGraph::add_lookup(LookupParameter p, const std::vector<unsigned>* pindices);

%rename(add_const_lookup_pindex) ComputationGraph::add_const_lookup(LookupParameter p, const unsigned* pindex);
%rename(add_const_lookup_pindces) ComputationGraph::add_const_lookup(LookupParameter p, const std::vector<unsigned>* pindices);

struct ComputationGraph {
  ComputationGraph();
  ~ComputationGraph();

  VariableIndex add_input(real s, Device* device);
  // VariableIndex add_input(const real* ps);
  VariableIndex add_input(const Dim& d, const std::vector<float>& data, Device* device);
  //VariableIndex add_input(const Dim& d, const std::vector<float>* pdata);
  VariableIndex add_input(const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, Device* device, float defdata = 0.f);

  VariableIndex add_parameters(Parameter p);
  VariableIndex add_const_parameters(Parameter p);
  VariableIndex add_lookup(LookupParameter p, const unsigned* pindex);
  VariableIndex add_lookup(LookupParameter p, unsigned index);
  VariableIndex add_lookup(LookupParameter p, const std::vector<unsigned>* pindices);
  // VariableIndex add_lookup(LookupParameter p, const std::vector<unsigned>& indices);
  VariableIndex add_const_lookup(LookupParameter p, const unsigned* pindex);
  VariableIndex add_const_lookup(LookupParameter p, unsigned index);
  VariableIndex add_const_lookup(LookupParameter p, const std::vector<unsigned>* pindices);
  // VariableIndex add_const_lookup(LookupParameter p, const std::vector<unsigned>& indices);

  void clear();
  void checkpoint();
  void revert();

  Dim& get_dimension(VariableIndex index) const;

  /* const Tensor& forward(const Expression& last); */
%extend {
  const Tensor forward_deref(const Expression& last) {
      return self->forward(last);
  }

  const Tensor incremental_forward_deref(const Expression& last) {
      return self->incremental_forward(last);
  }

  const Tensor get_value_deref(const Expression& e) {
      return self->get_value(e);
  }
};
  //const Tensor& forward(VariableIndex i);
  /* const Tensor& incremental_forward(const Expression& last); */
  //const Tensor& incremental_forward(VariableIndex i);
  //const Tensor& get_value(VariableIndex i);
  /* const Tensor& get_value(const Expression& e); */
  void invalidate();
  void backward(const Expression& last);
  //void backward(VariableIndex i);

  void print_graphviz() const;

  std::vector<Node*> nodes;
  std::vector<VariableIndex> parameter_nodes;
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

///////////////////////////////////
// declarations from dynet/rnn.h //
///////////////////////////////////

%nodefaultctor RNNBuilder;
struct RNNBuilder {
  RNNPointer state() const;
  void new_graph(ComputationGraph& cg, bool update = true);
  void start_new_sequence(const std::vector<Expression>& h_0 = {});
  Expression set_h(const RNNPointer& prev, const std::vector<Expression>& h_new = {});
  Expression set_s(const RNNPointer& prev, const std::vector<Expression>& s_new = {});
  Expression add_input(const Expression& x);
  Expression add_input(const RNNPointer& prev, const Expression& x);
  void rewind_one_step();
  RNNPointer get_head(const RNNPointer& p);
  void set_dropout(float d);
  void disable_dropout();

  virtual Expression back() const;
  virtual std::vector<Expression> final_h() const = 0;
  virtual std::vector<Expression> get_h(RNNPointer i) const = 0;

  virtual std::vector<Expression> final_s() const = 0;
  virtual std::vector<Expression> get_s(RNNPointer i) const = 0;

  virtual unsigned num_h0_components() const = 0;
  virtual void copy(const RNNBuilder& params) = 0;

  virtual ParameterCollection & get_parameter_collection() = 0;
};

struct SimpleRNNBuilder : public RNNBuilder {
  SimpleRNNBuilder() = default;

  explicit SimpleRNNBuilder(unsigned layers,
                            unsigned input_dim,
                            unsigned hidden_dim,
                            ParameterCollection& model,
                            bool support_lags = false);

  Expression add_auxiliary_input(const Expression& x, const Expression& aux);

  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;

  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;
  void copy(const RNNBuilder& params) override;

  unsigned num_h0_components() const override;

  ParameterCollection & get_parameter_collection() override;
};

////////////////////////////////////
// declarations from dynet/lstm.h //
////////////////////////////////////

struct CoupledLSTMBuilder : public RNNBuilder {
  CoupledLSTMBuilder() = default;
  explicit CoupledLSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model);
  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;
  unsigned num_h0_components() const override;

  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;

  void copy(const RNNBuilder& params) override;

  void set_dropout(float d);
  void set_dropout(float d, float d_h, float d_c);
  void disable_dropout();
  void set_dropout_masks(unsigned batch_size = 1);

  ParameterCollection & get_parameter_collection() override;

  ParameterCollection local_model;

  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is layer, then ...
  // masks for Gal dropout
  std::vector<std::vector<Expression>> masks;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
  unsigned input_dim = 0;
  unsigned hid = 0;
  bool dropout_masks_valid;

  float dropout_rate_h = 0.f, dropout_rate_c = 0.f;
};

struct VanillaLSTMBuilder : public RNNBuilder {
  VanillaLSTMBuilder() = default;
  explicit VanillaLSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model,
                       bool ln_lstm = false,
                       float forget_bias = 1.f);

  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;
  unsigned num_h0_components() const override;

  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;

  void copy(const RNNBuilder & params) override;

  void set_dropout(float d);
  void set_dropout(float d, float d_r);
  void disable_dropout();
  void set_dropout_masks(unsigned batch_size = 1);

  ParameterCollection & get_parameter_collection() override;

  ParameterCollection local_model;

  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;
  // first index is layer, then ...
  std::vector<std::vector<Parameter>> ln_params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;
  // first index is layer, then ...
  std::vector<std::vector<Expression>> ln_param_vars;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> masks;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
  unsigned input_dim, hid;
  float dropout_rate_h;
  bool ln_lstm;
  float forget_bias;
  bool dropout_masks_valid;
};

typedef VanillaLSTMBuilder LSTMBuilder;

struct CompactVanillaLSTMBuilder : public RNNBuilder {
  CompactVanillaLSTMBuilder();
  explicit CompactVanillaLSTMBuilder(unsigned layers,
                              unsigned input_dim,
                              unsigned hidden_dim,
                              ParameterCollection& model);

  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;
  unsigned num_h0_components() const override;

  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;

  void copy(const RNNBuilder & params) override;

  void set_dropout(float d);
  void set_dropout(float d, float d_r);
  void disable_dropout();
  void set_dropout_masks(unsigned batch_size = 1);
  void set_weightnoise(float std);

  ParameterCollection & get_parameter_collection() override;

  ParameterCollection local_model;

  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> masks;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
  unsigned input_dim, hid;
  float dropout_rate_h;
  float weightnoise_std;
  bool dropout_masks_valid;
};

///////////////////////////////////
// declarations from dynet/gru.h //
///////////////////////////////////

struct GRUBuilder : public RNNBuilder {
  GRUBuilder() = default;
  explicit GRUBuilder(unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      ParameterCollection& model);
  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;
  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;
  unsigned num_h0_components() const override;
  void copy(const RNNBuilder & params) override;
  ParameterCollection & get_parameter_collection() override;
};


/////////////////////////////////////////
// declarations from dynet/fast-lstm.h //
/////////////////////////////////////////

struct FastLSTMBuilder : public RNNBuilder {
  FastLSTMBuilder() = default;
  explicit FastLSTMBuilder(unsigned layers,
                           unsigned input_dim,
                           unsigned hidden_dim,
                           ParameterCollection& model);

  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;
  unsigned num_h0_components() const override;

  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;

  void copy(const RNNBuilder & params) override;

  ParameterCollection & get_parameter_collection() override;

  ParameterCollection local_model;

  std::vector<std::vector<Parameter>> params;
  std::vector<std::vector<Expression>> param_vars;

  std::vector<std::vector<Expression>> h, c;

  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
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


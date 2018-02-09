
open Swig
open Dynet_swig

module Parameter =
struct
    type t = c_obj

    let zero p = ignore (p -> zero ())
    let dim p = p -> dim ()
    
    let set_updated p b = ignore (p -> set_updated ((b to bool)))
    let is_updated p = (p -> is_updated ()) as bool
    let values p = p -> values ()
end

module LookupParameter =
struct
    type t = c_obj

    let initialize p i vs = ignore (p -> initialize ((i to int), vs))
    let zero p = ignore (p -> zero ())
    let dim p = p -> dim ()
    
    let values p = p -> values ()
    let set_updated p b = ignore (p -> set_updated ((b to bool)))
    let is_updated p = (p -> is_updated ()) as bool
end

module ParameterInit =
struct
    type t = c_obj
end
(*
struct ParameterInit {
  ParameterInit() {}
  virtual ~ParameterInit() {}
  virtual void initialize_params(Tensor & values) const = 0;
};
*)
let parameter_init_normal ?(m=0.0) ?(v=1.0) =
    new_ParameterInitNormal '((m to float), (v to float))
(*
struct ParameterInitNormal : public ParameterInit {
  ParameterInitNormal(float m = 0.0f, float v = 1.0f) : mean(m), var(v) {}
  virtual void initialize_params(Tensor& values) const override;
 private:
  float mean, var;
};
*)
let parameter_init_uniform scale =
    new_ParameterInitUniform '((scale to float))
(*
struct ParameterInitUniform : public ParameterInit {
  ParameterInitUniform(float scale) :
    left(-scale), right(scale) { assert(scale != 0.0f); }
  ParameterInitUniform(float l, float r) : left(l), right(r) { assert(l != r); }
  virtual void initialize_params(Tensor & values) const override;
 private:
  float left, right;
};
*)
let parameter_init_const c =
    new_ParameterInitConst ((c to float))
(*
struct ParameterInitConst : public ParameterInit {
  ParameterInitConst(float c) : cnst(c) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float cnst;
};
*)

let parameter_init_identity () =
    new_ParameterInitIdentity '()
(*
struct ParameterInitIdentity : public ParameterInit {
  ParameterInitIdentity() {}
  virtual void initialize_params(Tensor & values) const override;
};
*)
let parameter_init_glorot ?(is_lookup=false) () =
    new_ParameterInitGlorot ((is_lookup to bool))
(*
struct ParameterInitGlorot : public ParameterInit {
  ParameterInitGlorot(bool is_lookup = false) : lookup(is_lookup) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  bool lookup;
};
*)

let parameter_init_from_file f =
    new_ParameterInitFromFile ((f to string))

(*
struct ParameterInitFromFile : public ParameterInit {
  ParameterInitFromFile(std::string f) : filename(f) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::string filename;
};
*)

let parameter_init_from_vector v =
    new_ParameterInitFromVector v
(*
struct ParameterInitFromVector : public ParameterInit {
  ParameterInitFromVector(std::vector<float> v) : vec(v) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::vector<float> vec;
};
*)



(*
struct ParameterStorageBase {
  virtual void scale_parameters(float a) = 0;
  virtual void zero() = 0;
  virtual void squared_l2norm(float* sqnorm) const = 0;
  virtual void g_squared_l2norm(float* sqnorm) const = 0;
  virtual size_t size() const = 0;
  virtual ~ParameterStorageBase();
};
*)


(*
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
*)

(*
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
*)

module ParameterCollection =
struct
    type t = c_obj

    let make () = new_ParameterCollection '()


    let gradient_l2_norm p = (p -> gradient_l2_norm ()) as float
    let reset_gradient p = ignore (p -> reset_gradient ())

    let add_parameters ?(init=None) p dim scale =
        let init = match init with
        | None -> if scale = 0.0 then
            parameter_init_glorot ()
        else
            parameter_init_uniform scale
        | Some init -> init
        in p -> add_parameters (dim, init)

    let add_lookup_parameters ?(init=None) p n dim =
        match init with
        | None -> p -> add_lookup_parameters ((n to int), dim)
        | Some init -> p -> add_lookup_parameters ((n to int), dim, init)
end
(*
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
*)



open Swig
open Dynet_swig
open Vectors

module Parameter =
struct
    type t = c_obj

    let zero p = ignore (p -> zero ())
    let dim p = Dim.from_ptr (p -> dim ())
    
    let set_updated p b = ignore (p -> set_updated ((b to bool)))
    let is_updated p = (p -> is_updated ()) as bool
    let values p = Tensor.from_ptr (p -> values ())

    let to_ptr t = t
    let from_ptr t = t
end

module ParameterVector
    : VECTOR with type value = Parameter.t = Vector (
    struct
        type t = Parameter.t
        let new_vector = new_ParameterVector
        let from_t i = i
        let to_t i = i
        let zero = C_void
        let show i = match '&i with
            | C_ptr (i, j) -> Printf.sprintf "Parameter at (%Ld, %Ld)" i j
            | _ -> invalid_arg "never occur"
    end
)

module LookupParameter =
struct
    type t = c_obj

    let initialize p i vs =
        ignore (p -> initialize ((i to uint), (FloatVector.to_ptr vs)))
    let zero p = ignore (p -> zero ())
    let dim p = Dim.from_ptr (p -> dim ())
    
    let values p = p -> values ()
    let set_updated p b = ignore (p -> set_updated ((b to bool)))
    let is_updated p = (p -> is_updated ()) as bool

    let to_ptr t = t
    let from_ptr t = t
end

module ParameterInit =
struct
    type t = c_obj

    let normal ?(m=0.0) ?(v=1.0) () =
        new_ParameterInitNormal '((m to float), (v to float))
    let uniform scale =
        new_ParameterInitUniform '((scale to float))
    let const c =
        new_ParameterInitConst ((c to float))
    let identity () =
        new_ParameterInitIdentity '()
    let glorot ?(is_lookup=false) () =
        new_ParameterInitGlorot ((is_lookup to bool))
    let from_file f =
        new_ParameterInitFromFile ((f to string))
    let from_vector v =
        new_ParameterInitFromVector (FloatVector.to_ptr v)

    let lecun_uniform ?(scale=1.0) fan_in =
        new_ParameterInitLeCunUniform '((fan_in to float), (scale to float))

    let initialize_params p t = ignore (p -> initialize_params (t))

    let to_ptr t = t
    let from_ptr t = t
end



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

    let make () = _make_ParameterCollection '()

    let gradient_l2_norm p = (p -> gradient_l2_norm ()) as float
    let reset_gradient p = ignore (p -> reset_gradient ())

    let add_parameters ?init ?(scale=1.0) p dim =
        let dim = Dim.to_ptr dim in
        let init = match init with
        | None -> if scale = 0.0 then
            ParameterInit.glorot ()
        else
            ParameterInit.uniform scale
        | Some init -> init
        in p -> add_parameters (dim, init)

    let add_lookup_parameters ?init p n dim =
        let dim = Dim.to_ptr dim in
        match init with
        | None -> p -> add_lookup_parameters ((n to uint), dim)
        | Some init -> p -> add_lookup_parameters ((n to uint), dim, init)

    let project_weights ?(radius=1.0) p = ignore (p -> project_weights ((radius to float)))
    let set_weight_decay_lambda p lambda = ignore (p -> set_weight_decay_lambda ((lambda to float)))

    let parameter_count p = (p -> parameter_count ()) as int
    let updated_parameter_count p = (p -> updated_parameter_count ()) as int

    let to_ptr t = t
    let from_ptr t = t
end

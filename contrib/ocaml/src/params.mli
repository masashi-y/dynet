
open Swig
open Dynet_swig
open Vectors

module Parameter :
sig
    type t

    val zero : t -> unit
    val dim : t -> Dim.t
    
    val set_updated : t -> bool -> unit
    val is_updated : t -> bool
    val values : t -> Tensor.t

    val to_ptr : t -> c_obj
    val from_ptr : c_obj -> t
end

module ParameterVector : VECTOR with type value = Parameter.t

module LookupParameter :
sig
    type t

    val initialize : t -> int -> FloatVector.t -> unit
    val zero : t -> unit
    val dim : t -> Dim.t
    
    (* val values : t -> TensorVector.t *)
    val set_updated : t -> bool -> unit
    val is_updated : t -> bool

    val to_ptr : t -> c_obj
    val from_ptr : c_obj -> t
end

module ParameterInit :
sig
    type t

    val normal : ?m:float -> ?v:float -> unit -> t
    val uniform : float -> t
    val const : float -> t
    val identity : unit -> t
    val glorot : ?is_lookup:bool -> unit -> t
    val from_file : string -> t
    val from_vector : FloatVector.t -> t
    val lecun_uniform : ?scale:float -> float -> t

    val initialize_params : t -> Tensor.t -> unit

    val to_ptr : t -> c_obj
    val from_ptr : c_obj -> t
end

module ParameterCollection :
sig
    type t

    val make : unit -> t

    val gradient_l2_norm : t -> float
    val reset_gradient : t -> unit

    val add_parameters : ?init:ParameterInit.t -> ?scale:float -> t -> Dim.t -> Parameter.t
    val add_lookup_parameters : ?init:ParameterInit.t -> t -> int -> Dim.t -> LookupParameter.t

    val project_weights : ?radius:float -> t -> unit
    val set_weight_decay_lambda : t -> float -> unit

    val parameter_count : t -> int
    val updated_parameter_count : t -> int

    val to_ptr : t -> c_obj
    val from_ptr : c_obj -> t
end

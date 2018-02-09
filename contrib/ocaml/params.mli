
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
end

module LookupParameter :
sig
    type t

    val initialize : t -> int -> FloatVector.t -> unit
    val zero : t -> unit
    val dim : t -> Dim.t
    
    (* val values : t -> TensorVector.t *)
    val set_updated : t -> bool -> unit
    val is_updated : t -> bool
end

module ParameterInit :
sig
    type t
end

val parameter_init_const : float -> ParameterInit.t

module ParameterCollection :
sig
    type t = c_obj

    val make : unit -> t

    val gradient_l2_norm : t -> float
    val reset_gradient : t -> unit

    val add_parameters : ?init:(ParameterInit.t option) -> t -> Dim.t -> float -> Parameter.t
    val add_lookup_parameters : ?init:(ParameterInit.t option) -> t -> int -> Dim.t -> LookupParameter.t
end

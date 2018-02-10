
open Swig
open Dynet_swig
open Vectors

module Parameter :
sig
    type t = c_obj

    val zero : t -> unit
    val dim : t -> Dim.t
    
    val set_updated : t -> bool -> unit
    val is_updated : t -> bool
    val values : t -> Tensor.t
end

module ParameterVector : VECTOR with type value = Parameter.t

module LookupParameter :
sig
    type t = c_obj

    val initialize : t -> int -> FloatVector.t -> unit
    val zero : t -> unit
    val dim : t -> Dim.t
    
    (* val values : t -> TensorVector.t *)
    val set_updated : t -> bool -> unit
    val is_updated : t -> bool
end

module ParameterInit :
sig
    type t = c_obj

    val normal : ?m:float -> ?v:float -> t
    val uniform : float -> t
    val const : float -> t
    val identity : unit -> t
    val glorot : ?is_lookup:bool -> unit -> t
    val from_file : string -> t
    val from_vector : FloatVector.t -> t

    val initialize_params : t -> Tensor.t -> unit

end

module ParameterCollection :
sig
    type t = c_obj

    val make : unit -> t

    val gradient_l2_norm : t -> float
    val reset_gradient : t -> unit

    val add_parameters : ?init:ParameterInit.t -> ?scale:float -> t -> Dim.t -> Parameter.t
    val add_lookup_parameters : ?init:ParameterInit.t -> t -> int -> Dim.t -> LookupParameter.t
end

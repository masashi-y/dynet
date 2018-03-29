
open Swig
open Dynet_swig
open Vectors

type t

val dim : t -> Dim.t

val as_scalar : t -> float
val as_vector : t -> FloatVector.t
val as_list : t -> float list
val as_array : t -> float array
val show : t -> string

val to_ptr : t -> c_obj
val from_ptr : c_obj -> t

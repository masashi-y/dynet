
open Swig
open Dynet_swig

module type VECTOR =
sig
    type t = c_obj
    type value
    val make : value array -> t

    val size : t -> int
    val empty : t -> bool
    val clear : t -> unit
    val push_back : t -> value -> unit
    val get : t -> int -> value
    val set : t -> int -> value -> unit
    val to_array : t -> value array
    val fold_left : ('a -> value -> 'a) -> 'a -> t -> 'a
    val fold_right : (value -> 'a -> 'a) -> t -> 'a -> 'a
    val map : (value -> 'a) -> t -> 'a list
    val iter : (value -> unit) -> t -> unit
end

module IntVector : VECTOR with type value = int

val print_intvector : IntVector.t -> unit

module LongVector : VECTOR with type value = int

module FloatVector : VECTOR with type value = float

module DoubleVector : VECTOR with type value = float


open Swig
open Dynet_swig

module type VECTORBASE =
sig
    type t
    val new_vector : c_obj -> c_obj
    val from_t : t -> c_obj
    val to_t : c_obj -> t
    val zero : t
    val show : t -> string
end

module type VECTOR =
sig
    type t
    type value
    val size : t -> int
    val of_array : value array -> t
    val to_array : t -> value array
    val init : int -> (int -> value) -> t
    val of_list : value list -> t
    val to_list : t -> value list
    val make : int -> value -> t
    val empty : t -> bool
    val clear : t -> unit
    val push_back : t -> value -> unit
    val get : t -> int -> value
    val set : t -> int -> value -> unit
    val fold_left : ('a -> value -> 'a) -> 'a -> t -> 'a
    val fold_right : (value -> 'a -> 'a) -> t -> 'a -> 'a
    val map : (value -> 'a) -> t -> 'a list
    val iter : (value -> unit) -> t -> unit
    val show : t -> string
    val print : t -> unit
    val to_ptr : t -> c_obj
    val from_ptr : c_obj -> t
end

module Vector (Base : VECTORBASE) : VECTOR with type value = Base.t

module IntVector : VECTOR with type value = int

module UnsignedVector : VECTOR with type value = int

module LongVector : VECTOR with type value = int

module FloatVector : VECTOR with type value = float

module DoubleVector : VECTOR with type value = float

module StringVector : VECTOR with type value = string

module ExpressionVector : VECTOR with type value = Expression.t

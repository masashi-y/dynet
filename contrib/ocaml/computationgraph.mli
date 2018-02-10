
open Swig
open Params
open Expr

type t = c_obj

val _with : (t -> 'a) -> 'a

val add_parameters : t -> Parameter.t -> int
val add_const_parameters : t -> Parameter.t -> int
val add_lookup : t -> Parameter.t -> int -> int
val add_const_lookup : t -> Parameter.t -> int -> int

val clear : t -> unit
val checkpoint : t -> unit
val revert : t -> unit

val get_dimension : t -> int -> Dim.t

val forward : t -> Expression.t -> Tensor.t
val incremental_forward : t -> Expression.t -> Tensor.t
val get_value : t -> int -> Tensor.t

val invalidate : t -> unit

val backward : t -> Expression.t -> unit

val print_graphviz : t -> unit

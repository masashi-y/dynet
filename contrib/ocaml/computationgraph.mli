
open Swig
open Params

type t

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

val to_ptr : t -> c_obj
val from_ptr : c_obj -> t

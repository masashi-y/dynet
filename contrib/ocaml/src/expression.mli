
open Swig

type t
val to_ptr : t -> c_obj
val from_ptr : c_obj -> t
val null : t


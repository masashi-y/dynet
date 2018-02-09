
open Swig

type t = c_obj

val make : ?batch:int -> int array -> t
val size : t -> int
val batch_size : t -> int
val truncate : t -> t
val single_batch : t -> t
val resize : t -> int -> unit
val ndims : t -> int
val rows : t -> int
val cols : t -> int
val batch_elems : t -> int

val set : t -> int -> int -> unit
val get : t -> int -> int

val transpose : t -> t

val show : t -> string

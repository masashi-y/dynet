
open Swig
open Dynet_swig
open Vectors

type t = c_obj

let make ?(batch=1) arr =
    let v = LongVector.of_array arr in
    let batch = batch to uint in
    new_Dim '((LongVector.to_ptr v), batch)

let size d = ((d -> size ()) as int)
let batch_size d = ((d -> batch_size ()) as int)
let sum_dims d = ((d -> sum_dims ()) as int)
let truncate d = d -> truncate ()
let single_batch d = d -> single_batch ()
let resize d i = ignore (d -> resize ((i to uint)))
let ndims d = ((d -> ndims ()) as int)
let rows d = ((d -> rows ()) as int)
let cols d = ((d -> cols ()) as int)
let batch_elems d = ((d -> batch_elems ()) as int)

let set d i s = ignore (d -> set ((i to uint), (s to uint)))
let get d i = ((d '[i to uint]) as int)

let transpose d = (d -> transpose ())

let show d = ((_dim_show d) as string)
(*

"size"
"delete_dim"

*)

let to_ptr t = t
let from_ptr t = t

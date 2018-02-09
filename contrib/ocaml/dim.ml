
open Swig
open Dynet_swig
open Vectors

type t = c_obj

let make ?(batch=1) arr =
    let v = LongVector.make arr in
    let batch = batch to int in
    new_Dim '(v, batch)

let size d = ((d -> size ()) as int)
let batch_size d = ((d -> batch_size ()) as int)
let sum_dims d = ((d -> sum_dims ()) as int)
let truncate d = d -> truncate ()
let single_batch d = d -> single_batch ()
let resize d i = ignore (d -> resize ((i to int)))
let ndims d = ((d -> ndims ()) as int)
let rows d = ((d -> rows ()) as int)
let cols d = ((d -> cols ()) as int)
let batch_elems d = ((d -> batch_elems ()) as int)

let set d i s = ignore (d -> set ((i to int), (s to int)))
let get d i = ((d '[i to int]) as int)

let transpose d = (d -> transpose ())

let show d = ((_dim_show d) as string)
(*

"size"
"delete_dim"

*)

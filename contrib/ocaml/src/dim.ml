
open Swig
open Dynet_swig
open Vectors

type t = c_obj

module List =
struct
    include List

    let rec initial = function
        | [] -> raise (Invalid_argument "init")
        | [x] -> []
        | x :: xs -> x :: initial xs

    let init n ~f =
        if n < 0 then raise (Invalid_argument "init");
        let rec aux i accum =
            if i = 0 then accum
            else aux (i-1) (f (i-1) :: accum) in
        aux n []
end

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

let to_pair d =
    let lst = List.init (ndims d) (get d) in
    let b = batch_elems d in
    (lst, b)

let to_pair_mat d = match to_pair d with
    | [n; m], b -> ((n, m), b)
    | _ -> raise (Invalid_argument "to_pair_mat")

let to_pair_vec d = match to_pair d with
    | [n], b -> (n, b)
    | _ -> raise (Invalid_argument "to_pair_vec")

(*

"size"
"delete_dim"

*)

let to_ptr t = t
let from_ptr t = t

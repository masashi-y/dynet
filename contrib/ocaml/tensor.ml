
open Swig
open Dynet_swig

type t = c_obj

let dim t = (t -> "[d]" ())
(*
struct Tensor {
  Dim d;
  float* v;
};
*)

let as_scalar t = (_as_scalar t) as float
let as_vector t = (_as_vector t)
let show t = ((_tensor_show t) as string)

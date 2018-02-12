
open Swig
open Dynet_swig
open Vectors

type t = c_obj

let dim t = Dim.from_ptr (t -> "[d]" ())
(*
struct Tensor {
  Dim d;
  float* v;
};
*)

let as_scalar t = (_as_scalar t) as float
let as_vector t = FloatVector.from_ptr (_as_vector t)
let show t = ((_tensor_show t) as string)

let to_ptr t = t
let from_ptr t = t

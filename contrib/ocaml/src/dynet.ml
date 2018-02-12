
open Params
open Vectors

module IntVector = IntVector
module LongVector = LongVector
module FloatVector = FloatVector
module DoubleVector = DoubleVector
module StringVector = StringVector
(* module ExpressionVector : VECTOR with type value = Expr.t *)
(* module ParameterVector : VECTOR with type value = Parameter.t *)

module Init = Init
module Model = ParameterCollection
module Parameter = Parameter
module LookupParameter = LookupParameter
module ParameterVector = ParameterVector
module Computationgraph = Computationgraph
module Dim = Dim
module Expr = Expr
module ExpressionVector = ExpressionVector
module Tensor = Tensor
module Trainer = Trainer
module RNN = Rnn
module Dict = Dict

let ( !@ ) = Dim.make

let print_dim d = print_endline (Dim.show d)

let print_tensor t = print_endline (Tensor.show t)

let argmax v =
    let arr = FloatVector.to_array v in
    if Array.length arr <= 0 then
        failwith "operating argmax on empty vector";
    let _, _, max = Array.fold_left (fun (x, i, mx) y ->
        if x > y then (x, i+1, mx)
        else (y, i+1, i))
    (arr.(0), 0, 0) arr in
    max

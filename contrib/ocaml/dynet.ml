
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

let ( !@ ) = Dim.make


open Swig
open Dynet_swig
open Vectors
open Params


let parameter cg p = Expression.from_ptr (
    _parameter_Parameter '((Computationgraph.to_ptr cg), (Parameter.to_ptr p)))

let input cg dim v = Expression.from_ptr (
    _input '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (FloatVector.to_ptr v)))

let tanh x = Expression.from_ptr (_tanh '((Expression.to_ptr x)))

let add x y = Expression.from_ptr (
    _exprPlus '((Expression.to_ptr x), (Expression.to_ptr y)))
let mul x y = Expression.from_ptr (
    _exprTimes '((Expression.to_ptr x), (Expression.to_ptr y)))
let sub x y = Expression.from_ptr (
    _exprMinus '((Expression.to_ptr x), (Expression.to_ptr y)))
let div x y = Expression.from_ptr (
    _exprDivide '((Expression.to_ptr x), (Expression.to_ptr y)))

let squared_distance x y = Expression.from_ptr (
    _squared_distance '((Expression.to_ptr x), (Expression.to_ptr y)))

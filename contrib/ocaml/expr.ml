
open Swig
open Dynet_swig
open Vectors

module Expression =
struct
    type t = c_obj
end

let parameter cg p = _parameter_Parameter '(cg, p)

let input cg dim v = _input '(cg, dim, v)

let tanh x = _tanh '(x)

let add x y = _exprPlus '(x, y)
let mul x y = _exprTimes '(x, y)
let sub x y = _exprMinus '(x, y)
let div x y = _exprDivide '(x, y)

let squared_distance x y = _squared_distance '(x, y)

module ExpressionVector
    : VECTOR with type value = Expression.t = Vector (
    struct
        type t = Expression.t
        let new_vector = new_ExpressionVector
        let from_t i = i
        let to_t i = i
        let zero = C_void
        let show i = match '& i with
            | C_ptr (i, j) -> Printf.sprintf "Expression at (%Ld, %Ld)" i j
            | _ -> invalid_arg "never occur"
    end
)


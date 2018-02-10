
open Swig
open Dynet_swig

module type VECTORBASE =
sig
    type t
    val new_vector : c_obj -> c_obj
    val from_t : t -> c_obj
    val to_t : c_obj -> t
    val zero : t
    val show : t -> string
end

module type VECTOR =
sig
    type t
    type value
    val make : value array -> t

    val size : t -> int
    val empty : t -> bool
    val clear : t -> unit
    val push_back : t -> value -> unit
    val get : t -> int -> value
    val set : t -> int -> value -> unit
    val to_array : t -> value array
    val fold_left : ('a -> value -> 'a) -> 'a -> t -> 'a
    val fold_right : (value -> 'a -> 'a) -> t -> 'a -> 'a
    val map : (value -> 'a) -> t -> 'a list
    val iter : (value -> unit) -> t -> unit
    val show : t -> string
    val to_ptr : t -> c_obj
    val from_ptr : c_obj -> t
end

module Vector (Base : VECTORBASE)
    : VECTOR with type value = Base.t =
struct
    type t = c_obj
    type value = Base.t

    let make arr =
        let v = Base.new_vector '() in
        array_to_vector v Base.from_t arr

    let size v = (v -> size ()) as int
    let empty v = (v -> empty ()) as bool
    let clear v = ignore (v -> clear ())
    let push_back v e = ignore (v -> push_back ((Base.from_t e)))
    let get v i = Base.to_t (v '[i to int])
    let set v i s = ignore (v -> set ((i to int), (Base.from_t s)))
    let to_array v =
        let arr = Array.make (size v) Base.zero in
        ignore (vector_to_array v Base.to_t arr);
        arr

    let fold_left f accum v =
        let len = size v in
        let rec aux i accum =
            if i < len then
                aux (i+1) (f accum (get v i))
            else accum
        in aux 0 accum

    let fold_right f v accum =
        let len = size v in
        let rec aux i accum =
            if i < len then
                f (get v i) (aux (i+1) accum)
            else accum
        in aux 0 accum

    let map f v = fold_left (fun accum e -> f e::accum) [] v
    let iter f v = fold_left (fun () e -> f e) () v

    let show v = 
        let buf = Buffer.create 10 in
        for i = 0 to (size v) - 1 do
            if i > 0 then
                Buffer.add_string buf ", ";
            Buffer.add_string buf (Base.show (get v i))
        done;
        Buffer.contents buf

    let to_ptr t = t
    let from_ptr t = t
end

module IntVector
    : VECTOR with type value = int = Vector (
    struct
        type t = int
        let new_vector = new_IntVector
        let from_t i = i to int
        let to_t i = i as int
        let zero = 0
        let show = string_of_int
    end
)

module LongVector
    : VECTOR with type value = int = Vector (
    struct
        type t = int
        let new_vector = new_LongVector
        let from_t i = i to int
        let to_t i = i as int
        let zero = 0
        let show = string_of_int
    end
)

module FloatVector
    : VECTOR with type value = float = Vector (
    struct
        type t = float
        let new_vector = new_FloatVector
        let from_t i = i to float
        let to_t i = i as float
        let zero = 0.0
        let show = string_of_float
    end
)

module DoubleVector
    : VECTOR with type value = float = Vector (
    struct
        type t = float
        let new_vector = new_DoubleVector
        let from_t i = i to float
        let to_t i = i as float
        let zero = 0.0
        let show = string_of_float
    end
)

module StringVector
    : VECTOR with type value = string = Vector (
    struct
        type t = string
        let new_vector = new_StringVector
        let from_t i = i to string
        let to_t i = i as string
        let zero = "0"
        let show x = x
    end
)

module ExpressionVector
    : VECTOR with type value = Expression.t = Vector (
    struct
        type t = Expression.t
        let new_vector = new_ExpressionVector
        let from_t = Expression.to_ptr
        let to_t = Expression.from_ptr
        let zero = Expression.null
        let show i = match '& (from_t i) with
            | C_ptr (i, j) -> Printf.sprintf "Expression at (%Ld, %Ld)" i j
            | _ -> invalid_arg "never occur"
    end
)

(*
let print_intvector v = IntVector.(
    for i = 0 to (size v) - 1 do
        if i > 0 then
            print_string ", ";
        print_int (get v i)
    done;
    print_endline ""
)

let () = IntVector.(
    let v = make [|1;2;3;4|] in
    print_intvector v; Printf.printf "%i\n" (size v);
    set v 5 1000;
    print_intvector v; Printf.printf "%i\n" (size v);
    fold_right (fun e () -> Printf.printf "%i" e) v ();
    print_endline "";
    print_intvector v; Printf.printf "%i\n" (size v);
    clear v;
    print_intvector v; Printf.printf "%i\n" (size v);
    push_back v 1;
    push_back v 2;
    print_intvector v; Printf.printf "%i\n" (size v);
    Printf.printf "%i\n" (get v 0);
    set v 0 1000;
    Printf.printf "%i\n" (get v 0);
    print_intvector v
)
  UnsignedVector
  StringVector
  ExpressionVector
  ParameterStorageVector
  LookupParameterStorageVector
  ExpressionVectorVector
  ParameterVector
  ParameterVectorVector
*)

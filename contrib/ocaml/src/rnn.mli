
open Swig
open Params
open Vectors

type t

val simple_rnn : ?support_lags:bool -> int -> int -> int -> ParameterCollection.t -> t

val coupled_lstm : int -> int -> int -> ParameterCollection.t -> t

val vanilla_lstm : ?ln_lstm:bool -> ?forget_bias:float -> int -> int -> int -> ParameterCollection.t -> t

val compact_vanilla_lstm : int -> int -> int -> ParameterCollection.t -> t

val gru : int -> int -> int -> ParameterCollection.t -> t

val fast_lstm : int -> int -> int -> ParameterCollection.t -> t

val state : t -> int
val new_graph : ?update:bool -> t -> Computationgraph.t -> unit
val start_new_sequence : ?init:ExpressionVector.t -> t -> unit

val set_h : t -> int -> ExpressionVector.t -> Expression.t
val set_s : t -> int -> ExpressionVector.t -> Expression.t

val add_input : ?prev:int -> t -> Expression.t -> Expression.t

val rewind_one_step : t -> unit
val get_head : t -> int -> int
val set_dropout : t -> float -> unit
val disable_dropout : t -> unit

val back : t -> Expression.t
val final_h : t -> ExpressionVector.t
val final_s : t -> ExpressionVector.t
val get_h : t -> int -> ExpressionVector.t
val get_s : t -> int -> ExpressionVector.t

val num_h0_components : t -> int
val copy : t -> t -> unit

val get_parameter_collection : t -> ParameterCollection.t

val to_ptr : t -> c_obj
val from_ptr : c_obj -> t

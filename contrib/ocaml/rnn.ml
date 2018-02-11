
open Swig
open Params
open Vectors
open Dynet_swig

type t = c_obj

let simple_rnn ?(support_lags=false) layers input_dim hidden_dim p = 
    new_SimpleRNNBuilder '((layers to int), (input_dim to int),
         (hidden_dim to int), (ParameterCollection.to_ptr p), (support_lags to bool))

let coupled_lstm layers input_dim hidden_dim p =
  new_CoupledLSTMBuilder '((layers to int), (input_dim to int), (hidden_dim to int), (ParameterCollection.to_ptr p))

let vanilla_lstm ?(ln_lstm=false) ?(forget_bias=1.0) layers input_dim hidden_dim p =
  new_VanillaLSTMBuilder '((layers to int), (input_dim to int), (hidden_dim to int), (ParameterCollection.to_ptr p), (ln_lstm to bool), (forget_bias to float))

let compact_vanilla_lstm layers input_dim hidden_dim p =
  new_CompactVanillaLSTMBuilder '((layers to int), (input_dim to int), (hidden_dim to int), (ParameterCollection.to_ptr p))

let gru layers input_dim hidden_dim p =
  new_GRUBuilder '((layers to int), (input_dim to int), (hidden_dim to int), (ParameterCollection.to_ptr p))

let fast_lstm layers input_dim hidden_dim p =
  new_FastLSTMBuilder '((layers to int), (input_dim to int), (hidden_dim to int), (ParameterCollection.to_ptr p))

let state rnn = (rnn -> _state ()) as int
let new_graph ?(update=true) rnn cg = ignore (rnn -> _new_graph ((Computationgraph.to_ptr cg), (update to bool)))
let start_new_sequence rnn h_0 = ignore (rnn -> _start_new_sequence ((ExpressionVector.to_ptr h_0)))

let set_h rnn prev h_new = Expression.from_ptr (rnn -> _set_h ((prev to int), (ExpressionVector.to_ptr h_new)))
let set_s rnn prev s_new = Expression.from_ptr (rnn -> _set_s ((prev to int), (ExpressionVector.to_ptr s_new)))

let add_input ?prev rnn x = match prev with
    | Some prev -> Expression.from_ptr (rnn -> _add_input ((prev to int), (Expression.to_ptr x)))
    | None -> Expression.from_ptr (rnn -> _add_input ((Expression.to_ptr x)))

let rewind_one_step rnn = ignore (rnn -> _rewind_one_step ())
let get_head rnn p = (rnn -> _get_head ((p to int))) as int
let set_dropout rnn d = ignore (rnn -> _set_dropout ((d to float)))
let disable_dropout rnn = ignore (rnn -> _disable_dropout ())

let back rnn = Expression.from_ptr (rnn -> _back ())
let final_h rnn = ExpressionVector.from_ptr (rnn -> _final_h ())
let final_s rnn = ExpressionVector.from_ptr (rnn -> _final_s ())
let get_h rnn i = ExpressionVector.from_ptr (rnn -> _get_h ((i to int)))
let get_s rnn i = ExpressionVector.from_ptr (rnn -> _get_s ((i to int)))

let num_h0_components rnn = (rnn -> _num_h0_components ()) as int
let copy rnn params = ignore (rnn -> _copy (params))

let get_parameter_collection rnn = ParameterCollection.from_ptr (rnn -> _get_parameter_collection ())

let to_ptr t = t
let from_ptr t = t

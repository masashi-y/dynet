
open Dynet

exception ParseError of string

type attention = Dot_product | Additive

let parse_attention = function
    | "dot_product" -> Dot_product
    | "additive" -> Additive
    | s -> raise (ParseError s)

let show_attention = function
    | Dot_product -> "dot product"
    | Additive -> "additive"

type activation = ReLU | Swish

let parse_activation = function
    | "relu" -> ReLU
    | "swish" -> Swish
    | s -> raise (ParseError s)

let show_activation = function
    | ReLU -> "relu"
    | Swish -> "swish"

type t = {
    src_vocab_size : int;
    tgt_vocab_size : int;
    num_units : int;
    nheads : int;
    nlayers : int;
    n_ff_units_factor : int;
    use_dropout : bool;
    encoder_emb_dropout_rate : float;
    encoder_sublayer_dropout_rate : float;
    decoder_emb_dropout_rate : float;
    decoder_sublayer_dropout_rate : float;
    attention_dropout_rate : float;
    ff_dropout_rate : float;
    use_label_smoothing : bool;
    label_smoothing_weight : float;
    position_encoding : int;
    max_length : int;
    attention_type : attention;
    ffl_activation : activation;
    use_hybrid_model : bool;
    is_training : bool;
    unk_id : int;
}


let perr_config c =
Printf.eprintf "
Options:
  -src-vocab-size : [%i]
  -tgt-vocab-size : [%i]
  -num-units : [%i]
  -nheads : [%i]
  -nlayers : [%i]
  -n-ff-units-factor : [%i]
  -use-dropout : [%B]
  -encoder-emb-dropout-rate : [%f]
  -encoder-sublayer-dropout-rate : [%f]
  -decoder-emb-dropout-rate : [%f]
  -decoder-sublayer-dropout-rate : [%f]
  -attention-dropout-rate : [%f]
  -ff-dropout-rate : [%f]
  -use-label-smoothing : [%B]
  -label-smoothing-weight : [%f]
  -position-encoding : [%i]
  -max-length : [%i]
  -attention-type : [%s]
  -ffl-activation : [%s]
  -use-hybrid-model : [%B]
  -is-training : [%B]\n\n%!"
c.src_vocab_size c.tgt_vocab_size c.num_units c.nheads
c.nlayers c.n_ff_units_factor c.use_dropout c.encoder_emb_dropout_rate
c.encoder_sublayer_dropout_rate c.decoder_emb_dropout_rate
c.decoder_sublayer_dropout_rate c.attention_dropout_rate c.ff_dropout_rate
c.use_label_smoothing c.label_smoothing_weight c.position_encoding c.max_length
(show_attention c.attention_type) (show_activation c.ffl_activation)
c.use_hybrid_model c.is_training


let default = {
    src_vocab_size = 0;
    tgt_vocab_size = 0;
    num_units = 512;
    nheads = 8;
    nlayers = 6;
    n_ff_units_factor = 4;
    use_dropout = true;
    encoder_emb_dropout_rate = 0.1;
    encoder_sublayer_dropout_rate = 0.1;
    decoder_emb_dropout_rate = 0.1;
    decoder_sublayer_dropout_rate = 0.1;
    attention_dropout_rate = 0.1;
    ff_dropout_rate = 0.1;
    use_label_smoothing = false;
    label_smoothing_weight = 0.1;
    position_encoding = 1;
    max_length = 500;
    attention_type = Dot_product;
    ffl_activation = ReLU;
    use_hybrid_model = false;
    is_training = true;
    unk_id = -1;
}

let global_ref = ref default

let global () = !global_ref

let set_dropout b =
    global_ref := {(global ()) with use_dropout = b}

module ParseUtils :
sig
    val int_of_string : string -> int
    val float_of_string : string -> float
    val bool_of_string : string -> bool
end =
struct
    let int_of_string v =
        try int_of_string v
        with _ -> raise (ParseError v)

    let float_of_string v =
        try float_of_string v
        with _ -> raise (ParseError v)

    let bool_of_string v =
        try bool_of_string v
        with _ -> raise (ParseError v)
end

let parse argv = 
    let rec aux cfg args = try ParseUtils.(match args with
        | [] -> (cfg, [])
        | "--" :: rest -> (cfg, rest)
        | "-src-vocab-size" :: i :: rest
            -> aux {cfg with src_vocab_size = int_of_string i} rest
        | "-tgt-vocab-size" :: i :: rest
            -> aux {cfg with tgt_vocab_size = int_of_string i} rest
        | "-num-units" :: i :: rest
            -> aux {cfg with num_units = int_of_string i} rest
        | "-nheads" :: i :: rest
            -> aux {cfg with nheads = int_of_string i} rest
        | "-nlayers" :: i :: rest
            -> aux {cfg with nlayers = int_of_string i} rest
        | "-n-ff-units-factor" :: i :: rest
            -> aux {cfg with n_ff_units_factor = int_of_string i} rest
        | "-not-use-dropout" :: rest
            -> aux {cfg with use_dropout = false} rest
        | "-encoder-emb-dropout-rate" :: f :: rest
            -> aux {cfg with encoder_emb_dropout_rate = float_of_string f} rest
        | "-encoder-sublayer-dropout-rate" :: f :: rest
            -> aux {cfg with encoder_sublayer_dropout_rate = float_of_string f} rest
        | "-decoder-emb-dropout-rate" :: f :: rest
            -> aux {cfg with decoder_emb_dropout_rate = float_of_string f} rest
        | "-decoder-sublayer-dropout-rate" :: f :: rest 
            -> aux {cfg with decoder_sublayer_dropout_rate = float_of_string f} rest
        | "-attention-dropout-rate" :: f :: rest
            -> aux {cfg with attention_dropout_rate = float_of_string f} rest
        | "-ff-dropout-rate" :: f :: rest
            -> aux {cfg with ff_dropout_rate = float_of_string f} rest
        | "-use-label-smoothing" :: rest
            -> aux {cfg with use_label_smoothing = true} rest
        | "-label-smoothing-weight" :: f :: rest
            -> aux {cfg with label_smoothing_weight = float_of_string f} rest
        | "-position-encoding" :: i :: rest
            -> aux {cfg with position_encoding = int_of_string i} rest
        | "-max-length":: i :: rest
            -> aux {cfg with max_length = int_of_string i} rest
        | "-attention-type" :: att :: rest
            -> aux {cfg with attention_type = parse_attention att} rest
        | "-ffl-activation-type" :: act :: rest
            -> aux {cfg with ffl_activation = parse_activation act} rest
        | "-use-hybrid-model" :: rest
            -> aux {cfg with use_hybrid_model = true} rest
        | "-is-training" :: rest (* TODO *)
            -> aux {cfg with is_training = true} rest
        | rest -> (cfg, rest))
    with ParseError s ->
        failwith (Printf.sprintf "Invalid argument for keyword option \"%s\": \"%s\"" (List.hd args) s) in
    let cfg, rest = aux (global ()) (Array.to_list argv) in
    global_ref := cfg;
    perr_config (global ());
    Array.of_list rest


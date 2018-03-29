
open Dynet
open Config

module Cg = Computationgraph

let pseudo_min_value = -999999.0

let print_dim0 x = print_dim (Expr.dim x); x

module List =
struct
    include List

    let rec initial = function
        | [] -> raise (Invalid_argument "init")
        | [x] -> []
        | x :: xs -> x :: initial xs

    let init n ~f =
        if n < 0 then raise (Invalid_argument "init");
        let rec aux i accum =
            if i = 0 then accum
            else aux (i-1) (f (i-1) :: accum) in
        aux n []

    let rec transpose = function
        | [] -> []
        | [] :: xss -> transpose xss
        | (x::xs) :: xss ->
            (x :: map hd xss) :: transpose (xs :: map tl xss)

    let maximum = function
        | [] -> raise (Invalid_argument "maximum")
        | x :: xs -> List.fold_left max x xs

    let maximum_by f = function
        | [] -> raise (Invalid_argument "maximum")
        | x :: xs -> List.fold_left (fun x y -> max x (f y)) (f x) xs
end


(* utilities to convert a batch of sentences *)
(* to Expression of dimension (nunits, max length) x batch *)
let rec map_with_padding f = 
    let rec aux n = function
        | [] when n <= 0 -> []
        | [] -> let res = f None in res :: aux (n - 1) []
        | x :: xs when n >= 0 -> let res = f (Some x) in res :: aux (n - 1) xs
        | _ -> raise (Invalid_argument "pad expects length lst <= n")
    in aux

let pad unk_id = map_with_padding (function None -> unk_id | Some x -> x)

let mask = map_with_padding (function None -> 0.0 | Some _ -> 1.0)

let position pad =
    let counter =
        let n = ref 0 in
        let f () =
            let res = !n in
            incr n ; res in
        f in
    map_with_padding (function None -> pad | Some _ -> counter ())


(** ((x, y), z) --> ((x, 1), y * z) *)
let make_time_distributed x =
    let d = Expr.dim x in
    let (n, m), b = Dim.to_pair_mat d in
    let batch = m * b in
    Expr.reshape x (Dim.make ~batch [|n; 1|])

let layer_norm_colwise_3 x g b =
    let x_td = make_time_distributed x in
    Expr.reshape (Expr.layer_norm x_td g b) (Expr.dim x)

let sinusoidal_position_encoding cg dim =
    let nunits = Dim.get dim 0 in
    let nwords = Dim.get dim 1 in
    let timescales = nunits / 2 in
    let log_timescale_increment = log 10000.0 /. (float_of_int timescales -. 1.0) in
    let ss = Array.make (nunits * nwords) 0.0 in
    for p = 0 to nwords - 1 do
        for i = 0 to timescales - 1 do
            let v = float_of_int p *. exp (float_of_int i *. -.log_timescale_increment) in
            ss.(p * nunits + i) <- sin v;
            ss.(p * nunits + timescales  + i) <- cos v;
        done
    done;
    let ss = FloatVector.of_array ss in
    Expr.input cg (!@[|nunits; nwords|]) ss


let split_batch x h =
    let d = Expr.dim x in
    let b = Dim.batch_elems d in
    let steps = b / h in
    ExpressionVector.init h (fun i ->
        let range = Array.make steps 0 in
        for j = 0 to steps - 1 do
            range.(j) <- i * steps + j
        done;
        Expr.pick_batch_elems x range)

let split_rows x h =
    let d = Expr.dim x in
    let steps = (Dim.get d 0) / h in
    ExpressionVector.init h (fun i ->
        Expr.pick_range x (i * steps) ((i+1) * steps))

let dropout0 rate x =
    let cfg = Config.global () in
    if cfg.use_dropout && rate > 0.0 then
        Expr.dropout_dim x 1 rate else x


module LinearLayer =
struct
    type t = {
        pW : Parameter.t;
        pb : Parameter.t option;
        in_dim : int;
        out_dim : int
    }

    let make ?(bias=true) ?(lecun=false) m in_dim out_dim = 
        let init = if lecun then
                   ParameterInit.lecun_uniform (float_of_int in_dim) else
                   ParameterInit.glorot () in
        let pW = Model.add_parameters ~init m (!@[|out_dim; in_dim|]) in
        let pb = if bias then
            Some (Model.add_parameters ~init m (!@[|out_dim|]))
            else None in
        { pW; pb; in_dim; out_dim }

    let apply_distributed ?(reconstruct_shape=false) l cg x =
        let param = Expr.parameter cg in
        let x0 = make_time_distributed x in
        let out = Expr.(match l.pb with
            | Some pb -> affine_transform [|param pb; param l.pW; x0|]
            | None    -> param l.pW * x0) in
        if reconstruct_shape then
            let dim = Expr.dim x in
            let (_, m), batch = Dim.to_pair_mat dim in
            Expr.reshape out (Dim.make ~batch [|l.out_dim; m|])
        else out

    let apply = apply_distributed ~reconstruct_shape:true
end


module HighwayNetworkLayer =
struct
    type t = {
        ln : LinearLayer.t
    }

    let make ?(bias=true) m in_dim = {
        ln = LinearLayer.make ~bias ~lecun:true m in_dim in_dim
    }

    let apply cg {ln} x gx =
        let t = Expr.logistic (LinearLayer.apply ln cg x) in
        Expr.(t *. gx + (1.0 $- t) *. x)

    let apply_distributed ?(reconstruct_shape=false) cg {ln} x gx =
        let t = Expr.logistic (LinearLayer.apply_distributed ~reconstruct_shape ln cg x) in
        Expr.(t *. gx + (1.0 $- t) *. x)
end


module FeedForwardLayer =
struct
    type t = {
        l_in : LinearLayer.t;
        l_out : LinearLayer.t;
        cfg : Config.t
    }

    let make m cfg = {
        l_in = LinearLayer.make m cfg.num_units (cfg.n_ff_units_factor * 4);
        l_out = LinearLayer.make m (cfg.n_ff_units_factor * 4) cfg.num_units;
        cfg
    }

    let apply {l_in; l_out; cfg} cg x =
        let dropout = dropout0 cfg.ff_dropout_rate in
        let activation = match cfg.ffl_activation with
            | ReLU -> Expr.rectify
            | Swish -> Expr.silu ~beta:1.0 in
        x |> LinearLayer.apply l_in cg
          |> activation |> LinearLayer.apply l_out cg |> dropout
end


module Mask =
struct
    type t = {
        (* sequence mask *)
        seq : Expression.t;

        (* 2 masks for padding positions *)
        padding_k : Expression.t;
        padding_q : Expression.t;

        (* 1 mask for future blinding *)
        blinding : Expression.t option
    }

    let sequence_mask ?(self=true) cg (seq_masks : float array list) =
        let seq_masks = List.map FloatVector.of_array seq_masks in
        let l = FloatVector.size (List.nth seq_masks 0) in
        let d = Dim.make (if self then [|l; 1|] else [|1; l|]) in
        let f mask = Expr.(
            let mask = input cg d mask in
            if self then mask *$ pseudo_min_value
                else 1.0 $- mask) in
        Expr.concatenate_to_batch (Array.of_list (List.map f seq_masks)) (* i_seq_mask *)

    let padding_positions_mask nheads seq_mask =
        let l = Dim.get (Expr.dim seq_mask) 0 in
        let padding_k = Expr.(concatenate_to_batch
                 (Array.make nheads @@ concatenate_cols (Array.make l seq_mask))) in
        let padding_q = Expr.(1.0 $- padding_k /$ pseudo_min_value) in
        padding_k, padding_q

    let padding_positions_mask_source nheads seq_mask src_seq_mask =
        let ly = Dim.get (Expr.dim seq_mask) 1 in
        let lx = Dim.get (Expr.dim src_seq_mask) 0 in
        let padding_k = Expr.(concatenate_to_batch
                 (Array.make nheads @@ concatenate_cols (Array.make ly src_seq_mask))) in
        let padding_q = Expr.(concatenate_to_batch
                 (Array.make nheads @@ concatenate (Array.make lx seq_mask))) in
        padding_k, padding_q

    let future_blinding_mask cg l =
        let mask = Array.make (l * l) 0.0 in
        for i = 0 to l - 1 do
            for j = 0 to i do
                mask.(j * l + i) <- 1.0
            done
        done;
        let mask = FloatVector.of_array mask in
        let mask = Expr.input cg (!@[|l; l|]) mask in
        Expr.((1.0 $- mask) *$ pseudo_min_value)
end


module MultiHeadAttentionLayer =
struct
    type t = {
        lnQ : LinearLayer.t;
        lnK : LinearLayer.t;
        lnV : LinearLayer.t;
        lnO : LinearLayer.t;
        scale : float;
        cfg : Config.t;
        blinding : bool
    }

    let make ?(blinding=false) m cfg = {
        lnQ = LinearLayer.make ~bias:false ~lecun:true m cfg.num_units cfg.num_units;
        lnK = LinearLayer.make ~bias:false ~lecun:true m cfg.num_units cfg.num_units;
        lnV = LinearLayer.make ~bias:false ~lecun:true m cfg.num_units cfg.num_units;
        lnO = LinearLayer.make ~bias:false ~lecun:true m cfg.num_units cfg.num_units;
        scale = 1.0 /. sqrt (float_of_int cfg.num_units /. float_of_int cfg.nheads);
        cfg; blinding
    }

    let apply ({cfg} as mha) cg y (* queries *) x (* keys and values *) masks = Expr.(
        let Mask.{padding_k; padding_q; blinding} = masks in
        let linear ln x =
            let x = LinearLayer.apply ln cg x in
            concatenate_to_batch_vec (split_rows x cfg.nheads) in
        let dropout = dropout0 cfg.attention_dropout_rate in
        let blinding x = match blinding with
            | Some bl -> x + bl | None -> x in
        let reconst x = concatenate_vec (split_batch x cfg.nheads) in
        let lQ, lK, lV = linear mha.lnQ y, linear mha.lnK x, linear mha.lnV x in
        transpose lK * lQ *$ mha.scale |> (+) padding_k
          |> blinding |> softmax |> ( *. ) padding_q |> dropout
          |> ( * ) lV (* looking up *) |> reconst
          |> LinearLayer.apply mha.lnO cg
    )
end


module LayerNorm =
struct
    type t = {
        g : Parameter.t;
        b : Parameter.t;
    }

    let make m cfg = {
        g = Model.add_parameters ~init:(ParameterInit.const 1.0) m (!@[|cfg.num_units|]);
        b = Model.add_parameters ~init:(ParameterInit.const 0.0) m (!@[|cfg.num_units|]);
    }

    let apply {g; b} cg x =
        Expr.(layer_norm_colwise_3 x (parameter cg g) (parameter cg b))
end


module EncoderLayer =
struct
    type t = {
        self_attention : MultiHeadAttentionLayer.t;
        feed_forward : FeedForwardLayer.t;
        ln1 : LayerNorm.t;
        ln2 : LayerNorm.t;
        cfg : Config.t
    }

    let make m cfg = {
        self_attention = MultiHeadAttentionLayer.make m cfg;
        feed_forward = FeedForwardLayer.make m cfg;
        ln1 = LayerNorm.make m cfg;
        ln2 = LayerNorm.make m cfg;
        cfg
    }

    let apply ({cfg} as encl) cg src mask = Expr.(
        let dropout = dropout0 cfg.encoder_sublayer_dropout_rate in
        MultiHeadAttentionLayer.apply encl.self_attention cg src src mask
          |> dropout |> (+) src |> LayerNorm.apply encl.ln1 cg
             |> fun ln -> FeedForwardLayer.apply encl.feed_forward cg ln |> (+) ln (* residual *)
             |> LayerNorm.apply encl.ln2 cg
    )
end


module Encoder =
struct
    type t = {
        embed_s : LookupParameter.t;
        embed_pos : LookupParameter.t option;
        layers : EncoderLayer.t list;
        src_rnns : RNN.t list;
        scale_emb : float;
        batch_length : int;
        cfg : Config.t
    }

    let make m cfg = {
        embed_s = Model.add_lookup_parameters m cfg.src_vocab_size (!@[|cfg.num_units|]);
        embed_pos = if cfg.use_hybrid_model && cfg.position_encoding = 1 then
            Some (Model.add_lookup_parameters m cfg.max_length (!@[|cfg.num_units|])) else None;
        layers = List.init cfg.nlayers (fun _ -> EncoderLayer.make m cfg);
        src_rnns = []; (* TODO *)
        scale_emb = sqrt (float_of_int cfg.num_units);
        batch_length = 0;
        cfg
    }

    let compute_embeddings_and_masks ({cfg} as encoder) cg xss (* sentences *) =
        let max_len = List.maximum_by List.length xss in
        let embed xs = Expr.lookup_batch cg encoder.embed_s (Array.of_list xs) in
        let pad_and_mask xs (xss, masks) =
            let ms = Array.of_list (mask max_len xs) in
            let xs = pad cfg.unk_id max_len xs in
            (xs :: xss, ms :: masks) in
        let xss0, masks = List.fold_right pad_and_mask xss ([], []) in
        let src = Expr.(xss0 |> List.transpose |> List.map embed
            |> Array.of_list |> concatenate_cols |> ($*) encoder.scale_emb) in
        let ss = match encoder.embed_pos with
            | None -> sinusoidal_position_encoding cg (Expr.dim src)
            | Some embed ->
                let embed xs = Expr.lookup_batch cg embed (Array.of_list xs) in
                let g xs = position (cfg.max_length - 1) max_len xs in
                xss |> List.map g |> List.transpose |> List.map embed
                    |> Array.of_list |> Expr.concatenate_cols in
        let dropout = dropout0 cfg.encoder_emb_dropout_rate in
        let seq = Mask.sequence_mask cg masks in
        let padding_k, padding_q = Mask.padding_positions_mask cfg.nheads seq in
        let masks = Mask.{ seq; padding_k; padding_q; blinding = None } in
        (dropout Expr.(src + ss), masks)

    let apply encoder cg xs =
        let xs_rep, masks = compute_embeddings_and_masks encoder cg xs in
        let f x l = EncoderLayer.apply l cg x masks in
        (List.fold_left f xs_rep encoder.layers, masks)
end


module DecoderLayer =
struct
    type t = {
        self_attention : MultiHeadAttentionLayer.t;
        src_attention : MultiHeadAttentionLayer.t;
        feed_forward : FeedForwardLayer.t;
        ln1 : LayerNorm.t;
        ln2 : LayerNorm.t;
        ln3 : LayerNorm.t;
        cfg : Config.t
    }

    let make m cfg = {
        self_attention = MultiHeadAttentionLayer.make ~blinding:true m cfg;
        src_attention = MultiHeadAttentionLayer.make m cfg;
        feed_forward = FeedForwardLayer.make m cfg;
        ln1 = LayerNorm.make m cfg;
        ln2 = LayerNorm.make m cfg;
        ln3 = LayerNorm.make m cfg;
        cfg
    }

    let apply ({cfg} as decl) cg enc_inp dec_inp self_mask src_mask = Expr.(
        let dropout = dropout0 cfg.decoder_sublayer_dropout_rate in
        MultiHeadAttentionLayer.apply decl.self_attention cg dec_inp dec_inp self_mask
          |> dropout |> (+) dec_inp |> LayerNorm.apply decl.ln1 cg
            |> fun ln -> MultiHeadAttentionLayer.apply decl.src_attention cg ln enc_inp src_mask
            |> dropout |> (+) ln |> LayerNorm.apply decl.ln2 cg
            |> fun ln -> FeedForwardLayer.apply decl.feed_forward cg ln |> (+) ln
            |> LayerNorm.apply decl.ln3 cg
    )
end


module Decoder =
struct
    type t = {
        embed_t : LookupParameter.t;
        embed_pos :  LookupParameter.t option;
        layers : DecoderLayer.t list;
        rnn : RNN.t option;
        scale_emb : float;
        cfg : Config.t
    }

    let make m cfg = {
        embed_t = Model.add_lookup_parameters m cfg.tgt_vocab_size (!@[|cfg.num_units|]);
        embed_pos = if cfg.use_hybrid_model && cfg.position_encoding = 1 then
            Some (Model.add_lookup_parameters m cfg.max_length (!@[|cfg.num_units|])) else None;
        layers = List.init cfg.nlayers (fun _ -> DecoderLayer.make m cfg);
        rnn = None; (* TODO *)
        scale_emb = sqrt (float_of_int cfg.num_units);
        cfg;
    }

    let embedding_matrix dec cg =
        Expr.lookup_parameter cg dec.embed_t

    let compute_embeddings_and_masks ({cfg} as dec) cg xss (* sentences *) src_seq_mask =
        let max_len = List.maximum_by List.length xss in
        let embed xs = Expr.lookup_batch cg dec.embed_t (Array.of_list xs) in
        let pad_len = max_len + if cfg.is_training then 1 else 0 in
        let pad_and_mask xs (xss, masks) =
            let ms = Array.of_list (mask pad_len xs) in
            let xs = pad cfg.unk_id pad_len xs in
            (xs :: xss, ms :: masks) in
        let xss0, masks = List.fold_right pad_and_mask xss ([], []) in
        let src = Expr.(xss0 |> List.transpose |> List.map embed
            |> Array.of_list |> concatenate_cols |> ($*) dec.scale_emb) in
        let ss = match dec.embed_pos with
            | None -> sinusoidal_position_encoding cg (Expr.dim src)
            | Some embed ->
                let embed xs = Expr.lookup_batch cg embed (Array.of_list xs) in
                let g xs = position (cfg.max_length - 1) pad_len xs in
                xss |> List.map g |> List.transpose |> List.map embed
                    |> Array.of_list |> Expr.concatenate_cols in
        let dropout = dropout0 cfg.decoder_emb_dropout_rate in
        let seq = Mask.sequence_mask cg masks in
        (* create maskings self-attention for future blinding *)
        let blinding = Some (Mask.future_blinding_mask cg pad_len) in
        (* for padding positions blinding *)
        let padding_k, padding_q = Mask.padding_positions_mask cfg.nheads seq in
        let self_masks = Mask.{ seq; padding_k; padding_q; blinding } in
        (* source-attention *)
        let seq = Mask.sequence_mask ~self:false cg masks in
        let padding_k, padding_q =
            Mask.padding_positions_mask_source cfg.nheads seq src_seq_mask in
        let src_masks = Mask.{ seq; padding_k; padding_q; blinding = None } in
        (dropout Expr.(src + ss), self_masks, src_masks)


    let apply dec cg sents src_rep src_seq_masks =
        let tgt_rep, self_masks, src_masks =
            compute_embeddings_and_masks dec cg sents src_seq_masks in
        let f x l = DecoderLayer.apply l cg src_rep x self_masks src_masks in
        List.fold_left f tgt_rep dec.layers
end


module Transformer =
struct
    type t = {
        encoder : Encoder.t;
        decoder : Decoder.t;
        bias : Parameter.t;
        sd : Dict.t;
        td : Dict.t;
        cfg : Config.t
    }

    let make m cfg sd td = {
        encoder = Encoder.make m cfg;
        decoder = Decoder.make m cfg;
        bias = Model.add_parameters m (!@[|cfg.tgt_vocab_size|]);
        sd; td; cfg
    }

    let compute_src_rep t cg sents =
        Encoder.apply t.encoder cg sents

    let apply ?(is_eval=false) ({cfg} as t) cg srcs tgts =
        let max_len = List.maximum_by List.length tgts in
        let src_rep, Mask.{seq} = Encoder.apply t.encoder cg srcs in
        let tgt_rep = Decoder.apply t.decoder cg tgts src_rep seq in
        let bias = Expr.parameter cg t.bias in
        let emb_tgt = Expr.transpose (Decoder.embedding_matrix t.decoder cg) in

        let neglogsoftmax_smoothed x y = 
            let delim = float_of_int cfg.tgt_vocab_size -. 1.0 in
            let min_smoothing_weight = 1.0 -. cfg.label_smoothing_weight in
            Expr.(let ls = log_softmax x in
            let pre = neg (pick_batch ls y) in
            let loss = neg (sum_elems ls /$ delim) in
            pre *$ min_smoothing_weight + loss *$ cfg.label_smoothing_weight) in

        let compute_loss i tgt =
            let tgt_xp = Expr.pick ~d:1 tgt_rep i in
            let r = Expr.affine_transform [|bias; emb_tgt; tgt_xp|] in
            let loss_fun = if cfg.use_label_smoothing && not is_eval then
                neglogsoftmax_smoothed else Expr.pickneglogsoftmax_batch in
            loss_fun r (Array.of_list tgt) in
        List.(tl @@ transpose @@ map (pad cfg.unk_id max_len) tgts)
            |> List.mapi compute_loss |> Array.of_list |> Expr.sum |> Expr.sum_batches

    let step_forward t cg src_rep src_mask sent log_prob =
        let tgt_rep = Decoder.apply t.decoder cg [sent] src_rep src_mask in
        let tgt_rep = if List.length sent = 1 then tgt_rep
            else Expr.pick ~d:1 tgt_rep (List.length sent - 1) in
        let bias = Expr.parameter cg t.bias in
        let emb_tgt = Expr.transpose (Decoder.embedding_matrix t.decoder cg) in
        let r = Expr.affine_transform [|bias; emb_tgt; tgt_rep|] in
        if log_prob then Expr.log_softmax r else Expr.softmax r

    let decode_greedy ({cfg} as t) cg src =
        let tgt = [Dict.convert_word t.td "<s>"] in
        let eos = Dict.convert_word t.td "</s>" in
        let src_rep, Mask.{seq} = Encoder.apply t.encoder cg [src] in
        let rec loop step tgt =
            Cg.checkpoint cg;
            let dist = step_forward t cg src_rep seq tgt false in
            let v = Tensor.as_vector (Cg.incremental_forward cg dist) in
            let res =  argmax v in
            Cg.revert cg;
            let keep_going = res <> eos || step < cfg.max_length in
            let tgt0 = tgt @ [res] in
            if keep_going then loop (step + 1) tgt0
                else tgt0 in
        loop 0 tgt

end

let load_vocab file =
    let d = Dict.make () in
    let read_one x = ignore (Dict.convert_word d x) in
    List.iter read_one ("<unk>" :: "<s>" :: "</s>" :: Utils.read_lines file);
    Dict.freeze d; Dict.set_unk d "<unk>"; d


let load_data sd td file =
    let rec sep_on s = function
        | [] -> raise (Invalid_argument "load_data")
        | x :: xs when x = s -> ([], xs)
        | x :: xs -> let y, ys = sep_on s xs in
                    (x :: y, ys) in
    let f line (xs, ys) =
        let x, y = sep_on "|||" (String.split_on_char ' ' line) in
        (List.map (Dict.convert_word sd) x :: xs,
             List.map (Dict.convert_word sd) y :: ys) in
    List.fold_right f (Utils.read_lines file) ([], [])


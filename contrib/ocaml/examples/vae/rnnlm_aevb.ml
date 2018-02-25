
open Dynet

module Cg = Computationgraph
module P = Pervasives

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
end

type t = {
    vocab : Dict.t;
    h     : Parameter.t;
    hb    : Parameter.t;
    h2m   : Parameter.t;
    mb    : Parameter.t;
    h2s   : Parameter.t;
    sb    : Parameter.t;
    z2h0  : Parameter.t;
    h0b   : Parameter.t;
    r     : Parameter.t;
    bias  : Parameter.t;
    c     : LookupParameter.t;
    ernn  : RNN.t;
    drnn  : RNN.t;
    input_dim   : int;
    hidden_dim  : int;
    hidden_dim2 : int;
    latent_dim  : int;
    layers : int
}

let make ~input_dim ~hidden_dim ~hidden_dim2 ~latent_dim ~layers m vocab =
    let vocab_size = Dict.length vocab in {
    vocab = vocab;
    h    = Model.add_parameters m (!@[|hidden_dim2; hidden_dim|]);
    hb   = Model.add_parameters m (!@[|hidden_dim2|]);
    h2m  = Model.add_parameters m (!@[|latent_dim; hidden_dim2|]);
    mb   = Model.add_parameters m (!@[|latent_dim|]);
    h2s  = Model.add_parameters m (!@[|latent_dim; hidden_dim2|]);
    sb   = Model.add_parameters m (!@[|latent_dim|]);
    z2h0 = Model.add_parameters m (!@[|hidden_dim * layers; latent_dim|]);
    h0b  = Model.add_parameters m (!@[|hidden_dim * layers|]);
    r    = Model.add_parameters m (!@[|vocab_size; hidden_dim|]);
    bias = Model.add_parameters m (!@[|vocab_size|]);
    c    = Model.add_lookup_parameters m  vocab_size (!@[|input_dim|]);
    ernn = RNN.gru layers input_dim hidden_dim m;
    drnn = RNN.gru layers input_dim hidden_dim m;
    input_dim; hidden_dim; hidden_dim2; latent_dim; layers
}

let split_rows x h =
    let d = Expr.dim x in
    let steps = (Dim.get d 0) / h in
    ExpressionVector.init h (fun i ->
        Expr.pick_range x (i * steps) ((i+1) * steps))

let build_graph ?(samples=10) ?(flag=false) xs p cg = Expr.(
    let param = parameter cg in
    let xs_init = List.initial xs in
    let _R = param p.r in
    let bias = param p.bias in
    RNN.new_graph p.ernn cg;
    RNN.start_new_sequence p.ernn;
    List.iter (fun x ->
        let x = lookup cg p.c x in
        ignore (RNN.add_input p.ernn x)
    ) xs_init;
    let h = tanh @@ param p.h * RNN.back p.ernn + param p.hb in
    let mu = param p.h2m * h + param p.mb in
    let log_sigma = 0.5 $* param p.h2s * h + param p.sb in 
    let log_prior =
        0.5 $* sum_cols (transpose (2.0 $* log_sigma - square mu - exp (2.0 $* log_sigma))) in
    RNN.new_graph p.drnn cg;
    let errs = List.init samples (fun _ ->
        let ceps = random_normal cg (!@[|p.latent_dim|]) in
        let z = mu + ceps *. exp log_sigma in
        let h0 = param p.z2h0 * z + param p.h0b in
        let h0s = split_rows h0 p.layers in
        RNN.start_new_sequence ~init:h0s p.drnn;
        List.map2 (fun t t' ->
            let x = lookup cg p.c t in
            let y = RNN.add_input p.drnn x in
            let r = affine_transform [|bias; _R; y|] in
            pickneglogsoftmax r t') xs_init (List.tl xs)
    ) in
    let errs = Array.of_list @@ List.concat errs in
    sum errs /$ (float_of_int samples) - log_prior
)

let read_sentences dict file =
    let f word = Dict.convert_word dict word in
    let g line = List.map f (String.split_on_char ' ' line) in
    List.map g (Utils.read_lines file)


let descr =
    format_of_string
"train size: %i
dev size: %i
vocab size\t: %i
num layers\t: %i
input dim\t: %i
hidden dim\t: %i
hidden dim2\t: %i
latent dim\t: %i\n%!"

let parse_argv = function
    | [train; dev] -> (train, dev)
    | _ -> prerr_endline "Usage: rnnlm_aevb train.txt dev.txt"; exit 1

let () =
    let iteration = 30
    and input_dim = 32
    and hidden_dim = 128
    and hidden_dim2 = 32
    and latent_dim = 2
    and layers = 2 in
    let argv = Init.initialize Sys.argv in
    let train_file, dev_file = parse_argv (Array.to_list argv) in
    let m = Model.make () in
    let vocab = Dict.make () in
    let _ = Dict.convert_word vocab "<s>" in
    let _ = Dict.convert_word vocab "</s>" in
    let trainer = Trainer.adam m in
    let train_xs = read_sentences vocab train_file in
    Dict.freeze vocab;
    let dev_xs = read_sentences vocab dev_file in
    let lm = make ~input_dim ~hidden_dim
             ~hidden_dim2 ~latent_dim ~layers m vocab in

    Printf.printf descr (List.length train_xs) (List.length dev_xs)
            (Dict.length vocab) layers input_dim hidden_dim hidden_dim2 latent_dim;

    let batches = Utils.make_batch 500 train_xs in
    let eval_epoch = min 10 (List.length batches) in

    for _ = 1 to iteration do
        List.iteri (fun i xs ->
            Cg._with (fun cg ->
                let expr = build_graph xs lm cg in
                let loss = Tensor.as_scalar (Cg.forward cg expr) in
                Cg.backward cg expr;
                Trainer.update trainer;

                Printf.printf "loss: %f\n%!" loss
(*
                if (i + 1) mod eval_epoch = 0 then begin
                    let preds = List.map (tag tagger) dev_xs in
                    let index = Random.int (List.length preds) in
                    Printf.printf "accuracy: %f\n\nin: %s\ngold: %s\npred: %s\n%!"
                    (Utils.accuracy preds dev_ys)
                    (String.concat " " @@ List.nth dev_xs index)
                    (String.concat " " @@ List.nth dev_ys index)
                    (String.concat " " @@ List.nth preds index)
                end)
            *)
            )
        ) train_xs
    done

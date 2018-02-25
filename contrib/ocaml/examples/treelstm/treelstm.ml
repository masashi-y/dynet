
open Dynet

module Cg = Computationgraph

module Tree =
struct
    type 'a t = Node of 'a * 'a t * 'a t
              | Leaf of 'a * string

    let value = function
        | Node (v, _, _) | Leaf (v, _) -> v

    let rec nonterms = function
        | Leaf (v, _) -> [v]
        | Node (v, c1, c2) ->
                v :: nonterms c1 @ nonterms c2

    let rec terms = function
        | Leaf (_, w) -> [w]
        | Node (_, c1, c2) -> terms c1 @ terms c2

    let parse str =
        let error () = invalid_arg ("failed to parse " ^ str) in
        let preprocess s = Str.(
            let regex = regexp "\\([()]\\)" in
            let s = global_replace regex " \\1 " s in
            split (regexp " +") s) in
        let rec aux stack = function
            | [] -> begin match stack with
               | [`Close x] -> x
               | _ -> error ()
            end
            | "(" :: score :: "(" :: rest
                -> let n x y = Node (int_of_string score, x, y) in
                   aux (`Open n :: stack) ("(" :: rest)
            | "(" :: score :: word :: ")" :: rest
                -> let n = Leaf (int_of_string score, word) in
                   aux (`Close n :: stack) rest
            | ")" :: rest -> begin match stack with
                | `Close c2 :: `Close c1 :: `Open p :: ss
                    -> let n = p c1 c2 in
                       aux (`Close n :: ss) rest
                | _ -> error ()
            end
            | _ -> error () in
        aux [] (preprocess str)
end

open Tree

let read_trees file = List.map Tree.parse (Utils.read_lines file)

type t = {
    vocab : Dict.t;
    pE : LookupParameter.t;
    pWS : Parameter.t array;
}


let make m vocab hdim wdim = {
    vocab = vocab;
    pWS = Model.(
        [|add_parameters m (!@[|hdim; wdim|]);      (* 0: Wi   *)
          add_parameters m (!@[|hdim; wdim|]);      (* 1: Wo   *)
          add_parameters m (!@[|hdim; wdim|]);      (* 2: Wu   *)
          add_parameters m (!@[|hdim; 2*hdim|]);    (* 3: Ui   *)
          add_parameters m (!@[|hdim; 2*hdim|]);    (* 4: Uo   *)
          add_parameters m (!@[|hdim; 2*hdim|]);    (* 5: Uu   *)
          add_parameters m (!@[|hdim; hdim|]);      (* 6: UFS1 *)
          add_parameters m (!@[|hdim; hdim|]);      (* 7: UFS2 *)
          add_parameters m (!@[|hdim|]);            (* 8: Bi   *)
          add_parameters m (!@[|hdim|]);            (* 9: Bo   *)
          add_parameters m (!@[|hdim|]);            (* 10: Bu  *)
          add_parameters m (!@[|hdim|])|] );        (* 11: Bf  *)
    pE = Model.add_lookup_parameters m (Dict.length vocab) (!@[|wdim|])
}


let expr_for_tree cg {vocab; pWS; pE} tree = Expr.(
    let _WS = Array.map (parameter cg) pWS in
    let rec aux = function
    | Leaf (_, word) ->
            let e = lookup cg pE (Dict.convert_word vocab word) in
            let i = logistic @@ affine_transform [|_WS.(8);  _WS.(0); e|]
            and o = logistic @@ affine_transform [|_WS.(9);  _WS.(1); e|]
            and u = tanh     @@ affine_transform [|_WS.(10); _WS.(2); e|] in
            let c = i *. u in
            let h = o *. (tanh c) in
            Leaf ((h, c), word)
    | Node (_, ch1, ch2) ->
            let n1 = aux ch1 in
            let n2 = aux ch2 in
            let h1, c1 = Tree.value n1 in
            let h2, c2 = Tree.value n2 in
            let e = concatenate [|h1; h2|] in
            let i  = logistic @@ affine_transform [|_WS.(8); _WS.(3); e|]
            and o  = logistic @@ affine_transform [|_WS.(9); _WS.(4); e|]
            and u  = tanh     @@ affine_transform [|_WS.(10); _WS.(5); e|]
            and f1 = logistic @@ affine_transform [|_WS.(11); _WS.(6); h1|]
            and f2 = logistic @@ affine_transform [|_WS.(11); _WS.(7); h2|] in
            let c  = i *. u + f1 *. c1 + f2 *. c2 in
            let h  = o *. (tanh c) in
            Node ((h, c), n1, n2)
    in aux tree)


let train trainer pW lstm xs yss =
    let f x ys = Cg._with (fun cg ->
        let _W = Expr.parameter cg pW in
        let y_preds = Tree.nonterms (expr_for_tree cg lstm x) in
        let losses = List.map2 Expr.(fun y (h, _) ->
            pickneglogsoftmax (_W * h) y
            ) ys y_preds in
        let loss_expr = Expr.sum (Array.of_list losses) in
        let loss = Tensor.as_scalar (Cg.forward cg loss_expr) in
        Cg.backward cg loss_expr;
        Trainer.update trainer;
        (loss, float_of_int @@ List.length ys)) in
    let (losses, lengths) = List.split (List.map2 f xs yss) in
    Utils.(sum losses /. sum lengths)


let tag pW lstm x =
    Cg._with (fun cg ->
        let _W = Expr.parameter cg pW in
        let h, _ = Tree.value @@ expr_for_tree cg lstm x in
        let v = Tensor.as_vector @@ Cg.forward cg Expr.(_W * h) in
        argmax v)


let make_vocab xs =
    let vocab = Dict.make () in
    Hashtbl.iter (fun w count ->
        if count > 2 then
            ignore (Dict.convert_word vocab w))
        (Utils.word_count @@ List.flatten xs);
    Dict.freeze vocab;
    Dict.set_unk vocab "<unk>";
    vocab

let descr =
    format_of_string
"train size: %i
dev size: %i
vocab size\t: %i
label size\t: %i
embed dim\t: %i
hidden dim\t: %i\n%!"

let parse_argv = function
    | [train; dev] -> (train, dev)
    | _ -> prerr_endline "Usage: treelstm train.txt dev.txt"; exit 1

let () =
    let emb_dim = 80 in
    let hidden_dim = 100 in
    let iteration = 30 in
    let labels = 5 in
    let argv = Init.initialize Sys.argv in
    let train_file, dev_file = parse_argv (Array.to_list argv) in
    let train_xs = read_trees train_file in
    let dev_xs = read_trees dev_file in
    let train_ys = List.map Tree.nonterms train_xs in
    let dev_ys = List.map Tree.value dev_xs in
    let vocab = make_vocab (List.map Tree.terms train_xs) in
    let m = Model.make () in
    let pW = Model.add_parameters m (!@[|labels; hidden_dim|]) in
    let trainer = Trainer.adam m in
    let lstm = make m vocab hidden_dim emb_dim in
    let dev_size = List.length dev_ys in

    Printf.printf descr (List.length train_xs) (List.length dev_xs)
            (Dict.length vocab) labels emb_dim hidden_dim;

    let batch_xs = Utils.make_batch 500 train_xs in
    let batch_ys = Utils.make_batch 500 train_ys in
    let batches = List.combine batch_xs batch_ys in
    let eval_epoch = min 10 (List.length batches) in

    let accuracy preds =
        let correct = List.fold_left2 
                (fun c p g -> if p = g then c + 1 else c)
            0 preds dev_ys in
        float_of_int correct /. float_of_int dev_size in

    for _ = 1 to iteration do
        List.iteri (fun i (xs, ys) ->
            let loss = train trainer pW lstm xs ys in
            Trainer.status trainer;
            Printf.printf "loss: %f\n%!" loss;
            if (i + 1) mod eval_epoch = 0 then begin
                let preds = List.map (tag pW lstm) dev_xs in
                Printf.printf "accuracy: %f\n%!" (accuracy preds)
            end
        ) batches
    done

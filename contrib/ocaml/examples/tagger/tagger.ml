
open Dynet

module Cg = Computationgraph

type t = {
    vocab : Dict.t;
    labels : Dict.t;
    embed : LookupParameter.t;
    pH : Parameter.t;
    pO : Parameter.t;
    fwd_lstm : RNN.t;
    bwd_lstm : RNN.t;
}

let make ~layers ~wembed_dim ~hidden_dim ~mlp_dim m vocab labels =
    let vocab_size = Dict.length vocab in
    let label_size = Dict.length labels in
    let res = {
        vocab; labels;
        embed = Model.add_lookup_parameters m vocab_size (!@[|wembed_dim|]);
        pH = Model.add_parameters m (!@[|mlp_dim; hidden_dim*2|]);
        pO = Model.add_parameters m (!@[|label_size; mlp_dim|]);
        fwd_lstm = RNN.vanilla_lstm layers wembed_dim hidden_dim m;
        bwd_lstm = RNN.vanilla_lstm layers wembed_dim hidden_dim m
    } in
    res


let build_graph {vocab; pH; pO; embed; fwd_lstm; bwd_lstm} cg xs = Expr.(
    let _H = parameter cg pH in
    let _O = parameter cg pO in
    RNN.new_graph fwd_lstm cg;
    RNN.new_graph bwd_lstm cg;
    RNN.start_new_sequence fwd_lstm;
    RNN.start_new_sequence bwd_lstm;

    let embs = List.map (fun x ->
        let id = Dict.convert_word vocab x in
        lookup cg embed id) xs in
    let fwds = List.map (fun e -> RNN.add_input fwd_lstm e) embs in
    let bwds = List.map (fun e -> RNN.add_input bwd_lstm e) (List.rev embs) in
    List.map2 (fun f b ->
        _O * tanh (_H * concatenate [|f; b|])) fwds (List.rev bwds)
)

let loss cg tagger xs ys =
    let exprs = build_graph tagger cg xs in
    let f expr y =
        let y' = Dict.convert_word tagger.labels y in
        Expr.pickneglogsoftmax expr y' in
    let errs = List.map2 f exprs ys in
    Expr.sum (Array.of_list errs)


let train trainer tagger xs ys =
    let f x y =
        Cg._with (fun cg ->
            let loss_expr = loss cg tagger x y in
            let loss = Tensor.as_scalar (Cg.forward cg loss_expr) in
            Cg.backward cg loss_expr;
            Trainer.update trainer;
            (loss, float_of_int (List.length x))) in
    let (losses, lengths) = List.split (List.map2 f xs ys) in
    Utils.sum losses /. Utils.sum lengths


let tag tagger xs =
    Cg._with (fun cg ->
        let exprs = build_graph tagger cg xs in
        List.map (fun expr ->
            let v = Tensor.as_vector (Cg.forward cg expr) in
            Dict.convert_id tagger.labels (argmax v)) exprs
        )

let read_data file = List.(
    let f tok = match String.split_on_char '|' tok with
        | [word; tag] -> (String.lowercase_ascii word, tag)
        | _ -> failwith "failed to load input file" in
    let g line = split (map f (String.split_on_char ' ' line)) in
    split (map g (Utils.read_lines file))
)


let make_vocab xs ys =
    let vocab = Dict.make () in
    let labels = Dict.make () in
    Hashtbl.iter (fun w count ->
        if count > 2 then
            ignore (Dict.convert_word vocab w))
        (Utils.word_count @@ List.flatten xs);
    List.iter (fun w ->
            ignore (Dict.convert_word labels w))
        (List.flatten ys);
    Dict.freeze vocab;
    Dict.set_unk vocab "<unk>";
    Dict.freeze labels;
    (vocab, labels)


let descr =
    format_of_string
"train size: %i
dev size: %i
vocab size\t: %i
label size\t: %i
num layers\t: %i
embed dim\t: %i
hidden dim\t: %i
mlp dim\t: %i\n%!"

let parse_argv = function
    | [train; dev] -> (train, dev)
    | _ -> prerr_endline "Usage: tagger train.txt dev.txt"; exit 1

let () =
    let iteration = 30
    and layers = 2
    and wembed_dim = 80
    and hidden_dim = 100
    and mlp_dim = 100 in
    let argv = Init.initialize Sys.argv in
    let train_file, dev_file = parse_argv argv in
    let m = Model.make () in
    let trainer = Trainer.adam m in
    let train_xs, train_ys = read_data train_file in
    let train_xs, train_ys = Utils.shuffle2 train_xs train_ys in
    let dev_xs, dev_ys = read_data dev_file in
    let vocab, labels = make_vocab train_xs train_ys in
    let tagger = make m vocab labels
         ~layers ~wembed_dim ~hidden_dim ~mlp_dim in

    Printf.printf descr (List.length train_xs) (List.length dev_xs)
            (Dict.length vocab) (Dict.length labels) layers wembed_dim hidden_dim mlp_dim;

    let batch_xs = Utils.make_batch 500 train_xs in
    let batch_ys = Utils.make_batch 500 train_ys in
    let batches = List.combine batch_xs batch_ys in
    let eval_epoch = min 10 (List.length batch_xs) in

    for _ = 1 to iteration do
        List.iteri (fun i (xs, ys) ->
            let loss = train trainer tagger xs ys in
            Trainer.status trainer;
            Printf.printf "loss = %f\n%!" loss;
            if i + 1 mod eval_epoch = 0 then begin
                let preds = List.map (tag tagger) dev_xs in
                let index = Random.int (List.length preds) in
                Printf.printf "accuracy: %f\n\nin: %s\ngold: %s\npred: %s\n%!"
                (Utils.accuracy preds dev_ys)
                (String.concat " " @@ List.nth dev_xs index)
                (String.concat " " @@ List.nth dev_ys index)
                (String.concat " " @@ List.nth preds index)
            end
        ) batches
    done

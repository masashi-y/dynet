
open Dynet

module List =
struct
    include List

    let init n ~f =
        if n < 0 then raise (Invalid_argument "init");
        let rec aux i accum =
            if i = 0 then accum
            else aux (i-1) (f (i-1) :: accum) in
        aux n []

    let index lst v = 
        let rec aux i lst v = match lst with
        | [] -> raise Not_found
        | hd :: tl -> if hd = v then i
                      else aux (i + 1) tl v in
        aux 0 lst v
end

module Cg = Computationgraph

type t = {
    e_R : LookupParameter.t;
    e_I : LookupParameter.t;
    r_R : LookupParameter.t;
    r_I : LookupParameter.t;
}

let make ?(embed_dim=200) m num_ents num_rels =
    let init = ParameterInit.glorot () in {
    e_R = Model.add_lookup_parameters m ~init num_ents (!@[|embed_dim|]);
    e_I = Model.add_lookup_parameters m ~init num_ents (!@[|embed_dim|]);
    r_R = Model.add_lookup_parameters m ~init num_rels (!@[|embed_dim|]);
    r_I = Model.add_lookup_parameters m ~init num_rels (!@[|embed_dim|]);
}


let make_graph ?drop_rate { e_R; e_I; r_R; r_I } cg (e1, rel, e2) = Expr.(
    let dropout x = match drop_rate with
        | Some r -> Expr.dropout x r
        | None -> x in
    let lookup = lookup_batch_vec cg in
    let e1_R  = dropout @@ lookup e_R e1 in
    let e2_R  = dropout @@ lookup e_R e2 in
    let rel_R = dropout @@ lookup r_R rel in
    let e1_I  = dropout @@ lookup e_I e1 in
    let e2_I  = dropout @@ lookup e_I e2 in
    let rel_I = dropout @@ lookup r_I rel in

    let rrr = e1_R *. rel_R *. e2_R in
    let rii = e1_R *. rel_I *. e2_I in
    let iri = e1_I *. rel_R *. e2_I in
    let iir = e1_I *. rel_I *. e2_R in
    logistic @@ sum_elems @@ rrr + rii + iri - iir
)

let calc_rank scores y =
    let pair x y = (x, y) in
    let compare (_, s1) (_, s2) = - compare s1 s2 in
    let rank = List.(scores |> mapi pair |> sort compare |> map fst) in
    List.index rank y

let evaluate complex xss =
    let mrr = List.fold_left (fun mrr ((e1, _, _) as xs) ->
        Cg._with (fun cg ->
            let expr = make_graph complex cg xs in
            ignore (Cg.forward cg expr);
            let rev_rank = List.mapi (fun i y ->
                let scores = Tensor.as_list Expr.(value @@ pick_batch_elem expr i) in
                let rank = calc_rank scores y in
                1.0 /. float_of_int rank) (UnsignedVector.to_list e1) in
            rev_rank @ mrr
        )
    ) [] xss in
    let mrr = Utils.sum mrr /. float_of_int (List.length mrr) in
    Printf.printf "MRR = %f\n%!" mrr


let make_vocab path =
    let vocab = Dict.make () in
    List.iter (fun w ->
            ignore (Dict.convert_word vocab w))
        (Utils.read_lines path);
    Dict.freeze vocab;
    Dict.set_unk vocab "<unk>";
    vocab

let read_triplets ?negative_sample ents rels path =
    let ents_len = Dict.length ents in
    let parse line (xs, ys) =
        match String.split_on_char '\t' line, negative_sample with
        | [e1; rel; e2], None ->
            let x = Dict.(convert_word ents e1, convert_word rels rel, convert_word ents e2) in
            (x :: xs, 1.0 :: ys)
        | [e1; rel; e2], Some n ->
            let e1 = Dict.convert_word ents e1 in
            let rel = Dict.convert_word rels rel in
            let e2 = Dict.convert_word ents e2 in
            let negs = List.init n (fun _ -> (e1, rel, Random.int ents_len)) in
            ((e1, rel, e2) :: negs @ xs, 1.0 :: List.init n (fun _ -> 0.0) @ ys)
        | _ -> raise (Invalid_argument line) in
    List.fold_right parse (Utils.read_lines path) ([], [])



type cfg = {
    rels : string;
        (** path to the list of relations *)
    ents : string;
        (** path to the list of entities *)
    train_file : string;
        (** path to training file *)
    valid_file : string;
        (** path to validation file *)
    iteration : int;
        (** the number of iteration in training *)
    embed_dim : int;
        (** the dimension size of embedding vectors *)
    batch_size : int;
        (** batch size *)
    eval_iteration : int;
        (** evaluate on validation data every this value *)
} [@@deriving show, argparse]

let default = {
    rels = "data/kb/wordnet-mlj12/train.rellist";
    ents = "data/kb/wordnet-mlj12/train.entlist";
    train_file = "data/kb/wordnet-mlj12/wordnet-mlj12-train.txt";
    valid_file = "data/kb/wordnet-mlj12/wordnet-mlj12-valid.txt";
    iteration = 300;
    embed_dim = 200;
    batch_size = 64;
    eval_iteration = 10;

}

let vectorize xs = 
    let (e1s, rels, e2s) = List.fold_right (fun (e1, rel, e2) (e1s, rels, e2s) ->
            (e1 :: e1s, rel :: rels, e2 :: e2s)) xs ([], [], []) in
    UnsignedVector.(of_list e1s, of_list rels, of_list e2s)

let () =
    let argv = Init.initialize Sys.argv in
    let cfg, _ = argparse_cfg default "complex" argv in
    print_endline (show_cfg cfg);
    let rels = make_vocab cfg.rels in
    let ents = make_vocab cfg.ents in
    prerr_endline "reading dataset...";
    let trainXs, trainYs = read_triplets ~negative_sample:5 ents rels cfg.train_file in
    let validXs, _ = read_triplets ents rels cfg.valid_file in
    let validXs = List.map vectorize (Utils.make_batch cfg.batch_size validXs) in
    let num_rels = Dict.length rels in
    let num_ents = Dict.length ents in
    let m = Model.make () in
    let trainer = Trainer.adam m in
    prerr_endline "creating ComplEx model...";
    let complex = make ~embed_dim:cfg.embed_dim m num_ents num_rels in
    let e1s  = UnsignedVector.make cfg.batch_size 0 in
    let e2s  = UnsignedVector.make cfg.batch_size 0 in
    let rels = UnsignedVector.make cfg.batch_size 0 in
    let xs = (e1s, rels, e2s) in
    let y_value = FloatVector.make cfg.batch_size 0.0 in
    prerr_endline "start training...";

    for i = 1 to cfg.iteration do
        let trainXs', trainYs' = Utils.shuffle2 trainXs trainYs in
        let trainXs' = Utils.make_batch cfg.batch_size trainXs' in
        let trainYs' = Utils.make_batch cfg.batch_size trainYs' in
        let losses = List.map2 (fun xs' ys' ->
            List.iteri UnsignedVector.(fun i (e1, rel, e2) ->
                set e1s i e1; set e2s i e2; set rels i rel) xs';
            List.iteri FloatVector.(fun i y ->
                set y_value i y) ys';
            Cg._with (fun cg ->
                let ys = Expr.input cg (Dim.make ~batch:cfg.batch_size [|1|]) y_value in
                let graph = Expr.(mean_batches @@ binary_log_loss (make_graph ~drop_rate:0.2 complex cg xs) ys) in
                let loss = Tensor.as_scalar @@ Cg.forward cg graph in
                Cg.backward cg graph;
                Trainer.update trainer;
                loss)
        ) trainXs' trainYs' in
        Trainer.status trainer;
        Printf.printf "iter = %i, loss = %f\n%!" i
            (Utils.sum losses /. float_of_int (List.length losses))
        (* if i mod cfg.eval_iteration = 0 then begin *)
            (* evaluate complex validXs *)
        (* end *)
    done



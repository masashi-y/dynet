
let read_lines file =
    let ch = open_in file in
    let rec parse () =
        let one = try Some (input_line ch)
                with End_of_file -> close_in_noerr ch; None
        in match one with
            | Some s -> s :: parse ()
            | None -> []
    in parse ()

let sum ls = List.fold_left (+.) 0.0 ls

let word_count lst =
    let table = Hashtbl.create 1000 in
    List.iter (fun x ->
        let prev_count =
            match Hashtbl.find_opt table x with
            | Some n -> n
            | None -> 0 in
        Hashtbl.replace table x (prev_count + 1)) lst;
    table

let list_split_at n lst =
    let rec aux n hs rest =
        if n <= 0 then (hs, rest) else begin
        match rest with
        | [] -> (hs, [])
        | h :: rest -> let (hs, ts) = aux (n - 1) hs rest
                       in (h :: hs, ts)
        end
    in aux n [] lst

let rec make_batch n lst =
    let (x, xs) = list_split_at n lst in
    match xs with
    | [] -> [x]
    | _  -> x :: make_batch n xs

let shuffle d =
    let nd = List.rev_map (fun c -> (Random.bits (), c)) d in
    let sond = List.sort compare nd in
    List.rev_map snd sond

let shuffle2 d1 d2 =
    let nd = List.rev_map2 (fun c1 c2 -> (Random.bits (), (c1, c2))) d1 d2 in
    let sond = List.sort compare nd in
    List.fold_left (fun (l1, l2) (_, (c1, c2)) ->
        (c1::l1, c2::l2)) ([], []) sond

let accuracy preds golds =
    let preds = List.flatten preds in
    let golds = List.flatten golds in
    let correct = List.fold_left2 (
        fun c p g -> if p = g then c + 1 else c)
            0 preds golds in
    float_of_int correct /. float_of_int (List.length preds)

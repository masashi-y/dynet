
type t = {
    ids : (string, int) Hashtbl.t;
    words : (int, string) Hashtbl.t;
    mutable frozen : bool;
    mutable map_unk : bool;
    mutable unk_id : int
}

let make ?(init=2000) () = {
    ids = Hashtbl.create init;
    words = Hashtbl.create init;
    frozen = false;
    map_unk = false;
    unk_id = -1
}

let length {ids} = Hashtbl.length ids

let mem {ids} w = Hashtbl.mem ids w

let freeze d = d.frozen <- true

let is_frozen {frozen} = frozen

let convert_word {ids; words; frozen; map_unk; unk_id} w =
    try Hashtbl.find ids w
    with Not_found -> if frozen then
        if map_unk then
            unk_id
        else
            failwith (
        Printf.sprintf "Unknown word encountered in frozen dictionary: %s" w)
    else
        let id = Hashtbl.length words in
        Hashtbl.add words id w;
        Hashtbl.add ids w id;
        id

let convert_id {words} id =
    try Hashtbl.find words id
    with Not_found -> failwith (
        Printf.sprintf "Out-of-bounds error in Dict.convert_id for word ID %i" id)

let set_unk ({frozen; map_unk} as d) w =
    if not frozen then
        failwith "Please call set_unk only after dictionary is frozen";
    if map_unk then
        failwith "Set UNK more than one time";
    d.frozen <- false;
    d.unk_id <- (convert_word d w);
    d.frozen <- true;
    d.map_unk <- true

let get_unk_id {unk_id} = unk_id

let get_words {ids} = ids

let clear {words; ids} = Hashtbl.reset words; Hashtbl.reset ids

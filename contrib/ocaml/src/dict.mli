
type t

val make : ?init:int -> unit -> t

val length : t -> int

val mem : t -> string -> bool

val freeze : t -> unit

val is_frozen : t -> bool

val convert_word : t -> string -> int

val convert_id : t -> int -> string

val set_unk : t -> string -> unit

val get_unk_id : t -> int

val get_words : t -> (string, int) Hashtbl.t

val clear : t -> unit

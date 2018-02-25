
open Params

module Saver :
sig
    type t
    val save_model : ?key:string -> t -> ParameterCollection.t -> unit
    val save_parameter : ?key:string -> t -> Parameter.t -> unit
    val save_lookup_parameter : ?key:string -> t -> LookupParameter.t -> unit

    val textfile : ?append:bool -> string -> t
end


module Loader :
sig
    type t
    val populate_model : ?key:string -> t -> ParameterCollection.t -> unit
    val populate_parameter : ?key:string -> t -> Parameter.t -> unit
    val populate_lookup_parameter : ?key:string -> t -> LookupParameter.t -> unit

    val load_param : t -> ParameterCollection.t -> string -> Parameter.t
    val load_lookup_param : t -> ParameterCollection.t -> string -> LookupParameter.t
    val textfile : string -> t
end


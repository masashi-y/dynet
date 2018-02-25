
open Swig
open Dynet_swig
open Params

module Saver =
struct
    type t = c_obj
    let save_model ?(key="") s p = ignore (s -> save ((ParameterCollection.to_ptr p), (key to string)))
    let save_parameter ?(key="") s p = ignore (s -> save ((Parameter.to_ptr p), (key to string)))
    let save_lookup_parameter ?(key="") s p = ignore (s -> save ((LookupParameter.to_ptr p), (key to string)))

    let textfile ?(append=false) filename =
        new_TextFileSaver '((filename to string), (append to bool))
end


module Loader =
struct
    type t = c_obj
    let populate_model ?(key="") l p = ignore (l -> populate ((ParameterCollection.to_ptr p), (key to string)))
    let populate_parameter ?(key="") l p = ignore (l -> populate ((Parameter.to_ptr p), (key to string)))
    let populate_lookup_parameter ?(key="") l p = ignore (l -> populate ((LookupParameter.to_ptr p), (key to string)))

    let load_param l m k = Parameter.from_ptr (l -> load_param ((ParameterCollection.to_ptr m), (k to string)))
    let load_lookup_param l m k = LookupParameter.from_ptr (l -> load_lookup_param((ParameterCollection.to_ptr m), (k to string)))
    let textfile filename =
        new_TextFileLoader '((filename to string))
end


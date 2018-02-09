
open Dynet
open Dynet__Params

let () = 
    Dynet.test ();Dynet.cleanup ()

let () =
    let argv = Dynet.initialize (Sys.argv) in
    let m = ParameterCollection.make () in
    let p = ParameterCollection.add_parameters
        ~init:(Some (parameter_init_const 100.0))
        m (Dynet__Dim.make [|2;2|]) 0.0 in
    let t = Parameter.values p in
    Printf.printf "%s\n" (Dynet__Tensor.show t)



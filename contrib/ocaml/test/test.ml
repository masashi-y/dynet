
open Dynet

let () = 
    let argv = Dynet.initialize (Sys.argv) in
    Dynet.test ();Dynet.cleanup ();
    List.iter (fun v -> Printf.printf "%s\n" v) argv


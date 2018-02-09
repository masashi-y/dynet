
(* This example was mostly lifted from the guile example directory *)

open Swig
open Dynet_swig

let () =
    let d = Dim.make ~batch:5 [|2; 3|] in
    Printf.printf "%s\n" (Dim.show d);
    Printf.printf "%d\n" (Dim.get d 1)

let cleanup () = ignore (_cleanup '())

let initialize argv = 
    let ps = new_DynetParams '() in
    let rest = ref [] in
    let spec = 
        [("-dynet-mem",         Arg.String (fun v -> ignore (ps -> "[mem_descriptor]" ((v to string)))), "");
         ("-random-seed",       Arg.Int    (fun v -> ignore (ps -> "[random_seed]" ((v to int)))),       "");
         ("-weight-decay",      Arg.Float  (fun v -> ignore (ps -> "[weight_decay]" ((v to float)))),    "");
         ("-shared-parameters", Arg.Bool   (fun v -> ignore (ps -> "[shared_parameters]" ((v to bool)))),"");
         ("-autobatch",         Arg.Int    (fun v -> ignore (ps -> "[autobatch]" ((v to int)))),         "");
         ("-profiling",         Arg.Int    (fun v -> ignore (ps -> "[profiling]" ((v to int)))),         "");
         ("-ngpus-requested",   Arg.Bool   (fun v -> ignore (ps -> "[ngpus_requested]" ((v to bool)))),  "");
         ("-ids-requested",     Arg.Bool   (fun v -> ignore (ps -> "[ids_requested]" ((v to bool)))),    "");
         ("-cpu-requested",     Arg.Bool   (fun v -> ignore (ps -> "[cpu_requested]" ((v to bool)))),    "");
         ("-requested-gpus",    Arg.Int    (fun v -> ignore (ps -> "[requested_gpus]" ((v to int)))),    "");
         (* std::vector<int> gpu_mask; /**< List of required GPUs by ids */ *)
         ("--", Arg.Rest (fun v -> rest := v :: !rest), "") ] in
    Arg.parse spec (fun v -> rest := v :: !rest) "";
    ignore (_initialize ps);
    List.rev @@ !rest

let with_vector v f =
  for i = 0 to ((v -> size()) as int) - 1 do
    f v i
  done

let print_DoubleVector v =
  begin
    with_vector v 
      (fun v i -> 
	 print_float ((v '[i to int]) as float) ;
	 print_string " ") ;
    print_endline 
  end

(* Call average with a Ocaml array... *)

let test () =
    let v = new_DoubleVector '() in
    let rec fill_dv v x =
      if x < 0.0001 then v else 
        begin
          v -> push_back ((x to float)) ;
          fill_dv v (x *. x)
        end in
    let _ = fill_dv v 0.999 in
    let _ = print_DoubleVector v ; print_endline "" in
    ()

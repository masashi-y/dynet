
open Swig
open Dynet_swig

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
    Arg.parse_argv argv spec (fun v -> rest := v :: !rest) "";
    ignore (_initialize ps);
    Array.of_list (Sys.argv.(0) :: (List.rev @@ !rest))


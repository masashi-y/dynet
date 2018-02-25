
open Dynet

module Cg = Computationgraph

let hidden_size = 8
let iteration = 30

let () =
    let argv = Init.initialize Sys.argv in
    let m = Model.make () in
    let sgd = Trainer.simple_sgd m in
    let pW = Model.add_parameters m (!@[|hidden_size; 2|])
    and pb = Model.add_parameters m (!@[|hidden_size|])
    and pV = Model.add_parameters m (!@[|1; hidden_size|])
    and pa = Model.add_parameters m (!@[|1|]) in
    (match (Array.to_list argv) with
        | fname :: _ -> Loader.(populate_model (textfile fname) m)
        | _ -> ());
    let x_values = FloatVector.make 2 0.0 in
    let y_values = FloatVector.make 1 0.0 in
    Cg._with Expr.(fun cg ->
        let _W = parameter cg pW
        and b = parameter cg pb
        and _V = parameter cg pV
        and a = parameter cg pa in
        let x = input cg (!@[|2|]) x_values in
        let y = input cg (!@[|1|]) y_values in
        let h = tanh (_W * x + b) in
        let y_pred = _V * h + a in
        let loss_expr = squared_distance y_pred y in
        Cg.print_graphviz cg;
        for i = 1 to iteration do
            let loss = ListLabels.fold_left [0; 1; 2; 3]
            ~init:0.0 ~f:(fun prev_loss mi ->
                let x1 = mi mod 2
                and x2 = (mi / 2) mod 2 in
                FloatVector.set x_values 0 (if x1 > 0 then 1.0 else -1.0);
                FloatVector.set x_values 1 (if x2 > 0 then 1.0 else -1.0);
                FloatVector.set y_values 0 (if x1 <> x2 then 1.0 else -1.0);
                let loss = Tensor.as_scalar (Cg.forward cg loss_expr) in
                Cg.backward cg loss_expr;
                Trainer.update sgd;
                prev_loss +. loss) in
            Printf.printf "E = %f\n" (loss /. 4.0)
        done);
        let saver = Saver.textfile "/tmp/xor.model" in
        Saver.save_model saver m


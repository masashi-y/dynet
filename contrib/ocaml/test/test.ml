
open Dynet

let () =
    let hidden_size = 8 in
    let iteration = 200 in
    let argv = Init.initialize (Sys.argv) in
    let m = Model.make () in
    let trainer = Trainer.simple_sgd m in
    let pW = Model.add_parameters m (Dim.make [|hidden_size; 2|])
    and pb = Model.add_parameters m (Dim.make [|hidden_size|])
    and pV = Model.add_parameters m (Dim.make [|1; hidden_size|])
    and pa = Model.add_parameters m (Dim.make [|1|]) in
    let x_values = FloatVector.make [|0.0; 0.0|] in
    let y_values = FloatVector.make [|0.0|] in
    let _ = Computationgraph._with Expr.(fun cg ->
        let _W = parameter cg pW
        and _b = parameter cg pb
        and _V = parameter cg pV
        and _a = parameter cg pa in
        let x = input cg (Dim.make [|2|]) x_values in
        let y = input cg (Dim.make [|1|]) y_values in
        let h = tanh (add (mul _W x) _b) in
        let y_pred = add (mul _V h) _a in
        let loss_expr = squared_distance y_pred y in
        Computationgraph.print_graphviz cg;
        for i = 1 to iteration do
            let loss = ListLabels.fold_left [0; 1; 2; 3] ~init:0.0
             ~f:(fun prev_loss mi ->
                let x1 = mi mod 2
                and x2 = (mi / 2) mod 2 in
                FloatVector.set x_values 0 (if x1 > 0 then 1.0 else -1.0);
                FloatVector.set x_values 1 (if x2 > 0 then 1.0 else -1.0);
                FloatVector.set y_values 0 (if x1 != x2 then 1.0 else -1.0);
                let loss = Tensor.as_scalar (Computationgraph.forward cg loss_expr) in
                Computationgraph.backward loss_expr;
                Trainer.update trainer;
                prev_loss +. loss) in
            Printf.printf "E = %f\n" (loss /. 4.0)
        done
    ) in
    ()


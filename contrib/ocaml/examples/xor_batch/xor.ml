
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
    let x_values = FloatVector.of_array
        [|1.0; 1.0; 1.0; -1.0; -1.0; 1.0; -1.0; -1.0|] in
    let y_values = FloatVector.of_array
        [|-1.0; 1.0; 1.0; -1.0|] in
    Cg._with Expr.(fun cg ->
        let _W = parameter cg pW
        and b = parameter cg pb
        and _V = parameter cg pV
        and a = parameter cg pa in
        let x = input cg (Dim.make ~batch:4 [|2|]) x_values in
        let y = input cg (Dim.make ~batch:4 [|1|]) y_values in
        let h = tanh (_W * x + b) in
        let y_pred = _V * h + a in
        let loss = squared_distance y_pred y in
        let sum_loss = sum_batches loss in
        Cg.print_graphviz cg;
        for i = 1 to iteration do
            let loss = Tensor.as_scalar (Cg.forward cg sum_loss) in
            Cg.backward cg sum_loss;
            Trainer.update sgd;
            Printf.printf "E = %f\n" (loss /. 4.0)
        done)



open Dynet
open Transformer
module Config = Transformer__Config

let () =
    let argv = Init.initialize Sys.argv in
    let _ = Config.parse argv in
    let m = Model.make () in
    let enc = Transformer.make m Config.{
        default with position_encoding = 1;
        src_vocab_size = 100;
        tgt_vocab_size = 100;
        use_label_smoothing = true} (Dict.make()) (Dict.make()) in
    let xss = [[1;2;3];[1;3;4;3];[1;3;3;3;3;3;3;]] in
    Cg._with (fun cg ->
        let res = Transformer.apply ~is_eval:false enc cg xss xss in
        print_dim (Expr.dim res);
        print_tensor (Expr.value res)
    )


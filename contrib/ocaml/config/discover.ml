
open Base
open Stdio

let write_sexp fn sexp =
  Out_channel.write_all fn ~data:(Sexp.to_string sexp)

let (!%) = Printf.sprintf

let getenv var =
    try Caml.Sys.getenv var
    with Not_found ->
        failwith (!%"Not found environment variable for %s. Please set the variable." var)


let () =
    let dynet = getenv "DYNETROOT" in
    let eigen = getenv "EIGEN3_INCLUDE_DIR" in
    let cflags = ["-xc++";
                  !%"-I%s" dynet;
                  !%"-I%s/dynet" dynet;
                  !%"-I%s" eigen;
                  "-std=c++11";
                  "-DCAML_NAME_SPACE=1"] in
    let libs = ["-lstdc++"; !%"-L%sbuild/dynet" dynet; "-ldynet"] in

    write_sexp "c_flags.sexp"         (sexp_of_list sexp_of_string cflags);
    write_sexp "c_library_flags.sexp" (sexp_of_list sexp_of_string libs)

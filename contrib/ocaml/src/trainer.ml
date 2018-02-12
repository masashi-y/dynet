
open Swig
open Dynet_swig
open Vectors
open Params

type t = c_obj

let simple_sgd ?(learning_rate=0.1) m =
    new_SimpleSGDTrainer '((ParameterCollection.to_ptr m), (learning_rate to float))
    
let cyclical_sgd ?(learning_rate_min=0.01) ?(learning_rate_max=0.1) ?(step_size=2000) ?(gamma=0.0) ?(edecay=0.0) m =
    new_CyclicalSGDTrainer '((ParameterCollection.to_ptr m), (learning_rate_min to float), (learning_rate_max to float),
        (step_size to int), (gamma to float), (edecay to float))

let momentum_sgd ?(learning_rate=0.01) ?(mom=0.9) m =
    new_MomentumSGDTrainer '((ParameterCollection.to_ptr m), (learning_rate to float), (mom to float))

let adagrad ?(learning_rate=0.1) ?(eps=1e-20) m =
    new_AdagradTrainer '((ParameterCollection.to_ptr m), (learning_rate to float), (eps to float))

let adadelta ?(eps=1e-6) ?(rho=0.95) m =
    new_AdadeltaTrainer '((ParameterCollection.to_ptr m), (eps to float), (rho to float))

let rms_prop ?(learning_rate=0.1) ?(eps=1e-20) ?(rho=0.95) m =
    new_RMSPropTrainer '((ParameterCollection.to_ptr m), (learning_rate to float), (eps to float), (rho to float))

let adam ?(learning_rate=0.001) ?(beta_1=0.9) ?(beta_2=0.999) ?(eps=1e-8) m =
    new_AdamTrainer '((ParameterCollection.to_ptr m), (learning_rate to float), (beta_1 to float), (beta_2 to float), (eps to float))

let amsgrad ?(learning_rate=0.001) ?(beta_1=0.9) ?(beta_2=0.999) ?(eps=1e-8) m =
    new_AmsgradTrainer '((ParameterCollection.to_ptr m), (learning_rate to float), (beta_1 to float), (beta_2 to float), (eps to float))


let update ?update_params ?update_lookup_params t =
    let make v = IntVector.(to_ptr (of_array v)) in
    ignore (match update_params, update_lookup_params with
    | Some ps, Some lps -> t -> update ((make ps), (make lps))
    | Some ps, None -> t -> update ((make ps), (make [||]))
    | None, Some lps -> t -> update ((make [||]), (make lps))
    | None, None -> t -> update ())

let update_epoch ?(rate=1.0) t = ignore (t -> update_epoch ((rate to float)))

let restart t lr = ignore (t -> restart ((lr to float)))

let clip_gradients t = (t -> clip_gradients ()) as float

let rescale_and_reset_weight_decay t = ignore (t -> rescale_and_reset_weight_decay ())

let status t = ignore (t -> status ())

let to_ptr t = t
let from_ptr t = t

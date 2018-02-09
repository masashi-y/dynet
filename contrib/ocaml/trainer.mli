
open Params
type t

val simple_sgd : ?learning_rate:float -> ParameterCollection.t -> t
    
val cyclical_sgd : ?learning_rate_min:float
                -> ?learning_rate_max:float
                -> ?step_size:int
                -> ?gamma:float
                -> ?edecay:float -> ParameterCollection.t -> t

val momentum_sgd : ?learning_rate:float
                -> ?mom:float -> ParameterCollection.t -> t

val adagrad : ?learning_rate:float
           -> ?eps:float -> ParameterCollection.t -> t

val adadelta : ?eps:float -> ?rho:float -> ParameterCollection.t -> t

val rms_prop : ?learning_rate:float
            -> ?eps:float
            -> ?rho:float -> ParameterCollection.t -> t

val adam : ?learning_rate:float
        -> ?beta_1:float
        -> ?beta_2:float
        -> ?eps:float -> ParameterCollection.t -> t

val amsgrad : ?learning_rate:float
           -> ?beta_1:float
           -> ?beta_2:float
           -> ?eps:float -> ParameterCollection.t -> t

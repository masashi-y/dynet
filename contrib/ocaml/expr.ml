
open Swig
open Dynet_swig
open Vectors
open Params

module E = Expression

let parameter cg p =
    E.from_ptr (_parameter_Parameter '((Computationgraph.to_ptr cg), (Parameter.to_ptr p)))
let lookup_parameter cg p =
    E.from_ptr (_parameter_LookupParameter '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p)))

let const_parameter cg p =
    E.from_ptr (_const_parameter_Parameter '((Computationgraph.to_ptr cg), (Parameter.to_ptr p)))
let const_lookup_parameter cg p =
    E.from_ptr (_const_parameter_LookupParameter '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p)))

let lookup cg p i = E.from_ptr (_lookup '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (i to uint)))
let const_lookup cg p i = E.from_ptr (_const_lookup '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (i to uint)))

let lookup_batch cg p v =
    E.from_ptr (_lookup_vector '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (UnsignedVector.(to_ptr (of_array v)))))
let const_lookup_batch cg p v =
    E.from_ptr (_const_lookup_vector '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (UnsignedVector.(to_ptr (of_array v)))))

let input cg dim v =
    E.from_ptr (_input '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (FloatVector.to_ptr v)))
let input_scalar cg v =
    E.from_ptr (_input_scalar '((Computationgraph.to_ptr cg), (v to float)))
let input_sparse ?(default=0.0) cg d ids data =
    E.from_ptr (_input_sparse '((Computationgraph.to_ptr cg), (Dim.to_ptr d),
     (UnsignedVector.(to_ptr (of_array ids))), (FloatVector.(to_ptr (of_array data))), (default to float)))


let neg x = E.from_ptr (_exprNeg '((E.to_ptr x)))
let add x y = E.from_ptr (_exprPlusExEx '((E.to_ptr x), (E.to_ptr y)))
let add_scalar x y = E.from_ptr (_exprPlusExRe '((E.to_ptr x), (y to float)))
let scalar_add x y = E.from_ptr (_exprPlusExRe '((E.to_ptr y), (x to float)))

let mul x y = E.from_ptr (_exprTimesExEx '((E.to_ptr x), (E.to_ptr y)))
let mul_scalar x y = E.from_ptr (_exprTimesExRe '((E.to_ptr x), (y to float)))
let scalar_mul x y = E.from_ptr (_exprTimesExRe '((E.to_ptr y), (x to float)))

let sub x y = E.from_ptr (_exprMinusExEx '((E.to_ptr x), (E.to_ptr y)))
let sub_scalar x y = E.from_ptr (_exprMinusExRe '((E.to_ptr x), (y to float)))
let scalar_sub x y = E.from_ptr (_exprMinusReEx '((x to float), (E.to_ptr y)))

let div_scalar x y = E.from_ptr (_exprDivide '((E.to_ptr x), (y to float)))

let zeros cg dim = E.from_ptr (_zeros '((Computationgraph.to_ptr cg), (Dim.to_ptr dim)))
let zeroes cg dim = E.from_ptr (_zeroes '((Computationgraph.to_ptr cg), (Dim.to_ptr dim)))
let ones cg dim = E.from_ptr (_ones '((Computationgraph.to_ptr cg), (Dim.to_ptr dim)))
let constant cg dim v = E.from_ptr (_constant '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (v to float)))
let random_normal cg dim = E.from_ptr (_random_normal '((Computationgraph.to_ptr cg), (Dim.to_ptr dim)))
let random_bernoulli ?(scale=1.0) cg dim p = E.from_ptr (_random_bernoulli '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (p to float), (scale to float)))
let random_uniform cg dim left right = E.from_ptr (_random_uniform '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (left to float), (right to float)))
let random_gumbel ?(mu=0.0) ?(beta=1.0) cg dim = E.from_ptr (_random_gumbel '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (mu to float), (beta to float)))

let sum xs = E.from_ptr (_sum '((ExpressionVector.(to_ptr (of_array xs)))))
let sum_elems x = E.from_ptr (_sum_elems '((E.to_ptr x)))
let moment_elems x r = E.from_ptr (_moment_elems '((E.to_ptr x), (r to uint)))
let mean_elems x = E.from_ptr (_mean_elems '((E.to_ptr x)))
let std_elems x = E.from_ptr (_std_elems '((E.to_ptr x)))

let sum_batches x = E.from_ptr (_sum_batches '((E.to_ptr x)))
let moment_batches x r = E.from_ptr (_moment_batches '((E.to_ptr x), (r to uint)))
let mean_batches x = E.from_ptr (_mean_batches '((E.to_ptr x)))
let std_batches x = E.from_ptr (_std_batches '((E.to_ptr x)))

let sum_dim ?(b=false) x dims = E.from_ptr (_sum_dim '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array dims))), (b to bool)))
let moment_dim ?(b=false) ?(n=0) x dims r = E.from_ptr (_moment_dim '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array dims))), (r to uint), (b to bool), (n to uint)))
let mean_dim ?(b=false) ?(n=0) x dims = E.from_ptr (_mean_dim '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array dims))), (b to bool), (n to uint)))
let std_dim ?(b=false) ?(n=0) x dims = E.from_ptr (_std_dim '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array dims))), (b to bool), (n to uint)))

let sqrt x = E.from_ptr (_sqrt '((E.to_ptr x)))
let abs x = E.from_ptr (_abs '((E.to_ptr x)))
let erf x = E.from_ptr (_erf '((E.to_ptr x)))
let tanh x = E.from_ptr (_tanh '((E.to_ptr x)))
let exp x = E.from_ptr (_exp '((E.to_ptr x)))
let square x = E.from_ptr (_square '((E.to_ptr x)))
let cube x = E.from_ptr (_cube '((E.to_ptr x)))
let lgamma x = E.from_ptr (_lgamma '((E.to_ptr x)))
let log x = E.from_ptr (_log '((E.to_ptr x)))
let logistic x = E.from_ptr (_logistic '((E.to_ptr x)))
let rectify x = E.from_ptr (_rectify '((E.to_ptr x)))
let elu ?(alpha=1.0) x = E.from_ptr (_elu '((E.to_ptr x), (alpha to float)))
let selu x = E.from_ptr (_selu '((E.to_ptr x)))
let silu ?(beta=1.0) x = E.from_ptr (_silu '((E.to_ptr x), (beta to float)))
let softsign x = E.from_ptr (_softsign '((E.to_ptr x)))
let pow x y = E.from_ptr (_pow '((E.to_ptr x), (E.to_ptr y)))

let min x y = E.from_ptr (_min '((E.to_ptr x), (E.to_ptr y)))
let max x y = E.from_ptr (_max '((E.to_ptr x), (E.to_ptr y)))
let max xs = E.from_ptr (_max '((ExpressionVector.(to_ptr (of_array xs)))))
let dot_product x y = E.from_ptr (_dot_product '((E.to_ptr x), (E.to_ptr y)))
let cmult x y = E.from_ptr (_cmult '((E.to_ptr x), (E.to_ptr y)))
let cdiv x y = E.from_ptr (_cdiv '((E.to_ptr x), (E.to_ptr y)))
let colwise_add x b = E.from_ptr (_colwise_add '((E.to_ptr x), (E.to_ptr b)))

let softmax x = E.from_ptr (_softmax '((E.to_ptr x)))
let log_softmax x = E.from_ptr (_log_softmax '((E.to_ptr x)))
let log_softmax x restriction = E.from_ptr (_log_softmax '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array restriction)))))
let logsumexp_dim x d = E.from_ptr (_logsumexp_dim '((E.to_ptr x), (d to uint)))

let pickneglogsoftmax x v = E.from_ptr (_pickneglogsoftmax '((E.to_ptr x), (v to uint)))
(* let pickneglogsoftmax x = E.from_ptr (_pickneglogsoftmax '((E.to_ptr x), const unsigned* pv)) *)
(* let pickneglogsoftmax x = E.from_ptr (_pickneglogsoftmax '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array v))))) *)

let hinge ?(m=1.0) x index = E.from_ptr (_hinge '((E.to_ptr x), (index to uint), (m to float)))
(* let hinge x = E.from_ptr (_hinge '((E.to_ptr x), unsigned* pindex, float m = 1.0)) *)
(* let hinge x = E.from_ptr (_hinge '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array indices))), float m = 1.0)) *)

let hinge_dim ?(d=0) ?(m=1.0) x indices = E.from_ptr (_hinge_dim '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array indices))), (d to uint), (m to float)))
(* let hinge_dim x = E.from_ptr (_hinge_dim '((E.to_ptr x), const std::vector<std::vector<unsigned> >& indices, unsigned d = 0, float m = 1.0)) *)

let sparsemax x = E.from_ptr (_sparsemax '((E.to_ptr x)))
let sparsemax_loss x target_support = E.from_ptr (_sparsemax_loss '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array target_support)))))

let squared_norm x = E.from_ptr (_squared_norm '((E.to_ptr x)))
let l2_norm x = E.from_ptr (_l2_norm '((E.to_ptr x)))
let squared_distance x y = E.from_ptr (_squared_distance '((E.to_ptr x), (E.to_ptr y)))

let l1_distance x y = E.from_ptr (_l1_distance '((E.to_ptr x), (E.to_ptr y)))
let huber_distance ?(c=1.345) x y = E.from_ptr (_huber_distance '((E.to_ptr x), (E.to_ptr y), (c to float)))
let binary_log_loss x y = E.from_ptr (_binary_log_loss '((E.to_ptr x), (E.to_ptr y)))
let pairwise_rank_loss ?(m=1.0) x y = E.from_ptr (_pairwise_rank_loss '((E.to_ptr x), (E.to_ptr y), (m to float)))
let poisson_loss x y = E.from_ptr (_poisson_loss '((E.to_ptr x), (y to uint)))
(* let poisson_loss x = E.from_ptr (_poisson_loss '((E.to_ptr x), const unsigned* py)) *)

let nobackprop x = E.from_ptr (_nobackprop '((E.to_ptr x)))
let flip_gradient x = E.from_ptr (_flip_gradient '((E.to_ptr x)))
let reshape x d = E.from_ptr (_reshape '((E.to_ptr x), (Dim.to_ptr d)))
let transpose x = E.from_ptr (_transpose '((E.to_ptr x)))

let select_rows x rows = E.from_ptr (_select_rows '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array rows)))))
let select_cols x cols = E.from_ptr (_select_cols '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array cols)))))

let pick ?(d=0) x v = E.from_ptr (_pick '((E.to_ptr x), (v to uint), (d to uint)))
(* let pick ?(d=0) x v = E.from_ptr (_pick '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array v))), (d to uint))) *)
(* let pick ?(d=0) x = E.from_ptr (_pick '((E.to_ptr x), const unsigned* v, (d = 0 to uint))) *)
let pick_range ?(d=0) x s e = E.from_ptr (_pick_range '((E.to_ptr x), (s to uint), (e to uint), (d to uint)))
let pick_batch_elem x v = E.from_ptr (_pick_batch_elem '((E.to_ptr x), (v to uint)))
let pick_batch_elems x v = E.from_ptr (_pick_batch_elems '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array v)))))

let concatenate_to_batch xs = E.from_ptr (_concatenate_to_batch '((ExpressionVector.(to_ptr (of_array xs)))))
let concatenate_cols xs = E.from_ptr (_concatenate_cols '((ExpressionVector.(to_ptr (of_array xs)))))
let concatenate ?(d=0) xs = E.from_ptr (_concatenate '((ExpressionVector.(to_ptr (of_array xs))), (d to uint)))

let max_dim x d = E.from_ptr (_max_dim '((E.to_ptr x), (d to uint)))
let min_dim x d = E.from_ptr (_min_dim '((E.to_ptr x), (d to uint)))

let noise x stddev = E.from_ptr (_noise '((E.to_ptr x), (stddev to float)))
let dropout x p = E.from_ptr (_dropout '((E.to_ptr x), (p to float)))
let dropout_dim x d p = E.from_ptr (_dropout_dim '((E.to_ptr x), (d to uint), (p to float)))
let dropout_batch x p = E.from_ptr (_dropout_batch '((E.to_ptr x), (p to float)))
let block_dropout x p = E.from_ptr (_block_dropout '((E.to_ptr x), (p to float)))

let filter1d_narrow x f = E.from_ptr (_filter1d_narrow '((E.to_ptr x), (E.to_ptr f)))
let kmax_pooling x k = E.from_ptr (_kmax_pooling '((E.to_ptr x), (k to uint)))
let fold_rows ?(nrows=2) x = E.from_ptr (_fold_rows '((E.to_ptr x), (nrows to uint)))
let average_cols x = E.from_ptr (_average_cols '((E.to_ptr x)))
let kmh_ngram x n = E.from_ptr (_kmh_ngram '((E.to_ptr x), (n to uint)))

let conv2d ?(is_valid=true) ?bias x f b stride = match bias with
    | Some bias -> E.from_ptr (_conv2d '((E.to_ptr x), (E.to_ptr f), (E.to_ptr bias), (UnsignedVector.(to_ptr (of_array stride))), (is_valid to bool)))
    | None -> E.from_ptr (_conv2d '((E.to_ptr x), (E.to_ptr f), (UnsignedVector.(to_ptr (of_array stride))), (is_valid to bool)))

let maxpooling2d ?(is_valid=true) x ksize stride =
    E.from_ptr (_maxpooling2d '((E.to_ptr x), (UnsignedVector.(to_ptr (of_array ksize))), (UnsignedVector.(to_ptr (of_array stride))), (is_valid to bool)))

let contract3d_1d ?bias x y = match bias with
    | Some b -> E.from_ptr (_contract3d_1d '((E.to_ptr x), (E.to_ptr y), (E.to_ptr b)))
    | None -> E.from_ptr (_contract3d_1d '((E.to_ptr x), (E.to_ptr y)))

let contract3d_1d_1d ?bias x y z = match bias with
    | Some b -> E.from_ptr (_contract3d_1d_1d '((E.to_ptr x), (E.to_ptr y), (E.to_ptr z), (E.to_ptr b)))
    | None -> E.from_ptr (_contract3d_1d_1d '((E.to_ptr x), (E.to_ptr y), (E.to_ptr z)))

let inverse x = E.from_ptr (_inverse '((E.to_ptr x)))
let logdet x = E.from_ptr (_logdet '((E.to_ptr x)))
let trace_of_product x y = E.from_ptr (_trace_of_product '((E.to_ptr x), (E.to_ptr y)))

let layer_norm x g b = E.from_ptr (_layer_norm '((E.to_ptr x), (E.to_ptr g), (E.to_ptr b)))
let weight_norm w g = E.from_ptr (_weight_norm '((E.to_ptr w), (E.to_ptr g)))


(* adopted from owl ... *)
let ( ~ ) = neg
let ( + ) = add
let ( - ) = sub
let ( * ) = mul
let ( +$ ) = add_scalar
let ( $+ ) = scalar_add
let ( -$ ) = sub_scalar
let ( $- ) = scalar_sub
let ( *$ ) = mul_scalar
let ( $* ) = scalar_mul
(* let ( / ) = cdiv *)
let ( /$ ) = div_scalar
let ( *@ ) = dot_product
let ( ** ) = pow
(* let ( *. ) = cmult *)
(* let ( /. ) = cdiv *)
(* let ( $/ ) = scalar_div *)


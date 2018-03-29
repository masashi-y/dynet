
open Swig
open Dynet_swig
open Vectors
open Params

let dim x = Dim.from_ptr ((Expression.to_ptr x) -> dim_deref ())

let value x = Tensor.from_ptr ((Expression.to_ptr x) -> value_deref ())

let parameter cg p =
    Expression.from_ptr (_parameter_Parameter '((Computationgraph.to_ptr cg), (Parameter.to_ptr p)))
let lookup_parameter cg p =
    Expression.from_ptr (_parameter_LookupParameter '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p)))

let const_parameter cg p =
    Expression.from_ptr (_const_parameter_Parameter '((Computationgraph.to_ptr cg), (Parameter.to_ptr p)))
let const_lookup_parameter cg p =
    Expression.from_ptr (_const_parameter_LookupParameter '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p)))

let lookup cg p i =
    Expression.from_ptr (_lookup '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (i to uint)))
let const_lookup cg p i =
    Expression.from_ptr (_const_lookup '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (i to uint)))

let lookup_batch cg p v =
    Expression.from_ptr (_lookup_vector '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (UnsignedVector.(to_ptr (of_array v)))))
let const_lookup_batch cg p v =
    Expression.from_ptr (_const_lookup_vector '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (UnsignedVector.(to_ptr (of_array v)))))
let lookup_batch_vec cg p v =
    Expression.from_ptr (_lookup_vector '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (UnsignedVector.to_ptr v)))
let const_lookup_batch_vec cg p v =
    Expression.from_ptr (_const_lookup_vector '((Computationgraph.to_ptr cg), (LookupParameter.to_ptr p), (UnsignedVector.to_ptr v)))

let input cg dim v =
    Expression.from_ptr (_input '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (FloatVector.to_ptr v)))
let input_scalar cg v =
    Expression.from_ptr (_input_scalar '((Computationgraph.to_ptr cg), (v to float)))
let input_sparse ?(default=0.0) cg d ids data =
    Expression.from_ptr (_input_sparse '((Computationgraph.to_ptr cg), (Dim.to_ptr d),
     (UnsignedVector.(to_ptr (of_array ids))), (FloatVector.(to_ptr (of_array data))), (default to float)))


let neg x = Expression.from_ptr (_exprNeg '((Expression.to_ptr x)))
let add x y = Expression.from_ptr (_exprPlusExEx '((Expression.to_ptr x), (Expression.to_ptr y)))
let add_scalar x y = Expression.from_ptr (_exprPlusExRe '((Expression.to_ptr x), (y to float)))
let scalar_add x y = Expression.from_ptr (_exprPlusExRe '((Expression.to_ptr y), (x to float)))

let mul x y = Expression.from_ptr (_exprTimesExEx '((Expression.to_ptr x), (Expression.to_ptr y)))
let mul_scalar x y = Expression.from_ptr (_exprTimesExRe '((Expression.to_ptr x), (y to float)))
let scalar_mul x y = Expression.from_ptr (_exprTimesExRe '((Expression.to_ptr y), (x to float)))

let sub x y = Expression.from_ptr (_exprMinusExEx '((Expression.to_ptr x), (Expression.to_ptr y)))
let sub_scalar x y = Expression.from_ptr (_exprMinusExRe '((Expression.to_ptr x), (y to float)))
let scalar_sub x y = Expression.from_ptr (_exprMinusReEx '((x to float), (Expression.to_ptr y)))

let div_scalar x y = Expression.from_ptr (_exprDivide '((Expression.to_ptr x), (y to float)))

let zeros cg dim = Expression.from_ptr (_zeros '((Computationgraph.to_ptr cg), (Dim.to_ptr dim)))
let zeroes cg dim = Expression.from_ptr (_zeroes '((Computationgraph.to_ptr cg), (Dim.to_ptr dim)))
let ones cg dim = Expression.from_ptr (_ones '((Computationgraph.to_ptr cg), (Dim.to_ptr dim)))
let constant cg dim v = Expression.from_ptr (_constant '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (v to float)))
let random_normal cg dim = Expression.from_ptr (_random_normal '((Computationgraph.to_ptr cg), (Dim.to_ptr dim)))
let random_bernoulli ?(scale=1.0) cg dim p = Expression.from_ptr (_random_bernoulli '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (p to float), (scale to float)))
let random_uniform cg dim left right = Expression.from_ptr (_random_uniform '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (left to float), (right to float)))
let random_gumbel ?(mu=0.0) ?(beta=1.0) cg dim = Expression.from_ptr (_random_gumbel '((Computationgraph.to_ptr cg), (Dim.to_ptr dim), (mu to float), (beta to float)))

let affine_transform xs = Expression.from_ptr (_affine_transform '((ExpressionVector.(to_ptr (of_array xs)))))
let affine_transform_vec xs = Expression.from_ptr (_affine_transform '((ExpressionVector.to_ptr xs)))

let sum xs = Expression.from_ptr (_sum '((ExpressionVector.(to_ptr (of_array xs)))))
let sum_vec xs = Expression.from_ptr (_sum '((ExpressionVector.(to_ptr xs))))
let sum_elems x = Expression.from_ptr (_sum_elems '((Expression.to_ptr x)))
let moment_elems x r = Expression.from_ptr (_moment_elems '((Expression.to_ptr x), (r to uint)))
let mean_elems x = Expression.from_ptr (_mean_elems '((Expression.to_ptr x)))
let std_elems x = Expression.from_ptr (_std_elems '((Expression.to_ptr x)))

let sum_batches x = Expression.from_ptr (_sum_batches '((Expression.to_ptr x)))
let moment_batches x r = Expression.from_ptr (_moment_batches '((Expression.to_ptr x), (r to uint)))
let mean_batches x = Expression.from_ptr (_mean_batches '((Expression.to_ptr x)))
let std_batches x = Expression.from_ptr (_std_batches '((Expression.to_ptr x)))

let sum_dim ?(b=false) x dims = Expression.from_ptr (_sum_dim '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array dims))), (b to bool)))
let moment_dim ?(b=false) ?(n=0) x dims r = Expression.from_ptr (_moment_dim '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array dims))), (r to uint), (b to bool), (n to uint)))
let mean_dim ?(b=false) ?(n=0) x dims = Expression.from_ptr (_mean_dim '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array dims))), (b to bool), (n to uint)))
let std_dim ?(b=false) ?(n=0) x dims = Expression.from_ptr (_std_dim '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array dims))), (b to bool), (n to uint)))

let sum_dim_vec ?(b=false) x dims = Expression.from_ptr (_sum_dim '((Expression.to_ptr x), (UnsignedVector.to_ptr dims), (b to bool)))
let moment_dim_vec ?(b=false) ?(n=0) x dims r = Expression.from_ptr (_moment_dim '((Expression.to_ptr x), (UnsignedVector.to_ptr dims), (r to uint), (b to bool), (n to uint)))
let mean_dim_vec ?(b=false) ?(n=0) x dims = Expression.from_ptr (_mean_dim '((Expression.to_ptr x), (UnsignedVector.to_ptr dims), (b to bool), (n to uint)))
let std_dim_vec ?(b=false) ?(n=0) x dims = Expression.from_ptr (_std_dim '((Expression.to_ptr x), (UnsignedVector.to_ptr dims), (b to bool), (n to uint)))

let sum_rows x = Expression.from_ptr (_sum_rows '((Expression.(to_ptr x))))
let sum_cols x = Expression.from_ptr (_sum_cols '((Expression.(to_ptr x))))

let sqrt x = Expression.from_ptr (_sqrt '((Expression.to_ptr x)))
let abs x = Expression.from_ptr (_abs '((Expression.to_ptr x)))
let erf x = Expression.from_ptr (_erf '((Expression.to_ptr x)))
let tanh x = Expression.from_ptr (_tanh '((Expression.to_ptr x)))
let exp x = Expression.from_ptr (_exp '((Expression.to_ptr x)))
let square x = Expression.from_ptr (_square '((Expression.to_ptr x)))
let cube x = Expression.from_ptr (_cube '((Expression.to_ptr x)))
let lgamma x = Expression.from_ptr (_lgamma '((Expression.to_ptr x)))
let log x = Expression.from_ptr (_log '((Expression.to_ptr x)))
let logistic x = Expression.from_ptr (_logistic '((Expression.to_ptr x)))
let rectify x = Expression.from_ptr (_rectify '((Expression.to_ptr x)))
let elu ?(alpha=1.0) x = Expression.from_ptr (_elu '((Expression.to_ptr x), (alpha to float)))
let selu x = Expression.from_ptr (_selu '((Expression.to_ptr x)))
let silu ?(beta=1.0) x = Expression.from_ptr (_silu '((Expression.to_ptr x), (beta to float)))
let softsign x = Expression.from_ptr (_softsign '((Expression.to_ptr x)))
let pow x y = Expression.from_ptr (_pow '((Expression.to_ptr x), (Expression.to_ptr y)))

let min x y = Expression.from_ptr (_min '((Expression.to_ptr x), (Expression.to_ptr y)))
let max x y = Expression.from_ptr (_max '((Expression.to_ptr x), (Expression.to_ptr y)))
let max xs = Expression.from_ptr (_max '((ExpressionVector.(to_ptr (of_array xs)))))
let max_vec xs = Expression.from_ptr (_max '((ExpressionVector.to_ptr xs)))
let dot_product x y = Expression.from_ptr (_dot_product '((Expression.to_ptr x), (Expression.to_ptr y)))
let cmult x y = Expression.from_ptr (_cmult '((Expression.to_ptr x), (Expression.to_ptr y)))
let cdiv x y = Expression.from_ptr (_cdiv '((Expression.to_ptr x), (Expression.to_ptr y)))
let colwise_add x b = Expression.from_ptr (_colwise_add '((Expression.to_ptr x), (Expression.to_ptr b)))

let softmax x = Expression.from_ptr (_softmax '((Expression.to_ptr x)))
let log_softmax ?restriction x = match restriction with
    | Some r -> Expression.from_ptr (_log_softmax '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array r)))))
    | None -> Expression.from_ptr (_log_softmax '((Expression.to_ptr x)))
let logsumexp_dim x d = Expression.from_ptr (_logsumexp_dim '((Expression.to_ptr x), (d to uint)))

let pickneglogsoftmax x v = Expression.from_ptr (_pickneglogsoftmax '((Expression.to_ptr x), (v to uint)))
(* let pickneglogsoftmax x = Expression.from_ptr (_pickneglogsoftmax '((Expression.to_ptr x), const unsigned* pv)) *)
let pickneglogsoftmax_batch x v = Expression.from_ptr (_pickneglogsoftmax '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array v)))))
let pickneglogsoftmax_batch_vec x v = Expression.from_ptr (_pickneglogsoftmax '((Expression.to_ptr x), (UnsignedVector.(to_ptr v))))

let hinge ?(m=1.0) x index = Expression.from_ptr (_hinge '((Expression.to_ptr x), (index to uint), (m to float)))
(* let hinge x = Expression.from_ptr (_hinge '((Expression.to_ptr x), unsigned* pindex, float m = 1.0)) *)
(* let hinge x = Expression.from_ptr (_hinge '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array indices))), float m = 1.0)) *)

let hinge_dim ?(d=0) ?(m=1.0) x indices = Expression.from_ptr (_hinge_dim '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array indices))), (d to uint), (m to float)))
let hinge_dim_vec ?(d=0) ?(m=1.0) x indices = Expression.from_ptr (_hinge_dim '((Expression.to_ptr x), (UnsignedVector.to_ptr indices), (d to uint), (m to float)))
(* let hinge_dim x = Expression.from_ptr (_hinge_dim '((Expression.to_ptr x), const std::vector<std::vector<unsigned> >& indices, unsigned d = 0, float m = 1.0)) *)

let sparsemax x = Expression.from_ptr (_sparsemax '((Expression.to_ptr x)))
let sparsemax_loss x target_support = Expression.from_ptr (_sparsemax_loss '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array target_support)))))
let sparsemax_loss_vec x target_support = Expression.from_ptr (_sparsemax_loss '((Expression.to_ptr x), (UnsignedVector.to_ptr target_support)))

let squared_norm x = Expression.from_ptr (_squared_norm '((Expression.to_ptr x)))
let l2_norm x = Expression.from_ptr (_l2_norm '((Expression.to_ptr x)))
let squared_distance x y = Expression.from_ptr (_squared_distance '((Expression.to_ptr x), (Expression.to_ptr y)))

let l1_distance x y = Expression.from_ptr (_l1_distance '((Expression.to_ptr x), (Expression.to_ptr y)))
let huber_distance ?(c=1.345) x y = Expression.from_ptr (_huber_distance '((Expression.to_ptr x), (Expression.to_ptr y), (c to float)))
let binary_log_loss x y = Expression.from_ptr (_binary_log_loss '((Expression.to_ptr x), (Expression.to_ptr y)))
let pairwise_rank_loss ?(m=1.0) x y = Expression.from_ptr (_pairwise_rank_loss '((Expression.to_ptr x), (Expression.to_ptr y), (m to float)))
let poisson_loss x y = Expression.from_ptr (_poisson_loss '((Expression.to_ptr x), (y to uint)))
(* let poisson_loss x = Expression.from_ptr (_poisson_loss '((Expression.to_ptr x), const unsigned* py)) *)

let nobackprop x = Expression.from_ptr (_nobackprop '((Expression.to_ptr x)))
let flip_gradient x = Expression.from_ptr (_flip_gradient '((Expression.to_ptr x)))
let reshape x d = Expression.from_ptr (_reshape '((Expression.to_ptr x), (Dim.to_ptr d)))
let transpose x = Expression.from_ptr (_transpose '((Expression.to_ptr x)))

let select_rows x rows = Expression.from_ptr (_select_rows '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array rows)))))
let select_cols x cols = Expression.from_ptr (_select_cols '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array cols)))))
let select_rows_vec x rows = Expression.from_ptr (_select_rows '((Expression.to_ptr x), (UnsignedVector.to_ptr rows)))
let select_cols_vec x cols = Expression.from_ptr (_select_cols '((Expression.to_ptr x), (UnsignedVector.to_ptr cols)))

let pick ?(d=0) x v = Expression.from_ptr (_pick '((Expression.to_ptr x), (v to uint), (d to uint)))
let pick_batch ?(d=0) x v = Expression.from_ptr (_pick '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array v))), (d to uint)))
(* let pick ?(d=0) x = Expression.from_ptr (_pick '((Expression.to_ptr x), const unsigned* v, (d = 0 to uint))) *)
let pick_range ?(d=0) x s e = Expression.from_ptr (_pick_range '((Expression.to_ptr x), (s to uint), (e to uint), (d to uint)))
let pick_batch_elem x v = Expression.from_ptr (_pick_batch_elem '((Expression.to_ptr x), (v to uint)))
let pick_batch_elems x v = Expression.from_ptr (_pick_batch_elems '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array v)))))
let pick_batch_elems_vec x v = Expression.from_ptr (_pick_batch_elems '((Expression.to_ptr x), (UnsignedVector.to_ptr v)))

let concatenate_to_batch xs = Expression.from_ptr (_concatenate_to_batch '((ExpressionVector.(to_ptr (of_array xs)))))
let concatenate_cols xs = Expression.from_ptr (_concatenate_cols '((ExpressionVector.(to_ptr (of_array xs)))))
let concatenate ?(d=0) xs = Expression.from_ptr (_concatenate '((ExpressionVector.(to_ptr (of_array xs))), (d to uint)))

let concatenate_to_batch_vec xs = Expression.from_ptr (_concatenate_to_batch '((ExpressionVector.to_ptr xs)))
let concatenate_cols_vec xs = Expression.from_ptr (_concatenate_cols '((ExpressionVector.to_ptr xs)))
let concatenate_vec ?(d=0) xs = Expression.from_ptr (_concatenate '((ExpressionVector.to_ptr xs), (d to uint)))

let max_dim x d = Expression.from_ptr (_max_dim '((Expression.to_ptr x), (d to uint)))
let min_dim x d = Expression.from_ptr (_min_dim '((Expression.to_ptr x), (d to uint)))

let noise x stddev = Expression.from_ptr (_noise '((Expression.to_ptr x), (stddev to float)))
let dropout x p = Expression.from_ptr (_dropout '((Expression.to_ptr x), (p to float)))
let dropout_dim x d p = Expression.from_ptr (_dropout_dim '((Expression.to_ptr x), (d to uint), (p to float)))
let dropout_batch x p = Expression.from_ptr (_dropout_batch '((Expression.to_ptr x), (p to float)))
let block_dropout x p = Expression.from_ptr (_block_dropout '((Expression.to_ptr x), (p to float)))

let filter1d_narrow x f = Expression.from_ptr (_filter1d_narrow '((Expression.to_ptr x), (Expression.to_ptr f)))
let kmax_pooling x k = Expression.from_ptr (_kmax_pooling '((Expression.to_ptr x), (k to uint)))
let fold_rows ?(nrows=2) x = Expression.from_ptr (_fold_rows '((Expression.to_ptr x), (nrows to uint)))
let average_cols x = Expression.from_ptr (_average_cols '((Expression.to_ptr x)))
let kmh_ngram x n = Expression.from_ptr (_kmh_ngram '((Expression.to_ptr x), (n to uint)))

let conv2d ?(is_valid=true) ?bias x f b stride = match bias with
    | Some bias -> Expression.from_ptr (_conv2d '((Expression.to_ptr x), (Expression.to_ptr f), (Expression.to_ptr bias), (UnsignedVector.(to_ptr (of_array stride))), (is_valid to bool)))
    | None -> Expression.from_ptr (_conv2d '((Expression.to_ptr x), (Expression.to_ptr f), (UnsignedVector.(to_ptr (of_array stride))), (is_valid to bool)))

let maxpooling2d ?(is_valid=true) x ksize stride =
    Expression.from_ptr (_maxpooling2d '((Expression.to_ptr x), (UnsignedVector.(to_ptr (of_array ksize))), (UnsignedVector.(to_ptr (of_array stride))), (is_valid to bool)))

let contract3d_1d ?bias x y = match bias with
    | Some b -> Expression.from_ptr (_contract3d_1d '((Expression.to_ptr x), (Expression.to_ptr y), (Expression.to_ptr b)))
    | None -> Expression.from_ptr (_contract3d_1d '((Expression.to_ptr x), (Expression.to_ptr y)))

let contract3d_1d_1d ?bias x y z = match bias with
    | Some b -> Expression.from_ptr (_contract3d_1d_1d '((Expression.to_ptr x), (Expression.to_ptr y), (Expression.to_ptr z), (Expression.to_ptr b)))
    | None -> Expression.from_ptr (_contract3d_1d_1d '((Expression.to_ptr x), (Expression.to_ptr y), (Expression.to_ptr z)))

let inverse x = Expression.from_ptr (_inverse '((Expression.to_ptr x)))
let logdet x = Expression.from_ptr (_logdet '((Expression.to_ptr x)))
let trace_of_product x y = Expression.from_ptr (_trace_of_product '((Expression.to_ptr x), (Expression.to_ptr y)))

let layer_norm x g b = Expression.from_ptr (_layer_norm '((Expression.to_ptr x), (Expression.to_ptr g), (Expression.to_ptr b)))
let weight_norm w g = Expression.from_ptr (_weight_norm '((Expression.to_ptr w), (Expression.to_ptr g)))


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
let ( *. ) = cmult
(* let ( /. ) = cdiv *)
(* let ( $/ ) = scalar_div *)


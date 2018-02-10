
open Swig
open Dynet_swig

type t = c_obj

let _with f =
    let cg = new_ComputationGraph '() in
    let res = f cg in
    '~ cg; res

let add_parameters cg p = (cg -> add_parameters (p)) as int
let add_const_parameters cg p = (cg -> add_const_parameters (p)) as int
let add_lookup cg p i = (cg -> add_lookup (p, (i to int))) as int
let add_const_lookup cg p i = (cg -> add_const_lookup (p, (i to int))) as int

let clear cg = ignore (cg -> clear ())
let checkpoint cg = ignore (cg -> checkpoint ())
let revert cg = ignore (cg -> revert ())

let get_dimension cg i = cg -> get_dimension (i)

let forward cg x = cg -> forward_one (x)
let incremental_forward cg x = cg -> incremental_forward (x)
let get_value cg x = cg -> get_value (x)

let invalidate cg = ignore (cg -> invalidate ())

let backward cg x = ignore (cg -> backward (x))

let print_graphviz cg = ignore (cg -> print_graphviz ())

(*
struct ComputationGraph {
  ComputationGraph();
  ~ComputationGraph();

  VariableIndex add_input(real s, Device* device);
  // VariableIndex add_input(const real* ps);
  VariableIndex add_input(const Dim& d, const std::vector<float>& data, Device* device);
  //VariableIndex add_input(const Dim& d, const std::vector<float>* pdata);
  VariableIndex add_input(const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, Device* device, float defdata = 0.f);

  VariableIndex add_parameters(Parameter p);
  VariableIndex add_const_parameters(Parameter p);
  VariableIndex add_lookup(LookupParameter p, const unsigned* pindex);
  VariableIndex add_lookup(LookupParameter p, unsigned index);
  VariableIndex add_lookup(LookupParameter p, const std::vector<unsigned>* pindices);
  // VariableIndex add_lookup(LookupParameter p, const std::vector<unsigned>& indices);
  VariableIndex add_const_lookup(LookupParameter p, const unsigned* pindex);
  VariableIndex add_const_lookup(LookupParameter p, unsigned index);
  VariableIndex add_const_lookup(LookupParameter p, const std::vector<unsigned>* pindices);
  // VariableIndex add_const_lookup(LookupParameter p, const std::vector<unsigned>& indices);
*)

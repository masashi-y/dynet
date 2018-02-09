
open Swig
open Dynet_swig

type t = c_obj

let simple_sgd ?(learning_rate=0.1) m =
    new_SimpleSGDTrainer '(m, (learning_rate to float))
    
let cyclical_sgd ?(learning_rate_min=0.01) ?(learning_rate_max=0.1) ?(step_size=2000) ?(gamma=0.0) ?(edecay=0.0) m =
    new_CyclicalSGDTrainer '(m, (learning_rate_min to float), (learning_rate_max to float),
        (step_size to int), (gamma to float), (edecay to float))

let momentum_sgd ?(learning_rate=0.01) ?(mom=0.9) m =
    new_MomentumSGDTrainer '(m, (learning_rate to float), (mom to float))

let adagrad ?(learning_rate=0.1) ?(eps=1e-20) m =
    new_AdagradTrainer '(m, (learning_rate to float), (eps to float))

let adadelta ?(eps=1e-6) ?(rho=0.95) m =
    new_AdadeltaTrainer '(m, (eps to float), (rho to float))

let rms_prop ?(learning_rate=0.1) ?(eps=1e-20) ?(rho=0.95) m =
    new_RMSPropTrainer '(m, (learning_rate to float), (eps to float), (rho to float))

let adam ?(learning_rate=0.001) ?(beta_1=0.9) ?(beta_2=0.999) ?(eps=1e-8) m =
    new_AdamTrainer '(m, (learning_rate to float), (beta_1 to float), (beta_2 to float), (eps to float))

let amsgrad ?(learning_rate=0.001) ?(beta_1=0.9) ?(beta_2=0.999) ?(eps=1e-8) m =
    new_AmsgradTrainer '(m, (learning_rate to float), (beta_1 to float), (beta_2 to float), (eps to float))

(*
// Need to disable constructor as SWIG gets confused otherwise
%nodefaultctor Trainer;
struct Trainer {
  virtual void update();
  //void update(const std::vector<unsigned> & updated_params, const std::vector<unsigned> & updated_lookup_params);
  void update_epoch(real r = 1.0);

  virtual void restart() = 0;
  void restart(real lr);

  float clip_gradients();
  void rescale_and_reset_weight_decay();

  real learning_rate;

  bool clipping_enabled;
  real clip_threshold;
  real clips;
  real updates;

  real clips_since_status;
  real updates_since_status;

  bool sparse_updates_enabled;
  unsigned aux_allocated;
  unsigned aux_allocated_lookup;

  void status();

  ParameterCollection* model;
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(ParameterCollection& m, real learning_rate = 0.1);
  void restart() override;
};

struct CyclicalSGDTrainer : public Trainer {
  explicit CyclicalSGDTrainer(ParameterCollection& m, float learning_rate_min = 0.01, float learning_rate_max = 0.1, float step_size = 2000, float gamma = 0.0, float edecay = 0.0);
  void restart() override;
  void update() override;
};

struct MomentumSGDTrainer : public Trainer {
  explicit MomentumSGDTrainer(ParameterCollection& m, real learning_rate = 0.01, real mom = 0.9);
  void restart() override;
};

struct AdagradTrainer : public Trainer {
  explicit AdagradTrainer(ParameterCollection& m, real learning_rate = 0.1, real eps = 1e-20);
  void restart() override;
};

struct AdadeltaTrainer : public Trainer {
  explicit AdadeltaTrainer(ParameterCollection& m, real eps = 1e-6, real rho = 0.95);
  void restart() override;
};

struct RMSPropTrainer : public Trainer {
   explicit RMSPropTrainer(ParameterCollection& m, real learning_rate = 0.1, real eps = 1e-20, real rho = 0.95);
   void restart() override;
};

struct AdamTrainer : public Trainer {
  explicit AdamTrainer(ParameterCollection& m, float learning_rate = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8);
  void restart() override;
};

struct AmsgradTrainer : public Trainer {
  explicit AmsgradTrainer(ParameterCollection& m, float learning_rate = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8);
  void restart() override;
};

//struct EGTrainer : public Trainer {
//  explicit EGTrainer(ParameterCollection& mod, real learning_rate = 0.1, real mom = 0.9, real ne = 0.0);
//  void enableCyclicalLR(float _learning_rate_min = 0.01, float _learning_rate_max = 0.1, float _step_size = 2000, float _gamma = 0.0);
//  void update() override;
//  void restart() override;
//};
*)

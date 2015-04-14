#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <string>
#include <vector>

#include "caffe/net.hpp"

// print the contents of a unidimensional blob.  used for debugging.
#define PRINT_VECTOR_BLOB(v) \
  for (int i = 0; i < v->shape(3); i++) { \
    std::cout << v->data_at(0,0,0,i) << ' '; \
  } \
  std::cout << "\n";

// print the contents of a vector
#define PRINT_VECTOR(v, Dtype) \
  std::copy(v.begin(), v.end(), std::ostream_iterator<Dtype>(std::cout, " ")); \
  std::cout << "\n";

namespace caffe {

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ComputeUpdateValue to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }

 protected:
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue() = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(SolverState* state) = 0;
  virtual void RestoreSolverState(const SolverState& state) = 0;
  void DisplayOutputBlobs(const int net_id);

  SolverParameter param_;
  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};


/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
  Dtype GetLearningRate();
  virtual void PreSolve();
  void DisplayIterInfo(Dtype rate);
  Dtype GetGradNorm();
  void TrackAvgGradNorm();
  Dtype ResetAvgGradNorm();
  Dtype grad_norm;
  int n_grad_norm_iters;
  virtual void ComputeUpdateValue();
  virtual void ClipGradients();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);

  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}

 protected:
  virtual void ComputeUpdateValue();

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }

 protected:
  virtual void ComputeUpdateValue();
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};

template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { PreSolve(); constructor_sanity_check(); }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { PreSolve(); constructor_sanity_check(); }

 protected:
  virtual void PreSolve();
  virtual void ComputeUpdateValue();
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.base_lr())
        << "Learning rate cannot be used with AdaDelta.";
    CHECK_EQ("", this->param_.lr_policy())
        << "Learning rate policy cannot be applied to AdaDelta.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};

template <typename Dtype>
class DucbSolver : public SGDSolver<Dtype> {
 public:
  explicit DucbSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { PreSolve(); constructor_sanity_check(); }
  explicit DucbSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { PreSolve(); constructor_sanity_check(); }

 protected:
  virtual void PreSolve();
  virtual void ComputeUpdateValue();
  void RegularizeGradient();

  void GrantReward(Dtype old_obj, Dtype new_obj, int lr_index);
  int GetStartingLrIndex();

  // back up net_params.data and net_params.diff to temp_.data and temp_.diff
  // for easy restoration in case the main copy fills with NaNs
  void BackupDataAndDiff();

  Dtype PrepareJumpFromBackup(Dtype next_alpha);

  //  takes param.diff from grad_alpha_current to
  // (param_alpha_next - param_alpha_current) * grad(theta) so that a
  // subsequent call to net.update() will move
  // param.data from theta - param_alpha_current * grad(theta) to
  // theta - param_alpha_next * grad(theta).  returns the new gradient
  // multiplier
  Dtype PrepareJumpToAlpha(Dtype param_alpha_next,
      Dtype param_alpha_current, Dtype grad_alpha_current);

  // make a vector of logarithmically spaced values
  void LogSpace(vector<Dtype> & vect, Dtype log_high_alpha = 2,
      Dtype log_low_alpha = -6, int n_alphas = 33, Dtype base = 10);

  void constructor_sanity_check() {
 	  // TODO: fill in
    CHECK_EQ(0, this->param_.base_lr())
        << "Learning rate cannot be used with DucbSolver.";
    CHECK_EQ("", this->param_.lr_policy())
        << "Learning rate policy cannot be applied to DucbSolver.";
  }

  // alphas_ is the set of all the learning rates we will consider
  // rewards_ tracks the reward values for each alpha
  // numbers_ tracks the number of times each alpha has been played
  // all three are needed in the snapshot
  vector<Dtype> alphas_, rewards_, numbers_, mus_, cs_, js_;

//  // used for storing temporary update values.  not needed in snapshot
//  vector<shared_ptr<Blob<Dtype> > > temp_;

  // the DUCB algorithm performs an initial sweep of all possible alpha values
  // at the start of each training run.  init_sweep_ind tracks the index of the
  // next alpha to try
  int init_sweep_ind;

  // base-10 log of the smallest learning rate value to be considered
  // default: -6
  Dtype log_low_alpha;
  // base-10 log of the largest learning rate value to be considered
  // default: 2
  Dtype log_high_alpha;
  // number of learning rates to be log-interpolated between log_low_alpha and log_high_alpha.
  // default: 33
  Dtype n_alphas;


  // DUCB hyperparameters
  // forgetting factor, \in [0, 1].
  // 0 means the bandit model has no memory.  1 means no forgetting.
  // default: 0.99
  Dtype ducb_gamma;
  // explore constant, positive float.
  // larger number means the bandit model will take more explore steps
  // default: 1e-8
  Dtype explore_const;

  DISABLE_COPY_AND_ASSIGN(DucbSolver);
};


template <typename Dtype>
Solver<Dtype>* GetSolver(const SolverParameter& param) {
  SolverParameter_SolverType type = param.solver_type();

  switch (type) {
  case SolverParameter_SolverType_SGD:
      return new SGDSolver<Dtype>(param);
  case SolverParameter_SolverType_NESTEROV:
      return new NesterovSolver<Dtype>(param);
  case SolverParameter_SolverType_ADAGRAD:
      return new AdaGradSolver<Dtype>(param);
  case SolverParameter_SolverType_ADADELTA:
      return new AdaDeltaSolver<Dtype>(param);
  case SolverParameter_SolverType_DUCB:
      return new DucbSolver<Dtype>(param);
  default:
      LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_

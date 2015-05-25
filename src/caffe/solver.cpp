#include <cstdio>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include <math.h> // for isnan, log
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_() {
  SolverParameter param;
  ReadProtoFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG(INFO) << "Initializing solver from parameters: " << std::endl
            << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  InitTestNets();
  LOG(INFO) << "Solver scaffolding done.";
  iter_ = 0;
  current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG(INFO) << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG(INFO) << "Creating training net from train_net file: "
              << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG(INFO) << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG(INFO) << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  net_.reset(new Net<Dtype>(net_param));
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;

  for (; iter_ < stop_iter; ++iter_) {
    if (param_.test_interval() && iter_ % param_.test_interval() == 0 \
        && (iter_ > 0 || param_.test_initialization()) ) {
      TestAll();
    }

    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    Dtype loss = net_->ForwardBackward(bottom_vec);
    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }
    if (display) {
      LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG(INFO) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }

    // DropoutLayers will persist their masks until told to update them.
    // Tell the DropoutLayers to update their masks.
//    Dtype obj = this->net()->ForwardFrom(1);
//    Dtype obj1 = this->net()->ForwardFrom(1);
//    assert(obj == obj1);
    net_->FlagDropoutLayersForUpdate();
//    Dtype obj2 = this->net()->ForwardFrom(1);
//
//    assert(obj != obj2);
//    Dtype obj3 = this->net()->ForwardFrom(1);
//
//    assert(obj2 != obj3);
//
//    LOG(INFO) << "SOLVER::SOLVE() objectives before: " << obj  << "\t" << obj1 << "\n";
//    LOG(INFO) << "SOLVER::SOLVE() objectives after:  " << obj2 << "\t" << obj3 << "\n\n";

    ComputeUpdateValue();
    net_->Update();

    // Save a snapshot if needed.
    if (param_.snapshot() && (iter_ + 1) % param_.snapshot() == 0) {
      Snapshot();
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    Dtype loss;
    net_->ForwardPrefilled(&loss);
    LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
        << mean_score << loss_msg_stream.str();
  }
}


template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  net_->ToProto(&net_param, param_.snapshot_diff());
  string filename(param_.snapshot_prefix());
  string model_filename, snapshot_filename;
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  // Add one to iter_ to get the number of iterations that have completed.
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", iter_ + 1);
  filename += iter_str_buffer;
  model_filename = filename + ".caffemodel";
  LOG(INFO) << "Snapshotting to " << model_filename;
  WriteProtoToBinaryFile(net_param, model_filename.c_str());
  SolverState state;
  SnapshotSolverState(&state);
  state.set_iter(iter_ + 1);
  state.set_learned_net(model_filename);
  state.set_current_step(current_step_);
  snapshot_filename = filename + ".solverstate";
  LOG(INFO) << "Snapshotting solver state to " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  SolverState state;
  NetParameter net_param;
  ReadProtoFromBinaryFile(state_file, &state);
  if (state.has_learned_net()) {
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    net_->CopyTrainedLayersFrom(net_param);
  }
  iter_ = state.iter();
  current_step_ = state.current_step();
  RestoreSolverState(state);
}


// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
  ResetAvgGradNorm();
}

template<typename Dtype>
void SGDSolver<Dtype>::DisplayIterInfo(Dtype rate) {
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
    if (this->param_.log_avg_gradient_norm()) {
      Dtype avg_grad_norm = ResetAvgGradNorm();
      LOG(INFO)<< "Iteration " << this->iter_ << \
          ", avg_grad_norm = " << avg_grad_norm;
    }
  }
}

template<typename Dtype>
Dtype SGDSolver<Dtype>::GetGradNorm() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    if (this->net_->param_owners()[i] < 0) {
      sumsq_diff += net_params[i]->sumsq_diff();
    }
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  return l2norm_diff;
}

template<typename Dtype>
void SGDSolver<Dtype>::TrackAvgGradNorm() {
  if (this->param_.log_avg_gradient_norm()) {
    // track sufficient statistics to be able to log the minibatch gradient
    // norm averaged across minibatches
    grad_norm += GetGradNorm();
    n_grad_norm_iters += 1;
  }
}

template<typename Dtype>
Dtype SGDSolver<Dtype>::ResetAvgGradNorm() {
  // reset sufficient statistic totals for logging average minibatch gradient
  // norm
  Dtype avg_grad_norm = grad_norm / n_grad_norm_iters;
  grad_norm = Dtype(0);
  n_grad_norm_iters = 0;
  return avg_grad_norm;
}

template<typename Dtype>
void SGDSolver<Dtype>::RegularizeGradient() {
  // perform L1 or L2 regularization
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  const vector<shared_ptr<Blob<Dtype> > >& temp_ = this->temp_;

  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_axpy(net_params[param_id]->count(), local_decay,
              net_params[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else if (regularization_type == "L1") {
          caffe_cpu_sign(net_params[param_id]->count(),
              net_params[param_id]->cpu_data(),
              temp_[param_id]->mutable_cpu_data());
          caffe_axpy(net_params[param_id]->count(), local_decay,
              temp_[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else {
          LOG(FATAL)<< "Unknown regularization type: " << regularization_type;
        }
      }
    }

    break;
    case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else if (regularization_type == "L1") {
          caffe_gpu_sign(net_params[param_id]->count(),
              net_params[param_id]->gpu_data(),
              temp_[param_id]->mutable_gpu_data());
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              temp_[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }
    }

#else
    NO_GPU;
#endif
    break;
    default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }

  Dtype l2norm_diff = GetGradNorm();
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
    for (int i = 0; i < net_params.size(); ++i) {
      if (this->net_->param_owners()[i] < 0) {
        net_params[i]->scale_diff(scale_factor);
      }
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  // get the learning rate
  Dtype rate = GetLearningRate();
  TrackAvgGradNorm();
  DisplayIterInfo(rate);
  ClipGradients();
  Dtype momentum = this->param_.momentum();
  this->RegularizeGradient();

  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];

      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->cpu_diff(), momentum,
                history_[param_id]->mutable_cpu_data());
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];

      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->gpu_diff(), momentum,
                history_[param_id]->mutable_gpu_data());
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  state->clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state->add_history();
    history_[i]->ToProto(history_blob);
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void NesterovSolver<Dtype>::ComputeUpdateValue() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  // get the learning rate
  Dtype rate = this->GetLearningRate();
  this->TrackAvgGradNorm();
  this->DisplayIterInfo(rate);
  SGDSolver<Dtype>::ClipGradients();

  this->RegularizeGradient();

  Dtype momentum = this->param_.momentum();

  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // save history momentum for stepping back
      caffe_copy(net_params[param_id]->count(),
          this->history_[param_id]->cpu_data(),
          this->update_[param_id]->mutable_cpu_data());

      Dtype local_rate = rate * net_params_lr[param_id];

      // update history
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->cpu_diff(), momentum,
                this->history_[param_id]->mutable_cpu_data());

      // compute udpate: step back then over step
      caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
          this->history_[param_id]->cpu_data(), -momentum,
          this->update_[param_id]->mutable_cpu_data());

      // copy
      caffe_copy(net_params[param_id]->count(),
          this->update_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // save history momentum for stepping back
      caffe_copy(net_params[param_id]->count(),
          this->history_[param_id]->gpu_data(),
          this->update_[param_id]->mutable_gpu_data());

      Dtype local_rate = rate * net_params_lr[param_id];

      // update history
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->gpu_diff(), momentum,
                this->history_[param_id]->mutable_gpu_data());

      // compute udpate: step back then over step
      caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
          this->history_[param_id]->gpu_data(), -momentum,
          this->update_[param_id]->mutable_gpu_data());

      // copy
      caffe_copy(net_params[param_id]->count(),
          this->update_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void AdaGradSolver<Dtype>::ComputeUpdateValue() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  // get the learning rate
  Dtype rate = this->GetLearningRate();
  Dtype delta = this->param_.delta();
  this->TrackAvgGradNorm();
  this->DisplayIterInfo(rate);
  SGDSolver<Dtype>::ClipGradients();

  this->RegularizeGradient();

  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype local_rate = rate * net_params_lr[param_id];

      // compute square of gradient in update
      caffe_powx(net_params[param_id]->count(),
          net_params[param_id]->cpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_cpu_data());

      // update history, which tracks sum of the squared gradients
      caffe_add(net_params[param_id]->count(),
          this->update_[param_id]->cpu_data(),
          this->history_[param_id]->cpu_data(),
          this->history_[param_id]->mutable_cpu_data());

      // prepare update
      caffe_powx(net_params[param_id]->count(),
                this->history_[param_id]->cpu_data(), Dtype(0.5),
                this->update_[param_id]->mutable_cpu_data());

      caffe_add_scalar(net_params[param_id]->count(),
                delta, this->update_[param_id]->mutable_cpu_data());

      caffe_div(net_params[param_id]->count(),
                net_params[param_id]->cpu_diff(),
                this->update_[param_id]->cpu_data(),
                this->update_[param_id]->mutable_cpu_data());

      // perform line search here

      // scale and copy
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
          this->update_[param_id]->cpu_data(), Dtype(0),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype local_rate = rate * net_params_lr[param_id];

      // compute square of gradient in update
      caffe_gpu_powx(net_params[param_id]->count(),
          net_params[param_id]->gpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_gpu_data());

      // update history, which tracks sum of the squared gradients
      caffe_gpu_add(net_params[param_id]->count(),
          this->update_[param_id]->gpu_data(),
          this->history_[param_id]->gpu_data(),
          this->history_[param_id]->mutable_gpu_data());

      // prepare update
      caffe_gpu_powx(net_params[param_id]->count(),
                this->history_[param_id]->gpu_data(), Dtype(0.5),
                this->update_[param_id]->mutable_gpu_data());

      caffe_gpu_add_scalar(net_params[param_id]->count(),
                delta, this->update_[param_id]->mutable_gpu_data());

      caffe_gpu_div(net_params[param_id]->count(),
                net_params[param_id]->gpu_diff(),
                this->update_[param_id]->gpu_data(),
                this->update_[param_id]->mutable_gpu_data());

      // scale and copy
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
          this->update_[param_id]->gpu_data(), Dtype(0),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void AdaGradLineSearchSolver<Dtype>::ComputeUpdateValue() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();

  // get the learning rate
  Dtype delta = this->param_.delta();
  this->TrackAvgGradNorm();
//  SGDSolver<Dtype>::ClipGradients();

//  this->RegularizeGradient();

  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype local_scale = net_params_lr[param_id];

      // compute square of gradient in update
      caffe_powx(net_params[param_id]->count(),
          net_params[param_id]->cpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_cpu_data());

      // update history, which tracks sum of the squared gradients
      caffe_add(net_params[param_id]->count(),
          this->update_[param_id]->cpu_data(),
          this->history_[param_id]->cpu_data(),
          this->history_[param_id]->mutable_cpu_data());

      // prepare update
      caffe_powx(net_params[param_id]->count(),
                this->history_[param_id]->cpu_data(), Dtype(0.5),
                this->update_[param_id]->mutable_cpu_data());

      caffe_add_scalar(net_params[param_id]->count(),
                delta, this->update_[param_id]->mutable_cpu_data());

      caffe_div(net_params[param_id]->count(),
                net_params[param_id]->cpu_diff(),
                this->update_[param_id]->cpu_data(),
                this->update_[param_id]->mutable_cpu_data());

      // scale and copy
      caffe_cpu_axpby(net_params[param_id]->count(), local_scale,
          this->update_[param_id]->cpu_data(), Dtype(0),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype local_scale = net_params_lr[param_id];

      // compute square of gradient in update
      caffe_gpu_powx(net_params[param_id]->count(),
          net_params[param_id]->gpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_gpu_data());

      // update history, which tracks sum of the squared gradients
      caffe_gpu_add(net_params[param_id]->count(),
          this->update_[param_id]->gpu_data(),
          this->history_[param_id]->gpu_data(),
          this->history_[param_id]->mutable_gpu_data());

      // prepare update
      caffe_gpu_powx(net_params[param_id]->count(),
                this->history_[param_id]->gpu_data(), Dtype(0.5),
                this->update_[param_id]->mutable_gpu_data());

      caffe_gpu_add_scalar(net_params[param_id]->count(),
                delta, this->update_[param_id]->mutable_gpu_data());

      caffe_gpu_div(net_params[param_id]->count(),
                net_params[param_id]->gpu_diff(),
                this->update_[param_id]->gpu_data(),
                this->update_[param_id]->mutable_gpu_data());

      // scale and copy
      caffe_gpu_axpby(net_params[param_id]->count(), local_scale,
          this->update_[param_id]->gpu_data(), Dtype(0),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }

  // perform a line search in the AdaGrad direction
  this->PerformLineSearch();
  Dtype chosen_alpha = this->alphas_[this->prev_alpha_index];
  this->DisplayIterInfo(chosen_alpha);


}

template <typename Dtype>
void AdaDeltaSolver<Dtype>::PreSolve() {
  // Add the extra history entries for AdaDelta after those from
  // SGDSolver::PreSolve. In the notation from the AdaDelta paper, the first
  // set of history entries track the  expected value of g^2.  The second set
  // of history entries track the expected value of (\Delta x)^2 (i.e.: step
  // sizes squared).
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  for (int i = 0; i < net_params.size(); ++i) {
        const vector<int>& shape = net_params[i]->shape();
        this->history_.push_back(
                shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

template <typename Dtype>
void AdaDeltaSolver<Dtype>::ComputeUpdateValue() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  Dtype delta = this->param_.delta();
  Dtype momentum = this->param_.momentum();

  // note: this solver doesn't do gradient clipping

  this->RegularizeGradient();

  size_t update_history_offset = net_params.size();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // compute square of gradient in update
      caffe_powx(net_params[param_id]->count(),
          net_params[param_id]->cpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_cpu_data());

      // update history of gradients
      caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
          this->update_[param_id]->cpu_data(), momentum,
          this->history_[param_id]->mutable_cpu_data());

      // add delta to history to guard against dividing by zero later
      caffe_set(net_params[param_id]->count(), delta,
          this->temp_[param_id]->mutable_cpu_data());

      caffe_add(net_params[param_id]->count(),
          this->temp_[param_id]->cpu_data(),
          this->history_[update_history_offset + param_id]->cpu_data(),
          this->update_[param_id]->mutable_cpu_data());

      caffe_add(net_params[param_id]->count(),
          this->temp_[param_id]->cpu_data(),
          this->history_[param_id]->cpu_data(),
          this->temp_[param_id]->mutable_cpu_data());

      // divide history of updates by history of gradients
      caffe_div(net_params[param_id]->count(),
          this->update_[param_id]->cpu_data(),
          this->temp_[param_id]->cpu_data(),
          this->update_[param_id]->mutable_cpu_data());

      // jointly compute the RMS of both for update and gradient history
      caffe_powx(net_params[param_id]->count(),
          this->update_[param_id]->cpu_data(), Dtype(0.5),
          this->update_[param_id]->mutable_cpu_data());

      // compute the update
      caffe_mul(net_params[param_id]->count(),
          net_params[param_id]->cpu_diff(),
          this->update_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());

      // perform line search here

      // compute square of update
      caffe_powx(net_params[param_id]->count(),
          net_params[param_id]->cpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_cpu_data());

      // update history of updates
      caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
          this->update_[param_id]->cpu_data(), momentum,
          this->history_[update_history_offset + param_id]->mutable_cpu_data());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // compute square of gradient in update
      caffe_gpu_powx(net_params[param_id]->count(),
          net_params[param_id]->gpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_gpu_data());

      // update history of gradients
      caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
          this->update_[param_id]->gpu_data(), momentum,
          this->history_[param_id]->mutable_gpu_data());

      // add delta to history to guard against dividing by zero later
      caffe_gpu_set(net_params[param_id]->count(), delta,
          this->temp_[param_id]->mutable_gpu_data());

      caffe_gpu_add(net_params[param_id]->count(),
          this->temp_[param_id]->gpu_data(),
          this->history_[update_history_offset + param_id]->gpu_data(),
          this->update_[param_id]->mutable_gpu_data());

      caffe_gpu_add(net_params[param_id]->count(),
          this->temp_[param_id]->gpu_data(),
          this->history_[param_id]->gpu_data(),
          this->temp_[param_id]->mutable_gpu_data());

      // divide history of updates by history of gradients
      caffe_gpu_div(net_params[param_id]->count(),
          this->update_[param_id]->gpu_data(),
          this->temp_[param_id]->gpu_data(),
          this->update_[param_id]->mutable_gpu_data());

      // jointly compute the RMS of both for update and gradient history
      caffe_gpu_powx(net_params[param_id]->count(),
          this->update_[param_id]->gpu_data(), Dtype(0.5),
          this->update_[param_id]->mutable_gpu_data());

      // compute the update and copy to net_diff
      caffe_gpu_mul(net_params[param_id]->count(),
          net_params[param_id]->gpu_diff(),
          this->update_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());

      // compute square of update
      caffe_gpu_powx(net_params[param_id]->count(),
          net_params[param_id]->gpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_gpu_data());

      // update history of updates
      caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
          this->update_[param_id]->gpu_data(), momentum,
          this->history_[update_history_offset + param_id]->mutable_gpu_data());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template<typename Dtype>
void LineSearchSolver<Dtype>::LogSpace(vector<Dtype > & vect,
    Dtype log_high_alpha, Dtype log_low_alpha, int n_alphas, Dtype base) {
  vect.clear();
  Dtype delta = (log_high_alpha - log_low_alpha) / (n_alphas - 1);
  for (int i = 0; i < n_alphas; ++i) {
    Dtype linear_val = log_high_alpha - i * delta;
    vect.push_back(pow(base, linear_val));
  }
}


template <typename Dtype>
void LineSearchSolver<Dtype>::PreSolve() {

  this->log_high_alpha = this->param_.log_high_alpha();
  this->log_low_alpha = this->param_.log_low_alpha();
  this->n_alphas = this->param_.n_alphas();

  LogSpace(this->alphas_,
      this->log_high_alpha, this->log_low_alpha, this->n_alphas);

  this->init_sweep_ind = 0;
  this->prev_alpha_index = -1;

// really i want this to replace the presolve from sgdsolver because
// i don't want history_, only temp_.
//
//  // initialize temporary update memory to same size as net parameters
//  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
//  temp_ = this->temp_;
//  temp_.clear();
//  for (int i = 0; i < net_params.size(); ++i) {
//    const vector<int>& shape = net_params[i]->shape();
//    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
//  }
}

template <typename Dtype>
void LineSearchSolver<Dtype>::BackupDataAndDiff() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<shared_ptr<Blob<Dtype> > >& temp_ = this->temp_;

    switch (Caffe::mode()) {
    case Caffe::CPU:
      for (int param_id = 0; param_id < net_params.size(); ++param_id) {
        caffe_copy(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());

        caffe_copy(net_params[param_id]->count(),
            net_params[param_id]->cpu_diff(),
            temp_[param_id]->mutable_cpu_diff());
      }
      break;
    case Caffe::GPU:
  #ifndef CPU_ONLY
      for (int param_id = 0; param_id < net_params.size(); ++param_id) {
        caffe_copy(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());

        caffe_copy(net_params[param_id]->count(),
            net_params[param_id]->gpu_diff(),
            temp_[param_id]->mutable_gpu_diff());
      }
  #else
      NO_GPU;
  #endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }

}

template <typename Dtype>
Dtype LineSearchSolver<Dtype>::PrepareJumpFromBackup(Dtype next_alpha) {
  // this method assumes that each net_param is undefined, that temp_.data is
  // at theta and that each temp_.diff is at grad(theta)
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<shared_ptr<Blob<Dtype> > >& temp_ = this->temp_;

  const vector<float>& net_params_lr = this->net_->params_lr();

  // copy from data/diff
  switch (Caffe::mode()) {
    case Caffe::CPU:
      for (int param_id = 0; param_id < net_params.size(); ++param_id) {
        caffe_copy(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());

        caffe_copy(net_params[param_id]->count(),
            net_params[param_id]->cpu_diff(),
            temp_[param_id]->mutable_cpu_diff());

      }
      break;
    case Caffe::GPU:
  #ifndef CPU_ONLY
      for (int param_id = 0; param_id < net_params.size(); ++param_id) {
        caffe_copy(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());

        caffe_copy(net_params[param_id]->count(),
            net_params[param_id]->gpu_diff(),
            temp_[param_id]->mutable_gpu_diff());
      }
  #else
      NO_GPU;
  #endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }

  // scale diff
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    Dtype local_alpha = next_alpha * net_params_lr[param_id];
    net_params[param_id]->scale_diff(local_alpha);
  }

  return next_alpha;
}

template <typename Dtype>
Dtype LineSearchSolver<Dtype>::PrepareJumpToAlpha(Dtype param_alpha_next,
    Dtype param_alpha_current, Dtype grad_alpha_current) {
  // this method assumes that each net_param.data is at
  // theta - param_alpha_current * grad(theta) and that each net_param.diff is
  // at grad_alpha_current * grad(theta) where theta represents net_param.data
  // at the beginning of this iteration.
  //
  // it leaves data at theta - param_alpha_current * grad(theta) and diff at
  // (param_alpha_next - param_alpha_current) * grad(theta) so that running
  // update on the net will leave the params at theta - param_alpha_next *
  // grad(theta)
  //
  // note that this update will leave local_rate multipliers intact without
  // having to deal with them explicitly (work out the algebra to convince
  // yourself of this)
  //
  // returns (param_alpha_next - param_alpha_current), the final multiplier to
  // grad(theta) in diff.

  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const Dtype grad_multiplier =
      (param_alpha_next - param_alpha_current) / grad_alpha_current;

  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    net_params[param_id]->scale_diff(grad_multiplier);
  }

  return param_alpha_next - param_alpha_current;
}

template<typename Dtype>
void LineSearchSolver<Dtype>::PerformLineSearch(
    Dtype & final_data_mult, Dtype & final_diff_mult) {
  // get the learning rate
  int start_ind = GetStartingLrIndex();

  Dtype alpha_start = alphas_.at(start_ind);

  BackupDataAndDiff();

  shared_ptr<Net<Dtype> > net = this->net();

  // ForwardFrom(1) prevents updating the input data in the bottom layer
  Dtype starting_obj = this->net()->ForwardFrom(1);
  Dtype alpha = alpha_start;
  Dtype best_alpha = Dtype(0);
  int best_alpha_ind = 0;

  Dtype best_obj = starting_obj;
  Dtype obj = starting_obj;
  Dtype prev_obj;

  Dtype alpha_param_current = Dtype(0);
  Dtype alpha_grad_current = Dtype(1);

  // perform a back-tracking line search
  bool have_found_better = false;
  for (int i = start_ind; i < n_alphas; ++i) {
    prev_obj = obj;
    alpha = alphas_.at(i);
    alpha_grad_current = PrepareJumpToAlpha(alpha, alpha_param_current,
        alpha_grad_current);
    net->Update(); // execute the jump

    alpha_param_current = alpha;
    obj = this->net()->ForwardFrom(1);

    // TODO: DEAL WITH INF/NAN OBJECTIVE.  Restore from backup because
    // there might be inf/nan entries in the parameters

    GrantReward(starting_obj, obj, i);

    if (obj < best_obj) {
      // this gets triggered both when 1) when we first drop below the starting
      // objective function value and 2) whenever we find a new, better
      // objective function value
      best_obj = obj;
      best_alpha = alpha;
      best_alpha_ind = i;
      have_found_better = true;
    }

    // the objective has dipped below the starting value and started to rise
    //  again. terminate line search for this minibatch.
    if (have_found_better && obj > prev_obj) {
      break;
    }
  }

  this->prev_alpha_index = best_alpha_ind;

  final_diff_mult =
      PrepareJumpToAlpha(best_alpha, alpha_param_current, alpha_grad_current);
  // don't execute the jump with net->update() because that will be done in
  // Solver::Step

  final_data_mult = alpha_param_current;
}

template<typename Dtype>
void LineSearchSolver<Dtype>::PerformLineSearch() {
  Dtype final_data_mult, final_diff_mult;
  PerformLineSearch(final_data_mult, final_diff_mult);
}

template <typename Dtype>
void LineSearchSolver<Dtype>::ScaleDiffByLocalLrParams() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    Dtype scale = net_params_lr[param_id];
    net_params[param_id]->scale_diff(scale);
  }
}


template <typename Dtype>
void LineSearchSolver<Dtype>::ComputeUpdateValue() {
  this->TrackAvgGradNorm();

  // perform L1 or L2 regularization
  this->RegularizeGradient();

  ScaleDiffByLocalLrParams();

  PerformLineSearch();

  Dtype chosen_alpha = this->alphas_[this->prev_alpha_index];
  this->DisplayIterInfo(chosen_alpha);
}

template <typename Dtype>
int LineSearchSolver<Dtype>::GetStartingLrIndex() {
  return 0;
}


template <typename Dtype>
void LineSearchSolver<Dtype>::GrantReward(Dtype old_obj,
    Dtype new_obj, int alpha_index) {
  return;
}

template <typename Dtype>
void LineSearchSolver<Dtype>::SetInOneDimBlob(
    shared_ptr<Blob<Dtype> >& blob_vect, int index, Dtype val) {
  blob_vect->mutable_cpu_data()[blob_vect->offset(0,0,0,index)] = val;
}

//// get the value from a Blob vector which is stored at a given index
template <typename Dtype>
Dtype LineSearchSolver<Dtype>::GetFromOneDimBlob(
    shared_ptr<Blob<Dtype> >& blob_vect, int index) {
  return blob_vect->cpu_data()[blob_vect->offset(0,0,0,index)];
}

template <typename Dtype>
shared_ptr<Blob<Dtype> > LineSearchSolver<Dtype>::Vect2Blob(const vector<Dtype> & vect) {
  int n_elements = vect.size();
  shared_ptr<Blob<Dtype> > blob =
      shared_ptr<Blob<Dtype> >(new Blob<Dtype>(1, 1, 1, n_elements));
  for (int i = 0; i < n_elements; ++i) {
    LineSearchSolver<Dtype>::SetInOneDimBlob(blob, i, vect.at(i));
  }
  return blob;
}

template <typename Dtype>
vector<Dtype> * LineSearchSolver<Dtype>::Blob2Vect(shared_ptr<Blob<Dtype> > & blob) {
  vector<Dtype> * vect = new vector<Dtype>();
  assert(blob.shape(0) == 1);
  assert(blob.shape(1) == 1);
  assert(blob.shape(2) == 1);

  int n_elements = blob->shape(3);
  for (int i = 0; i < n_elements; ++i) {
    vect->push_back(LineSearchSolver<Dtype>::GetFromOneDimBlob(blob, i));
  }

  return vect;
}

template <typename Dtype>
void LineSearchSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  // convert vectors to blobs so they can be saved
  shared_ptr<Blob<Dtype> > alphas_blob = this->Vect2Blob(alphas_);

  // save blob versions of alphas
  state->clear_history();

  BlobProto* save_blob = state->add_history();
  alphas_blob->ToProto(save_blob);
}

template <typename Dtype>
void LineSearchSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), 1)
      << "SolverState should have exactly 1 elements in its history:\n" \
          "a 1-dimensional blob representing alphas.\n" \
          "Instead found " << state.history_size() << " elements.";
  LOG(INFO) << "LineSearchSolver: restoring history";

  shared_ptr<Blob<Dtype> > alphas_blob =
      shared_ptr<Blob<Dtype> >(new Blob<Dtype>());

  alphas_blob->FromProto(state.history(0));

  int n_loaded = alphas_blob->shape(3);
  int n_expected = alphas_.size();
  CHECK_EQ(n_loaded, n_expected)
    << "SolverState file with " << n_loaded << " alpha values loaded into " \
    "LineSearchSolver which was set to have " << n_expected << "possible alphas";

  alphas_ = *Blob2Vect(alphas_blob);
}















template <typename Dtype>
void LineSearchCurrentSolver<Dtype>::PreSolve() {

  this->log_high_alpha = this->param_.log_high_alpha();
  this->log_low_alpha = this->param_.log_low_alpha();
  this->n_alphas = this->param_.n_alphas();

  this->LogSpace(this->alphas_,
      this->log_high_alpha, this->log_low_alpha, this->n_alphas);

  this->ALPHA_GROW_RATE = 1;
  this->prev_alpha_index = 0;

// really i want this to replace the presolve from sgdsolver because
// i don't want history_, only temp_.
//
//  // initialize temporary update memory to same size as net parameters
//  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
//  temp_ = this->temp_;
//  temp_.clear();
//  for (int i = 0; i < net_params.size(); ++i) {
//    const vector<int>& shape = net_params[i]->shape();
//    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
//  }
}

template <typename Dtype>
int LineSearchCurrentSolver<Dtype>::GetStartingLrIndex() {
  return std::max(this->prev_alpha_index - this->ALPHA_GROW_RATE, 0);
}

template <typename Dtype>
void LineSearchCurrentSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  // convert vectors to blobs so they can be saved
  shared_ptr<Blob<Dtype> > alphas_blob = this->Vect2Blob(this->alphas_);

  // save blob version of alphas_ into SolverState
  state->clear_history();

  BlobProto* save_blob = state->add_history();
  alphas_blob->ToProto(save_blob);
}

template <typename Dtype>
void LineSearchCurrentSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), 3)
      << "SolverState should have exactly 3 elements in its history:\n" \
          "1-dimensional blobs representing alphas, rewards, and numbers.\n" \
          "Instead found: " << state.history_size() ;
  LOG(INFO) << "LineSearchCurrentSolver: restoring history";
  LOG(INFO) << "WARNING: UNTESTED!";
  LOG(INFO) << "WARNING: UNTESTED!";
  LOG(INFO) << "WARNING: UNTESTED!";
  LOG(INFO) << "WARNING: UNTESTED!";
  LOG(INFO) << "WARNING: UNTESTED!";
  LOG(INFO) << "WARNING: UNTESTED!";

  shared_ptr<Blob<Dtype> > alphas_blob =
      shared_ptr<Blob<Dtype> >(new Blob<Dtype>());

  alphas_blob->FromProto(state.history(0));

  int n_loaded = alphas_blob->shape(3);
  int n_expected = this->alphas_.size();
  CHECK_EQ(n_loaded, n_expected)
    << "SolverState file with " << n_loaded << " alpha values loaded into " \
    "LineSearchCurrentSolver which was set to have " << n_expected << "possible alphas";

  this->alphas_ = *this->Blob2Vect(alphas_blob);

  // setting prev_alpha_index to 0 means that the LineSearchCurrentSolver won't
  // remember where it was when the snapshot was taken.  that's probably ok
  // in practice because it will just backtrack down to where it needs to be
  // during the first iteration anyway.
  this->ALPHA_GROW_RATE = 1;
  this->prev_alpha_index = 0;
}























template <typename Dtype>
void DucbSolver<Dtype>::PreSolve() {

  this->log_high_alpha = this->param_.log_high_alpha();
  this->log_low_alpha = this->param_.log_low_alpha();
  this->n_alphas = this->param_.n_alphas();

  ducb_gamma = this->param_.ducb_gamma();
  explore_const = this->param_.explore_const();

  this->LogSpace(this->alphas_,
      this->log_high_alpha, this->log_low_alpha, this->n_alphas);

  // sufficient statistics for the bandit model
  this->rewards_.resize(this->n_alphas);
  this->numbers_.resize(this->n_alphas);

  this->mus_.resize(this->n_alphas);
  this->cs_.resize(this->n_alphas);
  this->js_.resize(this->n_alphas);

  this->init_sweep_ind = 0;

// really i want this to replace the presolve from sgdsolver because
// i don't want history_, only temp_.
//
//  // initialize temporary update memory to same size as net parameters
//  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
//  temp_ = this->temp_;
//  temp_.clear();
//  for (int i = 0; i < net_params.size(); ++i) {
//    const vector<int>& shape = net_params[i]->shape();
//    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
//  }
}


template <typename Dtype>
int DucbSolver<Dtype>::GetStartingLrIndex() {
  int n_alphas = this->n_alphas;
  if (init_sweep_ind < n_alphas)
    return init_sweep_ind++;

  Dtype n_total = 0;
  for (int i = 0; i < n_alphas; ++i) {
    rewards_[i] *= ducb_gamma;
    numbers_[i] *= ducb_gamma;
    n_total += numbers_[i];
  }

  int best_ind = 0;
  Dtype best_j = std::numeric_limits<Dtype>::min();
  Dtype j;

  for (int i = 0; i < n_alphas; ++i) {
    assert(numbers_[i] > 0);
    mus_[i] = rewards_[i] / numbers_[i];
    cs_[i] = sqrt(explore_const * log(n_total) / numbers_[i]);
    j = mus_[i] + cs_[i];
    js_[i] = j;
    assert(!isnan(j));
    if (j > best_j) { best_j = j; best_ind = i; }
  }

  return best_ind;
}


template <typename Dtype>
void DucbSolver<Dtype>::GrantReward(Dtype old_obj,
    Dtype new_obj, int alpha_index) {
  Dtype stability_const = 1e-15;
  Dtype reward_amount =
      log(old_obj + stability_const) - log(new_obj + stability_const);

  assert(!isnan(reward_amount));
  assert(alpha_index < n_alphas && alpha_index >= 0);
  numbers_[alpha_index] += 1;
  rewards_[alpha_index] += reward_amount;
  return;
}

template <typename Dtype>
void DucbSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  // convert vectors to blobs so they can be saved
  shared_ptr<Blob<Dtype> > alphas_blob = this->Vect2Blob(this->alphas_);
  shared_ptr<Blob<Dtype> > rewards_blob = this->Vect2Blob(rewards_);
  shared_ptr<Blob<Dtype> > numbers_blob = this->Vect2Blob(numbers_);

  // save blob versions of alphas, rewards, and numbers into SolverState
  state->clear_history();

  BlobProto* save_blob = state->add_history();
  alphas_blob->ToProto(save_blob);

  save_blob = state->add_history();
  rewards_blob->ToProto(save_blob);

  save_blob = state->add_history();
  numbers_blob->ToProto(save_blob);
}

template <typename Dtype>
void DucbSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), 3)
      << "SolverState should have exactly 3 elements in its history:\n" \
          "1-dimensional blobs representing alphas, rewards, and numbers.\n" \
          "Instead found: " << state.history_size() ;
  LOG(INFO) << "DucbSolver: restoring history";

  shared_ptr<Blob<Dtype> > alphas_blob =
      shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  shared_ptr<Blob<Dtype> > rewards_blob =
      shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  shared_ptr<Blob<Dtype> > numbers_blob =
      shared_ptr<Blob<Dtype> >(new Blob<Dtype>());

  alphas_blob->FromProto(state.history(0));
  rewards_blob->FromProto(state.history(1));
  numbers_blob->FromProto(state.history(2));

  int n_loaded = alphas_blob->shape(3);
  int n_expected = this->alphas_.size();
  CHECK_EQ(n_loaded, n_expected)
    << "SolverState file with " << n_loaded << " alpha values loaded into " \
    "DucbSolver which was set to have " << n_expected << "possible alphas";

  this->alphas_ = *this->Blob2Vect(alphas_blob);
  rewards_ = *this->Blob2Vect(rewards_blob);
  numbers_ = *this->Blob2Vect(numbers_blob);

  // setting init_sweep_ind to n_expected means that we are assuming that the
  // DucbSolver was run for at least n_expected iterations before the snapshot
  // was taken.  this should toe a reasonable assumption in practice.
  init_sweep_ind = n_expected;
}


INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);
INSTANTIATE_CLASS(NesterovSolver);
INSTANTIATE_CLASS(AdaGradSolver);
INSTANTIATE_CLASS(AdaGradLineSearchSolver);
INSTANTIATE_CLASS(AdaDeltaSolver);
INSTANTIATE_CLASS(LineSearchSolver);
INSTANTIATE_CLASS(LineSearchCurrentSolver);
INSTANTIATE_CLASS(DucbSolver);

}  // namespace caffe

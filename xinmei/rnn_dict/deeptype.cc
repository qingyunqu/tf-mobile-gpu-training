//
// by Echo.
//

#include <stdio.h>
#include <vector>
#include <time.h>
#include <sstream>
#include "deeptype.h"
#include "utils.h"

using namespace tensorflow;

namespace {
    template<class TFlat>
    inline std::vector<std::pair<int, float>> topK(const TFlat &flat, int topN) {
      std::vector<std::pair<int, float>> top;
      top.reserve((size_t) (topN + 1));
      for (int i = 0, size = static_cast<int>(flat.size()); i < size; i++) {
        float p = flat(i);
        if (top.size() < topN || p > top.rbegin()->second) {
          top.push_back(std::make_pair(i, p));
          std::make_heap(top.rbegin(), top.rend(),
                         [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
                             return a.second > b.second;
                         });
        }
        while (top.size() > topN) {
          top.erase(top.begin() + (top.size() - 1));
          std::make_heap(top.rbegin(), top.rend(),
                         [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
                             return a.second > b.second;
                         });
        }
      }
      std::sort(top.begin(), top.end(),
                [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
                    return a.second > b.second;
                });
      return top;
    }

    inline TensorShape shapeOf(int *shape, size_t size) {
      TensorShape s;
      for (size_t i = 0; i < size; i++) {
        s.AddDim(shape[i]);
      }
      return s;
    }

    template<DataType DT>
    inline Tensor zeroTensor(const TensorShape &shape) {
      Tensor tensor(DT, shape);
      auto flat = tensor.flat<typename EnumToDataType<DT>::Type>();
      for (int i = 0; i < flat.size(); i++) {
        flat(i) = 0;
      }
      return tensor;
    }

    template<typename T>
    inline std::unique_ptr<T> make_unique(T *ptr) {
      return std::unique_ptr<T>(ptr);
    }

    struct SubSequenceFinder {
        bool found;
        size_t begin;
        size_t length;

        inline SubSequenceFinder(const std::vector<int> &sequence, const int *subSequence, size_t subSequenceLen,
                                 bool exact)
            : found(false), begin(0), length(0) {
          if (sequence.empty() || subSequenceLen == 0)
            return;

          for (size_t sequenceBeg = 0, sequenceCheckSize = exact ? std::min<size_t>(1, sequence.size())
                                                                 : sequence.size();
               sequenceBeg < sequenceCheckSize;
               sequenceBeg++) {
            size_t compareLen = std::min(sequence.size() - sequenceBeg, subSequenceLen);
            size_t matchLen = 0;
            for (size_t offset = 0; offset < compareLen; offset++) {
              if (sequence[sequenceBeg + offset] == subSequence[offset])
                matchLen++;
              else
                break;
            }

            if (matchLen > length) {
              found = true;
              begin = sequenceBeg;
              length = matchLen;
            }
          }
        }
    };
}

DeepType::DeepType(std::string model_prefix, char *infer_prefix, char *pName, char *wordIdName, int *wordIdShape,
                  size_t maxStateStepCacheCount, char *train_prefix, char *logits_op, char *sm_input_op, char *train_op,
                  char *mask_op, char *train_input_op, char *target_op, char* lr_op)
    : mPName(pName), mWordIdName(wordIdName), mLogitsName(logits_op), mTrainPrefix(train_prefix), model_path(model_prefix), mSmInputName(sm_input_op),
      mTrainName(train_op), mMaskName(mask_op), mTargetName(target_op), mTrainInputName(train_input_op), mLRName(lr_op), mInferPrefix(infer_prefix),
      mWordIdShape(shapeOf(wordIdShape, 2)), mMaxStateStepCacheCount(maxStateStepCacheCount) {
  LOGD("DeepType() begin");
  printf("DeepType()\n");

  SessionOptions options;
  mSession.reset(NewSession(options));
  LOGD("DeepType() session created");

  Status status;
  // Read in the protobuf graph we exported
  std::string path_to_graph = model_prefix + ".meta";
  status = ReadBinaryProto(Env::Default(), path_to_graph, &meta_graph_def);
  if (!status.ok()) {
      LOGE("Error reading graph definition from %s: %s\n", path_to_graph.c_str(), status.ToString().c_str());
      return;
  }
  // printGraphInfo(meta_graph_def.graph_def());

  // Add the graph to the session
  status = mSession->Create(meta_graph_def.graph_def());
  if (!status.ok()) {
      LOGE("Error creating graph: %s\n", status.ToString().c_str());
      return;
  }

  // Read weights from the saved checkpoint
  Tensor checkpointPathTensor(DT_STRING, TensorShape());
  checkpointPathTensor.scalar<std::string>()() = model_prefix;
  status = mSession->Run(
        {{ meta_graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
        {},
        {meta_graph_def.saver_def().restore_op_name()},
        nullptr);
  if (!status.ok()) {
      LOGE("Error loading checkpoint from %s: %s\n", model_prefix.c_str(), status.ToString().c_str());
      return;
  }

  LOGD("checkpoint loaded successfully!\n");

  // GraphData *graphData = nullptr;
  // mGraphData.reset(graphData);
  LOGD("DeepType() finish");

  _initialized = true;
}

DeepType::~DeepType() {
  // saveModel(model_path.c_str() + ".bak");
}

#define LOCK std::lock_guard<std::mutex> lock(mLock);

void DeepType::saveModel(const char *out_file) {
  LOCK
  Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointPathTensor.scalar<std::string>()() = out_file;
  std::vector<std::pair<std::string, Tensor>> inputs;
  inputs.push_back(std::make_pair(meta_graph_def.saver_def().filename_tensor_name(), checkpointPathTensor));
  Status status = mSession->Run(inputs, {}, {meta_graph_def.saver_def().save_tensor_name()}, nullptr);
  if (!status.ok()) {
      LOGE("Error saving checkpoint to %s: %s\n", out_file, status.ToString().c_str());
      return;
  }
}

void DeepType::saveModel() {
  saveModel(model_path.c_str());
}

void DeepType::addState(char *inName, char *outName, int *shape, size_t size, bool required) {
  LOCK
  LOGD("DeepType::addState");
  std::unique_ptr<State> state(new State());
  state->inName = inName;
  state->outName = outName;
  state->required = required;
  state->shape = shapeOf(shape, size);
  mStates.push_back(std::move(state));
}

void DeepType::predict(const int *ids, size_t idCount, bool partialSequence, int topN, bool save_logits) {
  LOCK

  LOGD("DeepType::predict begin");

  struct PredictTimeLogger {
      struct timespec begin;
      struct timespec end;
      double &cost;
      size_t &steps;

      inline PredictTimeLogger(double &_cost, size_t &_steps) : cost(_cost), steps(_steps) {
        clock_gettime(CLOCK_REALTIME, &begin);
      }

      inline ~PredictTimeLogger() {
        clock_gettime(CLOCK_REALTIME, &end);
        cost = 1000.0 * (end.tv_sec - begin.tv_sec) + (double) (end.tv_nsec - begin.tv_nsec) / 1e6;
        LOGD("predict %d step, cost %fms", static_cast<int>(steps), cost);
      }
  } predictTimeLogger(mLastPredictDuration, mLastPredictStepCount);

  // try restore states
  SubSequenceFinder subSequenceFinder(mStateSteps, ids, idCount, partialSequence);
  LOGD("sub sequence beg %d, len %d",
       static_cast<int>(subSequenceFinder.begin),
       static_cast<int>(subSequenceFinder.length));

  size_t stepKeepLen = 0;
  if (subSequenceFinder.found) {
    if (subSequenceFinder.length == idCount) // predict at least one step, assure that we have predict result
      subSequenceFinder.length--;
    stepKeepLen = subSequenceFinder.begin + subSequenceFinder.length;
  }
  // LOGD("stepKeepLen %d", static_cast<int>(stepKeepLen));
  for (auto &state : mStates) {
    state->steps.resize(stepKeepLen);
  }
  mStateSteps.resize(stepKeepLen);
  mLogits.resize(stepKeepLen);
  mSmInputs.resize(stepKeepLen);

  // prepare graph info
  std::vector<std::string> outputNames;
  for (auto &state : mStates) {
    outputNames.push_back(mInferPrefix + state->outName);
  }

  std::vector<std::pair<std::string, Tensor>> baseInputs;
  baseInputs.push_back(std::make_pair(mInferPrefix + mWordIdName, Tensor(DT_INT32, mWordIdShape)));
  // if (mGraphData) {
  //   mGraphData->append(baseInputs);
  // }

  // each id
  std::vector<Tensor> outputs;
  outputs.reserve(mStates.size() + 2);
  mLastPredictStepCount = 0;
  for (size_t idIndex = subSequenceFinder.length; idIndex < idCount; idIndex++) {
    LOGD("run id %d", ids[idIndex]);
    std::vector<std::pair<std::string, Tensor>> inputs(baseInputs);
    inputs[0].second.flat<int>()(0) = ids[idIndex];

    for (auto &state : mStates) {
      // input state
      if (state->steps.empty()) {
        if (state->required) {
          inputs.push_back(std::make_pair(mInferPrefix + state->inName, zeroTensor<DT_FLOAT>(state->shape)));
        }
      } else {
        inputs.push_back(std::make_pair(mInferPrefix + state->inName, **state->steps.rbegin()));
      }
    }


    // calculate output p at last predict step
    if (topN > 0 && idIndex == idCount - 1) {
      outputNames.push_back(mInferPrefix + mPName);
    }

    if (save_logits) {
      outputNames.push_back(mInferPrefix + mLogitsName);
      outputNames.push_back(mInferPrefix + mSmInputName);
    }

    outputs.clear();
    auto status = mSession->Run(inputs, outputNames, {}, &outputs);
    if (!status.ok())
      error("run failed: %d %s", static_cast<int>(status.code()), status.error_message().c_str());
    mLastPredictStepCount++;
    mStateSteps.push_back(ids[idIndex]);
    for (size_t i = 0; i < mStates.size(); i++) {
      mStates[i]->steps.push_back(make_unique(new Tensor(std::move(outputs[i]))));
    }
    if (save_logits) {
      mLogits.push_back(outputs[outputs.size() - 2]);
      mSmInputs.push_back(outputs[outputs.size() - 1]);
    }
  }

  // get result
  if (topN > 0)
    mPredicted = topK(outputs[mStates.size()].flat<float>(), topN);
  else
    mPredicted.clear();

  LOGD("DeepType::predict finish");
}

void DeepType::train_online(const int *ids, size_t wordCount, size_t letterCount, int id_out, float learning_rate, bool infer_reuse) {
  LOCK
  // LOGD("train_online begin\n");

  // cut down the id count. max word count is 3
  size_t idCount = wordCount + letterCount;
  if (idCount > mMaxStepSize) {
    LOGD("Cut down ids word: %d, letter: %d, max: %d\n", wordCount, letterCount, mMaxStepSize);
    letterCount = mMaxStepSize - wordCount;
    idCount = wordCount + letterCount;
  }

  std::ostringstream oss;
  oss << mTrainPrefix << int(idCount);
  std::string prefix = oss.str();

  struct TrainTimeLogger {
      struct timespec begin;
      struct timespec end;
      double cost;
      size_t &steps;

      inline TrainTimeLogger(size_t _steps) : steps(_steps) {
        clock_gettime(CLOCK_REALTIME, &begin);
      }

      inline ~TrainTimeLogger() {
        clock_gettime(CLOCK_REALTIME, &end);
        cost = 1000.0 * (end.tv_sec - begin.tv_sec) + (double) (end.tv_nsec - begin.tv_nsec) / 1e6;
        LOGD("train %d step, cost %fms", static_cast<int>(steps), cost);
      }
  } trainTimeLogger(idCount);

  std::vector<std::pair<std::string, Tensor>> inputs;

  int inputShape[2] = {1, idCount};
  
  Tensor word_tensor(DT_INT32, shapeOf(inputShape, 2));
  for (int i = 0; i < idCount; i ++) {
    word_tensor.flat<int>()(i) = ids[i];
  }
  inputs.push_back(std::make_pair(prefix + mTrainInputName, word_tensor));
  
  Tensor target_tensor(DT_INT32, shapeOf(inputShape, 2));
  for (int i = 0; i < idCount; i ++) {
    target_tensor.flat<int>()(i) = id_out;
  }
  inputs.push_back(std::make_pair(prefix + mTargetName, target_tensor));
  
  Tensor mask_tensor(DT_FLOAT, shapeOf(inputShape, 2));
  for (int i = 0; i < idCount; i ++) {
    mask_tensor.flat<float>()(i) = (i < ((int)wordCount - 1)) ? 0 : 1;
  }
  inputs.push_back(std::make_pair(prefix + mMaskName, mask_tensor));

  Tensor lr_tensor(DT_FLOAT, shapeOf(nullptr, 0)); // learning rate is a scalar
  lr_tensor.flat<float>()(0) = learning_rate;
  inputs.push_back(std::make_pair(prefix + mLRName, lr_tensor));


  // Check if we can reuse the results
  // TODO: maybe we can use subsequence as in prediction
  bool can_reuse = infer_reuse && (idCount > 0) && (idCount == mStateSteps.size());
  if (infer_reuse) {
    for (int i = 0; i < idCount; i ++) {
      if (ids[i] != mStateSteps[i]) {
        can_reuse = false;
        break;
      }
    }
  }

  if (infer_reuse && !can_reuse) {
    LOGD("Fail to reuse inference results!");
  }

  if (can_reuse) {
    struct TrainTimeLogger logger(0);
    // softmax_output: (batch_size * step_size) * out_vocab_size
    // We need to concat the saved logits to one
    int logits_shape[2] = {1 * idCount, mLogits[0].dim_size(1)};
    Tensor logits_tensor(DT_FLOAT, shapeOf(logits_shape, 2));
    // LOGD("%s %s\n", mLogits[0].DebugString().c_str(), logits_tensor.DebugString().c_str());
    void* logits_ptr = logits_tensor.flat<float>().data();
    size_t logits_chunk_size = sizeof(float) * mLogits[0].dim_size(1);
    for (int i = 0; i < idCount; i ++) {
      const void* temp_ptr = mLogits[i].tensor_data().data();
      //memcpy(logits_ptr + i * logits_chunk_size, temp_ptr, logits_chunk_size);
    }
    // This way is too slow
    // for (int i = 0; i < idCount; i ++)
    //   for (int j = 0; j < logits_tensor.dim_size(1); j ++)
    //     logits_tensor.tensor<float, 2>()(i, j) = mLogits[i].tensor<float, 2>()(0, j);
    inputs.push_back(std::make_pair(prefix + mLogitsName, logits_tensor));

    LOGD("softmax input: %s\n", mSmInputs[0].DebugString().c_str());
    int sm_shape[2] = {1 * idCount, mSmInputs[0].dim_size(1)};
    Tensor sm_tensor(DT_FLOAT, shapeOf(sm_shape, 2));
    void* sm_ptr = sm_tensor.flat<float>().data();
    size_t sm_chunk_size = sizeof(float) * mSmInputs[0].dim_size(1);
    for (int i = 0; i < idCount; i ++) {
      const void* temp_ptr = mSmInputs[i].tensor_data().data();
      //memcpy(sm_ptr + i * sm_chunk_size, temp_ptr, sm_chunk_size);
    }
    inputs.push_back(std::make_pair(prefix + mSmInputName, sm_tensor));
  }

  std::vector<Tensor> outputs;
  outputs.reserve(1);

  auto status = mSession->Run(inputs, {}, {prefix + mTrainName}, &outputs);
  if (!status.ok())
    error("run failed: %d %s", static_cast<int>(status.code()), status.error_message().c_str());
  // LOGD("Loss: %f\n", outputs[0].flat<float>()(0));


  // Clear everything we have cached so that next prediction
  // will be performed on the new model params
  if (can_reuse) {
    clearState();
  }
}

void DeepType::clearState() {
  mLogits.resize(1);
  mStates.resize(1);
  mStateSteps.resize(1);
  mSmInputs.resize(1);
}

int DeepType::getWordId(int index) {
  LOCK
  if (index < 0 || index >= mPredicted.size())
    error("index out of bounds, index %d not in [0, %d)", index, mPredicted.size());
  return mPredicted[index].first;
}

float DeepType::getP(int index) {
  LOCK
  if (index < 0 || index >= mPredicted.size())
    error("index out of bounds, index %d not in [0, %d)", index, mPredicted.size());
  return mPredicted[index].second;
}

size_t DeepType::lastPredictStepCount() {
  LOCK
  return mLastPredictStepCount;
}

int DeepType::lastPredictTimeInMillis() {
  LOCK
  return mLastPredictDuration;
}

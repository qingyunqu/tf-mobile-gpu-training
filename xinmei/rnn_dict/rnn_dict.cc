//
// by afpro.
//

#include <stdio.h>
#include <vector>
#include "rnn_dict.h"
#include "utils.h"

#ifdef NO_LOG
#define LOGD(...)
#define LOGE(...)
#elif defined(__ANDROID__)

#include <android/log.h>

namespace {
    template<typename...TArgs>
    void LOGD(const TArgs &...args) {
      __android_log_print(ANDROID_LOG_DEBUG, "rnn_dict", args...);
    }

    template<typename...TArgs>
    void LOGE(const TArgs &...args) {
      __android_log_print(ANDROID_LOG_ERROR, "rnn_dict", args...);
    }
}
#else
namespace {
    template <typename...TArgs>
    void LOGD(const TArgs &...args) {
        printf(args...);
        printf("\n");
    }

    template <typename...TArgs>
    void LOGE(const TArgs &...args) {
        fprintf(stderr, args...);
        fprintf(stderr, "\n");
    }
}
#endif

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

    inline tensorflow::TensorShape shapeOf(int *shape, size_t size) {
      tensorflow::TensorShape s;
      for (size_t i = 0; i < size; i++) {
        s.AddDim(shape[i]);
      }
      return s;
    }

    template<tensorflow::DataType DT>
    inline tensorflow::Tensor zeroTensor(const tensorflow::TensorShape &shape) {
      tensorflow::Tensor tensor(DT, shape);
      auto flat = tensor.flat<typename tensorflow::EnumToDataType<DT>::Type>();
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

RnnDict::RnnDict(char *data, size_t size,
                 char *pName,
                 char *wordIdName, int *wordIdShape, size_t wordIdShapeSize,
                 size_t maxStateStepCacheCount,
                 bool quantized)
    : mPName(pName), mWordIdName(wordIdName), mWordIdShape(shapeOf(wordIdShape, wordIdShapeSize)),
      mMaxStateStepCacheCount(maxStateStepCacheCount) {
  LOGD("RnnDict() begin");

  tensorflow::SessionOptions options;
  mSession.reset(tensorflow::NewSession(options));
  LOGD("RnnDict() session created");

  tensorflow::Status status;
  GraphData *graphData = nullptr;
  if (quantized) {
    status = loadQuantizedGraph(mSession.get(), &graphData, data, static_cast<int>(size));
    LOGD("RnnDict() quantized graph loaded");
  } else {
    tensorflow::GraphDef graph;
    if (!graph.ParseFromArray(data, (int) size))
      error("parse graph failed");
    status = mSession->Create(graph);
    LOGD("RnnDict() normal graph loaded");
  }
  mGraphData.reset(graphData);
  if (!status.ok())
    error("init RnnDict failed (quantized=%d): %s", (int) quantized, status.error_message().c_str());
  LOGD("RnnDict() finish");
}

RnnDict::~RnnDict() = default;

#define LOCK std::lock_guard<std::mutex> lock(mLock);

void RnnDict::addState(char *inName, char *outName, int *shape, size_t size, bool required) {
  LOCK
  std::unique_ptr<State> state(new State());
  state->inName = inName;
  state->outName = outName;
  state->required = required;
  state->shape = shapeOf(shape, size);
  mStates.push_back(std::move(state));
}

void RnnDict::predict(const int *ids, size_t idCount, bool partialSequence, int topN) {
  LOCK

  struct PredictTimeLogger {
      clock_t begin;
      clock_t &cost;
      size_t &steps;

      inline PredictTimeLogger(clock_t &_cost, size_t &_steps) : begin(clock()), cost(_cost), steps(_steps) {
      }

      inline ~PredictTimeLogger() {
        cost = clock() - begin;
        LOGD("predict %d step, cost %dms", static_cast<int>(steps), static_cast<int>(cost / (CLOCKS_PER_SEC / 1000)));
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
  LOGD("stepKeepLen %d", static_cast<int>(stepKeepLen));
  for (auto &state : mStates) {
    state->steps.resize(stepKeepLen);
  }
  mStateSteps.resize(stepKeepLen);

  // prepare graph info
  std::vector<std::string> outputNames;
  for (auto &state : mStates) {
    outputNames.push_back(state->outName);
  }

  std::vector<std::pair<std::string, tensorflow::Tensor>> baseInputs;
  baseInputs.push_back(std::make_pair(mWordIdName, tensorflow::Tensor(tensorflow::DT_INT32, mWordIdShape)));
  if (mGraphData) {
    mGraphData->append(baseInputs);
  }

  // each id
  std::vector<tensorflow::Tensor> outputs;
  outputs.reserve(mStates.size() + 1);
  mLastPredictStepCount = 0;
  for (size_t idIndex = subSequenceFinder.length; idIndex < idCount; idIndex++) {
    LOGD("run id %d", ids[idIndex]);
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs(baseInputs);
    inputs[0].second.flat<int>()(0) = ids[idIndex];

    for (auto &state : mStates) {
      // input state
      if (state->steps.empty()) {
        if (state->required) {
          inputs.push_back(std::make_pair(state->inName, zeroTensor<tensorflow::DT_FLOAT>(state->shape)));
        }
      } else {
        inputs.push_back(std::make_pair(state->inName, **state->steps.rbegin()));
      }
    }

    // calculate output p at last predict step
    if (topN > 0 && idIndex == idCount - 1) {
      outputNames.push_back(mPName);
    }

    outputs.clear();
    auto status = mSession->Run(inputs, outputNames, {}, &outputs);
    if (!status.ok())
      error("run failed: %d %s", static_cast<int>(status.code()), status.error_message().c_str());

    mLastPredictStepCount++;
    mStateSteps.push_back(ids[idIndex]);
    for (size_t i = 0; i < mStates.size(); i++) {
      mStates[i]->steps.push_back(make_unique(new tensorflow::Tensor(std::move(outputs[i]))));
    }
  }

  // get result
  if (topN > 0)
    mPredicted = topK(outputs[mStates.size()].flat<float>(), topN);
  else
    mPredicted.clear();
}

int RnnDict::getWordId(int index) {
  LOCK
  if (index < 0 || index >= mPredicted.size())
    error("index out of bounds, index %d not in [0, %d)", index, mPredicted.size());
  return mPredicted[index].first;
}

float RnnDict::getP(int index) {
  LOCK
  if (index < 0 || index >= mPredicted.size())
    error("index out of bounds, index %d not in [0, %d)", index, mPredicted.size());
  return mPredicted[index].second;
}

size_t RnnDict::lastPredictStepCount() {
  LOCK
  return mLastPredictStepCount;
}

int RnnDict::lastPredictTimeInMillis() {
  LOCK
  return static_cast<int>(mLastPredictDuration / (CLOCKS_PER_SEC / 1000));
}

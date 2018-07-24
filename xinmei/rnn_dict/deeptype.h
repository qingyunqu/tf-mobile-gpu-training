//
// by Echo.
//


#include <exception>
#include <string>
#include <map>
#include <vector>
#include <stack>
#include <mutex>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#define log_to_std true

#ifdef NO_LOG
#define LOGD(...)
#define LOGE(...)

#elif log_to_std || !defined(__ANDROID__)
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

#else
#include <android/log.h>

namespace {
    template<typename...TArgs>
    void LOGD(const TArgs &...args) {
      __android_log_print(ANDROID_LOG_DEBUG, "deeptype", args...);
    }

    template<typename...TArgs>
    void LOGE(const TArgs &...args) {
      __android_log_print(ANDROID_LOG_ERROR, "deeptype", args...);
    }
}

#endif


struct State {
    std::string inName;
    std::string outName;
    tensorflow::TensorShape shape;
    bool required;
    std::vector<std::unique_ptr<tensorflow::Tensor>> steps;
};

class DeepType {
public:
    DeepType(std::string model_prefix, char *infer_prefix, char *pName, char *wordIdName, int *wordIdShape,
            size_t maxStateStepCacheCount, char *train_prefix, char *logits_op, char *sm_input_op, char *train_op,
            char *mask_op, char *train_input_op, char *target_op, char* lr_op);
    ~DeepType();

    void addState(char *inName, char *outName, int *shape, size_t size, bool required);
    void predict(const int *ids, size_t idCount, bool partialSequence, int topN, bool save_logits);
    void train_online(const int *ids, size_t wordCount, size_t letterCount, int id_out, float learning_rate, bool infer_reuse);
    int getWordId(int index);
    float getP(int index);
    void saveModel(const char *out_file);
    void saveModel();
    void clearState();

    size_t lastPredictStepCount();
    int lastPredictTimeInMillis();
    bool initialized() {return _initialized;}

private:
    std::mutex mLock;

    bool _initialized = false;

    std::unique_ptr<tensorflow::Session> mSession;
    // std::unique_ptr<GraphData> mGraphData;
    std::vector<std::pair<int, float>> mPredicted;
    std::string model_path;
    tensorflow::MetaGraphDef meta_graph_def;

    std::string mInferPrefix;
    std::string mPName;
    std::string mWordIdName;
    tensorflow::TensorShape mWordIdShape;
    std::vector<std::unique_ptr<State>> mStates;

    size_t mMaxStateStepCacheCount;
    std::vector<int> mStateSteps;

    size_t mLastPredictStepCount;
    double mLastPredictDuration;

    // For training
    int mMaxStepSize = 15;
    std::string mTrainPrefix;
    std::string mMaskName;
    std::string mTargetName;
    std::string mTrainName;
    std::string mLogitsName;
    std::string mTrainInputName;
    std::string mSmInputName;
    std::vector<tensorflow::Tensor> mSmInputs;
    std::string mLRName;
    std::vector<tensorflow::Tensor> mLogits;
};

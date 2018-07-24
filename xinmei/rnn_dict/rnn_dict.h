//
// by afpro.
//

#pragma once

#ifdef SWIG
%module RnnModule
%include "arrays_java.i"
%apply (char *STRING, size_t LENGTH) { (char *data, size_t size) };
%apply int[] {int *};
%ignore State;
%inline %{
#include "rnn_dict.h"
%};
%exception {
    try {
        $action;
    } catch (std::runtime_error &e) {
        jclass clazz = jenv->FindClass("java/lang/RuntimeException");
        jenv->ThrowNew(clazz, e.what());
        return $null;
    }
};
#endif

#include <exception>
#include <string>
#include <map>
#include <vector>
#include <stack>
#include <mutex>
#include "tensorflow/core/public/session.h"
#include "xinmei/quantized_graph_loader/quantized_graph_loader.h"


struct State {
    std::string inName;
    std::string outName;
    tensorflow::TensorShape shape;
    bool required;
    std::vector<std::unique_ptr<tensorflow::Tensor>> steps;
};

class RnnDict {
public:
    RnnDict(char *data, size_t size,
            char *pName,
            char *wordIdName, int *wordIdShape, size_t wordIdShapeSize,
            size_t maxStateStepCacheCount,
            bool quantized);
    ~RnnDict();

    void addState(char *inName, char *outName, int *shape, size_t size, bool required);
    void predict(const int *ids, size_t idCount, bool partialSequence, int topN);
    int getWordId(int index);
    float getP(int index);

    size_t lastPredictStepCount();
    int lastPredictTimeInMillis();

private:
    std::mutex mLock;

    std::unique_ptr<tensorflow::Session> mSession;
    std::unique_ptr<GraphData> mGraphData;
    std::vector<std::pair<int, float>> mPredicted;

    std::string mPName;
    std::string mWordIdName;
    tensorflow::TensorShape mWordIdShape;
    std::vector<std::unique_ptr<State>> mStates;

    size_t mMaxStateStepCacheCount;
    std::vector<int> mStateSteps;

    size_t mLastPredictStepCount;
    clock_t mLastPredictDuration;
};

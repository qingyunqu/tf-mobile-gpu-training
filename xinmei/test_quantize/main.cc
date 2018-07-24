#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/op.h"

namespace tf = tensorflow;

inline void successOrDie(const tf::Status &status, const char *msg) {
  if (!status.ok()) {
    printf("error(s): %s code:%d msg:%s\n", msg, (int) status.code(), status.error_message().c_str());
    exit(-1);
  }
  printf("success(s): %s\n", msg);
}

inline void successOrDie(bool success, const char *msg) {
  if (!success) {
    printf("error(b): %s\n", msg);
    exit(-1);
  }
  printf("success(b): %s\n", msg);
}


namespace {
  struct TimeSpot {
    clock_t t;

    void init() {
      t = clock();
    }

    void point(const char *msg) {
      clock_t now = clock();
      printf("%s cost %dms\n", msg, (int)((now - t) * 1000L / CLOCKS_PER_SEC));
      t = now;
    }
  } timeSpot;
}


int main() {
  {
    std::vector<tensorflow::OpDef> ops;
    tensorflow::OpRegistry::Global()->GetRegisteredOps(&ops);
    printf("\nops count %d\n", (int) ops.size());
  }

  timeSpot.init();

  tf::GraphDef gd;
  {
    #if __ANDROID__
    std::ifstream input("/data/local/tmp/t.pb");
    #else
    std::ifstream input("/tmp/fuck/t.pb");
    #endif
    successOrDie(gd.ParseFromIstream(&input), "parse graph");
  }
  timeSpot.point("parse graph");

  std::unique_ptr<tf::Session> session(tf::NewSession(tf::SessionOptions()));
  printf("new session %p\n", session.get());
  successOrDie(session->Create(gd), "create session");
  timeSpot.point("create session");

  while (!std::cin.eof()) {
    printf("===========NEW=LINE===========\n");
    std::string line;
    std::getline(std::cin, line);

    if (line.empty())
      continue;

    tf::Tensor state(tf::DT_FLOAT, tf::TensorShape({2, 2, 1, 400}));
    std::fill_n(state.flat<float>().data(), state.flat<float>().size(), 0);

    timeSpot.point("init state");

    std::istringstream is(line);
    while (!is.eof()) {
      int id = -1;
      is >> id;
      if (id < 0)
        break;

      printf("id: %d\n", id);

      std::vector<tensorflow::Tensor> outputs;
      std::vector<std::pair<std::string, tf::Tensor>> inputs;
      inputs.emplace_back("Test/Model/state", state);

      tf::Tensor seqLen(tf::DT_INT32, tf::TensorShape({1}));
      seqLen.flat<int>()(0) = 1;
      inputs.emplace_back("Test/Model/seqlen", seqLen);

      tf::Tensor wordIds(tf::DT_INT32, tf::TensorShape({1, 1}));
      wordIds.flat<int>()(0) = id;
      inputs.emplace_back("Test/Model/batched_input_word_ids", wordIds);

      timeSpot.point("prepare input");

      successOrDie(session->Run(inputs, {"Test/Model/probabilities", "Test/Model/state_out"}, {}, &outputs), "run");
      timeSpot.point("run");

      state = outputs[1];
      auto p = outputs[0].flat<float>();
      std::vector<std::pair<int, float>> ids;
      for (int i = 0; i < 10000; i++) {
        float pi = p(i);
        if (ids.size() < 10 || pi > ids[ids.size() - 1].second) {
          ids.emplace_back(i, pi);
        }
        while (ids.size() > 10) {
          std::make_heap(ids.rbegin(), ids.rend(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
            return a.second > b.second;
          });
          ids.erase(ids.end() - 1);
        }
      }

      std::sort(ids.rbegin(), ids.rend(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
        return a.second > b.second;
      });

      for (int i = 0; i < ids.size(); i++) {
        printf("\tpredict %d %f\n", ids[i].first, ids[i].second);
      }
    }
  }

  return 0;
}

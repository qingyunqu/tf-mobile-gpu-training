#include <ctime>

#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/public/session.h"
// #include "tensorflow/core/protobuf/meta_graph.pb.h"

#define  LOG_TO_LOGCAT false

#if LOG_TO_LOGCAT
#include <android/log.h>
#define  LOG_TAG    "rs_benchmark"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#else
#define  LOGI(...)  printf(__VA_ARGS__)
#endif

using namespace tensorflow;

inline void printGraphInfo(GraphDef graph) {
	LOGI("Start to print graph info\n");
	for (int i = 0; i < graph.node_size(); i ++) {
		NodeDef node = graph.node(i);
		LOGI("N%d: %s %s\n", i, node.name().c_str(), node.op().c_str());
	}
}

Session* loadGraphAndWeights(std::string path_to_graph) {
	Status status;
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), path_to_graph, &graph_def);
	if (!status.ok()) {
	    LOGI("Error reading graph definition from %s: %s\n",
	    	path_to_graph.c_str(), status.ToString().c_str());
	    return NULL;
	}
	LOGI("graph successfully loaded: %s\n", path_to_graph.c_str());
	// printGraphInfo(graph_def);

	// Add the graph to the session
	tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  // config.set_intra_op_parallelism_threads(1);
	Session* session = NewSession(options);
	if (session == nullptr) {
	    LOGI("Could not create Tensorflow session.\n");
	}
	status = session->Create(graph_def);
	if (!status.ok()) {
	    LOGI("Error creating graph: %s\n", status.ToString().c_str());
	    return NULL;
	}
	return session;
}

template <class T>
void InitializeTensorZero(Tensor* input_tensor) {
  auto type_tensor = input_tensor->flat<T>();
  type_tensor = type_tensor.constant(0);
  for (int i = 0; i < input_tensor->NumElements(); ++i) {
  	type_tensor(i) = static_cast<T>(0);
  }
}

int main(int argc, char** argv) {
	int batch_size = 32;
	int step_size = 10;
	int run_loop = 10;
	std::string graph_path = "./OnlineTraining.pb";
	std::string init_op_name = "OnlineTraining/Model/init_all_vars_op";
	std::string output_node = "OnlineTraining/Model/train_op/update_Model/Softmax/softmax_w/ApplyGradientDescent";
	std::vector<Flag> flag_list = {
      Flag("graph", &graph_path, "graph file name"),
      Flag("output_node", &output_node, "output node name"),
      Flag("batch_size", &batch_size, "batch size"),
      Flag("step_size", &step_size, "step size"),
      Flag("run_loop", &run_loop, "run times to average"),
      Flag("init_op", &init_op_name, "op name for initializing all vars")
  	};
  	string usage = Flags::Usage(argv[0], flag_list);
  	const bool parse_result = Flags::Parse(&argc, argv, flag_list);

  	if (!parse_result) {
    	LOG(ERROR) << usage;
    	return -1;
  	}

	Session* session = loadGraphAndWeights(graph_path);
	Status status;

	// feed input
	Tensor batched_input_word_ids(DT_INT32, TensorShape({batch_size, step_size}));
	Tensor batched_output_word_ids(DT_INT32, TensorShape({batch_size, step_size}));
	Tensor batched_output_word_masks(DT_FLOAT, TensorShape({batch_size, step_size}));
	InitializeTensorZero<int32>(&batched_input_word_ids);
	InitializeTensorZero<int32>(&batched_output_word_ids);

	std::vector<std::pair<string, Tensor>> feed_dict = {
		{"OnlineTraining/Model/batched_input_word_ids", batched_input_word_ids},
		{"OnlineTraining/Model/batched_output_word_ids", batched_output_word_ids},
		{"OnlineTraining/Model/batched_output_word_masks", batched_output_word_masks}
	};

	status = session->Run({}, {}, {init_op_name.c_str()}, nullptr);
	if (!status.ok()) {
		LOGI("Initialize error: %s\n", status.ToString().c_str());
		return -1;
	}

	struct timeval tv_begin, tv_end;
	LOGI("Start training for %d times...\n", run_loop);
	
	// warmup
	int warmups = 3;
	for (int i = 0; i < warmups; i ++)
		status = session->Run(feed_dict, {output_node.c_str()}, {}, nullptr);

	status = session->Run(feed_dict, {output_node.c_str()}, {}, nullptr);
	gettimeofday(&tv_begin, NULL);
	for (int i = 0; i < run_loop; i ++)
		status = session->Run(feed_dict, {output_node.c_str()}, {}, nullptr);
	gettimeofday(&tv_end, NULL);
	double elasped = tv_end.tv_sec - tv_begin.tv_sec + (tv_end.tv_usec - tv_begin.tv_usec) / 1000000.0f;
	LOGI("Run time: %lfs\n", elasped / run_loop);
	if (!status.ok()) {
		LOGI("Run error: %s\n", status.ToString().c_str());
		return -1;
	}
}
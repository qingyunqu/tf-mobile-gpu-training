#include "deeptype.h"
#include "deeptype_v0.1_val.h"

#include "tensorflow/core/util/command_line_flags.h"

int main(int argc, char** argv) {
	std::string model = "./deeptype.ckpt";
	int infer_reuse = 0;

	std::vector<tensorflow::Flag> flag_list = {
    tensorflow::Flag("reuse", &infer_reuse, "reuse inference results"),
  };
 	std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);

  if (!parse_result) {
    printf(usage.c_str());
    return -1;
  }


	int wordIdShape[2] = {1, 1}; // batch_size * step_size
	DeepType deeptype(model, deeptype_val::infer_prefix, deeptype_val::prob_op, deeptype_val::infer_input_op,
		wordIdShape, 20, deeptype_val::train_prefix, deeptype_val::logits_op, deeptype_val::sm_input_op,
		deeptype_val::train_op, deeptype_val::mask_op, deeptype_val::train_input_op, deeptype_val::target_op, deeptype_val::lr_op);
	int inf_stateShape[4] = {2, 2, 1, 400};
	deeptype.addState(deeptype_val::state_in_name, deeptype_val::state_out_name, inf_stateShape, 4, false);

	int round = 3;
	for (int i = 0; i < round; i ++) {
		printf("-----------------------");
		printf("Inference round %d\n", i);
		// int inf_ids[4] = {14, 11, 18, 18}; // 'h e l l'
		int inf_ids[10] = {307, 105, 8, 15, 24, 26, 14, 10, 7, 31}; // 'baby happy b i r t h d a y'
		for (int i = 1; i <= 10; i ++) {
			deeptype.predict(inf_ids, i, false, 3, true);
			printf("Prediction results: <%d, %f> <%d, %f> <%d, %f>\n\n",
				deeptype.getWordId(0), deeptype.getP(0),
				deeptype.getWordId(1), deeptype.getP(1),
				deeptype.getWordId(2), deeptype.getP(2));
		}

		printf("Train round %d\n", i);
		// int train_out = 520; // hello
		int train_out = 220; // birthday
		bool reuse = (infer_reuse == 0) ? false : true;
		deeptype.train_online(inf_ids, 2, 8, train_out, 0.2, reuse);
	}

	deeptype.saveModel("./deeptype_2.ckpt");
}
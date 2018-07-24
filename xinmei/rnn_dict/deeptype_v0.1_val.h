#ifndef DEEPTYPE_INCLUDE_VAL_V0_1
#define DEEPTYPE_INCLUDE_VAL_V0_1

namespace deeptype_val {
	// For inference
	char* infer_prefix		= "Online";
	char* prob_op					= "/Model/probabilities";
	char* infer_input_op	= "/Model/batched_input_word_ids";
	char* logits_op				= "/Model/Softmax/add";
	char* sm_input_op			= "/Model/Reshape";
	char* state_in_name 	= "/Model/state";
	char* state_out_name 	= "/Model/state_out";
	// For training
	char* train_prefix		= "OnlineTraining_";
	char* train_op  			= "/Model/train_op";
	char* mask_op					= "/Model/batched_output_word_masks";
	char* target_op 			= "/Model/batched_output_word_ids";
	char* train_input_op	= "/Model/batched_input_word_ids";
	char* lr_op						= "/Model/learning_rate";
}

#endif
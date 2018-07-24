#include <jni.h>

#include "deeptype.h"
#include "deeptype_v0.1_val.h"

#define JNI_METHOD(return_type, method_name) \
  JNIEXPORT return_type JNICALL              \
      Java_edu_pku_sei_deeptype_##method_name


DeepType* deeptype;
bool model_initialized = false;

extern "C" {

JNI_METHOD(void, init)
(JNIEnv* env, jobject thiz, jstring model_path) {
	if (deeptype != nullptr) {
		deeptype->clearState();
		delete deeptype;
	}
	LOGD("JNI_init");
	int wordIdShape[2] = {1, 1}; // batch_size * step_size
	const char *model = env->GetStringUTFChars(model_path, JNI_FALSE);
	deeptype = new DeepType(std::string(model), deeptype_val::infer_prefix, deeptype_val::prob_op, deeptype_val::infer_input_op,
		wordIdShape, 20, deeptype_val::train_prefix, deeptype_val::logits_op, deeptype_val::sm_input_op,
		deeptype_val::train_op, deeptype_val::mask_op, deeptype_val::train_input_op, deeptype_val::target_op, deeptype_val::lr_op);
	model_initialized = deeptype->initialized();
	if (model_initialized) {
		int inf_stateShape[4] = {2, 2, 1, 400};
		deeptype->addState(deeptype_val::state_in_name, deeptype_val::state_out_name, inf_stateShape, 4, false);
	}
}

JNI_METHOD(void, predict)
(JNIEnv* env, jobject thiz, jintArray jids, jint jidCount, jboolean jpartialSequence, jint jtopN, jboolean jsave_logits) {
	LOGD("JNI_predict");
	deeptype->predict(env->GetIntArrayElements(jids, 0), jidCount, jpartialSequence, jtopN, jsave_logits);
}

JNI_METHOD(jint, getWordId)
(JNIEnv* env, jobject thiz, jint index) {
	// LOGD("JNI_getWordId");
	return deeptype->getWordId(index);
}

JNI_METHOD(jfloat, getP)
(JNIEnv* env, jobject thiz, jint index) {
	// LOGD("JNI_getP");
	return deeptype->getP(index);
}

JNI_METHOD(jint, lastPredictStepCount)
(JNIEnv* env, jobject thiz, jint index) {
	// LOGD("JNI_lastPredictStepCount");
	return deeptype->lastPredictStepCount();
}

JNI_METHOD(jfloat, lastPredictTimeInMillis)
(JNIEnv* env, jobject thiz) {
	// LOGD("JNI_lastPredictTimeInMillis");
	return deeptype->lastPredictTimeInMillis();
}

JNI_METHOD(jfloat, initialized)
(JNIEnv* env, jobject thiz) {
	// LOGD("JNI_initialized");
	return model_initialized;
}

} // extern "C"
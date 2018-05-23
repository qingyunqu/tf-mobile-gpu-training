package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl",
     "tf_copts",
     "if_android",
     "if_not_mobile")

LINKER_SCRIPT = ":android_version_script.lds"

# bazel build //tensorflow/examples/rs_test:rs_benchmark --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a --copt=-D__ANDROID_TYPES_FULL__
cc_binary(
    name = "rs_benchmark",
    srcs = [
            "rs_benchmark.cc",
            "utils.h",
        ]
        + if_android([
            "abort_wrapper.cc",
            "//tensorflow/core:android_op_registrations_and_gradients",
			
          ]),
    copts = tf_copts() + [
            "-fexceptions",
            "-DSUPPORT_SELECTIVE_REGISTRATION",
          ],
    linkopts = ["-lm"] + if_android([
        "-pie",
        "-landroid",
        "-llog",
        "-ldl",
        "-lz",
        "-z defs",
        "-Wl,--version-script", LINKER_SCRIPT,
        "-Wl,--wrap=abort",
    ]),
    deps = [
        ] + if_android([
            "@gemmlowp//:gemmlowp",
            # "//tensorflow/core:android_tensorflow_lib_selective_registration",
            "//tensorflow/core/kernels:android_tensorflow_kernels_rs_training",
		# renderscript support
        	#"//tensorflow/contrib/android_renderscript_ops/rs:rs_lib_so",
        	# renderscript support
            LINKER_SCRIPT]),
)

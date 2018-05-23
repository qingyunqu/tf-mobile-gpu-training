Extension to TensorFlow to support on-mobile-gpu training.

## Build example demo
$ export ANDROID_HOME=/path/to/sdk
$ export ANDROID_NDK_HOME=/path/to/ndk
$ bazel build //tensorflow/examples/rs_test:rs_benchmark --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a --copt=-D__ANDROID_TYPES_FULL__

## Test the demo
$ adb push bazel-bin/tensorflow/examples/rs_test/rs_benchmark /data/local/tmp
in an adb shell, execute
$ cd /data/local/tmp
$ ./rs_benchmark --graph=graph/OnlineTraining_128_10_True_40_20000.pb --batch_size=128
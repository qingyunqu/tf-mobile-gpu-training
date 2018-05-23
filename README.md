Extension to TensorFlow to support on-mobile-gpu training.

## Install bazel
install the bazel

## Build example demo
$ bazel build //tensorflow/examples/rs_test:rs_benchmark --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a --copt=-D__ANDROID_TYPES_FULL__

## Test the demo
$ adb push tensorflow/contrib/android_renderscript_ops/rs/so/libRSSupport.so /data/local/tmp
$ adb push tensorflow/contrib/android_renderscript_ops/rs/so/libblasV8.so /data/local/tmp
$ adb push bazel-bin/tensorflow/examples/rs_test/rs_benchmark /data/local/tmp
$ adb shell
$ cd /data/local/tmp
$ export LD_LIBRARY_PATH=./
$ ./rs_benchmark --graph=graph/OnlineTraining_128_10_True_40_20000.pb --batch_size=128

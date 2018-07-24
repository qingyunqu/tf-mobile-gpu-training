# quantized graph loader
#
# Public targets:
#   ":quantized_graph_loader_src" quantized graph loader source
#   ":quantized_graph_proto_cc" quantized graph proto library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library")

load("//tensorflow:tensorflow.bzl",
     "tf_copts",
     "if_android",
     "if_not_mobile")

tf_proto_library(
    name = "quantized_graph_proto",
    srcs = [":quantized.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "quantized_graph_loader_src",
    visibility = ["//visibility:public"],
    srcs = [
        ":quantized_graph_loader.h",
        ":quantized_graph_loader.cc",
    ],
)

xinmei tensorflow native code
====
所有tensorflow相关native代码都放在这个文件夹中 （也可能包含少量java代码）  
bazel build xinmei/rnn_dict:deeptype_test --copt=-DSELECTIVE_REGISTRATION --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a

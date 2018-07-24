//
// by afpro.
//

#include <cstdio>
#include <cstring>
#include <ctime>
#include <android/log.h>

// extern jni wrapper method
extern "C" __attribute__ ((visibility("default"))) void init_anchor_fun() {}

// init fun type
typedef void (*init_fun_p)(int, char **, char **);

//  init func
static void init_log_method_pos(int argc, char **argv, char **env_p) {
  // get package name
  char pkg_name[256];
  {
    // read cmdline
    FILE *fp = fopen("/proc/self/cmdline", "r");
    size_t package_name_len = fread(pkg_name, 1, sizeof(pkg_name) - 1, fp);
    fclose(fp);

    // null terminate
    pkg_name[package_name_len] = 0;

    // for process sep ':'
    for (int i = static_cast<int>(package_name_len) - 1; i >= 0; i--) {
      if (pkg_name[i] == ':') {
        pkg_name[i] = 0;
        break;
      }
    }
  }

  // path
  char path[512] = "/data/data/";
  strcat(path, pkg_name);
  strcat(path, "/rnn_dict_ldd_pos.log");
  __android_log_print(ANDROID_LOG_DEBUG, "test", "log to %s", path);

  // write pos
  time_t now = time(nullptr);
  char now_s[64];
  ctime_r(&now, now_s);

  FILE *fp = fopen(path, "a");
  fprintf(fp, "%ld %p %s\n", static_cast<long>(now), &init_anchor_fun, now_s);
  fclose(fp);
  __android_log_print(ANDROID_LOG_DEBUG, "test", "at %ld %s", static_cast<long>(now), now_s);
}

//  init section data
__attribute__((section(".init_array"), used)) static init_fun_p init_funcs[1] = {init_log_method_pos};

//
// by afpro.
//

#include "utils.h"

extern "C" {
void __real_abort();
void __wrap_abort();
}

void __wrap_abort() {
  error("abort called!");
}

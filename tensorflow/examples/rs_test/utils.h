//
// by afpro.
//
#pragma once

#include <string>
#include <vector>
#include <stdexcept>

template<class...TArgs>
std::string format(const char *fmt, const TArgs &...args) {
  int len = snprintf(0, 0, fmt, args...);
  std::vector<char> buf;
  buf.resize(static_cast<size_t> (len + 1));
  snprintf(buf.data(), static_cast<size_t>(len + 1), fmt, args...);
  return std::string(buf.data());
}

#define error(...) throw std::runtime_error(format(__VA_ARGS__))

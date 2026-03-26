#pragma once

#include <fmt/core.h>
#include <fmt/format.h>
#include <utility>

namespace spdlog {

namespace fmt_lib = fmt;

template<typename... Args>
inline void trace(fmt::format_string<Args...>, Args&&...) {}

template<typename... Args>
inline void debug(fmt::format_string<Args...>, Args&&...) {}

template<typename... Args>
inline void info(fmt::format_string<Args...>, Args&&...) {}

template<typename... Args>
inline void warn(fmt::format_string<Args...>, Args&&...) {}

template<typename... Args>
inline void error(fmt::format_string<Args...>, Args&&...) {}

} // namespace spdlog

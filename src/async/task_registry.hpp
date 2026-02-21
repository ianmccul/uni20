#pragma once

#include "config.hpp"

#if UNI20_DEBUG_ASYNC_TASKS
#include "task_registry_debug.hpp"
#else
#include "task_registry_dummy.hpp"
#endif

#pragma once

/**
 * \file task_registry.hpp
 * \brief Selects the task-registry implementation for the current build mode.
 */

#include <uni20/config.hpp>

#if UNI20_DEBUG_ASYNC_TASKS
#include "task_registry_debug.hpp"
#else
#include "task_registry_dummy.hpp"
#endif

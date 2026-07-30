#pragma once

/**
 * \file compiler_attributes.hpp
 * \ingroup core
 * \brief Project compiler annotations shared by host and device code.
 */

#if defined(__CUDACC__)
#define UNI20_HOST_DEVICE __host__ __device__
#else
#define UNI20_HOST_DEVICE
#endif

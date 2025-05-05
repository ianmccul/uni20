#pragma once

#include "common/types.hpp"
#include "config.hpp"

// intended as a hook to add tracing/logging
#define UNI20_API_CALL(...)

#define BLASCHAR s
#define BLASTYPE float32
#define BLASCOMPLEX 0
#include "uni20/blas/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASCOMPLEX

#define BLASCHAR d
#define BLASTYPE float64
#define BLASCOMPLEX 0
#include "uni20/blas/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASCOMPLEX

#define BLASCHAR c
#define BLASTYPE complex64
#define BLASCOMPLEX 1
#include "uni20/blas/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASCOMPLEX

#define BLASCHAR z
#define BLASTYPE complex128
#define BLASCOMPLEX 1
#include "uni20/blas/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASCOMPLEX

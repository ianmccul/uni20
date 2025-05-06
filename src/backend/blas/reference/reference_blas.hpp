#pragma once

#include "backend/backend.hpp"
#include "common/types.hpp"
#include "config.hpp"

#define BLASCHAR s
#define BLASTYPE float32
#define BLASCOMPLEX 0
#include "detail/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASCOMPLEX

#define BLASCHAR d
#define BLASTYPE float64
#define BLASCOMPLEX 0
#include "detail/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASCOMPLEX

#define BLASCHAR c
#define BLASTYPE complex64
#define BLASREALTYPE float32
#define BLASCOMPLEX 1
#include "detail/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASREALTYPE
#undef BLASCOMPLEX

#define BLASCHAR z
#define BLASTYPE complex128
#define BLASREALTYPE float64
#define BLASCOMPLEX 1
#include "detail/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASREALTYPE
#undef BLASCOMPLEX

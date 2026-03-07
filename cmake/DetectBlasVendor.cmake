#[[
Detect and normalize the BLAS vendor after `find_package(BLAS)`.

Output cache variables:
  - UNI20_DETECTED_BLAS_VENDOR
  - UNI20_DETECTED_BLAS_VENDOR_MACRO
  - UNI20_DETECTED_BLAS_LIBRARIES
]]

function(_uni20_map_blas_vendor_name input_value output_var)
  if(input_value STREQUAL "")
    set(${output_var} "" PARENT_SCOPE)
    return()
  endif()

  # Prefer file basename over full path to avoid false positives from directory names.
  get_filename_component(_candidate_file "${input_value}" NAME)
  string(TOLOWER "${_candidate_file}" _candidate_file_lower)
  string(TOLOWER "${input_value}" _candidate_path_lower)

  if(_candidate_file_lower MATCHES "mkl")
    set(${output_var} "MKL" PARENT_SCOPE)
    return()
  endif()
  if(_candidate_file_lower MATCHES "openblas")
    set(${output_var} "OpenBLAS" PARENT_SCOPE)
    return()
  endif()
  if(_candidate_file_lower MATCHES "flexiblas")
    set(${output_var} "FlexiBLAS" PARENT_SCOPE)
    return()
  endif()
  if(_candidate_file_lower MATCHES "blis")
    set(${output_var} "BLIS" PARENT_SCOPE)
    return()
  endif()
  if(_candidate_file_lower MATCHES "atlas")
    set(${output_var} "ATLAS" PARENT_SCOPE)
    return()
  endif()
  if(_candidate_file_lower MATCHES "goto")
    set(${output_var} "GotoBLAS" PARENT_SCOPE)
    return()
  endif()

  if(_candidate_file_lower MATCHES "accelerate|veclib")
    set(${output_var} "Accelerate" PARENT_SCOPE)
    return()
  endif()

  # Fallback for non-file tokens (e.g., BLA_VENDOR names or unusual linker tokens).
  if(_candidate_path_lower MATCHES "mkl|intel")
    set(${output_var} "MKL" PARENT_SCOPE)
  elseif(_candidate_path_lower MATCHES "openblas")
    set(${output_var} "OpenBLAS" PARENT_SCOPE)
  elseif(_candidate_path_lower MATCHES "accelerate|veclib")
    set(${output_var} "Accelerate" PARENT_SCOPE)
  elseif(_candidate_path_lower MATCHES "flexiblas")
    set(${output_var} "FlexiBLAS" PARENT_SCOPE)
  elseif(_candidate_path_lower MATCHES "blis")
    set(${output_var} "BLIS" PARENT_SCOPE)
  elseif(_candidate_path_lower MATCHES "atlas")
    set(${output_var} "ATLAS" PARENT_SCOPE)
  elseif(_candidate_path_lower MATCHES "goto")
    set(${output_var} "GotoBLAS" PARENT_SCOPE)
  else()
    set(${output_var} "" PARENT_SCOPE)
  endif()
endfunction()

function(_uni20_vendor_macro_name vendor_name output_var)
  string(TOUPPER "${vendor_name}" _vendor_upper)
  string(REGEX REPLACE "[^A-Z0-9]+" "_" _vendor_upper "${_vendor_upper}")
  string(REGEX REPLACE "_+" "_" _vendor_upper "${_vendor_upper}")
  string(REGEX REPLACE "^_" "" _vendor_upper "${_vendor_upper}")
  string(REGEX REPLACE "_$" "" _vendor_upper "${_vendor_upper}")

  if(_vendor_upper STREQUAL "")
    set(_vendor_upper "GENERIC")
  endif()

  set(${output_var} "UNI20_BLAS_VENDOR_${_vendor_upper}" PARENT_SCOPE)
endfunction()

function(detect_blas_vendor)
  unset(UNI20_REQUESTED_BLAS_VENDOR CACHE)
  unset(UNI20_DETECTED_BLAS_REQUESTED_VENDOR CACHE)

  if(NOT BLAS_FOUND)
    set(UNI20_DETECTED_BLAS_VENDOR "None" CACHE INTERNAL "Detected BLAS vendor" FORCE)
    _uni20_vendor_macro_name("None" _none_vendor_macro)
    set(UNI20_DETECTED_BLAS_VENDOR_MACRO "${_none_vendor_macro}" CACHE INTERNAL "Vendor-specific macro for BLAS" FORCE)
    set(UNI20_DETECTED_BLAS_LIBRARIES "" CACHE INTERNAL "Detected BLAS libraries" FORCE)
    message(STATUS "BLAS not found; UNI20_DETECTED_BLAS_VENDOR set to None")
    return()
  endif()

  # Prefer detection from the actual library list selected by FindBLAS.
  set(_detected_vendor "")
  foreach(_blas_lib IN LISTS BLAS_LIBRARIES)
    _uni20_map_blas_vendor_name("${_blas_lib}" _mapped_vendor)
    if(NOT _mapped_vendor STREQUAL "")
      set(_detected_vendor "${_mapped_vendor}")
      break()
    endif()
  endforeach()

  # Fall back to user-requested vendor when BLAS_LIBRARIES has no recognizable hints.
  if(_detected_vendor STREQUAL "")
    if(DEFINED BLA_VENDOR AND NOT BLA_VENDOR STREQUAL "" AND NOT BLA_VENDOR STREQUAL "All")
      _uni20_map_blas_vendor_name("${BLA_VENDOR}" _mapped_requested_vendor)
      if(_mapped_requested_vendor STREQUAL "")
        set(_detected_vendor "${BLA_VENDOR}")
      else()
        set(_detected_vendor "${_mapped_requested_vendor}")
      endif()
    else()
      set(_detected_vendor "Generic")
    endif()
  endif()

  _uni20_vendor_macro_name("${_detected_vendor}" _vendor_macro)

  set(UNI20_DETECTED_BLAS_VENDOR "${_detected_vendor}" CACHE INTERNAL "Detected BLAS vendor" FORCE)
  set(UNI20_DETECTED_BLAS_VENDOR_MACRO "${_vendor_macro}" CACHE INTERNAL "Vendor-specific macro for BLAS" FORCE)
  set(UNI20_DETECTED_BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE INTERNAL "Detected BLAS libraries" FORCE)

  message(STATUS "Detected UNI20_DETECTED_BLAS_VENDOR: ${UNI20_DETECTED_BLAS_VENDOR}")
endfunction()

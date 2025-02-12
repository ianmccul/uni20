# cmake/DetectBlasVendor.cmake
#
# This module defines a function 'detect_blas_vendor()' that sets the variable
# UNI20_BLAS_VENDOR based on the BLAS installation. It also sets
# UNI20_BLAS_VENDOR_xxxx symbol, where xxxx is the UPPERCASE vendor name.
#
# Use AFTER calling find_package(BLAS), to set UNI20_BLAS_VENDOR according
# to the found BLAS library.
#
# It first checks if BLAS is found (BLAS_FOUND). If yes, then:
#  - If BLAS_VENDOR is defined and non-empty, it uses that.
#  - Otherwise, it scans the cache for variables matching BLAS_*_LIBRARY,
#    and tries to detect common vendors (e.g. OpenBLAS, MKL, or GotoBLAS).
#  - If none of these match, it defaults to "Generic".
#
# If BLAS is not found, UNI20_BLAS_VENDOR is set to "None".
#
# This module depends on variables set internally by the FindBLAS module.
# It might not work with future changes to that module!

function(detect_blas_vendor)
  if(BLAS_FOUND)
    if(DEFINED BLAS_VENDOR AND NOT BLAS_VENDOR STREQUAL "")
      set(UNI20_BLAS_VENDOR "${BLAS_VENDOR}" CACHE INTERNAL "Detected BLAS vendor from FindBLAS" FORCE)
    else()
      # Auto-detect vendor by scanning for variables of the form BLAS_XXXX_LIBRARY.
      get_cmake_property(cacheVars CACHE_VARIABLES)
      set(_detected_vendor "")
      foreach(var ${cacheVars})
        if(var MATCHES "^BLAS_([A-Za-z0-9_]+)_LIBRARY$")
          if(NOT "${${var}}" MATCHES "NOTFOUND")
            # Extract the vendor hint from the variable name.
            string(REGEX REPLACE "^BLAS_([A-Za-z0-9_]+)_LIBRARY$" "\\1" vendor_hint ${var})
            message(STATUS "Found BLAS library variable: ${var} = ${${var}} (hint: ${vendor_hint})")
            # Convert the vendor hint to lowercase for case-insensitive matching.
            string(TOLOWER "${vendor_hint}" vendor_hint_lower)
            if(vendor_hint_lower MATCHES "openblas")
              set(_detected_vendor "OpenBLAS")
              break()
            elseif(vendor_hint_lower MATCHES "mkl")
              set(_detected_vendor "MKL")
              break()
            elseif(vendor_hint_lower MATCHES "goto")
              set(_detected_vendor "GotoBLAS")
              break()
            endif()
          endif()
        endif()
      endforeach()
      if(_detected_vendor STREQUAL "")
        set(_detected_vendor "Generic")
      endif()
      set(UNI20_BLAS_VENDOR "${_detected_vendor}" CACHE INTERNAL "Detected BLAS vendor (auto-detected)" FORCE)
    endif()
  else()
    set(UNI20_BLAS_VENDOR "None" CACHE INTERNAL "BLAS not found" FORCE)
  endif()

  # Convert the detected vendor to uppercase.
  string(TOUPPER "${UNI20_BLAS_VENDOR}" UNI20_BLAS_VENDOR_UPPER)
  # Build the vendor macro name, e.g., UNI20_BLAS_VENDOR_MKL.
  set(UNI20_BLAS_VENDOR_MACRO "UNI20_BLAS_VENDOR_${UNI20_BLAS_VENDOR_UPPER}" CACHE INTERNAL "Vendor-specific macro for BLAS" FORCE)

  message(STATUS "Detected UNI20_BLAS_VENDOR: ${UNI20_BLAS_VENDOR}")
endfunction()

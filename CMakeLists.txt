cmake_minimum_required(VERSION 3.16)

project(vine
  VERSION 0.2.0
  LANGUAGES C)

# Use C99 
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---- PHAST dependency --------------------------------------------------
# Packagers / users can do: -DPHAST_ROOT=/path/to/phast
set(PHAST_ROOT "" CACHE PATH "Root of PHAST installation (with include/ and lib/)")

if (NOT PHAST_ROOT)
  message(FATAL_ERROR "PHAST_ROOT is not set. Configure with -DPHAST_ROOT=/path/to/phast")
endif()

set(PHAST_INCLUDE_DIR "${PHAST_ROOT}/include" CACHE PATH "PHAST include directory")
set(PHAST_LIBRARY_DIR "${PHAST_ROOT}/lib")

set(PHAST_LIBRARY "${PHAST_LIBRARY_DIR}/libphast.a" CACHE FILEPATH "PHAST static library")

if (NOT EXISTS "${PHAST_INCLUDE_DIR}/phast/misc.h")
  message(FATAL_ERROR "Could not find phast headers in ${PHAST_INCLUDE_DIR}")
endif()

if (NOT EXISTS "${PHAST_LIBRARY}")
  message(FATAL_ERROR "Could not find libphast.a at ${PHAST_LIBRARY}")
endif()

# Make PHAST info visible to subdirs
add_library(phast STATIC IMPORTED)
set_target_properties(phast PROPERTIES
  IMPORTED_LOCATION "${PHAST_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${PHAST_INCLUDE_DIR}"
)

# LAPACK handling
if(APPLE)
    add_compile_definitions(VECLIB ACCELERATE_NEW_LAPACK)
#   - VECLIB tells external_libs.h to use Accelerate
#   - ACCELERATE_NEW_LAPACK selects the new (non-CLAPACK) interface
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    add_compile_definitions(PHAST_LAPACK_GENERIC)
endif()

# ---- PCRE dependency via PHAST -----------------------------------------
find_library(PCRE_LIBRARY NAMES pcre)

if (NOT PCRE_LIBRARY)
  message(FATAL_ERROR "Could not find libpcre; install it (e.g. Homebrew 'pcre').")
endif()

# ---- Project layout ----------------------------------------------------

# public headers 
set(VINE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

add_subdirectory(src/lib)
add_subdirectory(src/progs)

# ---- Installation (for Homebrew, etc.) ---------------------------------
# Install public headers
install(DIRECTORY "${VINE_INCLUDE_DIR}/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")


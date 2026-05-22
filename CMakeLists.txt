cmake_minimum_required(VERSION 3.16)

project(vine
  VERSION 0.2.0
  LANGUAGES C)

# Use C99 
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# allows use of upper case PHAST_ROOT
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# ---- PHAST dependency --------------------------------------------------
# Users/packagers provide -DPHAST_ROOT=/path/to/prefix (the install prefix)
set(PHAST_ROOT "" CACHE PATH "PHAST install prefix (contains lib/cmake/phast)")

if (NOT PHAST_ROOT)
  message(FATAL_ERROR "PHAST_ROOT is not set. Configure with -DPHAST_ROOT=/path/to/phast-prefix")
endif()

# Tell CMake where to find phastConfig.cmake
list(PREPEND CMAKE_PREFIX_PATH "${PHAST_ROOT}")

find_package(phast CONFIG REQUIRED)

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

# ---- Exported CMake package (so downstream projects can find_package(vine)) ---
include(CMakePackageConfigHelpers)

# Record whether vine_core ended up linking OpenMP, for the generated config.
# OpenMP is detected inside src/lib (a subdirectory), so its result variables
# are not visible in this scope; re-run the (cached, idempotent) detection here
# to learn whether the exported vine::vine_core target references OpenMP.
find_package(OpenMP)
if(OpenMP_C_FOUND)
  set(VINE_BUILT_WITH_OPENMP TRUE)
else()
  set(VINE_BUILT_WITH_OPENMP FALSE)
endif()

install(EXPORT vineTargets
        FILE vineTargets.cmake
        NAMESPACE vine::
        DESTINATION lib/cmake/vine)

configure_package_config_file(
  "${CMAKE_SOURCE_DIR}/cmake/vineConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/vineConfig.cmake"
  INSTALL_DESTINATION lib/cmake/vine)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/vineConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/vineConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/vineConfigVersion.cmake"
  DESTINATION lib/cmake/vine)


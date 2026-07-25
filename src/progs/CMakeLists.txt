# one executable per program, linked against vine + PHAST

add_executable(vine vine.c)
add_executable(evalTrees evalTrees.c)
add_executable(compareTrees compareTrees.c)
add_executable(crisprLnl crisprLnl.c)

# ---- Optional codesigning (macOS) ----
# Default is ON on macOS for normal developer builds, but OFF under conda-build.
set(_vine_codesign_default OFF)
if(APPLE)
  set(_vine_codesign_default ON)
  if(DEFINED ENV{CONDA_BUILD})
    set(_vine_codesign_default OFF)
  endif()
endif()

option(VINE_CODESIGN "Ad-hoc codesign executables on macOS after build" ${_vine_codesign_default})
message(STATUS "VINE_CODESIGN=${VINE_CODESIGN} (CONDA_BUILD=$ENV{CONDA_BUILD})")

# ---- Optional OpenBLAS thread control ----
# evalTrees/compareTrees can cap OpenBLAS's internal thread pool (to avoid
# oversubscription with vine's OpenMP threads).  This is OpenBLAS-specific;
# other BLAS backends (Apple Accelerate, MKL) lack openblas_set_num_threads.
# It cannot be auto-detected reliably: vine pulls BLAS in transitively via
# PHAST, so probing find_package(BLAS) can report a symbol that the actual
# (phast-provided) BLAS does not export, breaking the link.  Off by default;
# packagers building against OpenBLAS (e.g. the bioconda recipe) opt in with
# -DVINE_ENABLE_OPENBLAS_THREADS=ON.
option(VINE_ENABLE_OPENBLAS_THREADS
       "Cap OpenBLAS thread pool in evalTrees/compareTrees (OpenBLAS-linked builds only)" OFF)

foreach(tgt vine evalTrees compareTrees crisprLnl)
    target_link_libraries(${tgt} PRIVATE vine_core)
    if(APPLE AND VINE_CODESIGN)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND /usr/bin/codesign --force --sign - "$<TARGET_FILE:${tgt}>"
            COMMENT "Ad-hoc code signing ${tgt}"
            VERBATIM
        )
    endif()
endforeach()

if(VINE_ENABLE_OPENBLAS_THREADS)
  target_compile_definitions(evalTrees PRIVATE VINE_HAVE_OPENBLAS)
  target_compile_definitions(compareTrees PRIVATE VINE_HAVE_OPENBLAS)
endif()

# Read version file and strip whitespace
file(READ "${CMAKE_SOURCE_DIR}/version" VINE_VERSION_RAW)
string(STRIP "${VINE_VERSION_RAW}" VINE_VERSION)
message(STATUS "VINE_VERSION='${VINE_VERSION}'")

# add to vine compile target
target_compile_definitions(vine PRIVATE VINE_VERSION="${VINE_VERSION}")


# Install all progs into bin/
install(TARGETS
  vine
  evalTrees
  compareTrees
  crisprLnl
  RUNTIME DESTINATION bin
)

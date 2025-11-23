# src/lib/CMakeLists.txt

# List your library sources here (mirror what your Makefile uses)
add_library(vine_core STATIC
    adam_scheduler.c
    multiDAG.c
    sparse_matrix.c
    bitset.c
    mvn.c
    sparse_vector.c
    crispr.c
    nj.c
    tree_prior.c
    heap.c
    planar_flow.c
    upgma.c
    migration.c
    radial_flow.c
    var_resampling.c
    multi_mvn.c
    rf.c
)

set_target_properties(vine_core PROPERTIES
    OUTPUT_NAME vine
)

target_compile_options(vine_core PUBLIC
    -g -fno-inline -Wall -fno-strict-aliasing
)

# Make sure we use the same LAPACK configuration as PHAST on macOS:
#   - VECLIB tells external_libs.h to use Accelerate
#   - ACCELERATE_NEW_LAPACK selects the new (non-CLAPACK) interface
if(APPLE)
    target_compile_definitions(vine_core PUBLIC VECLIB ACCELERATE_NEW_LAPACK)
endif()

target_include_directories(vine_core
    PUBLIC
        "${VINE_INCLUDE_DIR}"     # vine/include
        "${PHAST_INCLUDE_DIR}"    # phast/include
        "${PHAST_ROOT}/src/lib/pcre"   # for pcre.h vendored in PHAST
)

target_link_libraries(vine_core
    PUBLIC
        phast        # libphast.a in ${PHAST_LIBRARY_DIR}
)

# PHAST was built against Accelerate/vecLib; add the same framework here
if(APPLE)
    target_link_libraries(vine_core PUBLIC "-framework Accelerate")
endif()

# Install the library
install(TARGETS vine_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

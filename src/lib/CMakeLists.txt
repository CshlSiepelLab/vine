# src/lib/CMakeLists.txt

# List your library sources here (mirror what your Makefile uses)
add_library(vine_core STATIC
    adam_scheduler.c
    backprop.c
    bitset.c
    covariance.c
    crispr.c
    geometry.c
    gradients.c
    heap.c
    hutchinson.c
    likelihoods.c
    migration.c
    multi_mvn.c
    multiDAG.c
    mvn.c
    nj.c
    nuisance.c
    planar_flow.c
    radial_flow.c
    rf.c
    sparse_matrix.c
    sparse_vector.c
    taylor.c
    tree_prior.c
    upgma.c
    var_resampling.c
    variational.c
)

set_target_properties(vine_core PROPERTIES
    OUTPUT_NAME vine
)

target_compile_options(vine_core PUBLIC
    -g -fno-inline -Wall -fno-strict-aliasing
)

target_include_directories(vine_core
    PUBLIC
        "${VINE_INCLUDE_DIR}"     # vine/include
        "${PHAST_INCLUDE_DIR}"    # phast/include
        "${PHAST_ROOT}/src/lib/pcre"   # for pcre.h vendored in PHAST
)

target_link_libraries(vine_core
    PUBLIC
        phast        # libphast.a in ${PHAST_LIBRARY_DIR}
        ${PCRE_LIBRARY}
)

# PHAST was built against Accelerate/vecLib; add the same framework here
if(APPLE)
    target_link_libraries(vine_core PUBLIC "-framework Accelerate")
else()
    target_link_libraries(vine_core PUBLIC LAPACK::LAPACK BLAS::BLAS)
endif()

# Install the library
install(TARGETS vine_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

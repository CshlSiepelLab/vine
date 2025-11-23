# src/progs/CMakeLists.txt

# one executable per program, linked against vine + PHAST

add_executable(vine vine.c)
add_executable(evalTrees evalTrees.c)

foreach(tgt vine evalTrees)
    target_link_libraries(${tgt} PRIVATE vine_core)
endforeach()

# Install all progs into bin/
install(TARGETS
  vine
  evalTrees
  RUNTIME DESTINATION bin
)

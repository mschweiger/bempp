include_directories ("${PROJECT_SOURCE_DIR}/lib")
add_executable(traversal traversal.cpp)
add_executable(traversal_benchmark traversal_benchmark.cpp)
add_executable(gmsh_import gmsh_import.cpp)
add_executable(vtk_export vtk_export.cpp)
target_link_libraries(traversal bempp)
target_link_libraries(traversal_benchmark bempp)
target_link_libraries(gmsh_import bempp)
target_link_libraries(vtk_export bempp)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/head.gmsh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

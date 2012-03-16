# Options (can be modified by user)
option(WITH_TESTS "Compile unit tests (can be run with 'make test')" ON)
option(WITH_OPENCL "Add OpenCL support for Fiber module" ON)

set(DUNE_PATH ${CMAKE_SOURCE_DIR}/contrib/dune CACHE PATH "Path to Dune")

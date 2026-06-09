# External KTDP dialect definitions from ktir-mlir-frontend.
# Only build the dialect library (include + lib), skip tools/test/python.
set(KTIR_MLIR_FRONTEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ktir-mlir-frontend")
if(NOT EXISTS "${KTIR_MLIR_FRONTEND_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "ktir-mlir-frontend not found at ${KTIR_MLIR_FRONTEND_DIR}. "
    "Run: git submodule update --init")
endif()
include_directories(${KTIR_MLIR_FRONTEND_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/ktir-mlir-frontend/include)
# Override add_mlir_doc to a no-op to skip doc generation when building as submodule
function(add_mlir_doc)
endfunction()
add_subdirectory(${KTIR_MLIR_FRONTEND_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/ktir-mlir-frontend/include)
add_subdirectory(${KTIR_MLIR_FRONTEND_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR}/ktir-mlir-frontend/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonSpyre
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_spyre.cc
    LINK_LIBS KTDPTransforms MLIRKtdp)
  target_link_libraries(TritonSpyre PRIVATE Python3::Module pybind11::headers)
endif()

set(SPYRE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
include_directories(${SPYRE_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../../..)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -name=KTDP)
add_public_tablegen_target(KTDPTransformsIncGen)

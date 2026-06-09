set(TRITON_LLVMIR_LINK_LIBS
  ${CMAKE_DL_LIBS}
  PUBLIC
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRIndexToLLVM
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTargetLLVMIRExport
)

if(NOT TRITON_BUILD_TTIR_ONLY)
  list(APPEND TRITON_LLVMIR_LINK_LIBS
    MLIRNVVMToLLVM
    MLIRNVVMToLLVMIRTranslation
    MLIRROCDLToLLVMIRTranslation
    TritonGPUToLLVM
  )
endif()

add_triton_library(TritonLLVMIR
        LLVMDIScope.cpp
        LLVMDILocalVariable.cpp
        LLVMIRBreakPhiStruct.cpp
        LLVMDIUtils.cpp

        DEPENDS
        LLVMIRIncGen

        LINK_LIBS ${TRITON_LLVMIR_LINK_LIBS}
        )

set_source_files_properties(
        LLVMIRTranslation.cpp
        PROPERTIES
        COMPILE_FLAGS "-D__BUILD_DIR__=\\\"${CMAKE_BINARY_DIR}\\\"")

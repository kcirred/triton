if(NOT TRITON_BUILD_TTIR_ONLY)
  add_subdirectory(TritonGPUToLLVM)
  add_subdirectory(TritonToTritonGPU)
endif()

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class SpyreDriver(DriverBase):
    """Spyre device driver stub.

    Spyre is a remote accelerator — there is no local device to query.
    This driver satisfies Triton's DriverBase interface for compilation.
    """

    @classmethod
    def is_active(cls) -> bool:
        return True

    def map_python_to_cpp_type(self, ty: str) -> str:
        mapping = {
            "i32": "int32_t",
            "f16": "half",
            "fp8": "fp8",
        }
        return mapping.get(ty, ty)

    def get_current_target(self) -> GPUTarget:
        return GPUTarget(backend="spyre", arch=1, warp_size=1)

    def get_active_torch_device(self):
        return None

    def get_benchmarker(self):
        raise NotImplementedError("Spyre does not support local benchmarking")

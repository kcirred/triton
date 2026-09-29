"""SIGNATURE + VARIANTS + reference oracle + input generators for ``where``.

Three variants, splitting ``tl.where(x > y, p, q)`` into the parts that can fail
independently:

- ``compare`` -- the comparison alone, storing its 1.0/0.0 mask.
  ``compare_device`` is its fp32 row on the device tier.
- ``select``  -- the select alone, over a mask the host wrote.
- ``spilled`` -- both in one kernel, with the mask spilled to its own buffer.

``compare``, ``select`` and ``spilled`` sweep fp16 and fp32. ``select``,
``spilled`` and ``compare_device`` are ``compiles_to_binary``; ``compare`` at
fp16 is not, and its entry says why.

The branch values are kept separate from the condition, which is what makes a
wrong branch visible. Written ``where(x > y, x, y)`` the kernel computes a
maximum, so an implementation that ignored the mask and returned ``max(x, y)``
would pass. Separate ``p``/``q`` buffers, held far apart (about +100 and -100),
make a wrong branch a wrong value. Every variant compares exactly
(``rtol``/``atol`` 0): a select copies its branch values without arithmetic,
and +/-100 survive the device's fp16 storage format unchanged (checked on the
device), so any difference at all is a wrong answer.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import numpy as np
from dataclasses import dataclass

import conftest
from . import kernel
from utils import sticksize, DTYPE_MAP


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input makers
# ---------------------------------------------------------------------------

# Branch values, held far apart so a wrong branch cannot pass as rounding.
_P_BASE = 100.0
_Q_BASE = -100.0


def _condition_inputs(n, np_dtype):
    """``x``/``y`` for the comparison: a ramp over [-1, 1] against its mirror.

    Deterministic, so a failure reproduces without a seed. The two cross once,
    at the midpoint, so both branches are taken over long runs. The ramp steps
    by about 1.6e-2 at n=128, far above fp16 eps, so no lane is a near-tie that
    a reduced-precision device comparison could break the other way.
    """
    ramp = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    return ramp.astype(np_dtype), (-ramp).astype(np_dtype)


def _branch_inputs(n, np_dtype) -> dict:
    return {"p_ptr": np.full(n, _P_BASE, dtype=np_dtype),
            "q_ptr": np.full(n, _Q_BASE, dtype=np_dtype)}


def make_compare_inputs(n_elements, DTYPE="fp16", **_unused) -> dict:
    np_dtype = DTYPE_MAP[DTYPE]
    x, y = _condition_inputs(n_elements, np_dtype)
    return {"x_ptr": x, "y_ptr": y,
            "mask_ptr": np.zeros(n_elements, dtype=np_dtype)}


def make_select_inputs(n_elements, DTYPE="fp16", **_unused) -> dict:
    """The mask is written by the host as 1.0/0.0, isolating the select.

    It alternates lane by lane rather than running in blocks, so a select that
    read the wrong lane -- a layout error rather than a logic one -- shows.
    """
    np_dtype = DTYPE_MAP[DTYPE]
    mask = np.zeros(n_elements, dtype=np_dtype)
    mask[::2] = 1.0
    return {"mask_ptr": mask, **_branch_inputs(n_elements, np_dtype),
            "output_ptr": np.zeros(n_elements, dtype=np_dtype)}


def make_where_inputs(n_elements, DTYPE="fp16", **_unused) -> dict:
    """``mask_ptr`` is scratch the kernel spills through; only ``output_ptr``
    is checked."""
    np_dtype = DTYPE_MAP[DTYPE]
    x, y = _condition_inputs(n_elements, np_dtype)
    return {"x_ptr": x, "y_ptr": y, **_branch_inputs(n_elements, np_dtype),
            "mask_ptr": np.zeros(n_elements, dtype=np_dtype),
            "output_ptr": np.zeros(n_elements, dtype=np_dtype)}


def run_compare(inputs) -> np.ndarray:
    return (inputs["x_ptr"] > inputs["y_ptr"]).astype(inputs["x_ptr"].dtype)


def run_select(inputs) -> np.ndarray:
    return np.where(inputs["mask_ptr"] != 0, inputs["p_ptr"], inputs["q_ptr"])


def run_where(inputs) -> np.ndarray:
    """Recomputes the comparison rather than reading ``mask_ptr``, so a wrong
    mask on the device makes this disagree instead of being inherited."""
    return np.where(inputs["x_ptr"] > inputs["y_ptr"],
                    inputs["p_ptr"], inputs["q_ptr"])


# ---------------------------------------------------------------------------
# Factory -- Where(VariantFactory)
#
# Only the SIGNATURE varies per combination: every pointer takes the swept
# DTYPE. The input makers read DTYPE from ``params`` themselves.
# ---------------------------------------------------------------------------

_PTR = {"fp16": "*fp16", "fp32": "*fp32"}

_SHAPE_ARGS = {
    "n_elements": "i32",
    "BLOCK_SIZE": "i32",
    "LAYOUT":     "constexpr",
}


@dataclass(frozen=True)
class Where(conftest.VariantFactory):
    """Factory for the where variants; ``ptrs`` names the kernel's pointer
    arguments in declaration order."""
    ptrs: tuple

    def signature(self, DTYPE, **_):
        return {**{n: _PTR[DTYPE] for n in self.ptrs}, **_SHAPE_ARGS}


# ---------------------------------------------------------------------------
# SIGNATURE -- dtype per @triton.jit arg. The module-level one is the default
# the registry needs; every variant's factory overrides it per DTYPE.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "x_ptr":      "*fp16",
    "y_ptr":      "*fp16",
    "mask_ptr":   "*fp16",
    **_SHAPE_ARGS,
}


def _stick_1d(dtype: str) -> tuple:
    """Labelled 1D stick layout at *dtype*, ``[n]`` -> ``[ceil(n/S), S]``."""
    stick = sticksize({"p": f"*{dtype}"}, "p")
    return ("stick", ((0, "floordiv", stick), (0, "mod", stick)))


# ---------------------------------------------------------------------------
# VARIANTS
#
# All three are loop-free and one tile total, which the device tier requires:
# dbo-opt rejects the scf.for a program-id distribution loop outlines.
#
# n_elements=128 is two sticks at fp16 (64 per stick) and four at fp32 (32).
# ---------------------------------------------------------------------------

_TAGS = [
    "descriptor-load-static", "descriptor-store-static",
    "simplified:no-loop", "spyre-tensor-layout", "where",
]

_PARAMS = {
    # One row per dtype, because the layout's stick width follows from it.
    ("DTYPE", "LAYOUT"): [
        ("fp16", _stick_1d("fp16")),
        ("fp32", _stick_1d("fp32")),
    ],
    "n_elements": [128],
    "BLOCK_SIZE": [128],
}

VARIANTS = {
    # -----------------------------------------------------------------------
    # The comparison alone, at the compute tier for both dtypes. Its fp32 row
    # also runs on the device, as ``compare_device``; the fp16 row cannot.
    # The fp16 kernel compiles and launches, and its mask is correct as a mask
    # -- every lane where the predicate holds is non-zero, which is all
    # spyreop.select asks of a condition. What fails is reading it back on the
    # host: the device stores fp16 as DF16, whose 1.0 is the bit pattern
    # 0x3800, and IEEE fp16 reads that as 0.5. Whether the store path lacks a
    # conversion or a host-visible fp16 mask is expected to be DF16-encoded is
    # not settled. ``spilled`` covers the fp16 comparison on the device without
    # handing the mask to the host.
    # -----------------------------------------------------------------------
    "compare": {
        "base": None,
        "tags": _TAGS,
        "summary": (
            "1D (x > y) over a single stick-tiled tile, no distribution loop. "
            "Sweeps fp16/fp32."
        ),
        "kernel_fn":    kernel.compare_1d_device,
        "factory":      Where(ptrs=("x_ptr", "y_ptr", "mask_ptr")),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT"],
        "params":       _PARAMS,
        "grid":         [1],
        "inputs":       make_compare_inputs,
        "reference":    run_compare,
        "output_key":   "mask_ptr",
        "rtol":         0.0,
        "atol":         0.0,
    },

    # -----------------------------------------------------------------------
    # ``compare``'s fp32 row on the device. fp32 has no DF16 re-encoding, so
    # the host reads the mask back as the 1.0/0.0 the oracle expects.
    # -----------------------------------------------------------------------
    "compare_device": {
        "base": "compare",
        "summary": (
            "1D fp32 (x > y) over a single stick-tiled tile on the device, "
            "no distribution loop."
        ),
        "params": {
            # The parent's group with its fp32 row alone. Redeclared in full
            # because ``params`` merges wholesale.
            ("DTYPE", "LAYOUT"): [
                ("fp32", _stick_1d("fp32")),
            ],
            "n_elements": [128],
            "BLOCK_SIZE": [128],
        },
        "compiles_to_binary": True,
    },

    # -----------------------------------------------------------------------
    # The select alone, over a host-written mask: no comparison and no mask
    # round trip precede it, so a failure here is spyreop.select's.
    # -----------------------------------------------------------------------
    "select": {
        "base": None,
        "tags": _TAGS,
        "summary": (
            "1D where(mask != 0, p, q) over a host-written mask, single tile, "
            "no distribution loop. Sweeps fp16/fp32."
        ),
        "kernel_fn":    kernel.select_1d_device,
        "factory":      Where(ptrs=("mask_ptr", "p_ptr", "q_ptr", "output_ptr")),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT"],
        "params":       _PARAMS,
        "grid":         [1],
        "compiles_to_binary": True,
        "inputs":       make_select_inputs,
        "reference":    run_select,
        "output_key":   "output_ptr",
        "rtol":         0.0,
        "atol":         0.0,
    },

    # -----------------------------------------------------------------------
    # Comparison and select in one kernel, the mask spilled between them
    # (kernel.py's module docstring says why it cannot stay in registers).
    # -----------------------------------------------------------------------
    "spilled": {
        "base": None,
        "tags": _TAGS,
        "summary": (
            "1D where(x > y, p, q) in one kernel, the mask spilled to its own "
            "buffer between compare and select. Sweeps fp16/fp32."
        ),
        "kernel_fn":    kernel.where_1d_device,
        "factory":      Where(ptrs=("x_ptr", "y_ptr", "p_ptr", "q_ptr",
                                    "mask_ptr", "output_ptr")),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT"],
        "params":       _PARAMS,
        "grid":         [1],
        "compiles_to_binary": True,
        "inputs":       make_where_inputs,
        "reference":    run_where,
        "output_key":   "output_ptr",
        "rtol":         0.0,
        "atol":         0.0,
    },
}

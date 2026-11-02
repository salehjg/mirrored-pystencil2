import pystencils as ps
import numpy as np
import sympy as sp
import pytest
import pickle

try:
    import randomgen
except ImportError:
    pytest.skip("randomgen not available", allow_module_level=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("target", ps.Target.available_targets())
@pytest.mark.parametrize(
    "offsets",
    [
        (0, 0),
        (12, 41),
        (
            ps.TypedSymbol("o0", ps.DynamicType.INDEX_TYPE),
            ps.TypedSymbol("o1", ps.DynamicType.INDEX_TYPE),
        ),
    ],
)
@pytest.mark.parametrize("c_value", [0, 41])
def test_philox(dtype, gen_config, xp, offsets, c_value):
    seed = 0x7EADBEEF
    c = ps.TypedSymbol("c", ps.DynamicType.INDEX_TYPE)

    rng = ps.random.Philox("philox", dtype, seed=seed, offsets=offsets)
    q = rng.vector_size

    target = gen_config.get_target()

    layout = "numpy" if target == ps.Target.SYCL else "fzyx"
    f, g = ps.fields(f"f({q}), g({q}): {dtype}[2D]", layout=layout)

    #   First invocation, write to f
    rx1, rasm1 = rng.get_random_vector(c)
    asms = [rasm1] + [ps.Assignment(f(i), rx1[i]) for i in range(q)]

    #   Second invocation, write to g
    rx2, rasm2 = rng.get_random_vector(c)
    asms += [rasm2] + [ps.Assignment(g(i), rx2[i]) for i in range(q)]

    cfg = gen_config.copy()
    cfg.default_dtype = dtype
    cfg.index_dtype = f"int{cfg.default_dtype.width}"

    ker = ps.create_kernel(asms, cfg).compile()

    shape = (12, 12)
    if target == ps.Target.SYCL:
        f_arr = xp.zeros(shape + (q,), dtype=dtype)
    else:
        layout_tuple = ps.field.layout_string_to_tuple(layout, 3)
        f_arr = ps.field.create_numpy_array_with_layout(
            shape + (q,), layout_tuple, dtype=dtype, xp=xp
        )

    f_arr[:] = np.dtype(dtype).type(0)
    g_arr = xp.zeros_like(f_arr)

    if isinstance(offsets[0], ps.TypedSymbol):
        offset_args = {"o0": 15, "o1": -3}
        actual_offsets = (offset_args["o0"], offset_args["o1"])
    else:
        offset_args = dict()
        actual_offsets = offsets

    ker(f=f_arr, g=g_arr, c=c_value, **offset_args)

    def get_reference(invocation_key):
        int_reference = np.empty(shape + (4,), dtype=int)

        for x in range(shape[0]):
            for y in range(shape[1]):
                cx = x + actual_offsets[0]
                cy = y + actual_offsets[1]

                def cast(v):
                    return int(np.uint32(np.int32(v)))

                #   Reflect counter construction of implementation exactly
                counter = cast(c_value) + cast(cx) * 2**32 + cast(cy) * 2**64 - 1
                #   128-bit wrap-around if negative
                counter = (2**128 + counter) & (2**128 - 1)

                keys = (seed, invocation_key)
                philox = randomgen.Philox(
                    counter=counter,
                    key=keys[0] + keys[1] * 2**32,
                    number=4,
                    width=32,
                )
                int_reference[x, y, :] = philox.random_raw(size=4)

        reference = np.empty(shape + (q,), dtype=dtype)

        match dtype:
            case "float32":
                reference[:] = int_reference * 2.0**-32 + 2.0**-33
            case "float64":
                x = int_reference[:, :, 0::2]
                y = int_reference[:, :, 1::2]
                z = x ^ y << (53 - 32)
                reference[:] = z * 2.0**-53 + 2.0**-54

        return reference

    eps = np.finfo(np.dtype(dtype)).eps

    if gen_config.get_target().is_gpu():
        f_arr = f_arr.get()
        g_arr = g_arr.get()

    f_reference = xp.array(get_reference(0))
    g_reference = xp.array(get_reference(1))

    xp.testing.assert_allclose(f_arr, f_reference, rtol=0, atol=eps)
    xp.testing.assert_allclose(g_arr, g_reference, rtol=0, atol=eps)


def test_compare_and_pickle():
    rng = ps.random.Philox(
        "phil", "float32", ps.TypedSymbol("seed", "uint32"), (13, 41, 2)
    )
    t = ps.TypedSymbol("t", "uint32")
    rx, rasm = rng.get_random_vector(t)

    assert isinstance(rasm.rhs, sp.Function)

    rasm_dump = pickle.dumps(rasm)

    rasm_reconstructed = pickle.loads(rasm_dump)
    assert rasm_reconstructed == rasm
    assert isinstance(rasm_reconstructed.rhs.state, ps.random.Philox.PhiloxState)
    assert isinstance(rasm_reconstructed.rhs, sp.Function)
    assert rasm_reconstructed.rhs.state == rasm.rhs.state

    rng2 = ps.random.Philox(
        "phil", "float32", ps.TypedSymbol("seed", "uint32"), (13, 41, 2)
    )
    rx2, rasm2 = rng2.get_random_vector(t)

    assert rx2 == rx
    assert rasm2 == rasm
    assert rasm2 == rasm_reconstructed

    rx3, rasm3 = rng2.get_random_vector(t)
    assert rx3 != rx2
    assert rasm3 != rasm2

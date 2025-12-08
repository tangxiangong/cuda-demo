import cupy as cp
import numpy as np
import cuda.tile as ct


@ct.kernel
def vector_dot(a, b, result, tile_size: ct.Constant[int]):
    pid = ct.bid(0)

    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    product = a_tile * b_tile

    partial_sum = ct.sum(product, axis=0)
    
    ct.atomic_add(result, ct.zeros(1, dtype=ct.int32), partial_sum)


def main():
    vector_size = 1 << 12
    tile_size = 1 << 4
    grid = (vector_size // tile_size, 1, 1)

    a = cp.random.uniform(-1, 1, vector_size)
    b = cp.random.uniform(-1, 1, vector_size)
    result = cp.zeros(1, dtype=a.dtype)

    ct.launch(cp.cuda.get_current_stream(), grid, vector_dot, (a, b, result, tile_size))

    expected = cp.dot(a, b)
    computed = result[0]

    print(f"Computed dot product: {computed}")
    print(f"Expected dot product: {expected}")
    np.testing.assert_allclose(cp.asnumpy(computed), cp.asnumpy(expected), rtol=1e-5)
    print("Success!")


if __name__ == "__main__":
    main()

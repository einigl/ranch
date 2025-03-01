from math import ceil, floor, trunc

import numpy as np
import pytest
from astropy.io import fits

from ranch import Cube, Map, Profile, create_header


@pytest.fixture(scope="module")
def cube() -> np.ndarray:
    nz, ny, nx = 2, 3, 5
    data = np.arange(nz * ny * nx).reshape(nz, ny, nx)
    header = create_header("cube", nx=nx, ny=ny, nz=nz)
    return Cube(data, header)


# Unary float operators


def test_neg(cube: Cube):
    res = -cube

    assert isinstance(res, Cube)
    assert (res.data == -cube.data).any()


def test_abs(cube: Cube):
    res = abs(cube)

    assert isinstance(res, Cube)
    assert (res.data == abs(cube.data)).any()


def test_round(cube: Cube):
    res = round(cube, 2)

    assert isinstance(res, Cube)
    assert (res.data == np.round(cube.data, 2)).any()


def test_floor(cube: Cube):
    res = floor(cube)

    assert isinstance(res, Cube)
    assert (res.data == np.floor(cube.data)).any()


def test_ceil(cube: Cube):
    res = ceil(cube)

    assert isinstance(res, Cube)
    assert (res.data == np.floor(cube.data)).any()


def test_trunc(cube: Cube):
    res = trunc(cube)

    assert isinstance(res, Cube)
    assert (res.data == np.trunc(cube.data)).any()


# Binary float operators


def test_add(cube: Cube):
    res_l = cube + 1
    res_r = 1 + cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l.data == cube.data + 1).any()
    assert (res_r.data == cube.data + 1).any()


def test_sub(cube: Cube):
    res_l = cube - 1
    res_r = 1 - cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l.data == cube.data - 1).any()
    assert (res_r.data == 1 - cube.data).any()


def test_mul(cube: Cube):
    res_l = cube * 2
    res_r = 2 * cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l.data == 2 * cube.data).any()
    assert (res_r.data == 2 * cube.data).any()


def test_truediv(cube: Cube):
    res_l = cube / 2
    res_r = 2 / (cube + 1)

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l.data == cube.data / 2).any()
    assert (res_r.data == 2 / (cube + 1)).any()


def test_floordiv(cube: Cube):
    res_l = cube // 2
    res_r = 10 // (cube + 1)

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l.data == cube.data // 2).any()
    assert (res_r.data == 10 // (cube + 1)).any()


def test_mod(cube: Cube):
    res_l = cube % 2
    res_r = 2 % cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l.data == cube.data % 2).any()
    assert (res_r.data == 2 % cube).any()


def test_pow(cube: Cube):
    res_l = cube**2
    res_r = 2**cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l.data == cube.data**2).any()
    assert (res_r.data == 2**cube).any()


# Augmented assignment float operators


def test_iadd(cube: Cube):
    cube += 1

    assert isinstance(cube, Cube)


def test_isub(cube: Cube):
    cube -= 1

    assert isinstance(cube, Cube)


def test_imul(cube: Cube):
    cube *= 2

    assert isinstance(cube, Cube)


def test_itruediv(cube: Cube):
    cube /= 2

    assert isinstance(cube, Cube)


def test_ifloordiv(cube: Cube):
    cube //= 2

    assert isinstance(cube, Cube)


def test_imod(cube: Cube):
    cube %= 2

    assert isinstance(cube, Cube)


def test_ipow(cube: Cube):
    cube **= 2

    assert isinstance(cube, Cube)


# Unary logical operators


def test_invert(cube: Cube):
    cube = cube > 10
    res = ~(cube > 10)

    assert isinstance(res, Cube)
    assert (res.data.astype(bool) == ~(cube.data.astype(bool))).any()


# Binary logical operators


def test_and(cube: Cube):
    cube_1 = cube >= 2
    cube_2 = cube < 6
    res = cube_1 & cube_2

    assert isinstance(res, Cube)
    assert res.sum() == 4


def test_or(cube: Cube):
    cube_1 = cube < 2
    cube_2 = cube < 6
    res = cube_1 | cube_2

    assert isinstance(res, Cube)
    assert res.sum() == 6


def test_xor(cube: Cube):
    cube_1 = cube < 2
    cube_2 = cube < 6
    res = cube_1 ^ cube_2

    assert isinstance(res, Cube)
    assert res.sum() == 4


# Augmented assignment logical operators


def test_iand(cube: Cube):
    res = cube >= 2
    res &= cube < 6

    assert isinstance(res, Cube)
    assert res.sum() == 4


def test_ior(cube: Cube):
    res = cube < 2
    res |= cube < 6

    assert isinstance(res, Cube)
    assert res.sum() == 6


def test_ixor(cube: Cube):
    res = cube < 2
    res ^= cube < 6

    assert isinstance(res, Cube)
    assert res.sum() == 4


# Comparison operators


def test_eq(cube: Cube):
    res_l = cube == 10
    res_r = 10 == cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l == res_r).any()


def test_ne(cube: Cube):
    res_l = cube != 10
    res_r = 10 != cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l == res_r).any()


def test_ge(cube: Cube):
    res_l = cube >= 10
    res_r = 10 <= cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l == res_r).any()
    assert res_l.sum() == res_l.size - 10


def test_le(cube: Cube):
    res_l = cube <= 10
    res_r = 10 >= cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l == res_r).any()
    assert res_l.sum() == 11


def test_gt(cube: Cube):
    res_l = cube > 10
    res_r = 10 < cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l == res_r).any()
    assert res_l.sum() == res_l.size - 11


def test_lt(cube: Cube):
    res_l = cube < 10
    res_r = 10 > cube

    assert isinstance(res_l, Cube) and isinstance(res_r, Cube)
    assert (res_l == res_r).any()
    assert res_l.sum() == 10


# Other operators


def test_contains(cube: Cube):
    assert 10 in cube
    assert -1 not in cube


# TODO: __getitem__ tests

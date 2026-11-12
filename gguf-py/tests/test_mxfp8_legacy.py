"""Reject ambiguous legacy type-43 tensors and preserve both FP8 encodings.

Set GGML_BASE_LIB to the freshly built libggml-base.so to check both readers.
"""
import ctypes
import os
import struct

import pytest
from gguf.gguf_reader import GGUFReader
from gguf.constants import GGMLQuantizationType


class _InitParams(ctypes.Structure):
    _fields_ = [('no_alloc', ctypes.c_bool), ('ctx', ctypes.POINTER(ctypes.c_void_p))]


def _write_file(path, alignment, tensors):
    # [(name, on-disk type, elements, bytes)], including explicit padding.
    key = b'general.alignment'
    out = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), 1)
    out += struct.pack('<Q', len(key)) + key + struct.pack('<II', 4, alignment)
    offset = 0
    for name, dtype, elems, data in tensors:
        name = name.encode()
        out += struct.pack('<Q', len(name)) + name
        out += struct.pack('<IQIQ', 1, elems, dtype, offset)
        offset += (len(data) + alignment - 1) // alignment * alignment
    out += b'\0' * (-len(out) % alignment)
    for _, _, _, data in tensors:
        out += data + b'\0' * (-len(data) % alignment)
    path.write_bytes(out)


def _native_types(path):
    library = os.getenv('GGML_BASE_LIB')
    if not library:
        return None
    lib = ctypes.CDLL(library)
    lib.gguf_init_from_file.argtypes = [ctypes.c_char_p, _InitParams]
    lib.gguf_init_from_file.restype = ctypes.c_void_p
    lib.gguf_get_tensor_type.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.gguf_get_tensor_type.restype = ctypes.c_int
    lib.gguf_get_n_tensors.argtypes = [ctypes.c_void_p]
    lib.gguf_get_n_tensors.restype = ctypes.c_int64
    lib.gguf_free.argtypes = [ctypes.c_void_p]
    ctx = lib.gguf_init_from_file(os.fsencode(path), _InitParams(True, None))
    if not ctx:
        return []
    try:
        return [lib.gguf_get_tensor_type(ctx, i) for i in range(lib.gguf_get_n_tensors(ctx))]
    finally:
        lib.gguf_free(ctx)


def test_type43_mxfp8_and_existing_f8_are_distinguished(tmp_path):
    path = tmp_path / 'mixed.gguf'
    _write_file(path, 32, [('legacy', 43, 128, bytes(132)),
                           ('native-f8', 43, 128, bytes(128)),
                           ('native-mxfp8', 63, 128, bytes(132))])
    got = [t.tensor_type for t in GGUFReader(path).tensors]
    expected = [GGMLQuantizationType.MXFP8, GGMLQuantizationType.F8_E4M3, GGMLQuantizationType.MXFP8]
    assert got == expected
    native = _native_types(path)
    if native is not None:
        assert native == [63, 43, 63]


def test_ambiguous_type43_refused_even_when_other_tensors_identify_format(tmp_path):
    path = tmp_path / 'ambiguous.gguf'
    _write_file(path, 64, [('known-f8', 43, 128, bytes(128)),
                           ('ambiguous', 43, 32, bytes(64))])
    with pytest.raises(ValueError, match='ambiguous type 43'):
        GGUFReader(path)
    native = _native_types(path)
    if native is not None:
        assert native == []


def test_aligned_unambiguous_f8_is_unchanged(tmp_path):
    path = tmp_path / 'f8.gguf'
    _write_file(path, 32, [('f8', 43, 256, bytes(256))])
    assert GGUFReader(path).tensors[0].tensor_type == GGMLQuantizationType.F8_E4M3
    native = _native_types(path)
    if native is not None:
        assert native == [43]


@pytest.mark.parametrize('size', [96, 192])
def test_invalid_type43_span_is_rejected(tmp_path, size):
    path = tmp_path / 'invalid.gguf'
    _write_file(path, 32, [('bad', 43, 128, bytes(size))])
    with pytest.raises(ValueError, match='invalid type 43 span'):
        GGUFReader(path)
    native = _native_types(path)
    if native is not None:
        assert native == []


def test_legacy_and_f8_are_resolved_independently_per_shard(tmp_path):
    shard1 = tmp_path / 'model-00001-of-00002.gguf'
    shard2 = tmp_path / 'model-00002-of-00002.gguf'
    _write_file(shard1, 32, [('embed.f8', 43, 128, bytes(128))])
    _write_file(shard2, 32, [('engram.mx', 43, 128, bytes(132))])
    assert GGUFReader(shard1).tensors[0].tensor_type == GGMLQuantizationType.F8_E4M3
    assert GGUFReader(shard2).tensors[0].tensor_type == GGMLQuantizationType.MXFP8
    one, two = _native_types(shard1), _native_types(shard2)
    if one is not None:
        assert one == [43] and two == [63]


@pytest.mark.skipif(os.getenv('GGML_RUN_LARGE_SPARSE_TESTS') != '1', reason='requires sparse-file support')
def test_real_engram_dimensions_without_downloading_large_model(tmp_path):
    if not os.getenv('GGML_BASE_LIB'):
        pytest.skip('GGML_BASE_LIB is required for the native no-allocation reader')
    path = tmp_path / 'real-shape-sparse.gguf'
    label = b'per_layer_token_embd.weight'
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)
    header += struct.pack('<Q', len(label)) + label
    header += struct.pack('<IQQIQ', 2, 256, 768022850, 43, 0)
    header += bytes(-len(header) % 32)
    # 8 MXFP8 blocks per row * 33 bytes * 768,022,850 rows.
    payload = (768022850 * 264 + 31) // 32 * 32
    try:
        with path.open('wb') as f:
            f.write(header)
            f.truncate(len(header) + payload)
    except OSError as exc:
        pytest.skip(f'filesystem cannot create the large sparse fixture: {exc}')
    stat = path.stat()
    if not hasattr(stat, 'st_blocks') or stat.st_blocks * 512 > 1024 * 1024:
        pytest.skip('filesystem did not create a sparse fixture')
    assert _native_types(path) == [63]

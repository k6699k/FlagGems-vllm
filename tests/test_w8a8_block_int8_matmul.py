# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flaggems_vllm

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax",
    reason="MetaX W8A8 Block INT8 backend",
)


def _reference(a, b, a_scale, b_scale, block_n, block_k):
    result = torch.zeros((a.shape[0], b.shape[0]), device="cpu", dtype=torch.float32)
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    a_scale = a_scale.cpu()
    b_scale = b_scale.cpu()
    for k_tile in range((a.shape[1] + block_k - 1) // block_k):
        ks = slice(k_tile * block_k, min((k_tile + 1) * block_k, a.shape[1]))
        for n_tile in range((b.shape[0] + block_n - 1) // block_n):
            ns = slice(n_tile * block_n, min((n_tile + 1) * block_n, b.shape[0]))
            result[:, ns] += (
                (a_cpu[:, ks] @ b_cpu[ns, ks].T)
                * a_scale[:, k_tile, None]
                * b_scale[n_tile, k_tile]
            )
    return result


@pytest.mark.w8a8_block_int8_matmul
@pytest.mark.parametrize("shape", [(1, 65, 130), (7, 129, 257), (32, 64, 384)])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_w8a8_block_int8_matmul(shape, output_dtype):
    m, n, k = shape
    block_n, block_k = 128, 128
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    a_scale = torch.rand((m, (k + block_k - 1) // block_k), device="cuda") * 0.01
    b_scale = (
        torch.rand(
            ((n + block_n - 1) // block_n, (k + block_k - 1) // block_k), device="cuda"
        )
        * 0.01
    )

    output = flaggems_vllm.w8a8_block_int8_matmul(
        a, b, a_scale.float(), b_scale.float(), [block_n, block_k], output_dtype
    )
    expected = _reference(a, b, a_scale, b_scale, block_n, block_k).to(output_dtype)
    atol = 1e-3 if output_dtype is torch.float32 else 2e-2
    torch.testing.assert_close(
        output.cpu(),
        expected,
        atol=atol,
        rtol=atol,
    )


def test_w8a8_block_int8_empty_shapes():
    for m, n in ((0, 65), (3, 0)):
        k = 130
        a = torch.empty((m, k), device="cuda", dtype=torch.int8)
        b = torch.empty((n, k), device="cuda", dtype=torch.int8)
        a_scale = torch.empty((m, 2), device="cuda", dtype=torch.float32)
        b_scale = torch.empty((1 if n else 0, 2), device="cuda", dtype=torch.float32)
        out = flaggems_vllm.w8a8_block_int8_matmul(
            a, b, a_scale, b_scale, [128, 128], torch.float32
        )
        assert out.shape == (m, n)


def test_w8a8_block_int8_block_k256_scales():
    m, n, k = 7, 129, 512
    block_n, block_k = 128, 256
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    a_scale = torch.rand((m, 2), device="cuda", dtype=torch.float32) * 0.01
    b_scale = torch.rand((2, 2), device="cuda", dtype=torch.float32) * 0.01
    output = flaggems_vllm.w8a8_block_int8_matmul(
        a, b, a_scale, b_scale, [block_n, block_k], torch.float32
    )
    torch.testing.assert_close(
        output.cpu(),
        _reference(a, b, a_scale, b_scale, block_n, block_k),
        atol=1e-3,
        rtol=1e-3,
    )


def test_w8a8_block_int8_rejects_invalid_inputs():
    a = torch.zeros((1, 128), device="cuda", dtype=torch.int8)
    b = torch.zeros((64, 128), device="cuda", dtype=torch.int8)
    a_scale = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    b_scale = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="block_k"):
        flaggems_vllm.w8a8_block_int8_matmul(
            a, b, a_scale, b_scale, [128, 16], torch.float32
        )
    with pytest.raises(NotImplementedError, match="contiguous"):
        flaggems_vllm.w8a8_block_int8_matmul(
            a[:, ::2], b[:, ::2], a_scale, b_scale, [128, 64], torch.float32
        )

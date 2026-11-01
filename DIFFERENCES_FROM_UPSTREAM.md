# Differences from upstream

CTranslate2-Reborn is upstream CTranslate2 4.8.4 plus the changes below, oldest first. Each one links to its commit.

## Speed (2026-10-14)

On an RTX 4090 under Windows, these make Whisper large-v3 about 2.3 times as fast, and Qwen3 and Gemma 3 text generation 2 to 4 times as fast at batch size 1. Most of the gain comes from the first change, which matters most on Windows (Linux hasn't been measured). Results don't change beyond floating-point rounding.

- [Skip cudaSetDevice in the CUDA allocators when the device is already current](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ac04aa6feb2f40e47d6fd24cb3daa41862df23d6)
- [Normalize short rows with one warp per row in the CUDA RMSNorm](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/aed2825870b9b33143ca6bf526cb5590380fc4a0)
- [Compute softmax with one warp per row for rows of up to 1024 values on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/46e324a3f543fb94c2f3c8462593085a6ebe9b2a)
- [Rotate attention heads with one warp per row and vector loads on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e6a55f42b0e387710095a4230088b22fc13a2879)
- [Attend with grouped-query keys and values without repeating their heads](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/439f6eb6a5aa811c50246fde8fcc128dd28f26ad)
- [Keep the padding on the GPU when it is under 5% of the batch](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/07a4eea57bd1e7ec005c81d83ca3f43c25d5c67d)
- [Split the fused QKV projection directly into heads](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/a04a95ef5f7c84ae9e83f05d874257973d6a3a8f)
- [Normalize and rotate query and key heads in one pass on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ae0a13625b03be69e4053a02611f241eb63b5dab)
- [Apply the gated activation of feed-forward networks in one pass](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/cb32112fa50de4fe7ec1dba2bc7f89f9a7fb7aec)
- [Compute softmax with one block per row for rows of up to 8192 values on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/dcb40147aa33d2405b69561305f063f3ae284755)
- [Concatenate along inner axes with one kernel on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ff0ae6f8066fb8621a3e8df4505ba4ce767aea99)

## Gemma 3 fix (2026-10-15)

- [Fix Gemma 3 sliding window attention and the converter's RoPE and score scaling](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/c07d979155fdff4600ae17a87eee9c659c9185d9)

Prompts longer than the sliding window no longer give degenerate output that varies from run to run. Gemma 3 4B, 12B and 27B models converted before this fix, or with upstream's converter, need to be converted again.

## Packaging (2026-10-15)

- [Run the CI workflow only when started from the Actions tab](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/203254a9b728db71e22c58a35a60557e4f50ea4f)
- [Publish the Python package as ctranslate2-reborn](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/78ec1b1b013fe4f6c79786d25caefd431ca98759)
- [Add Blair Chintella to the copyright notice](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/258d3aff5cbf456ad28bcf443c3b3dac717094a5)

## Documentation (2026-10-15)

- [Replace the README with a short description of the fork](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/0ca6baa83b5d99edb78363f5bea024ef782ffedc)
- [Remove upstream's contributing guide, release checklist and security policy](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/dc19c89b361e4b0532f57e24b8c1e695c1c6a334)
- [Remove upstream's documentation website, examples and tools](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/d4123e2677463cca057c10ce2fba81b29587f9e8)

## Release pipeline (2026-10-15)

- [Read only the X.Y.Z part of the version in CMake](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/afe95d0438fc7ab705a15fc86f66b9341791897f)
- [Ignore the egg-info folder under the package's new name](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/dd0026f79f262d1288efed8cecc7afdd0ee28c5f)
- [Run the release jobs when a GitHub release is published](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/f5e73c5ed4f35753502188b24cca264395154e04)
- [Publish to PyPI with Trusted Publishing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/d159e961a640efb5477f0c9cf2c505ed76dfe24a)
- [Push the Docker images to the fork's GitHub registry](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e0eb6f16112bc54c35d3fa59f2c999c7abdb8efd)
- [Set the version to 4.8.4.post1](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/c13036762220b0bf225119895b9574c64f8734eb)

## Wav2Vec2 test (2026-10-15)

Upstream's 4.8.4 has the same fix as this fork for Wav2Vec2 and Wav2Vec2-BERT models on Intel CPUs, where MKL packs their weights, so only the fork's test remains.

- [Test Wav2Vec2 on the CPU with MKL-packed weights](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/007d44841ff9db0d004fc26ee4b240136c78d72f)

## Docker images removed (2026-10-15)

- [Remove the Docker images and their CI job](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/d246fa269242e334b2e690492e1d9dccbc65512b)

## CUDA 13 (2026-10-16)

The wheels are built with CUDA 13.0.3 instead of 12.8. That drops GPUs older than Turing and needs NVIDIA driver 580 or newer, and `pip install ctranslate2-reborn[cuda]` also installs cuBLAS 13.

- [Upgrade CCCL to 3.0.1 for CUDA 13 support](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/b399a6c5f895e481f001bb6cae0fdcde49af1061)
- [Support CUDA 13 GPU architectures](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/23a96bd7389d5592c7156fcc96e6a74fa5fb237d)
- [Add native Ada, Hopper and Blackwell code to the Common architecture list](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e9d4200b4422fb244d85b3fa42e6ab0f45dc1bf0)
- [Find CUDA 13 DLLs on Windows](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/2d361f27e1b173161fcaed53d3ce26da5fb6cc4e)
- [Build the CI wheels with CUDA 13.0.3](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/24b0310ae85f678212c28e0ff8c4da94ad8963e0)
- [Add a cuda extra that installs cuBLAS from NVIDIA's pip packages](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ea078d5f6acf7859bd26645aca990924c08b654c)
- [Load cuBLAS from PyTorch's folder on Windows when the cuda extra is missing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/3d089519633d5a00515ef665d315e1a5763e04ac)
- [Prefer PyTorch's cuBLAS on Windows over NVIDIA's pip copy](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/4cd95bb8b3e89fb2942a84f5fe0bdc8477841823)

## Python 3.9 dropped (2026-10-16)

- [Drop Python 3.9](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/9d13eff05472279ae665d949e9e9a87499327723)

## CUDA 12.8 wheels on GitHub releases (2026-10-17)

PyPI keeps the CUDA 13 wheels. Each GitHub release also gets CUDA 12.8 wheels for Linux and Windows, labeled `+cu128`, for GPUs older than Turing and NVIDIA drivers older than 580.

- [Build CUDA 12.8 wheels and attach them to GitHub releases](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ca1035b71328a1a491ec99064e5d471a957dc0bb)

## Empty batches (2026-10-18)

An empty batch no longer crashes the process on the CPU or raises misleading errors on CUDA: `Generator.forward_batch`, `Encoder.forward_batch`, Whisper's `encode` and `detect_language` and the Wav2Vec2 and Wav2Vec2-BERT encoders return empty outputs with the shape a non-empty call has.

- [Return empty outputs for empty batches instead of crashing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/6649170fa9ee0766d476f6d8af0226e2dff21834)
- [Test empty batches on every model type](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/10d85eed69d6e1debe1ce2439d34f4eba617f2ff)

## BERT-family conversion with Transformers 5 (2026-10-18)

BERT, RoBERTa, XLM-RoBERTa and CamemBERT models now convert with Transformers 5, which removed a configuration setting the converter required; conversions with Transformers 4 are unchanged.

- [Convert BERT-family models with Transformers 5](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/533c110bcf1ba72f320c95747967284c8fd1cf64)
- [Test the BERT-family converters on tiny models](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/8732ac241b6f9f00c14d1596d09a9b30fcb4b76b)

## Qwen3-Embedding (2026-10-18)

Qwen3-Embedding models, and other Qwen3-based embedding models such as Yuan-embedding-2.0-en and harrier-oss-v1-0.6b, convert with `--model_type Qwen3Embedding` and run with `ctranslate2.Encoder`, whose `pooler_output` is each text's last-token embedding, unnormalized. Models that add layers to Qwen3 are refused instead of being converted without them.

- [Add Qwen3-Embedding support (decoder-as-encoder)](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/8e0cd913cd671db8caa17d23c8f4c23e48548835)
- [Return last-token pooled output for decoder-based encoders](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/20571167700afbb18e652ff3a41a2987dcc11619)
- [Format the Qwen3-Embedding loader with black](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ad031908a06d90dd488330c579097d5488d7439c)
- [Skip packing the output projection when loading a decoder for an encoder](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/2376ce824cc6fb3c7a3e4a22c2ed05910de26ee5)
- [Return empty outputs for an empty batch in decoder-based encoders](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/8804955f28571c1460cf4d39ea97f60d0ab82430)
- [Test Qwen3-Embedding conversion of both checkpoint layouts](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/91f37b803082c2f948dc7519517344093b35d908)
- [Refuse to convert Qwen3 models that add layers as Qwen3Embedding](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e352c4bb53dc5e705831e420279f2142c1bdf997)
- [Test an empty batch on a Qwen3-Embedding model](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e18c2d5a274a1f7a7c83b4256b582074ea1906cb)

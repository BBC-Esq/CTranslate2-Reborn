# Differences from upstream

CTranslate2-Reborn is upstream CTranslate2 4.8.3 plus the changes below, oldest first. Each one links to its commit.

## Speed (2026-10-14)

On an RTX 4090 under Windows, these make Whisper large-v3 about 2.3 times as fast, and Qwen3 and Gemma 3 text generation 2 to 4 times as fast at batch size 1. Most of the gain comes from the first change, which matters most on Windows (Linux hasn't been measured). Results don't change beyond floating-point rounding.

- [Skip cudaSetDevice in the CUDA allocators when the device is already current](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/9eb7a9cae845c740ebedfdd0f3bb1de538553991)
- [Normalize short rows with one warp per row in the CUDA RMSNorm](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/fb99de3b1af13810a8f95df7c249aaf71af42b0a)
- [Compute softmax with one warp per row for rows of up to 1024 values on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/794b846afde15f5e10bbf3c2fa16b13bd5808ddc)
- [Rotate attention heads with one warp per row and vector loads on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/304a2edb93f9b253d7fb765e21ae61602ac8043c)
- [Attend with grouped-query keys and values without repeating their heads](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/69f76b4b561dc3c9f2727a877ae84fba7c59b97f)
- [Keep the padding on the GPU when it is under 5% of the batch](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/5ef6406007cca54b583c67b000b9bf35d4d3edb9)
- [Split the fused QKV projection directly into heads](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/d20068fdb824ffbb2bad6e541428107161251dd8)
- [Normalize and rotate query and key heads in one pass on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/bbf18afc80141a7615d49de51f02f8ee9e269e7b)
- [Apply the gated activation of feed-forward networks in one pass](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e1be4707d9420d193e21160dec2efb3726918c75)
- [Compute softmax with one block per row for rows of up to 8192 values on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/5ce6fff4791c017a3860b214f2765e9922a2ef29)
- [Concatenate along inner axes with one kernel on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/d397417438a377e35c377438c6e96d68f07ff633)

## Gemma 3 fix (2026-10-15)

- [Fix Gemma 3 sliding window attention and the converter's RoPE and score scaling](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/0de269e25eaf694dc27d7c948cdbf4f6915e55ee)

Prompts longer than the sliding window no longer give degenerate output that varies from run to run. Gemma 3 4B, 12B and 27B models converted before this fix, or with upstream's converter, need to be converted again.

## Packaging (2026-10-15)

- [Run the CI workflow only when started from the Actions tab](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e314f7a7b377cb839559833be34164db86253240)
- [Publish the Python package as ctranslate2-reborn](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/c467d7c88562a67798eeabe16e738407a919f6ef)
- [Add Blair Chintella to the copyright notice](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/bc8ef1590054a0c765f5c2916fe46a20f4d2d7d7)

## Documentation (2026-10-15)

- [Replace the README with a short description of the fork](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/874ef071f9cb15a26a4525445d46dbcb5d6d7fb8)
- [Remove upstream's contributing guide, release checklist and security policy](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/fce80090a6a9dab0fdba0c3d57f7d44842186191)
- [Remove upstream's documentation website, examples and tools](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/aadab0039fc1bcdec8a710ee4b5a47bcc3eed604)

## Release pipeline (2026-10-15)

- [Read only the X.Y.Z part of the version in CMake](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/10baf788d329892fa0f76ab174a67b0071ee0706)
- [Ignore the egg-info folder under the package's new name](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/25973b9eff65a9ef8d905a2cca3d3c792d6d62e2)
- [Run the release jobs when a GitHub release is published](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/f2d68ebb1ab32f5e72e9623415edf26a8d36064a)
- [Publish to PyPI with Trusted Publishing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/4ab5f2e3647a817c36b0126312d7313f56c5f38d)
- [Push the Docker images to the fork's GitHub registry](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/eebbdede065d18c4ac65aa9c932c8fa006d16c4e)
- [Set the version to 4.8.3.post1](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/bead902dc1e72ac771012ce797076ca2048d5307)

## Wav2Vec2 fix (2026-10-15)

- [Detect packed weights in the Wav2Vec2 and Wav2Vec2-BERT encoders](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/42b43fe5dadbaf9e222e6726295801c5d1b5988b)
- [Test Wav2Vec2 on the CPU with MKL-packed weights](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/614f9d589a51e6109b7ac35b25c49b3f3c8350a6)

## Docker images removed (2026-10-15)

- [Remove the Docker images and their CI job](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/05b3372112f76d3a3bad9e9f75412a690725550e)

## CUDA 13 (2026-10-16)

The wheels are built with CUDA 13.0.3 instead of 12.8. That drops GPUs older than Turing and needs NVIDIA driver 580 or newer, and `pip install ctranslate2-reborn[cuda]` also installs cuBLAS 13.

- [Upgrade CCCL to 3.0.1 for CUDA 13 support](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/3c6418a962ddc9f529aeacc496d27096cbff9371)
- [Support CUDA 13 GPU architectures](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/01f3a67ae8e86e27bf6d1ba75583250e42f0fcfd)
- [Add native Ada, Hopper and Blackwell code to the Common architecture list](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/179a7ad8a952ef5f141ae9ffe7bb7949da8aff8e)
- [Find CUDA 13 DLLs on Windows](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/3483daff418e21d7a74b4a8424a014f303c7109a)
- [Build the CI wheels with CUDA 13.0.3](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/f68cfd19360832bce8c112799febfcdb591c0ef3)
- [Add a cuda extra that installs cuBLAS from NVIDIA's pip packages](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/b8083d3fdd1f2ec94decdd1ae91871f6bbbcba8a)
- [Load cuBLAS from PyTorch's folder on Windows when the cuda extra is missing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/a16a5bbb7738ac9e39018a43b50272f4fc3be7cd)
- [Prefer PyTorch's cuBLAS on Windows over NVIDIA's pip copy](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/95bc3829376866533d354d58d1659029e45f610a)

## Python 3.9 dropped (2026-10-16)

- [Drop Python 3.9](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/838997341c47f3391b70261ad49a383d6be769a4)

## CUDA 12.8 wheels on GitHub releases (2026-10-17)

PyPI keeps the CUDA 13 wheels. Each GitHub release also gets CUDA 12.8 wheels for Linux and Windows, labeled `+cu128`, for GPUs older than Turing and NVIDIA drivers older than 580.

- [Build CUDA 12.8 wheels and attach them to GitHub releases](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/3f906885f31bdb5a8d8f153580badd204c356117)

## Empty batches (2026-10-18)

An empty batch no longer crashes the process on the CPU or raises misleading errors on CUDA: `Generator.forward_batch`, `Encoder.forward_batch`, Whisper's `encode` and `detect_language` and the Wav2Vec2 and Wav2Vec2-BERT encoders return empty outputs with the shape a non-empty call has.

- [Return empty outputs for empty batches instead of crashing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/1aae5337fdd47a39847b6d41a9b4143d672ee886)
- [Test empty batches on every model type](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/6a59878e996639b924c5bc48f29c7cb7704ab8b6)

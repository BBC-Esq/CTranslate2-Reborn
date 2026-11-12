# Differences from upstream

CTranslate2-Reborn is upstream CTranslate2 4.9.0 plus the changes below, oldest first. Each one links to its commit.

## Speed (2026-10-14)

On an RTX 4090 under Windows, these make Whisper large-v3 about 2.3 times as fast, and Qwen3 and Gemma 3 text generation 2 to 4 times as fast at batch size 1. Most of the gain comes from the first change, which matters most on Windows (Linux hasn't been measured). Results don't change beyond floating-point rounding.

- [Skip cudaSetDevice in the CUDA allocators when the device is already current](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/124a0270f734669caf492566f87a1ef4384e1ce3)
- [Normalize short rows with one warp per row in the CUDA RMSNorm](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/98c47144cc18a10fca9a57341889b92bcc0a3cd0)
- [Compute softmax with one warp per row for rows of up to 1024 values on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/deccfe4e9a1cdb87c3d453a85363ea9bd5bb7d66)
- [Rotate attention heads with one warp per row and vector loads on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ff40d83cd2e4cc86205f2a12dc9aeb57b707d146)
- [Attend with grouped-query keys and values without repeating their heads](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/df17436de8efd2b333bc417dd4b6abcfe43972b2)
- [Keep the padding on the GPU when it is under 5% of the batch](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/bfe7dcde5323edc8aa2bdece6f350bcddd66c84a)
- [Split the fused QKV projection directly into heads](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/b07863ee97d944a036cefa32663884a27cb971d9)
- [Normalize and rotate query and key heads in one pass on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/35b875cb15238375f6e94678767b52a41cf525f9)
- [Apply the gated activation of feed-forward networks in one pass](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/675236bec1c43804fafae968f6f2fcc3807a4e18)
- [Compute softmax with one block per row for rows of up to 8192 values on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ff175f5a6641d9104311b65c797140e95fb9bb79)
- [Concatenate along inner axes with one kernel on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/2eb32f176bbcba455f558bb60ba59e200248be7c)

## Gemma 3 fix (2026-10-15)

- [Fix Gemma 3 sliding window attention and the converter's RoPE and score scaling](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/62fdf97c82c10fb3f919381fad685041e313bed5)

Prompts longer than the sliding window no longer give degenerate output that varies from run to run. Gemma 3 4B, 12B and 27B models converted before this fix, or with upstream's converter, need to be converted again.

## Packaging (2026-10-15)

- [Run the CI workflow only when started from the Actions tab](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/24692df8dd1761fef95d7d21ba457365985ddc62)
- [Publish the Python package as ctranslate2-reborn](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/d51b07df96234b965ff2bd8ece8ba239d0f5c282)
- [Add Blair Chintella to the copyright notice](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/f11dbd9d08d84cb55e78d82701c5bff02db29748)

## Documentation (2026-10-15)

- [Replace the README with a short description of the fork](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ca2c62a72a7677f98b1bd8d8fd2c7910fd4e50e6)
- [Remove upstream's contributing guide, release checklist and security policy](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/371c12cb0456fafcbf94efe3520860d1ce0fea74)
- [Remove upstream's documentation website, examples and tools](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/00f324a6fdb39d91c75d9dea89ef84e882b39827)

## Release pipeline (2026-10-15)

- [Read only the X.Y.Z part of the version in CMake](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/d235a7deef7c20bb20467b98166344004440688a)
- [Ignore the egg-info folder under the package's new name](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/7976bb5344a267031c70cbe91edc05265c2e6e7a)
- [Run the release jobs when a GitHub release is published](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/8a8825067464ac4c4ebab247aba35b55f9bfddd1)
- [Publish to PyPI with Trusted Publishing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/7ba232071a9eeb4d4b156b5b24a7ddfb66830953)
- [Push the Docker images to the fork's GitHub registry](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/7297f31e0bdc5b18392167a33eaf59fb76758e53)
- [Set the version to 4.9.0.post1](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/131839b147f1d7e98c3933ad44a90b00a3b9e60c)

## Wav2Vec2 test (2026-10-15)

Since 4.8.4, upstream has the same fix as this fork for Wav2Vec2 and Wav2Vec2-BERT models on Intel CPUs, where MKL packs their weights, so only the fork's test remains.

- [Test Wav2Vec2 on the CPU with MKL-packed weights](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/c3179077898d60a100a7c8d9be16723204f763e2)

## Docker images removed (2026-10-15)

- [Remove the Docker images and their CI job](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/27be402794cbaa65c171c3f1ae6a3081002234ec)

## CUDA 13 (2026-10-16)

The wheels are built with CUDA 13.0.3 instead of 12.8. That drops GPUs older than Turing and needs NVIDIA driver 580 or newer, and `pip install ctranslate2-reborn[cuda]` also installs cuBLAS 13. The wheels also include native code for Ada, Hopper and Blackwell GPUs; since upstream's 4.9.0 builds CUDA code with CMake's own CUDA support, the list of GPU architectures is set in the wheel build scripts.

- [Upgrade CCCL to 3.0.1 for CUDA 13 support](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ffa71140b44e07fb6f8ce9dcff51170af535a109)
- [Find CUDA 13 DLLs on Windows](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/b2e5f244c81743d0eb08c71336d1975e46a63362)
- [Build the CI wheels with CUDA 13.0.3](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/063f1e0c04b2fe3ee4afa01897fb9e62b54502f8)
- [Add a cuda extra that installs cuBLAS from NVIDIA's pip packages](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/4f30ff780dc2ef7d4c629dffe88befe704918636)
- [Load cuBLAS from PyTorch's folder on Windows when the cuda extra is missing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/4191156073a75c23fe35da7ffbfac7ac8d329a71)
- [Prefer PyTorch's cuBLAS on Windows over NVIDIA's pip copy](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ea74cd145bd141d9a23d4c2ed7c6fdcc55d11bdf)

## Python 3.9 dropped (2026-10-16)

- [Drop Python 3.9](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/47f9fd9547c8721d5bceacb4fe871bf2c0decf39)

## CUDA 12.8 wheels on GitHub releases (2026-10-17)

PyPI keeps the CUDA 13 wheels. Each GitHub release also gets CUDA 12.8 wheels for Linux and Windows, labeled `+cu128`, for GPUs older than Turing and NVIDIA drivers older than 580.

- [Build CUDA 12.8 wheels and attach them to GitHub releases](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/0ab21c7ca865950269aebb2637546e149387b779)

## Empty batches (2026-10-18)

An empty batch no longer crashes the process on the CPU or raises misleading errors on CUDA: `Generator.forward_batch`, `Encoder.forward_batch`, Whisper's `encode` and `detect_language` and the Wav2Vec2 and Wav2Vec2-BERT encoders return empty outputs with the shape a non-empty call has.

- [Return empty outputs for empty batches instead of crashing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/9d133cc2af365ad33b9a1c045d1ae714867e1922)
- [Test empty batches on every model type](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/53daa57725607e0815472471a22eb8577a1450a2)

## BERT-family conversion with Transformers 5 (2026-10-18)

BERT, RoBERTa, XLM-RoBERTa and CamemBERT models now convert with Transformers 5, which removed a configuration setting the converter required; conversions with Transformers 4 are unchanged.

- [Convert BERT-family models with Transformers 5](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/d237b0693ee0bde45920f12b8fdc9c3113a4200e)
- [Test the BERT-family converters on tiny models](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/3dcdeb1c2a53d4b32712401601efbe1ab7a2be40)

## Qwen3-Embedding (2026-10-18)

Qwen3-Embedding models, and other Qwen3-based embedding models such as Yuan-embedding-2.0-en and harrier-oss-v1-0.6b, convert with `--model_type Qwen3Embedding` and run with `ctranslate2.Encoder`, whose `pooler_output` is each text's last-token embedding, unnormalized. Models that add layers to Qwen3 are refused instead of being converted without them.

- [Add Qwen3-Embedding support (decoder-as-encoder)](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/f65eeb479c2845bd3d53f33f0ebf78c453720c02)
- [Return last-token pooled output for decoder-based encoders](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/56ec739b95fdaf67e53e57daec93a7bbb48c85b6)
- [Format the Qwen3-Embedding loader with black](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/c5af40d9a20ef59d34253e2c619102d00a7090b4)
- [Skip packing the output projection when loading a decoder for an encoder](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/4d7c9e477e6999d901e08f8a83c3ed9ac768c5b6)
- [Return empty outputs for an empty batch in decoder-based encoders](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/80785401d23139e1313a40af8a55b22481a855c4)
- [Test Qwen3-Embedding conversion of both checkpoint layouts](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/b7e9d113dd012511e7d21c786dfc4165195bb142)
- [Refuse to convert Qwen3 models that add layers as Qwen3Embedding](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/8d6c3cd0745669eb86ceeed34026c2efe4b1c190)
- [Test an empty batch on a Qwen3-Embedding model](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/2d1ed8b16772df13fa7147966f00d854e1a42d8a)

## Apple Metal backend (2026-11-11)

Upstream's 4.9.0 adds an experimental backend for Apple GPUs (the `mps` device). This fork's speed changes are written for CUDA and the CPU; on Metal, models run through upstream's code paths.

- [Keep upstream's code paths on Apple's Metal backend](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/34a085ec1c5ab88e7ded45c62531bb893db7545a)

## Release gate (2026-11-11)

Upstream's 4.9.0 adds a release gate: before a release is published, CI compares the new wheels with the latest release on PyPI, on the CPU, and fails on worse accuracy or a confirmed 10% drop in speed. Here the gate compares with the last release of `ctranslate2-reborn` instead of upstream's, and runs when a GitHub release is published or the workflow is started from the Actions tab.

- [Compare the release gate with ctranslate2-reborn's last release](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/6120ad446ae03b6f2451dced936c5638ffd83a02)

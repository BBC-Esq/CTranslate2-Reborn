# Differences from upstream

CTranslate2-Reborn is upstream CTranslate2 4.9.1 plus the changes below, oldest first. Each one links to its commit.

## Speed (2026-10-14)

On an RTX 4090 under Windows, these make Whisper large-v3 about 2.3 times as fast, and Qwen3 and Gemma 3 text generation 2 to 4 times as fast at batch size 1. Most of the gain comes from the first change, which matters most on Windows (Linux hasn't been measured). Results don't change beyond floating-point rounding.

- [Skip cudaSetDevice in the CUDA allocators when the device is already current](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/7ed86ac36735d0d7fc1778f035fe2793065ec49d)
- [Normalize short rows with one warp per row in the CUDA RMSNorm](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/95b8b1ee6f04eba5a99e25c59a5b8fb6151984fc)
- [Compute softmax with one warp per row for rows of up to 1024 values on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/aedad6098e1e5d2c63b13db20cee67070436b445)
- [Rotate attention heads with one warp per row and vector loads on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/97ae9ce98541f43b3ae8a782fc9c2a306822542f)
- [Attend with grouped-query keys and values without repeating their heads](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/c759efd2089f5e1d392d7db5813adc49fa61810e)
- [Keep the padding on the GPU when it is under 5% of the batch](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/c4fbc897313e96e883f5481639a00c0c0f28584a)
- [Split the fused QKV projection directly into heads](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/8ff52eb9139b57f8ad7122afdcf5854e7410c153)
- [Normalize and rotate query and key heads in one pass on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/7a84b2b827e4e52281d227373d4246cdc179febf)
- [Apply the gated activation of feed-forward networks in one pass](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/980a315ac900230b147c500b2d9cda4e554bdc9b)
- [Compute softmax with one block per row for rows of up to 8192 values on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/46f47e7dbd45790daf71af5cc7fb865be341862b)
- [Concatenate along inner axes with one kernel on CUDA](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/b47447980b4cad037eade25c18a9f53002d14910)

## Gemma 3 fix (2026-10-15)

- [Fix Gemma 3 sliding window attention](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/49ba596456382b15db34063fcdfe8c69e09a9dca)

Prompts longer than the sliding window no longer give degenerate output that varies from run to run. Since 4.9.1, upstream's converter has the same fix as this fork's had for the RoPE and score scaling of Gemma 3 4B, 12B and 27B, so only the sliding window fix remains; models converted with an older upstream converter need to be converted again.

## Packaging (2026-10-15)

- [Run the CI workflow only when started from the Actions tab](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/310bea978efe2b6784ab0b3954815977dd1d0a48)
- [Publish the Python package as ctranslate2-reborn](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/9ab6d7078709f4198be81528d93755784e36c1ae)
- [Add Blair Chintella to the copyright notice](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/cd4c74375119b2d11657156652dcc9f65e75323f)

## Documentation (2026-10-15)

- [Replace the README with a short description of the fork](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/4663c664a817ed403664824b67e8e4a9567bebe8)
- [Remove upstream's contributing guide, release checklist and security policy](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/84bf0f2f37af0d9888ac46aacb16a7df2161dd8c)
- [Remove upstream's documentation website, examples and tools](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ea794802334d06994ad96e8cdbae635ebfdd92b4)

## Release pipeline (2026-10-15)

- [Read only the X.Y.Z part of the version in CMake](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/b7b36b7297e844fba6184f30ae8cdf5ae6118693)
- [Ignore the egg-info folder under the package's new name](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/fb3811974329c2228b5f6933392423a2e0187380)
- [Run the release jobs when a GitHub release is published](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/6a485529f3486483a6b7ea9f7e1d1783613dfeb6)
- [Publish to PyPI with Trusted Publishing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/b0dca0eaa1f5c9fdbd1644cc4f11e60e4a0d820e)
- [Push the Docker images to the fork's GitHub registry](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ba887642f1bbbd374e8083abde87e84b098b20de)
- [Set the version to 4.9.1.post1](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/ec67f8d68e98ec0587fdbb87d16213a856f6dbfe)

## Wav2Vec2 test (2026-10-15)

Since 4.8.4, upstream has the same fix as this fork for Wav2Vec2 and Wav2Vec2-BERT models on Intel CPUs, where MKL packs their weights, so only the fork's test remains.

- [Test Wav2Vec2 on the CPU with MKL-packed weights](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/651694cddf5dba457e55255768ca422705d50765)

## Docker images removed (2026-10-15)

- [Remove the Docker images and their CI job](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/9a7aca1c9ba1e095033c2b63a6efc9b58da1c33a)

## CUDA 13 (2026-10-16)

The wheels are built with CUDA 13.0.3 instead of 12.8. That drops GPUs older than Turing and needs NVIDIA driver 580 or newer, and `pip install ctranslate2-reborn[cuda]` also installs cuBLAS 13. The wheels also include native code for Ada, Hopper and Blackwell GPUs; since upstream's 4.9.0 builds CUDA code with CMake's own CUDA support, the list of GPU architectures is set in the wheel build scripts.

- [Upgrade CCCL to 3.0.1 for CUDA 13 support](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/c21f8f6982085e76d679d9aacee104fd1197c751)
- [Find CUDA 13 DLLs on Windows](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/7b74eb21d697124588517a185db43992c956b3d8)
- [Build the CI wheels with CUDA 13.0.3](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/30014ac952c777ba9a312a406a0e29f3a93a5c1c)
- [Add a cuda extra that installs cuBLAS from NVIDIA's pip packages](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e4d66589de75f5385361005bba5f2aa3680dd251)
- [Load cuBLAS from PyTorch's folder on Windows when the cuda extra is missing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/4a7e4607f0e1e974826c6a9b6c09486e94cda7fd)
- [Prefer PyTorch's cuBLAS on Windows over NVIDIA's pip copy](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/dff019f78a79d636e0d22e545789665759127eee)

## Python 3.9 dropped (2026-10-16)

- [Drop Python 3.9](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/7c06424d21eca4ae807b7ef1b7f99718097c8a67)

## CUDA 12.8 wheels on GitHub releases (2026-10-17)

PyPI keeps the CUDA 13 wheels. Each GitHub release also gets CUDA 12.8 wheels for Linux and Windows, labeled `+cu128`, for GPUs older than Turing and NVIDIA drivers older than 580.

- [Build CUDA 12.8 wheels and attach them to GitHub releases](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/07e655ece22e0b798bc47304595cf1568a21e2ea)

## Empty batches (2026-10-18)

An empty batch no longer crashes the process on the CPU or raises misleading errors on CUDA: `Generator.forward_batch`, `Encoder.forward_batch`, Whisper's `encode` and `detect_language` and the Wav2Vec2 and Wav2Vec2-BERT encoders return empty outputs with the shape a non-empty call has.

- [Return empty outputs for empty batches instead of crashing](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/3f27d0f4af8c02f824decf682c0205f3d30a1432)
- [Test empty batches on every model type](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/6bd685ef2f472c8824833587d60a17942a0a0c29)

## BERT-family converter tests (2026-10-18)

Since 4.9.1, upstream has the same fix as this fork for converting BERT, RoBERTa, XLM-RoBERTa and CamemBERT models with Transformers 5, so only the fork's tests remain.

- [Test the BERT-family converters on tiny models](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/bfcee4f9da37becaac83bd7523dcefa69ec10202)

## Qwen3-Embedding (2026-10-18)

Qwen3-Embedding models, and other Qwen3-based embedding models such as Yuan-embedding-2.0-en and harrier-oss-v1-0.6b, convert with `--model_type Qwen3Embedding` and run with `ctranslate2.Encoder`, whose `pooler_output` is each text's last-token embedding, unnormalized. Models that add layers to Qwen3 are refused instead of being converted without them.

- [Add Qwen3-Embedding support (decoder-as-encoder)](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/5e95f0ef45b47f9e0540b88fe755372c73ca6363)
- [Return last-token pooled output for decoder-based encoders](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e22ffcd5ec09a4261632a99ead2a24acd1bea56e)
- [Format the Qwen3-Embedding loader with black](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/05862d0d0bdfc1f7295c7e5be250ab30ef3910a1)
- [Skip packing the output projection when loading a decoder for an encoder](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/f95f9d19047ec0175754fa1461c7a316809bb2c8)
- [Return empty outputs for an empty batch in decoder-based encoders](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/39fb56d24e2c42cea4722d0c64ae0ebb193da126)
- [Test Qwen3-Embedding conversion of both checkpoint layouts](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/57285a77edc9468dfb70e30f6ae5a006a97efd40)
- [Refuse to convert Qwen3 models that add layers as Qwen3Embedding](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/e0386a8f52b360dadf5e13f5f8793785ff48aa3b)
- [Test an empty batch on a Qwen3-Embedding model](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/cededd95ccab54cdee4ec4a1f7be954a8f0ccd45)

## Apple Metal backend (2026-11-11)

Upstream's 4.9.0 adds an experimental backend for Apple GPUs (the `mps` device). This fork's speed changes are written for CUDA and the CPU; on Metal, models run through upstream's code paths.

- [Keep upstream's code paths on Apple's Metal backend](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/1ab0cf1101c5c81291ea20276d919ae8b5aeb27a)

## Release gate (2026-11-11)

Upstream's 4.9.0 adds a release gate: before a release is published, CI compares the new wheels with the latest release on PyPI, on the CPU, and fails on worse accuracy or a confirmed 10% drop in speed. Here the gate compares with the last release of `ctranslate2-reborn` instead of upstream's, and runs when a GitHub release is published or the workflow is started from the Actions tab.

- [Compare the release gate with ctranslate2-reborn's last release](https://github.com/BBC-Esq/CTranslate2-Reborn/commit/566e955712db00b0d18f9f709e3236041239848f)

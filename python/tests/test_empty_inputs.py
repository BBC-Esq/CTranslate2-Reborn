import json
import os
import shutil

import numpy as np
import pytest
import test_utils

import ctranslate2

# An empty batch used to reach layers that don't take empty inputs: the process crashed on
# the CPU and CUDA raised misleading errors. Each entry point now returns empty outputs with
# the shape a non-empty call has. The models are tiny random ones, except Whisper.


def teardown_module():
    if os.environ.get("CI") == "true":
        from huggingface_hub import constants

        shutil.rmtree(constants.HF_HUB_CACHE, ignore_errors=True)


def _to_numpy(storage):
    if storage.device != "cpu":
        storage = storage.to_device(ctranslate2.Device.cpu)
    return np.array(storage)


def _features(shape):
    return ctranslate2.StorageView.from_array(np.zeros(shape, dtype=np.float32))


def _save_word_level_tokenizer(model_dir, tokens, **special_tokens):
    import tokenizers
    import transformers

    vocabulary = {token: i for i, token in enumerate(tokens)}
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel(vocabulary, unk_token="<unk>")
        ),
        unk_token="<unk>",
        **special_tokens,
    )
    tokenizer.save_pretrained(model_dir)


@test_utils.only_on_linux
@test_utils.on_available_devices
def test_generator_forward_empty_batches(tmp_dir, device):
    import torch
    import transformers

    model_dir = str(tmp_dir.join("model"))
    tokens = ["<unk>", "<s>", "</s>"] + ["tok%d" % i for i in range(61)]
    config = transformers.LlamaConfig(
        vocab_size=len(tokens),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(config).save_pretrained(model_dir)
    _save_word_level_tokenizer(model_dir, tokens, bos_token="<s>", eos_token="</s>")
    output_dir = ctranslate2.converters.TransformersConverter(model_dir).convert(
        str(tmp_dir.join("ctranslate2_model"))
    )

    generator = ctranslate2.Generator(output_dir, device=device)
    vocabulary_size = generator.forward_batch([[1, 5, 9]]).shape[-1]

    for batch, shape in (([], [0, 0]), ([[]], [1, 0]), ([[], []], [2, 0])):
        for return_log_probs in (False, True):
            logits = generator.forward_batch(batch, return_log_probs=return_log_probs)
            assert logits.shape == shape + [vocabulary_size]
            assert _to_numpy(logits).shape == tuple(shape + [vocabulary_size])

    ids = ctranslate2.StorageView.from_array(np.zeros((0, 0), dtype=np.int32))
    lengths = ctranslate2.StorageView.from_array(np.zeros((0,), dtype=np.int32))
    assert generator.forward_batch(ids, lengths).shape == [0, 0, vocabulary_size]


@test_utils.only_on_linux
@test_utils.on_available_devices
def test_encoder_forward_empty_batches(tmp_dir, device):
    import torch
    import transformers

    model_dir = str(tmp_dir.join("model"))
    os.makedirs(model_dir)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocabulary += ["w%d" % i for i in range(59)]
    vocabulary_path = os.path.join(model_dir, "vocab.txt")
    with open(vocabulary_path, "w", encoding="utf-8") as vocabulary_file:
        vocabulary_file.write("\n".join(vocabulary) + "\n")
    config = transformers.BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    torch.manual_seed(0)
    transformers.BertModel(config).save_pretrained(model_dir)
    transformers.BertTokenizer(vocabulary_path).save_pretrained(model_dir)
    output_dir = ctranslate2.converters.TransformersConverter(model_dir).convert(
        str(tmp_dir.join("ctranslate2_model"))
    )

    encoder = ctranslate2.Encoder(output_dir, device=device)
    sample = encoder.forward_batch([["[CLS]", "w1", "[SEP]"]])
    hidden_size = sample.pooler_output.shape[-1]

    output = encoder.forward_batch([])
    assert output.last_hidden_state.shape == [0, 0, hidden_size]
    assert output.pooler_output.shape == [0, hidden_size]
    assert _to_numpy(output.last_hidden_state).shape == (0, 0, hidden_size)
    assert _to_numpy(output.pooler_output).shape == (0, hidden_size)

    # Sequences without a token have no first token to pool.
    with pytest.raises(ValueError, match="at least one token"):
        encoder.forward_batch([[]])


@test_utils.only_on_linux
@test_utils.on_available_devices
def test_qwen3_embedding_empty_batches(tmp_dir, device):
    import torch
    import transformers

    model_dir = str(tmp_dir.join("model"))
    tokens = ["<unk>", "<|endoftext|>", "<|im_end|>"] + ["tok%d" % i for i in range(61)]
    config = transformers.Qwen3Config(
        vocab_size=len(tokens),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    transformers.Qwen3Model(config).save_pretrained(model_dir)
    _save_word_level_tokenizer(
        model_dir, tokens, pad_token="<|endoftext|>", eos_token="<|im_end|>"
    )
    converter = ctranslate2.converters.TransformersConverter(
        model_dir, model_type="Qwen3Embedding"
    )
    output_dir = converter.convert(str(tmp_dir.join("ctranslate2_model")))

    encoder = ctranslate2.Encoder(output_dir, device=device)
    hidden_size = encoder.forward_batch([[5, 9, 14]]).pooler_output.shape[-1]

    output = encoder.forward_batch([])
    assert output.last_hidden_state.shape == [0, 0, hidden_size]
    assert output.pooler_output.shape == [0, hidden_size]
    assert _to_numpy(output.pooler_output).shape == (0, hidden_size)

    # The last token is pooled, so every sequence needs one.
    with pytest.raises(ValueError, match="at least one token"):
        encoder.forward_batch([[]])


@test_utils.only_on_linux
@test_utils.on_available_devices
def test_whisper_empty_batches(tmp_dir, device):
    converter = ctranslate2.converters.TransformersConverter("openai/whisper-tiny")
    output_dir = converter.convert(str(tmp_dir.join("ctranslate2_model")))
    whisper = ctranslate2.models.Whisper(output_dir, device=device)

    expected_shape = whisper.encode(_features((1, 80, 3000))).shape
    output = whisper.encode(_features((0, 80, 3000)), to_cpu=True)
    assert output.shape == [0] + expected_shape[1:]
    assert np.array(output).shape == tuple([0] + expected_shape[1:])

    assert whisper.detect_language(_features((0, 80, 3000))) == []
    assert whisper.generate(_features((0, 80, 3000)), []) == []


@test_utils.only_on_linux
@test_utils.on_available_devices
def test_wav2vec2_empty_batches(tmp_dir, device):
    import torch
    import transformers

    model_dir = str(tmp_dir.join("model"))
    vocabulary_path = str(tmp_dir.join("vocab.json"))
    with open(vocabulary_path, "w", encoding="utf-8") as vocabulary_file:
        json.dump({"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4}, vocabulary_file)
    tokenizer = transformers.Wav2Vec2CTCTokenizer(vocabulary_path)
    config = transformers.Wav2Vec2Config(
        vocab_size=len(tokenizer.get_vocab()),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
        conv_dim=(32,) * 7,
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_bias=True,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
    )
    torch.manual_seed(0)
    transformers.Wav2Vec2ForCTC(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    output_dir = ctranslate2.converters.TransformersConverter(model_dir).convert(
        str(tmp_dir.join("ctranslate2_model"))
    )

    model = ctranslate2.models.Wav2Vec2(output_dir, device=device)
    expected_shape = model.encode(_features((1, 1, 16000)), to_cpu=True).shape
    output = model.encode(_features((0, 1, 16000)), to_cpu=True)
    assert output.shape == [0] + expected_shape[1:]
    assert np.array(output).shape == tuple([0] + expected_shape[1:])


@test_utils.only_on_linux
@test_utils.on_available_devices
def test_wav2vec2bert_empty_batches(tmp_dir, device):
    import torch
    import transformers

    model_dir = str(tmp_dir.join("model"))
    vocabulary_path = str(tmp_dir.join("vocab.json"))
    with open(vocabulary_path, "w", encoding="utf-8") as vocabulary_file:
        json.dump({"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4}, vocabulary_file)
    tokenizer = transformers.Wav2Vec2CTCTokenizer(vocabulary_path)
    # The runtime expects 16 attention heads and the hidden size of the released model.
    config = transformers.Wav2Vec2BertConfig(
        vocab_size=len(tokenizer.get_vocab()),
        hidden_size=1024,
        num_hidden_layers=1,
        num_attention_heads=16,
        intermediate_size=256,
        feature_projection_input_dim=160,
        add_adapter=True,
        num_adapter_layers=1,
        output_hidden_size=1024,
    )
    torch.manual_seed(0)
    transformers.Wav2Vec2BertForCTC(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    output_dir = ctranslate2.converters.TransformersConverter(model_dir).convert(
        str(tmp_dir.join("ctranslate2_model"))
    )

    model = ctranslate2.models.Wav2Vec2Bert(output_dir, device=device)
    expected_shape = model.encode(_features((1, 100, 160)), to_cpu=True).shape
    output = model.encode(_features((0, 100, 160)), to_cpu=True)
    assert output.shape == [0] + expected_shape[1:]
    assert np.array(output).shape == tuple([0] + expected_shape[1:])

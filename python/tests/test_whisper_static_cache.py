import numpy as np
import pytest

import ctranslate2

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")


_TEXT_TOKENS = ["t%d" % i for i in range(48)]
_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|en|>",
    "<|fr|>",
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nocaptions|>",
    "<|notimestamps|>",
]
_NUM_TIMESTAMPS = 40
_TOKENS = _TEXT_TOKENS + _SPECIAL_TOKENS
_EOT, _SOT, _EN, _FR, _TRANSLATE, _TRANSCRIBE, _, _PREV, _, _NO_TIMESTAMPS = range(
    len(_TEXT_TOKENS), len(_TOKENS)
)


class _Tokenizer:
    """Stands in for the Whisper tokenizer, which converting random weights doesn't need."""

    additional_special_tokens = _SPECIAL_TOKENS[1:]

    def __init__(self, tokens):
        self._ids = {token: i for i, token in enumerate(tokens)}

    def get_vocab(self):
        return dict(self._ids)

    def convert_tokens_to_ids(self, token):
        return self._ids[token]


@pytest.fixture(scope="module")
def tiny_whisper_dir(tmp_path_factory):
    """Converts a small Whisper model with random weights, built without downloads."""
    from ctranslate2.converters.transformers import WhisperLoader

    config = transformers.WhisperConfig(
        vocab_size=len(_TOKENS) + _NUM_TIMESTAMPS,
        num_mel_bins=80,
        d_model=64,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_layers=2,
        decoder_attention_heads=4,
        decoder_ffn_dim=128,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=_EOT,
        bos_token_id=_EOT,
        eos_token_id=_EOT,
        decoder_start_token_id=_SOT,
        suppress_tokens=[1, 2],
        begin_suppress_tokens=[_EOT],
        # Larger weights than the default make the predictions less uniform.
        init_std=0.2,
    )

    torch.manual_seed(0)
    model = transformers.WhisperForConditionalGeneration(config).eval()

    # The end of text gets the weights of a token this model often predicts, so that the
    # hypotheses end at different steps and the batch shrinks during the decoding.
    with torch.no_grad():
        weights = model.model.decoder.embed_tokens.weight
        weights[_EOT] = 1.02 * weights[68]

    spec = WhisperLoader()(model, _Tokenizer(_TOKENS))
    spec.validate()
    spec.optimize()

    output_dir = str(tmp_path_factory.mktemp("tiny_whisper"))
    spec.save(output_dir)
    return output_dir


def _get_features(batch_size):
    features = np.random.RandomState(1).randn(batch_size, 80, 3000).astype(np.float32)
    return ctranslate2.StorageView.from_array(features)


def _assert_same_results(results, expected_results):
    assert len(results) == len(expected_results)
    for result, expected in zip(results, expected_results):
        assert result.sequences_ids == expected.sequences_ids
        np.testing.assert_allclose(result.scores, expected.scores, rtol=0, atol=1e-5)
        assert result.no_speech_prob == pytest.approx(expected.no_speech_prob, abs=1e-6)


_PROMPTS = {
    "task": [[_SOT, _EN, _TRANSCRIBE, _NO_TIMESTAMPS]] * 3,
    "timestamps": [[_SOT, _EN, _TRANSCRIBE], [_SOT, _FR, _TRANSLATE]],
    "previous-text": [
        [_PREV, 3, 4, 5, _SOT, _EN, _TRANSCRIBE, _NO_TIMESTAMPS],
        [_PREV, 9, 8, 7, _SOT, _FR, _TRANSCRIBE, _NO_TIMESTAMPS],
    ],
    "prefix": [
        [_SOT, _EN, _TRANSCRIBE, _NO_TIMESTAMPS, 7, 8, 9],
        [_SOT, _FR, _TRANSCRIBE, _NO_TIMESTAMPS, 1],
        [_SOT, _EN, _TRANSCRIBE, _NO_TIMESTAMPS],
    ],
    "sot-only": [[_SOT]] * 2,
}

_OPTIONS = {
    # The end of text is suppressed so that the decoding goes to the maximum length.
    "greedy-to-max-length": dict(beam_size=1, suppress_tokens=[_EOT], return_scores=True),
    "greedy": dict(beam_size=1, return_scores=True, return_no_speech_prob=True),
    "beam": dict(beam_size=5, return_scores=True),
    "beam-to-max-length": dict(beam_size=3, suppress_tokens=[_EOT], return_scores=True),
    "beam-hypotheses": dict(
        beam_size=4, num_hypotheses=3, patience=2, length_penalty=0.6, return_scores=True
    ),
    "penalties": dict(beam_size=3, repetition_penalty=1.3, no_repeat_ngram_size=3),
    "short": dict(beam_size=2, max_length=12, return_scores=True),
}


@pytest.mark.parametrize("prompts_name", list(_PROMPTS))
@pytest.mark.parametrize("options_name", list(_OPTIONS))
def test_whisper_static_kv_cache(tiny_whisper_dir, prompts_name, options_name):
    prompts = _PROMPTS[prompts_name]
    options = _OPTIONS[options_name]
    model = ctranslate2.models.Whisper(tiny_whisper_dir, device="cpu")
    features = _get_features(len(prompts))

    expected = model.generate(features, prompts, **options)
    results = model.generate(features, prompts, static_kv_cache=True, **options)
    _assert_same_results(results, expected)

    # The next generation grows the cache again.
    _assert_same_results(model.generate(features, prompts, **options), expected)


def test_whisper_static_kv_cache_sampling(tiny_whisper_dir):
    # Each model samples with the generator of its thread, seeded when it is first used, so
    # both models draw the same random numbers.
    prompts = _PROMPTS["task"]
    features = _get_features(len(prompts))
    options = dict(beam_size=1, sampling_topk=10, sampling_temperature=1.5, return_scores=True)

    ctranslate2.set_random_seed(42)
    growing_model = ctranslate2.models.Whisper(tiny_whisper_dir, device="cpu")
    static_model = ctranslate2.models.Whisper(tiny_whisper_dir, device="cpu")

    # Several hypotheses replicate the cache for each of them.
    for num_hypotheses in (1, 3):
        expected = growing_model.generate(
            features, prompts, num_hypotheses=num_hypotheses, **options
        )
        results = static_model.generate(
            features, prompts, num_hypotheses=num_hypotheses, static_kv_cache=True, **options
        )
        _assert_same_results(results, expected)


def test_whisper_static_kv_cache_then_align(tiny_whisper_dir):
    model = ctranslate2.models.Whisper(tiny_whisper_dir, device="cpu")
    features = _get_features(2)
    start_sequence = [_SOT, _EN, _TRANSCRIBE, _NO_TIMESTAMPS]
    text_tokens = [[3, 4, 5], [6, 7]]

    expected = model.align(features, start_sequence, text_tokens, 3000)
    model.generate(features, [start_sequence] * 2, static_kv_cache=True)
    results = model.align(features, start_sequence, text_tokens, 3000)

    for result, expected_result in zip(results, expected):
        assert result.alignments == expected_result.alignments
        np.testing.assert_allclose(
            result.text_token_probs, expected_result.text_token_probs, rtol=0, atol=1e-6
        )

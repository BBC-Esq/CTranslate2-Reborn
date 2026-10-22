#include "ctranslate2/layers/transformer.h"

#include <cmath>

namespace ctranslate2 {
  namespace layers {

    // The gated activation op applies the activation and the product in one pass. Quantized
    // layers keep applying the activation themselves.
    static bool use_gated_activation_op(const models::Model& model, const std::string& scope) {
      if (!model.layer_exists(scope + "/linear_0_noact"))
        return false;
      const StorageView* weight = model.get_variable_if_exists(scope + "/linear_0/weight");
      return weight && is_float_type(weight->dtype());
    }

    FeedForwardNetwork::FeedForwardNetwork(const models::Model& model,
                                           const std::string& scope,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
      : _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _pre_norm(pre_norm)
      , _activation_type(activation_type)
      , _gated_activation(use_gated_activation_op(model, scope))
      , _ff1(model, scope + "/linear_0", _gated_activation ? nullptr : &_activation_type)
      , _ff1_noact(build_optional_layer<Dense>(model, scope + "/linear_0_noact"))
      , _ff2(model, scope + "/linear_1", nullptr, true)
      , _tensor_parallel(model.tensor_parallel()) {
    }

    void FeedForwardNetwork::operator()(const StorageView& input, StorageView& output) const {
      const StorageView* x = &input;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(input, output);
        x = &output;
      }

      const Device device = input.device();
      const DataType dtype = input.dtype();

      StorageView inner(dtype, device);
      _ff1(*x, inner);
      if (_ff1_noact) {
        StorageView linear(dtype, device);
        (*_ff1_noact)(*x, linear);
        if (_gated_activation) {
          const ops::GatedActivation gated_activation_op(_activation_type);
          gated_activation_op(inner, linear, inner);
        } else
          ops::Mul()(linear, inner, inner);
      }

      _ff2(inner, output, _layer_norm ? &input : nullptr);

      if (_tensor_parallel) {
        Shape shape = output.shape();
        StorageView tmp(std::move(shape), output.dtype(), output.device());
        ops::ReduceAll red_op(ops::ReduceAll::RED_OP::SUM);
        red_op(output, tmp);
        output = std::move(tmp);
      }

      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type,
                                                     const bool use_flash_attention)
      : _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_feedforward_layer_norm"))
      , _ff(model, scope + "/ffn", pre_norm, activation_type) {
    }


    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView* lengths,
                                             StorageView& output,
                                             const Padder* padder,
                                             StorageView* position_bias) const {
      PROFILE("TransformerEncoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      // Check if using pre_post_layer_norm pattern (T5Gemma style)
      const bool pre_post_layer_norm = _input_layer_norm && _post_attention_layer_norm
                                        && _pre_feedforward_layer_norm && _post_feedforward_layer_norm;

      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);

        (*_input_layer_norm)(input, hidden);

        if (_self_attention)
          (*_self_attention)(hidden,
                          hidden,
                          lengths,
                          context,
                          nullptr,
                          nullptr,
                          nullptr,
                          padder,
                          padder,
                          true,
                          position_bias);

        // post_self_attn_layernorm
        (*_post_attention_layer_norm)(context, output);

        // residual + hidden_states
        ops::Add()(input, output, output);

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        // mlp
        _ff(hidden, output);

        // post_feedforward_layernorm
        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);

        // residual + hidden_states
        ops::Add()(context, output, output);
        return;
      }

      // Original path for standard pre-norm/post-norm architectures
      StorageView context(dtype, device);
      if (_self_attention)
        (*_self_attention)(input,
                        input,
                        lengths,
                        context,
                        nullptr,
                        nullptr,
                        nullptr,
                        padder,
                        padder,
                        true,
                        position_bias);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type,
                                                     const bool use_flash_attention,
                                                     Alibi* alibi)
      : _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true,
                        alibi)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true,
                        alibi)))
      , _shared_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/shared_layer_norm"))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_feedforward_layer_norm"))
      , _encoder_attention(build_optional_layer<MultiHeadAttention>(model,
                                                                    scope + "/attention",
                                                                    num_heads,
                                                                    /*self_attention=*/false,
                                                                    pre_norm,
                                                                    /*is_decoder=*/true))
      , _ff(model, scope + "/ffn", pre_norm, activation_type)
      , _external_pre_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_pre_encoder_attention_layer_norm"))
      , _external_post_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_post_encoder_attention_layer_norm"))
      , _layer_scalar(model.get_attribute_with_default<float>(scope + "/layer_scalar", 1.f))
      , _has_merged_encoder_attention(
          !use_flash_attention
          && static_cast<MultiHeadAttention*>(_self_attention.get())->has_merged_encoder_attention())
      {
    }

    void TransformerDecoderLayer::operator()(const StorageView& input,
                                             const StorageView* input_length,
                                             const StorageView* memory,
                                             const StorageView* memory_lengths,
                                             StorageView* cached_self_attn_keys,
                                             StorageView* cached_self_attn_values,
                                             StorageView* cached_attn_keys,
                                             StorageView* cached_attn_values,
                                             StorageView& output,
                                             StorageView* attention,
                                             const Padder* input_padder,
                                             const Padder* memory_padder,
                                             bool return_normalized_attention,
                                             StorageView* position_bias,
                                             dim_t offset,
                                             dim_t static_cache_capacity) const {
      PROFILE("TransformerDecoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      const bool pre_post_layer_norm = _post_feedforward_layer_norm && _pre_feedforward_layer_norm;
      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);
        (*_input_layer_norm)(input, hidden);

        if (_has_merged_encoder_attention) {
          if (static_cache_capacity > 0)
            throw std::invalid_argument("A static cache is not supported with merged encoder "
                                        "attention");
          static_cast<MultiHeadAttention*>(_self_attention.get())->forward_merged(
            hidden,
            memory,
            memory_lengths,
            input_length,
            context,
            cached_self_attn_keys,
            cached_self_attn_values,
            cached_attn_keys,
            cached_attn_values,
            input_padder,
            memory_padder,
            offset);
        } else if (_self_attention) {
          (*_self_attention)(hidden,
                             hidden,
                             input_length,
                             context,
                             cached_self_attn_keys,
                             cached_self_attn_values,
                             nullptr,
                             input_padder,
                             input_padder,
                             true,
                             position_bias,
                             offset,
                             static_cache_capacity);
        }
        (*_post_attention_layer_norm)(context, output);
        ops::Add()(output, input, output);

        if (_encoder_attention) {
            StorageView cross_attn_in = output;  // save for residual

            StorageView query_normalized(dtype, device);
            if (_external_pre_encoder_attention_layer_norm) {
                (*_external_pre_encoder_attention_layer_norm)(output, query_normalized);
            }
            else {
                query_normalized.shallow_copy(output);
            }

            (*_encoder_attention)(query_normalized,
                                  *memory,
                                  memory_lengths,
                                  context,
                                  cached_attn_keys,
                                  cached_attn_values,
                                  attention,
                                  input_padder,
                                  memory_padder,
                                  return_normalized_attention);

            if (_external_post_encoder_attention_layer_norm) {
                (*_external_post_encoder_attention_layer_norm)(context, context);
            }
            ops::Add()(context, cross_attn_in, output);
        }

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        _ff(hidden, output);

        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);
        ops::Add()(output, context, output);

        // Gemma 4 layer scalar
        if (_layer_scalar != 1.f)
          ops::Mul()(output, StorageView(_layer_scalar).to(dtype), output);

        return;
      }

      const bool use_parallel_residual = _shared_layer_norm || _input_layer_norm;

      if (use_parallel_residual) {
        // The parallel residual implementation assumes there is no cross attention.
        StorageView hidden(dtype, device);

        if (_shared_layer_norm)
          (*_shared_layer_norm)(input, hidden);
        else
          (*_input_layer_norm)(input, hidden);

        StorageView attn(dtype, device);
        if (_self_attention)
          (*_self_attention)(hidden,
                        hidden,
                        input_length,
                        attn,
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        nullptr,
                        input_padder,
                        input_padder,
                        true,
                        position_bias,
                        offset,
                        static_cache_capacity);

        if (_post_attention_layer_norm)
          (*_post_attention_layer_norm)(input, hidden);

        _ff(hidden, output);

        ops::Add()(output, input, output);
        ops::Add()(output, attn, output);

        return;
      }
      if (_self_attention)
        (*_self_attention)(input,
                      input,
                      input_length,
                      output,
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      nullptr,
                      input_padder,
                      input_padder,
                      true,
                      position_bias,
                      offset,
                      static_cache_capacity);

      StorageView context(dtype, device);
      if (_encoder_attention) {
        (*_encoder_attention)(output,
                              *memory,
                              memory_lengths,
                              context,
                              cached_attn_keys,
                              cached_attn_values,
                              attention,
                              input_padder,
                              memory_padder,
                              return_normalized_attention);
      }
      else {
        context = std::move(output);
      }

      _ff(context, output);
    }


    static std::unique_ptr<PositionEncoder>
    build_position_encoder(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      if (model.get_variable_if_exists(scope + "/encodings"))
        return std::make_unique<PositionEmbedding>(model, scope);
      else
        return std::make_unique<SinusoidalPositionEncoder>(embeddings.output_size(),
                                                           embeddings.output_type(),
                                                           model.device());
    }

    static std::unique_ptr<const StorageView>
    build_embeddings_scale(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      const auto* scale = model.get_variable_if_exists(scope + "/scale_embeddings");

      // Backward compatibility with older models.
      if (!scale)
        scale = model.get_variable_if_exists(scope + "/embeddings/multiply_by_sqrt_depth");

      StorageView value;

      // The attribute can either be a boolean flag or the actual scale value.
      if (!scale || (scale->dtype() == DataType::INT8 && scale->as_scalar<int8_t>()))
        value = StorageView(std::sqrt(static_cast<float>(embeddings.output_size())));
      else if (scale->dtype() != DataType::INT8 && scale->as_scalar<float>() != 1.f)
        value = *scale;
      else
        return nullptr;

      return std::make_unique<StorageView>(value.to(embeddings.output_type()));
    }


    // T5-style models compute the relative attention bias in the first layer and share it
    // with the following layers. UMT5-style models give every layer its own bias table.
    // Identical tables are serialized as aliases resolving to the same StorageView, so
    // comparing the resolved pointers tells the two cases apart.
    static bool has_shared_position_bias(const models::Model& model,
                                         const std::string& scope,
                                         const size_t num_layers) {
      const auto bias_name = [&scope](size_t i) {
        return scope + "/layer_" + std::to_string(i) + "/self_attention/relative_attention_bias";
      };

      const StorageView* first = model.get_variable_if_exists(bias_name(0));
      if (!first)
        return true;  // no relative attention bias: the shared buffer is never filled

      for (size_t i = 1; i < num_layers; ++i) {
        if (model.get_variable_if_exists(bias_name(i)) != first)
          return false;
      }

      return true;
    }


    TransformerEncoder::TransformerEncoder(const models::Model& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings",
                    model.get_enum_value<EmbeddingsMerge>(scope + "/embeddings_merge"))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _use_flash_attention(model.use_flash_attention())
      , _layers(build_layers_list<const TransformerEncoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation")))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _tensor_parallel(model.tensor_parallel())
      , _shared_position_bias(has_shared_position_bias(model, scope, _layers.size()))
    {
    }

    void TransformerEncoder::operator()(const std::vector<StorageView>& ids,
                                        const StorageView* lengths,
                                        StorageView& output) {
      PROFILE("TransformerEncoder");
      StorageView input(output.dtype(), output.device());
      _embeddings(ids, input);
      if (_embeddings_scale)
        ops::Mul()(input, *_embeddings_scale, input);
      if (_position_encoder)
        (*_position_encoder)(input);
      if (_layernorm_embedding)
        (*_layernorm_embedding)(input, input);

      const dim_t max_time = input.dim(1);

      // Remove padding to reduce the amount of computation.
      std::unique_ptr<Padder> padder;
      std::unique_ptr<StorageView> lengths_mask;

      if (lengths) {
        if (Padder::allow_padding_removal(output.device(), _compute_type)) {
          padder = std::make_unique<Padder>(*lengths, max_time, 1,
                                            Padder::min_padding_to_remove(output.device()));
          padder->remove_padding(input);
        }

        int num_heads = _num_heads;
        if (_tensor_parallel) {
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
        }
        lengths_mask = std::make_unique<StorageView>(
          layers::MultiHeadAttention::prepare_length_mask(*lengths, num_heads, max_time));
      }

      StorageView position_bias(output.dtype(), output.device());
      StorageView* position_bias_ptr = _shared_position_bias ? &position_bias : nullptr;

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths_mask.get(), output, padder.get(), position_bias_ptr);
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
      if (_output_norm)
        (*_output_norm)(output, output);
      if (padder)
        padder->add_padding(output);
    }


    static std::unique_ptr<Alibi> make_alibi(const models::Model& model, const std::string& scope) {
      const bool use_alibi = model.get_flag_with_default(scope + "/alibi", false);
      if (!use_alibi)
        return nullptr;

      const bool use_positive_positions = model.get_flag_with_default(
        scope + "/alibi_use_positive_positions", true);
      const bool scale_alibi = model.get_flag_with_default(
        scope + "/scale_alibi", false);

      return std::make_unique<Alibi>(use_positive_positions, scale_alibi);
    }

    TransformerDecoder::TransformerDecoder(const models::Model& model, const std::string& scope)
      : Decoder(model.device())
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _start_from_zero_embedding(model.get_flag_with_default(scope + "/start_from_zero_embedding",
                                                               false))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _project_in(build_optional_layer<Dense>(model, scope + "/project_in"))
      , _project_out(build_optional_layer<Dense>(model, scope + "/project_out"))
      , _alibi(make_alibi(model, scope))
      , _use_flash_attention(model.use_flash_attention())
      , _layers(build_layers_list<const TransformerDecoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation"),
                  _use_flash_attention,
                  _alibi.get()))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _with_encoder_attention(_layers.front()->has_cross_attention())
      , _proj(model, scope + "/projection")
      , _sliding_window(model.get_attribute_with_default<int32_t>(scope + "/sliding_window", 0))
      , _tensor_parallel(model.tensor_parallel())
      , _final_logit_softcapping(model.get_attribute_with_default<float>(scope + "/final_logit_softcapping", 0.f))
      , _shared_position_bias(has_shared_position_bias(model, scope, _layers.size())) {

      dim_t alignment_layer = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_layer", -1));
      dim_t alignment_heads = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_heads", 1));

      if (alignment_layer < 0)
        alignment_layer = _layers.size() + alignment_layer;
      if (alignment_heads == 0)
        alignment_heads = _num_heads;

      set_alignment_heads(alignment_layer, alignment_heads);

      const auto* outputs_scale = model.get_variable_if_exists(scope + "/scale_outputs");
      if (outputs_scale) {
        const DataType dtype = get_default_float_type(_compute_type);
        _outputs_scale = std::make_unique<StorageView>(outputs_scale->to(dtype));
      }
    }

    DecoderState TransformerDecoder::initial_state(bool iterative_decoding) const {
      DecoderState state;

      if (iterative_decoding) {
        const size_t state_size = _layers.size() * (_with_encoder_attention ? 4 : 2);
        state.reserve(state_size);

        const DataType dtype = output_type();

        for (size_t i = 0; i < _layers.size(); ++i) {
          const std::string i_str = std::to_string(i);
          state.emplace("self_keys_" + i_str, StorageView(dtype, _device));
          state.emplace("self_values_" + i_str, StorageView(dtype, _device));
          if (_with_encoder_attention) {
            state.emplace("memory_keys_" + i_str, StorageView(dtype, _device));
            state.emplace("memory_values_" + i_str, StorageView(dtype, _device));
          }
        }
      }

      return state;
    }

    bool TransformerDecoder::replicate_state(const std::string& name) const {
      // No need to replicate projected memory keys and values as they are the same for each beam.
      return !_with_encoder_attention || !starts_with(name, "memory");
    }

    void TransformerDecoder::set_static_cache_capacity(dim_t capacity) {
      if (capacity < 0)
        throw std::invalid_argument("The static cache capacity must be positive, or 0 to disable "
                                    "the static cache");

      if (capacity > 0) {
        if (_sliding_window > 0)
          throw std::invalid_argument("A static cache is not supported with sliding window "
                                      "attention");
        if (_use_flash_attention)
          throw std::invalid_argument("A static cache is not supported with Flash Attention");
        for (const auto& layer : _layers) {
          if (layer->has_merged_encoder_attention())
            throw std::invalid_argument("A static cache is not supported with merged encoder "
                                        "attention");
        }
      }

      _static_cache_capacity = capacity;
      _static_cache_length = 0;
    }

    void TransformerDecoder::gather_state(const std::string& name,
                                          StorageView& value,
                                          const StorageView& indices) const {
      // Only the first positions of a static cache were written, so only they are gathered.
      // The time dimension of the cache is the one before the depth, whatever its layout.
      const bool is_static_cache = (_static_cache_capacity > 0
                                    && value
                                    && value.rank() >= 3
                                    && value.dim(-2) == _static_cache_capacity
                                    && (starts_with(name, "self_keys")
                                        || starts_with(name, "self_values")));

      if (is_static_cache)
        ops::GatherPrefix(value.rank() - 2, _static_cache_length)(value, indices);
      else
        Decoder::gather_state(name, value, indices);
    }

    void TransformerDecoder::set_alignment_heads(const dim_t layer,
                                                 const dim_t num_heads_to_average) {
      std::vector<dim_t> range(num_heads_to_average);
      std::iota(range.begin(), range.end(), dim_t(0));

      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      _alignment_heads[layer] = std::move(range);

      _average_alignment_heads = true;
    }

    void TransformerDecoder::set_alignment_heads(const std::vector<std::pair<dim_t, dim_t>>& alignment_heads) {
      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      for (const auto& [layer, head] : alignment_heads)
        _alignment_heads[layer].push_back(head);

      _average_alignment_heads = false;
    }

    std::unique_ptr<StorageView>
    TransformerDecoder::get_layer_alignment_heads(const dim_t layer, const dim_t batch_size) const {
      if (_alignment_heads.empty())
        return nullptr;

      const auto& heads = _alignment_heads[layer];
      const dim_t num_heads = heads.size();

      if (heads.empty())
        return nullptr;

      std::vector<int32_t> indices;
      indices.reserve(batch_size * num_heads);
      for (dim_t i = 0; i < batch_size; ++i)
        indices.insert(indices.end(), heads.begin(), heads.end());

      return std::make_unique<StorageView>(Shape{batch_size, num_heads}, indices, _device);
    }

    void TransformerDecoder::operator()(dim_t step,
                                        const StorageView& ids,
                                        DecoderState& state,
                                        StorageView* logits,
                                        StorageView* attention) {
      return decode(ids, nullptr, step, state, logits, attention);
    }

    void TransformerDecoder::operator()(const StorageView& ids,
                                        const StorageView& lengths,
                                        DecoderState& state,
                                        StorageView& logits,
                                        StorageView* attention) {
      return decode(ids, &lengths, -1, state, &logits, attention);
    }

    void TransformerDecoder::decode(const StorageView& ids,
                                    const StorageView* lengths,
                                    dim_t step,
                                    DecoderState& state,
                                    StorageView* outputs,
                                    StorageView* attention,
                                    bool return_logits) {
      PROFILE("TransformerDecoder");
      const DataType dtype = output_type();
      const Device device = ids.device();
      const bool is_sequence = ids.rank() > 1;

      StorageView layer_in(dtype, device);
      StorageView layer_out(dtype, device);

      _embeddings(ids, layer_in);
      if (_start_from_zero_embedding)
        zero_first_timestep(layer_in, step);
      if (_embeddings_scale && (!_start_from_zero_embedding || step != 0))
        ops::Mul()(layer_in, *_embeddings_scale, layer_in);
      if (_project_in) {
        (*_project_in)(layer_in, layer_out);
        layer_in = std::move(layer_out);
      }
      if (layer_in.rank() == 2)
        layer_in.expand_dims(1);
      if (_position_encoder)
        (*_position_encoder)(layer_in, std::max(step, dim_t(0)));
      if (_layernorm_embedding)
        (*_layernorm_embedding)(layer_in, layer_in);

      const dim_t batch_size = layer_in.dim(0);
      const dim_t max_time = layer_in.dim(1);

      // A prompt longer than the sliding window fills the cache in chunks of the window size,
      // which bounds the size of the attention scores. The chunks see the previous ones through
      // the cache, so this is only done for a decoder-only model that fills its cache without
      // padding and returns nothing.
      const bool chunked = (_sliding_window > 0
                            && max_time > _sliding_window
                            && step >= 0
                            && !lengths
                            && !outputs
                            && !attention
                            && !_with_encoder_attention
                            && !_use_flash_attention);

      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths;
      std::unique_ptr<const StorageView> input_lengths_mask;

      if (is_sequence && !lengths) {
        input_lengths = std::make_unique<StorageView>(Shape{ids.dim(0)}, int32_t(max_time), device);
        lengths = input_lengths.get();
      }

      bool multi_query = _layers.front()->get_self_attention().multi_query();

      if (lengths) {
        if (allow_padding_removal) {
          input_padder = std::make_unique<Padder>(*lengths, max_time, 1,
                                                  Padder::min_padding_to_remove(_device));
          input_padder->remove_padding(layer_in);
        }

        dim_t num_heads = _num_heads;
        if (_tensor_parallel) {
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
        }

        StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
          *lengths,
          num_heads,
          max_time,
          /*mask_future=*/true,
          multi_query);


        if (step > 0)
          ops::Add()(lengths_mask, StorageView(int32_t(step)), lengths_mask);

        input_lengths_mask = std::make_unique<StorageView>(std::move(lengths_mask));
      }

      StorageView* memory = nullptr;
      std::unique_ptr<const StorageView> memory_lengths_mask;
      std::unique_ptr<const Padder> memory_padder;
      if (_with_encoder_attention) {
        const auto it = state.find("memory_lengths");
        const StorageView* memory_lengths = it != state.end() ? &it->second : nullptr;

        if (step <= 0) {
          memory = &state.at("memory");

          if (memory_lengths && allow_padding_removal) {
            memory_padder = std::make_unique<Padder>(*memory_lengths, memory->dim(1), 1,
                                                     Padder::min_padding_to_remove(_device));
            memory_padder->remove_padding(*memory);
          }
        }

        if (memory_lengths) {
          dim_t num_heads = _num_heads;
          if (_tensor_parallel) {
            num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
          }
          const dim_t beam_size = batch_size / memory_lengths->dim(0);
          memory_lengths_mask = std::make_unique<StorageView>(
            layers::MultiHeadAttention::prepare_length_mask(*memory_lengths,
                                                            num_heads,
                                                            beam_size > 1 ? beam_size : max_time));
        }
      }

      std::vector<StorageView> alignment_heads;
      if (attention)
        alignment_heads.reserve(_layers.size());

      StorageView position_bias(dtype, device);
      StorageView* position_bias_ptr = _shared_position_bias ? &position_bias : nullptr;

      // A static cache only applies when the keys and values are cached at each step.
      const dim_t static_cache_capacity = step >= 0 ? _static_cache_capacity : 0;

      std::vector<StorageView> chunks;
      if (chunked) {
        std::vector<dim_t> chunk_sizes;
        for (dim_t start = 0; start < max_time; start += _sliding_window)
          chunk_sizes.emplace_back(std::min(_sliding_window, max_time - start));
        chunks.resize(chunk_sizes.size(), StorageView(dtype, device));
        std::vector<StorageView*> chunk_ptrs;
        for (auto& chunk : chunks)
          chunk_ptrs.emplace_back(&chunk);
        ops::Split(1, chunk_sizes)(layer_in, chunk_ptrs);
      }

      dim_t start = 0;
      for (size_t i = 0; i < std::max(chunks.size(), size_t(1)); ++i) {
        StorageView& layer_in_chunk = chunked ? chunks[i] : layer_in;
        const dim_t chunk_time = layer_in_chunk.dim(1);
        const dim_t offset = std::max(step, dim_t(0)) + start;

        if (chunked) {
          // Causal mask of the chunk, counting the positions before it.
          dim_t num_heads = _num_heads;
          if (_tensor_parallel) {
            num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
          }
          const StorageView chunk_lengths(Shape{batch_size}, int32_t(chunk_time), device);
          StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
            chunk_lengths,
            num_heads,
            chunk_time,
            /*mask_future=*/true,
            multi_query);
          if (offset > 0)
            ops::Add()(lengths_mask, StorageView(int32_t(offset)), lengths_mask);
          input_lengths_mask = std::make_unique<StorageView>(std::move(lengths_mask));
        }

        for (size_t l = 0; l < _layers.size(); ++l) {
          StorageView* cached_self_attn_keys = nullptr;
          StorageView* cached_self_attn_values = nullptr;
          StorageView* cached_attn_keys = nullptr;
          StorageView* cached_attn_values = nullptr;

          if (step >= 0) {
            const std::string l_str = std::to_string(l);
            cached_self_attn_keys = &state.at("self_keys_" + l_str);
            cached_self_attn_values = &state.at("self_values_" + l_str);
            if (_with_encoder_attention) {
              cached_attn_keys = &state.at("memory_keys_" + l_str);
              cached_attn_values = &state.at("memory_values_" + l_str);
            }
          }

          std::unique_ptr<StorageView> heads_to_select = get_layer_alignment_heads(l, batch_size);
          std::unique_ptr<StorageView> layer_attention;
          if (attention && heads_to_select)
            layer_attention = std::make_unique<StorageView>(dtype, device);

          (*_layers[l])(layer_in_chunk,
                        input_lengths_mask.get(),
                        memory,
                        memory_lengths_mask.get(),
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        cached_attn_keys,
                        cached_attn_values,
                        layer_out,
                        layer_attention.get(),
                        input_padder.get(),
                        memory_padder.get(),
                        return_normalized_attention(),
                        position_bias_ptr,
                        offset,
                        static_cache_capacity);
          layer_in_chunk = std::move(layer_out);

          if (layer_attention) {
            alignment_heads.emplace_back(dtype, device);
            ops::Gather(1, 1)(*layer_attention, *heads_to_select, alignment_heads.back());
          }
        }
        start += chunk_time;
      }

      if (chunked)
        layer_in = std::move(chunks.back());

      // The next update of the state gathers the positions written so far.
      if (static_cache_capacity > 0)
        _static_cache_length = std::max(step, dim_t(0)) + max_time;

      if (step == 0) {
        // The memory is no longer needed as its projections were cached in the first step.
        state.erase("memory");
      }

      if (attention && !alignment_heads.empty()) {
        if (_average_alignment_heads) {
          ops::Mean(1)(alignment_heads[0], *attention);
          if (!is_sequence)
            attention->squeeze(1);

        } else {
          std::vector<const StorageView*> alignment_heads_ptr;
          alignment_heads_ptr.reserve(alignment_heads.size());
          for (const auto& heads : alignment_heads)
            alignment_heads_ptr.emplace_back(&heads);

          ops::Concat(1)(alignment_heads_ptr, *attention);
          if (!is_sequence)
            attention->squeeze(2);
        }
      }

      if (outputs) {
        if (_output_norm)
          (*_output_norm)(layer_in, layer_in);
        if (_project_out) {
          (*_project_out)(layer_in, layer_out);
          layer_in = std::move(layer_out);
        }

        if (_outputs_scale)
          ops::Mul()(layer_in, *_outputs_scale, layer_in);

        if (return_logits) {
          _proj(layer_in, *outputs);
          if (_final_logit_softcapping != 0.f) {
            // logits = tanh(logits / cap) * cap  — squashes logits to (-cap, cap)
            const auto dtype = outputs->dtype();
            const auto device = outputs->device();
            ops::Mul()(*outputs, StorageView(1.f / _final_logit_softcapping).to(dtype), *outputs);
            ops::Tanh()(*outputs, *outputs);
            ops::Mul()(*outputs, StorageView(_final_logit_softcapping).to(dtype), *outputs);
          }
        } else
          *outputs = std::move(layer_in);

        if (!is_sequence)
          outputs->squeeze(1);
        else if (input_padder)
          input_padder->add_padding(*outputs);
      }
    }

  }
}

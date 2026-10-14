#include <cmath>

#include "test_utils.h"
#include "ctranslate2/layers/attention.h"

class MockModel : public models::Model {
public:
  MockModel(dim_t num_heads, dim_t num_heads_kv) {
    const dim_t d_model = 64;
    const dim_t d_head = d_model / num_heads;
    
    std::vector<float> linear_0_data(num_heads * d_head * d_model, 0.01f);
    std::vector<float> linear_1_data(2 * num_heads_kv * d_head * d_model, 0.01f);
    
    register_variable("attn/linear_0/weight",
                      StorageView({num_heads * d_head, d_model}, linear_0_data));
    register_variable("attn/linear_1/weight",
                      StorageView({2 * num_heads_kv * d_head, d_model}, linear_1_data));
    register_variable("attn/linear_2/weight",
                      StorageView({d_model, num_heads * d_head}, DataType::FLOAT32));
    register_variable("attn/q_norm/gamma",
                      StorageView({d_model}, std::vector<float>(d_model, 1.0f)));
    register_variable("attn/k_norm/gamma",
                      StorageView({d_head}, std::vector<float>(d_head, 1.0f)));
    
    register_variable("attn/num_heads_kv",
                      StorageView(static_cast<int32_t>(num_heads_kv)));

    set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, false);
  }
protected:
  std::unique_ptr<Model> clone() const override { return nullptr; }
};

class TestableAttention : public layers::MultiHeadAttention {
public:
  using MultiHeadAttention::MultiHeadAttention;
  using MultiHeadAttention::process_cross_attention;
};

class CrossAttentionTest : public ::testing::Test {
protected:
  static constexpr dim_t NUM_HEADS = 4;
  static constexpr dim_t D_MODEL = 64;
  static constexpr dim_t D_HEAD = D_MODEL / NUM_HEADS;
  static constexpr dim_t BATCH = 2;
  static constexpr dim_t Q_LEN = 6;
  static constexpr dim_t V_LEN = 8;

  float get_4d(const StorageView& view, dim_t b, dim_t h, dim_t t, dim_t d) {
    const auto& shape = view.shape();
    return view.data<float>()[b * shape[1] * shape[2] * shape[3] +
                              h * shape[2] * shape[3] + t * shape[3] + d];
  }

};

// MQA: All heads share same K/V
TEST_F(CrossAttentionTest, MultiQueryAttention) {
  MockModel model(NUM_HEADS, /*num_heads_kv=*/1);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);
  // Use non-uniform values to verify normalization is applied
  std::vector<float> value_data(BATCH * V_LEN * D_MODEL);
  for (size_t i = 0; i < value_data.size(); ++i)
    value_data[i] = static_cast<float>(i % 10 + 1);
  std::vector<float> fused_data(BATCH * Q_LEN * NUM_HEADS * D_HEAD);
  for (size_t i = 0; i < fused_data.size(); ++i)
    fused_data[i] = static_cast<float>(i % 10 + 1);
  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, value_data);
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, fused_data);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;
  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);
  // MQA: K/V are replicated to 4D format [batch, num_heads, time, d_head]
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  ASSERT_EQ(cached_values.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  // Verify K/V values are consistent across batch and time dimensions
  // (In MQA, there's only one set of K/V, so we just verify the tensor is valid)
  float k0 = get_4d(cached_keys, 0, 0, 0, 0);
  float v0 = get_4d(cached_values, 0, 0, 0, 0);
  EXPECT_NE(k0, 0.0f) << "K values should be non-zero after projection";
  EXPECT_NE(v0, 0.0f) << "V values should be non-zero after projection";
  // Verify q_norm and k_norm are applied (RMSNorm normalizes to ~1.0 magnitude)
  float q_val = q_proj.data<float>()[0];
  float k_val = cached_keys.data<float>()[0];
  EXPECT_GT(std::abs(q_val), 0.1f) << "q_norm should produce non-zero output";
  EXPECT_LT(std::abs(q_val), 2.0f) << "q_norm should normalize values";
  EXPECT_GT(std::abs(k_val), 0.1f) << "k_norm should produce non-zero output";
  EXPECT_LT(std::abs(k_val), 2.0f) << "k_norm should normalize values";
}

// GQA: Heads within same group share K/V
TEST_F(CrossAttentionTest, GroupedQueryAttention) {
  constexpr dim_t NUM_KV_HEADS = 2;
  constexpr dim_t HEADS_PER_GROUP = NUM_HEADS / NUM_KV_HEADS;

  MockModel model(NUM_HEADS, NUM_KV_HEADS);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);

  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, std::vector<float>(BATCH * V_LEN * D_MODEL, 1.0f));
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, DataType::FLOAT32);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;

  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);

  // GQA: After head replication, shape is [batch, num_heads, time, d_head]
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));

  // Heads in same group share K/V
  for (dim_t group = 0; group < NUM_KV_HEADS; ++group) {
    dim_t first = group * HEADS_PER_GROUP;
    float k_group = get_4d(cached_keys, 0, first, 0, 0);
    float v_group = get_4d(cached_values, 0, first, 0, 0);
    for (dim_t h = first + 1; h < first + HEADS_PER_GROUP; ++h) {
      EXPECT_EQ(get_4d(cached_keys, 0, h, 0, 0), k_group);
      EXPECT_EQ(get_4d(cached_values, 0, h, 0, 0), v_group);
    }
  }
}

// Merged self+cross attention (T5Gemma2 style): self-attention layer also
// projects encoder memory through a separate `memory_kv` linear and concatenates
// the result onto the self-attention K/V before softmax.
TEST(MergedAttentionTest, ForwardMergedProducesOutput) {
  constexpr dim_t NUM_HEADS = 4, NUM_KV = 1, D_HEAD = 16;
  constexpr dim_t D_MODEL = NUM_HEADS * D_HEAD;
  constexpr dim_t QKV_ROWS = (NUM_HEADS + 2 * NUM_KV) * D_HEAD;
  constexpr dim_t KV_ROWS = 2 * NUM_KV * D_HEAD;

  class MergedMockModel : public models::Model {
  public:
    MergedMockModel() {
      register_variable("attn/linear_0/weight",
                        StorageView({QKV_ROWS, D_MODEL}, std::vector<float>(QKV_ROWS * D_MODEL, 0.01f)));
      register_variable("attn/linear_1/weight",
                        StorageView({D_MODEL, NUM_HEADS * D_HEAD},
                                    std::vector<float>(D_MODEL * NUM_HEADS * D_HEAD, 0.01f)));
      register_variable("attn/memory_kv/weight",
                        StorageView({KV_ROWS, D_MODEL}, std::vector<float>(KV_ROWS * D_MODEL, 0.01f)));
      register_variable("attn/q_norm/gamma", StorageView({D_HEAD}, std::vector<float>(D_HEAD, 1.0f)));
      register_variable("attn/k_norm/gamma", StorageView({D_HEAD}, std::vector<float>(D_HEAD, 1.0f)));
      register_variable("attn/num_heads_kv", StorageView(static_cast<int32_t>(NUM_KV)));
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, false);
    }
  protected:
    std::unique_ptr<Model> clone() const override { return nullptr; }
  };

  MergedMockModel model;
  layers::MultiHeadAttention attention(model, "attn", NUM_HEADS, /*self_attention=*/true);
  ASSERT_TRUE(attention.has_merged_encoder_attention());

  constexpr dim_t B = 1, Q_LEN = 1, MEM_LEN = 5;
  StorageView queries({B, Q_LEN, D_MODEL}, std::vector<float>(B * Q_LEN * D_MODEL, 1.0f));
  StorageView memory({B, MEM_LEN, D_MODEL}, std::vector<float>(B * MEM_LEN * D_MODEL, 1.0f));
  StorageView output(DataType::FLOAT32);
  StorageView self_k(DataType::FLOAT32), self_v(DataType::FLOAT32);
  StorageView mem_k(DataType::FLOAT32), mem_v(DataType::FLOAT32);

  attention.forward_merged(queries, &memory, nullptr, nullptr, output,
                           &self_k, &self_v, &mem_k, &mem_v, nullptr, nullptr, /*offset=*/0);

  EXPECT_EQ(output.shape(), (Shape{B, Q_LEN, D_MODEL}));
  EXPECT_EQ(mem_k.shape(), (Shape{B, NUM_HEADS, MEM_LEN, D_HEAD}));
  EXPECT_EQ(self_k.shape(), (Shape{B, NUM_HEADS, Q_LEN, D_HEAD}));
}

// GQA self-attention keeps one key/value head per group instead of replicating it for every query
// head, and gives the same output as multi-head attention with the key/value heads repeated.
TEST(GroupedQueryAttentionTest, SelfAttentionMatchesRepeatedHeads) {
  constexpr dim_t NUM_HEADS = 4, NUM_KV = 2, D_HEAD = 16;
  constexpr dim_t D_MODEL = NUM_HEADS * D_HEAD;
  constexpr dim_t GROUP_SIZE = NUM_HEADS / NUM_KV;
  constexpr dim_t BATCH = 2, TIME = 5;

  const auto make_weights = [](dim_t size, float seed) {
    std::vector<float> weights(size);
    for (dim_t i = 0; i < size; ++i)
      weights[i] = 0.05f * std::sin(seed + 0.37f * i);
    return weights;
  };
  const std::vector<float> wq = make_weights(NUM_HEADS * D_HEAD * D_MODEL, 1);
  const std::vector<float> wk = make_weights(NUM_KV * D_HEAD * D_MODEL, 2);
  const std::vector<float> wv = make_weights(NUM_KV * D_HEAD * D_MODEL, 3);
  const std::vector<float> wo = make_weights(D_MODEL * NUM_HEADS * D_HEAD, 4);

  // Query head h uses key/value head h / GROUP_SIZE.
  const auto repeat_heads = [&](const std::vector<float>& weights) {
    std::vector<float> repeated;
    for (dim_t h = 0; h < NUM_HEADS; ++h) {
      const auto head = weights.begin() + (h / GROUP_SIZE) * D_HEAD * D_MODEL;
      repeated.insert(repeated.end(), head, head + D_HEAD * D_MODEL);
    }
    return repeated;
  };
  const auto fuse = [&](const std::vector<float>& k, const std::vector<float>& v) {
    std::vector<float> qkv = wq;
    qkv.insert(qkv.end(), k.begin(), k.end());
    qkv.insert(qkv.end(), v.begin(), v.end());
    return qkv;
  };

  class SelfAttentionModel : public models::Model {
  public:
    SelfAttentionModel(dim_t num_heads_kv, const std::vector<float>& qkv, const std::vector<float>& o) {
      register_variable("attn/linear_0/weight",
                        StorageView({dim_t(qkv.size()) / D_MODEL, D_MODEL}, qkv));
      register_variable("attn/linear_1/weight", StorageView({D_MODEL, NUM_HEADS * D_HEAD}, o));
      register_variable("attn/num_heads_kv", StorageView(static_cast<int32_t>(num_heads_kv)));
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, false);
    }
  protected:
    std::unique_ptr<Model> clone() const override { return nullptr; }
  };

  const SelfAttentionModel gqa_model(NUM_KV, fuse(wk, wv), wo);
  const SelfAttentionModel mha_model(NUM_HEADS, fuse(repeat_heads(wk), repeat_heads(wv)), wo);
  const layers::MultiHeadAttention gqa(gqa_model, "attn", NUM_HEADS, /*self_attention=*/true);
  const layers::MultiHeadAttention mha(mha_model, "attn", NUM_HEADS, /*self_attention=*/true);

  std::vector<float> x(BATCH * (TIME + 1) * D_MODEL);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = std::cos(0.13f * i);
  const StorageView inputs({BATCH, TIME, D_MODEL}, std::vector<float>(x.begin(), x.begin() + BATCH * TIME * D_MODEL));
  const StorageView step({BATCH, 1, D_MODEL}, std::vector<float>(x.begin() + BATCH * TIME * D_MODEL, x.end()));
  const StorageView lengths({BATCH}, std::vector<int32_t>{TIME, TIME - 2});
  const StorageView mask = layers::MultiHeadAttention::prepare_length_mask(
    lengths, NUM_HEADS, TIME, /*mask_future=*/true);

  StorageView gqa_output, gqa_keys, gqa_values;
  StorageView mha_output, mha_keys, mha_values;
  gqa(inputs, inputs, &mask, gqa_output, &gqa_keys, &gqa_values);
  mha(inputs, inputs, &mask, mha_output, &mha_keys, &mha_values);
  EXPECT_EQ(gqa_keys.shape(), (Shape{BATCH, NUM_KV, TIME, D_HEAD}));
  EXPECT_EQ(gqa_values.shape(), (Shape{BATCH, NUM_KV, TIME, D_HEAD}));
  expect_storage_eq(gqa_output, mha_output, 1e-5);

  // A decoding step that attends to the cached keys and values.
  gqa(step, step, nullptr, gqa_output, &gqa_keys, &gqa_values,
      nullptr, nullptr, nullptr, true, nullptr, /*offset=*/TIME);
  mha(step, step, nullptr, mha_output, &mha_keys, &mha_values,
      nullptr, nullptr, nullptr, true, nullptr, /*offset=*/TIME);
  EXPECT_EQ(gqa_keys.shape(), (Shape{BATCH, NUM_KV, TIME + 1, D_HEAD}));
  expect_storage_eq(gqa_output, mha_output, 1e-5);
}

// MHA: Each head has independent K/V
TEST_F(CrossAttentionTest, StandardMultiHeadAttention) {
  MockModel model(NUM_HEADS, NUM_HEADS);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);

  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, std::vector<float>(BATCH * V_LEN * D_MODEL, 1.0f));
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, DataType::FLOAT32);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;

  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);

  // Shape: [batch, num_heads, time, d_head] - each head has own K/V
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  ASSERT_EQ(cached_values.shape(), cached_keys.shape());
}

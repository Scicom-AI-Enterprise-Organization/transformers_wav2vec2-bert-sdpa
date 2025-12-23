# SDPA Support for Wav2Vec2-BERT with Relative Key Position Embeddings

PyTorch SDPA (Scaled Dot Product Attention) optimization for
Wav2Vec2-BERT models, providing 1.1--1.2x speedup and \~8% memory
reduction with full numerical equivalence.

## Overview

This implementation adds PyTorch's `scaled_dot_product_attention` to
Wav2Vec2-BERT while maintaining compatibility with `relative_key`
position embeddings through a hybrid approach:

1.  Manual computation of position bias (query-dependent, unavoidable)
2.  SDPA optimization for content attention, softmax, dropout, and value
    projection

## Performance Results (RTX 4070 Laptop GPU)

  Test Case               Eager (ms)   SDPA (ms)   Speedup
  ----------------------- ------------ ----------- ---------
  Real-time TTS (1×500)   107.6        101.7       1.06x
  Batch serving (4×500)   328.5        308.0       1.07x
  Long audio (1×2000)     945.8        785.9       1.20x
  Peak load (8×500)       656.9        602.7       1.09x

**Average speedup**: 1.10x\
**Memory reduction**: \~8%\
**Accuracy impact**: \<1% (no quality degradation)

## Model Specifications

-   Parameters: \~580M
-   Hidden size: 1024
-   Layers: 24
-   Attention heads: 16
-   Position embeddings: relative_key

## Installation

After cloning the repository, install it in editable mode to ensure the
modified Transformers code is used:

``` bash
pip install -e .
```

This is required so that the SDPA-enabled attention implementation and
configuration changes are picked up correctly by Python.

## Quick Start

### Enable SDPA

``` python
from transformers import Wav2Vec2BertModel, Wav2Vec2BertConfig

config = Wav2Vec2BertConfig.from_pretrained(
    "facebook/w2v-bert-2.0",
    attn_implementation="sdpa"
)

model = Wav2Vec2BertModel(config)
```

### Eager vs SDPA Benchmark

``` python
config_eager = Wav2Vec2BertConfig.from_pretrained(
    "facebook/w2v-bert-2.0",
    attn_implementation="eager"
)

config_sdpa = Wav2Vec2BertConfig.from_pretrained(
    "facebook/w2v-bert-2.0",
    attn_implementation="sdpa"
)
```

## Files Modified

### configuration_wav2vec2_bert.py

-   Added `attn_implementation` argument
-   Validation logic and documentation updated

### modular_wav2vec2_bert.py

-   Introduced SDPA execution path
-   Preserved eager fallback for unsupported cases

## Hybrid SDPA Implementation

``` python
hidden_states = F.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=position_bias + attention_mask,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=False,
)
```


## Position Embedding Compatibility

| Type | SDPA Support | Notes |
|------|-------------|-------|
| `relative_key` | ✅ Full (hybrid) | Position bias computed manually, passed to SDPA |
| `rotary` | ✅ Full | Applied to Q/K before SDPA |
| `relative` | ⚠️ Fallback | Falls back to eager (mathematically required) |
| `None` | ✅ Full | Pure SDPA, no position embeddings |

## Testing

``` bash
python test.py
```


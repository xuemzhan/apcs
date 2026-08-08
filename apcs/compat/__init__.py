"""T00 Compatibility Scanner (design.md §28)。

自动读取 teacher / student 的：
- tokenizer / vocab
- layers / attention_heads / kv_heads / head_dim / hidden_size
- RoPE config / position / dtype / cache_layout / attention_implementation

输出 model_compatibility.json。
"""
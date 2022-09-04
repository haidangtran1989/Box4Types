def tokenize_inputs(inputs, tokenizer, max_len):
    return tokenizer.batch_encode_plus(
        inputs,
        add_special_tokens=True,
        max_length=max_len,
        truncation_strategy="only_second",
        pad_to_max_length=True,
        return_tensors="pt",
        truncation=True
    )

# N-Gram Sequence Generation for LSTM Input

## Code Overview
This code generates input sequences (n-grams) from text for training an LSTM model, processing the text line by line.

```python
input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
```

## What It Does
- **Splits Text**: Breaks `text` into lines using `\n`.
- **Tokenizes**: Converts each line into a list of integers (`token_list`) using a `tokenizer`.
- **Generates N-Grams**: For each `token_list`, creates subsequences starting with length 2 up to the full length.
  - Example: `token_list = [5, 6, 7, 8]` → `[[5, 6], [5, 6, 7], [5, 6, 7, 8]]`.
- **Stores**: Appends each n-gram to `input_sequences`.

## Behavior Across Lines
- **Restarts with 2-Grams**: For each new line, it begins anew with a 2-gram.
  - After `[5, 6, 7, 8]` (e.g., "and then some cheese"), the next line (e.g., "now I want more" → `[9, 1, 10, 11]`) starts with `[9, 1]`, then builds to `[9, 1, 10]`, `[9, 1, 10, 11]`.
- **No Continuity**: Lines are independent; no sequences like `[8, 9]` are created across lines.

## Example
For `text = "I ate a samosa\nand then some cheese\nnow I want more"`:
- Line 1: `[1, 2, 3, 4]` → `[[1, 2], [1, 2, 3], [1, 2, 3, 4]]`
- Line 2: `[5, 6, 7, 8]` → `[[5, 6], [5, 6, 7], [5, 6, 7, 8]]`
- Line 3: `[9, 1, 10, 11]` → `[[9, 1], [9, 1, 10], [9, 1, 10, 11]]`

## Purpose for LSTM
- Prepares data for sequence prediction (e.g., predict "6" given "5").
- Each line’s sequences are separate training examples; LSTM learns patterns within lines, restarting memory per line.

## Key Insight
- After finishing a line (e.g., `[5, 6, 7, 8]`), it doesn’t bridge to the next (e.g., no `[8, 9]`). It resets to a 2-gram (e.g., `[9, 1]`) for the new line, building up again.

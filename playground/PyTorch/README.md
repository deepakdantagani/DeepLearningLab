# PyTorch Playground

A collection of interactive examples and tutorials for understanding PyTorch operations commonly used in deep learning and natural language processing.

## Examples

### 1. [Scatter Operation](01_scatter_.py) | [Documentation](01_scatter_.md)

**`torch.scatter_`** - In-place operation that distributes values from a source tensor into a target tensor based on indices.

**Key Concepts:**

- In-place vs out-of-place operations
- Restoring original ordering after sorting
- Top-k and top-p filtering applications

**Run:** `python 01_scatter_.py`

### 2. [Argmax Operation](02_argmax.py) | [Documentation](02_argmax.md)

**`torch.argmax`** - Returns indices of maximum values along specified dimensions.

**Key Concepts:**

- Classification predictions
- Attention mechanisms
- Dimension reduction
- Tie breaking behavior

**Run:** `python 02_argmax.py`

### 3. [Multinomial Sampling](03_multinomial.py) | [Documentation](03_multinomial.md)

**`torch.multinomial`** - Samples from probability distributions for stochastic selection.

**Key Concepts:**

- Temperature sampling for text generation
- Batch processing with multinomial
- Replacement vs non-replacement sampling
- Comparison with greedy (argmax) selection

**Run:** `python 03_multinomial.py`

## Common Use Cases in Your Codebase

These operations are frequently used in your generation pipeline:

### Scatter (`scatter_`)

- **Top-p filtering**: Restoring original positions after sorting and filtering
- **Batch processing**: Efficient in-place operations
- **Memory optimization**: Avoiding unnecessary tensor allocations

### Argmax (`argmax`)

- **Greedy decoding**: Selecting most likely next token
- **Classification**: Converting logits to predictions
- **Attention**: Finding attended positions

### Multinomial (`multinomial`)

- **Temperature sampling**: Controlling randomness in text generation
- **Stochastic selection**: Probabilistic token selection
- **Batch generation**: Efficient sampling for multiple sequences

## Running Examples

```bash
cd playground/PyTorch

# Run all examples
python 01_scatter_.py
python 02_argmax.py
python 03_multinomial.py

# Or run individual functions
python -c "from 01_scatter_ import basic_scatter_example; basic_scatter_example()"
python -c "from 03_multinomial import temperature_sampling_example; temperature_sampling_example()"
```

## Contributing

To add new examples:

1. Create a Python file with interactive examples
2. Create a corresponding markdown documentation file
3. Update this README with the new example
4. Follow the naming convention: `XX_operation_name.py`

## Learning Path

1. Start with **scatter_** to understand tensor manipulation
2. Move to **argmax** for classification and selection operations
3. Learn **multinomial** for probabilistic sampling and text generation
4. Combine all three for advanced generation techniques

## Related Files

- `src/dllab/generation/filters.py` - Uses `scatter_` for top-p filtering
- `src/dllab/generation/strategies.py` - Uses `multinomial` for temperature sampling
- `src/dllab/generation/base.py` - Abstract interfaces for generation
- `playground/tensor/` - Additional tensor operation examples

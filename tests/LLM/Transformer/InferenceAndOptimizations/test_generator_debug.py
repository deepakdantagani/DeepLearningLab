# pylint: disable=all
# pylint: disable=W0621
# flake8: noqa
"""Debug version of the generation test with breakpoints and detailed logging."""

import logging
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from huggingface_hub.inference._generated.types import text_generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: F401, E402

# Import dllab modules
from dllab.generation.generator import Generator  # noqa: E402
from dllab.generation.generator import (RepetitionPenalty,
                                        TemperatureLogitsProcessor, TopKTopP)
from dllab.generation.strategies import GreedyStrategy  # noqa: F401, E402
from dllab.generation.strategies import (MultinomialSampling,
                                         TemperatureSampling)

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "sshleifer/tiny-gpt2"

# Load model and tokenizer globally for reuse
logger.info("Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
logger.info("Model and tokenizer loaded successfully")

INPUT_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""
input_tokens = tokenizer(INPUT_TEXT, return_tensors="pt").input_ids


def test_greedy_with_debug():
    """Test greedy generation strategy with detailed debugging."""
    text_generator = Generator(model, strategy=GreedyStrategy())

    # Generate output
    output = text_generator.generate(input_tokens, max_new_tokens=3)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: '{decoded_output}'")

    return output


def test_temperature_sampling_with_debug():
    """Test temperature sampling generation strategy with detailed debugging."""
    # Use the new approach: TemperatureLogitsProcessor + MultinomialSampling
    temperatures = [0.5, 1.0, 2.0]

    for temperature in temperatures:
        text_generator = Generator(
            model,
            strategy=MultinomialSampling(),
            logits_processors=[TemperatureLogitsProcessor(temperature=temperature)],
        )
        output = text_generator.generate(input_tokens, max_new_tokens=3)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated: '{decoded_output}'")


def test_multinomial_sampling_with_debug():
    """Test multinomial sampling generation strategy with detailed debugging."""
    text_generation = Generator(model, strategy=MultinomialSampling())
    output = text_generation.generate(input_tokens, max_new_tokens=3)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: '{decoded_output}'")


def test_top_k_top_p_filtering():
    # Correct order: Temperature first, then TopK/TopP filtering
    # This follows the standard approach used by Hugging Face and other libraries
    text_generation = Generator(
        model,
        strategy=MultinomialSampling(),  # Use multinomial sampling after processors
        logits_processors=[
            TemperatureLogitsProcessor(temperature=1.2),  # Temperature first
            TopKTopP(k=2, p=0.9),  # Then TopK/TopP filtering
        ],
    )
    output = text_generation.generate(input_tokens, max_new_tokens=3)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: '{decoded_output}'")


def test_repetition_penalty():
    text_generation = Generator(
        model,
        strategy=GreedyStrategy(),
        logits_processors=[
            TemperatureLogitsProcessor(temperature=2.0),
            RepetitionPenalty(penalty=2.0),
        ],
    )
    output = text_generation.generate(input_tokens, max_new_tokens=32)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: '{decoded_output}'")


if __name__ == "__main__":
    # Run the debug test
    # test_greedy_with_debug()
    # test_temperature_sampling_with_debug()
    # test_multinomial_sampling_with_debug()
    # test_top_k_top_p_filtering()
    test_repetition_penalty()

    # Uncomment to run interactive test
    # test_greedy_interactive()
    pass

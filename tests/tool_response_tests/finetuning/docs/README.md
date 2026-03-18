# Tool-Calling Fine-Tuning Dataset

Fine-tuning dataset for training a Qwen 8B model to make tool calls for the HomeAssist voice assistant.

## Dataset Files

| File | Description | Examples |
|------|-------------|----------|
| `combined_training_data.jsonl` | **Main dataset** - Use this for training | 604 |
| `training_data.jsonl` | Single-turn examples only | 589 |
| `multiturn_training_data.jsonl` | Multi-turn conversation examples | 81 |
| `tool_schemas.json` | Tool definitions (embedded in system prompt) | 11 tools |
| `system_prompt.txt` | System prompt template | - |

## Data Format

Each line in the JSONL files is a training example in ChatML format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a tool-calling assistant..."},
    {"role": "user", "content": "Play some jazz music"},
    {"role": "assistant", "content": "[{\"name\": \"spotify_playback\", \"arguments\": {\"action\": \"play\", \"query\": \"jazz\", \"search_type\": \"artist\"}}]"}
  ]
}
```

The assistant's response is always a JSON array:
- `[]` when no tool is needed
- `[{...}]` for single tool calls
- `[{...}, {...}]` for multiple parallel tool calls

## Fine-Tuning with Ollama

### Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the base model
ollama pull qwen2:8b
```

### Create a Modelfile

Create a file called `Modelfile`:

```dockerfile
FROM qwen2:8b

# Set parameters for tool calling
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"

# System message is embedded in training data
```

### Fine-Tuning with LoRA

Ollama's built-in fine-tuning uses LoRA adapters. Create your fine-tuned model:

```bash
# From the finetuning directory
ollama create homeassist-tools -f Modelfile --quantize q4_K_M

# Train with the dataset (when Ollama supports fine-tuning CLI)
# Note: As of 2024, Ollama fine-tuning is via their API/cloud
```

### Alternative: Local LoRA Fine-Tuning with Unsloth

For local LoRA training with more control:

```bash
pip install unsloth
```

```python
from unsloth import FastLanguageModel
import json

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2-7B-bnb-4bit",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Load training data
def load_dataset(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

training_data = load_dataset("combined_training_data.jsonl")

# Format for training
def format_example(example):
    messages = example["messages"]
    # Convert to ChatML format
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": text}

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=[format_example(ex) for ex in training_data],
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        output_dir="outputs",
    ),
)

trainer.train()

# Save LoRA adapters
model.save_pretrained("homeassist-tools-lora")
```

### Convert to Ollama Format

After training with Unsloth:

```bash
# Merge LoRA adapters
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('homeassist-tools-lora')
model.save_pretrained_merged('homeassist-tools-merged', tokenizer)
"

# Create GGUF for Ollama
python llama.cpp/convert.py homeassist-tools-merged --outfile homeassist-tools.gguf

# Create Ollama model
ollama create homeassist-tools -f Modelfile
```

## Regenerating the Dataset

To regenerate or expand the training data:

```bash
# Generate single-turn examples
python generate_training_data.py

# Generate multi-turn examples
python generate_multiturn.py

# Combine all sources (includes test cases)
python combine_datasets.py
```

## Adding More Examples

Edit `generate_training_data.py` to add more examples to `TRAINING_EXAMPLES`:

```python
TRAINING_EXAMPLES = [
    # Format: (user_prompt, expected_tool_calls)
    ("Your new prompt", [{"name": "tool_name", "arguments": {...}}]),
    ...
]
```

For multi-turn conversations, edit `generate_multiturn.py`:

```python
MULTI_TURN_CONVERSATIONS = [
    [
        ("First user message", [tool_call_1]),
        ("Follow-up message", [tool_call_2]),
    ],
    ...
]
```

## Tips for Better Fine-Tuning

1. **Specialized model** - This dataset trains a model that ONLY does tool calling. Every input maps to a tool call output.

2. **Diverse phrasing** - Many ways to express the same intent are included to improve generalization.

3. **Multi-turn context** - The model learns to understand follow-ups like "and tomorrow?" after a weather query.

4. **Tool distribution** - Examples are weighted toward frequently-used tools (spotify, weather, lights, calendar).

5. **Validation** - All examples are validated against tool schemas before training.

## Testing Your Fine-Tuned Model

```bash
# Test with Ollama
ollama run homeassist-tools "What's the weather tomorrow?"

# Expected output:
# [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]
```

## Dataset Statistics

- Total examples: 604
- Tool call examples: 604 (100%) - **Specialized for tool-calling only**
- Single-tool calls: 507
- Multi-tool calls: 82
- Multi-turn conversations: 30 (81 total turns)
- Available tools: 12

# Un-LOCC Wrapper: An OpenAI SDK Wrapper Building Upon the Research of UN-LOCC

Un-LOCC (Universal Lossy Optical Context Compression) is a Python library that wraps the OpenAI SDK to enable optical compression of text inputs. By rendering text into images, it leverages Vision-Language Models (VLMs) for more efficient token usage, especially when dealing with large text contexts.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MaxDevv/Un-LOCC-Wrapper&type=date&legend=top-left)](https://www.star-history.com/#MaxDevv/Un-LOCC-Wrapper&type=date&legend=top-left) 

## Features

- **Optical Compression**: Converts text into images for VLM-compatible input.
- **Seamless Integration**: Drop-in replacement for OpenAI client with compression support.
- **Synchronous and Asynchronous**: Supports both sync and async OpenAI operations.
- **Flexible Compression**: Customize font, size, dimensions, and more.
- **Efficient Rendering**: Uses fast libraries like ReportLab and pypdfium2 when available, falls back to PIL.

## Installation

```bash
pip install un-locc
```

### Dependencies

- `openai`
- `Pillow` (PIL)
- Optional: `reportlab`, `pypdfium2`, `aggdraw` for enhanced performance

## Quickstart

### Basic Usage

```python
from un_locc import UnLOCC

# Initialize with your OpenAI API key
client = UnLOCC(api_key="your-api-key")

# Compress a large text context while keeping instructions plain
large_text = "Your large text content here..."
messages = [
    {"role": "user", "content": "Summarize the following text."},
    {"role": "user", "content": large_text, "compressed": True}
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)
```

### Asynchronous Usage

```python
import asyncio
from un_locc import AsyncUnLOCC

async def main():
    client = AsyncUnLOCC(api_key="your-api-key")
    large_text = "Your large text content here..."
    messages = [
        {"role": "user", "content": "Analyze the following document."},
        {"role": "user", "content": large_text, "compressed": True}
    ]
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    print(response)

asyncio.run(main())
```

### Responses API with Compression

```python
from un_locc import UnLOCC

client = UnLOCC(api_key="your-api-key")

response = client.responses.create(
    model="gpt-4o",
    input="Large text to compress",
    compression=True
)
```

## Documentation

### Classes

- **`UnLOCC`**: Synchronous wrapper for OpenAI client.
- **`AsyncUnLOCC`**: Asynchronous wrapper for OpenAI client.

Both classes initialize like the OpenAI client: `UnLOCC(api_key="...")`.

### Compression Parameters

Default compression settings (uses built-in Atkinson Hyperlegible Regular font):

```python
{
    'font_path': 'AtkinsonHyperlegible-Regular.ttf',  # Built-in font
    'font_size': 15,
    'max_width': 864,
    'max_height': 864,
    'padding': 20
}
```

Customize by passing a dict to `compressed`:

```python
messages = [
    {
        "role": "user", 
        "content": large_text,
        "compressed": {
            "font_size": 12,
            "max_width": 1024,
            "max_height": 1024
        }
    }
]
```

For `responses.create`, pass `compression` as a dict or `True` for defaults.

### Methods

#### Chat Completions

- `client.chat.completions.create(messages, **kwargs)`: Compresses messages with `"compressed"` key.
- `client.chat.completions.create(**kwargs)`: Standard usage.

#### Responses

- `client.responses.create(input, compression=None, **kwargs)`: Compresses `input` if `compression` is provided.

### Content Handling

- **String Content**: Directly compressed into images.
- **List Content**: Processes parts; text parts are compressed, others remain unchanged.

### Rendering Methods

The library selects the fastest available rendering method:

1. **ReportLab + pypdfium2** (fastest, recommended).
2. **ReportLab only**.
3. **PIL fallback** (ultra-fast bitmap).

Ensure fonts are available; defaults to system fonts if not found.

## Tips

Through several trials, I've found that it's much better to embed instructions into plain text and then only compress the large context like this:

```python
messages = [
    {
        "role": "user", 
        "content": "Instructions: Summarize the following text."
    },
    {
        "role": "user", 
        "content": long_text,
        "compressed": True
    },
]
```

This approach keeps instructions clear and readable while compressing only the bulky content. Alternatively, use it to compress prior chat history for efficient context management.

## License
MIT License see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Related Research

For more details on the library and optimal per model configurations, check out [github.com/MaxDevv/UN-LOCC](https://github.com/MaxDevv/UN-LOCC).

Based on UN-LOCC research for optical context compression in VLMs.

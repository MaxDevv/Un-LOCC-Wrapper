"""
Un-LOCC Wrapper: An OpenAI SDK Wrapper Building Upon the Research of Un-LOCC

A wrapper library for the OpenAI SDK that adds optical compression capabilities
to text inputs in chat completions and responses APIs through Un-LOCC (Universal Lossy Optical Context Compression).

This allows compressing large text contexts into images for better token efficiency
in Vision-Language Models (VLMs).
"""

import os
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI, AsyncOpenAI

# Try to import fast rendering libraries
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import pypdfium2 as pdfium
    PYPDFIUM_AVAILABLE = True
except ImportError:
    PYPDFIUM_AVAILABLE = False

try:
    import aggdraw
    AGGDRAW_AVAILABLE = True
except ImportError:
    AGGDRAW_AVAILABLE = False


class UnLOCC:
    """
    Synchronous wrapper for OpenAI client with optical compression support.
    """

    def __init__(self, *args, **kwargs):
        self.client = OpenAI(*args, **kwargs)
        # Default font path - users should provide their own font file
        # This is a fallback that may not work depending on installation
        self.default_compression = {
            'font_path': os.path.join(os.path.dirname(__file__), 'AtkinsonHyperlegible-Regular.ttf'),
            'font_size': 15,
            'max_width': 864,
            'max_height': 864,
            'padding': 20
        }
        # Cache for font objects and metrics
        self._font_cache = {}
        self._font_metrics_cache = {}

    def _wrap_text(self, text, font, max_width):
        """Helper function to wrap text into lines that fit within a max_width."""
        lines = []
        words = text.split()
        if not words:
            return []
        current_line = words[0]
        for word in words[1:]:
            # Use getbbox for PIL >= 10.0.0
            bbox = font.getbbox(current_line + " " + word)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines



    def _compress_content(self, content, compression_params):
        """
        Compress text content into images.
        Handles string content and lists of content parts.
        """
        if isinstance(content, str):
            images = self._generate_images(content, **compression_params)
            return [{"type": "image_url", "image_url": {"url": img}} for img in images]
        elif isinstance(content, list):
            new_content = []
            for part in content:
                if part.get("type") == "text":
                    images = self._generate_images(part["text"], **compression_params)
                    new_content.extend([{"type": "image_url", "image_url": {"url": img}} for img in images])
                else:
                    new_content.append(part)
            return new_content
        else:
            return content

    def _compress_messages(self, messages):
        """Compress text content in chat messages if 'compressed' is specified per message."""
        new_messages = []
        for msg in messages:
            new_msg = msg.copy()
            compressed = msg.get("compressed")
            if compressed and "content" in msg:
                if isinstance(compressed, bool):
                    params = self.default_compression.copy()
                else:
                    params = {**self.default_compression, **compressed}
                new_msg["content"] = self._compress_content(msg["content"], params)
                # Remove the 'compressed' key from the message
                new_msg.pop("compressed", None)
            new_messages.append(new_msg)
        return new_messages

    def _compress_content(self, content, compression_params):
        """
        Compress text content into images.
        Handles string content and lists of content parts.
        """
        if isinstance(content, str):
            print(f"📝 Compressing string content: {len(content)} chars")
            images = self._generate_images(content, **compression_params)
            return [{"type": "image_url", "image_url": {"url": img}} for img in images]
        elif isinstance(content, list):
            new_content = []
            for part in content:
                if part.get("type") == "text":
                    print(f"📝 Compressing text part: {len(part['text'])} chars")
                    images = self._generate_images(part["text"], **compression_params)
                    new_content.extend([{"type": "image_url", "image_url": {"url": img}} for img in images])
                else:
                    new_content.append(part)
            return new_content
        else:
            return content

    # Chat completions
    def chat_completions_create(self, messages=None, **kwargs):
        if messages is not None and 'messages' not in kwargs:
            kwargs['messages'] = messages
        compressed_messages = self._compress_messages(kwargs['messages'])
        kwargs['messages'] = compressed_messages
        return self.client.chat.completions.create(**kwargs)

    # Responses
    def responses_create(self, input=None, compression=None, **kwargs):
        if input is not None and 'input' not in kwargs:
            kwargs['input'] = input
        if compression:
            if isinstance(compression, bool):
                params = self.default_compression.copy()
            else:
                params = {**self.default_compression, **compression}
            compressed_input = self._compress_content(kwargs['input'], params)
            kwargs['input'] = compressed_input
        return self.client.responses.create(**kwargs)

    # For convenience, add the nested structure
    def _generate_images_ultra_fast(self, text, font_path, font_size, max_width, max_height, padding):
        """
        LIGHTNING FAST: Skip font loading entirely, use bitmap-style rendering.
        """
        words = text.split()
        images = []

        while words:
            # Maximize chunk size to minimize images
            chunk_size = min(800, len(words))  # Even larger chunks
            chunk_text = " ".join(words[:chunk_size])
            words = words[chunk_size:]

            # Create image directly
            image = Image.new('RGB', (max_width, max_height), 'white')
            draw = ImageDraw.Draw(image)

            # Use default bitmap font - no TTF loading at all
            font = ImageFont.load_default()

            # Ultra-simple text wrapping: fixed-width character wrapping
            chars_per_line = (max_width - 2 * padding) // 6  # Very conservative for default font
            lines = []
            remaining = chunk_text

            while remaining:
                if len(remaining) <= chars_per_line:
                    lines.append(remaining)
                    break

                # Simple break at word boundary
                break_at = chars_per_line
                while break_at > 0 and remaining[break_at - 1] != ' ':
                    break_at -= 1

                if break_at == 0:
                    break_at = chars_per_line

                lines.append(remaining[:break_at].rstrip())
                remaining = remaining[break_at:].lstrip()

                if len(lines) >= 60:  # Allow more lines
                    break

            # Render all lines at once - batch operation
            y_pos = padding
            for line in lines:
                if y_pos + 12 <= max_height - padding:  # Default font is ~11px
                    draw.text((padding, y_pos), line, font=font, fill='black')
                    y_pos += 12  # Fixed line spacing for default font
                else:
                    break

            # Minimal PNG encoding
            buffer = BytesIO()
            image.save(buffer, format='PNG', optimize=False)
            img_bytes = buffer.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            images.append(f"data:image/png;base64,{img_b64}")

        return images

    def _generate_images_reportlab(self, text, font_path, font_size, max_width, max_height, padding):
        """
        HIGH-SPEED ReportLab text rendering to images.
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_images_ultra_fast(text, font_path, font_size, max_width, max_height, padding)

        words = text.split()
        images = []

        while words:
            # Take large chunks for efficiency
            words_per_chunk = min(300, len(words))
            chunk_text = " ".join(words[:words_per_chunk])
            words = words[words_per_chunk:]

            # Use PIL to create image and draw text directly (simpler than PDF->image conversion)
            image = Image.new('RGB', (max_width, max_height), 'white')
            draw = ImageDraw.Draw(image)

            # Get cached font
            try:
                metrics = self._get_font_metrics(font_path, font_size)
                font = metrics['font']
            except:
                font = ImageFont.load_default()

            # Simple line breaking
            chars_per_line = (max_width - 2 * padding) // 8  # Conservative estimate
            words_in_chunk = chunk_text.split()
            lines = []
            current_line = ""

            for word in words_in_chunk:
                if len(current_line + " " + word) <= chars_per_line:
                    current_line += " " + word if current_line else word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                    if len(lines) >= 40:  # Limit lines
                        break

            if current_line:
                lines.append(current_line)

            # Render lines quickly
            y_pos = padding
            line_spacing = font_size + 2

            for line in lines:
                if y_pos + line_spacing <= max_height - padding:
                    draw.text((padding, y_pos), line, font=font, fill='black')
                    y_pos += line_spacing
                else:
                    break

            # Fast PNG encoding
            buffer = BytesIO()
            image.save(buffer, format='PNG', optimize=False)
            img_bytes = buffer.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            images.append(f"data:image/png;base64,{img_b64}")

        return images

    def _generate_images_lightning_fast(self, text, font_path, font_size, max_width, max_height, padding):
        """
        LIGHTNING FAST: ReportLab + pypdfium2 for maximum speed.
        Calculate exact text capacity and split accordingly.
        """
        # Register font once per session
        font_name = f"CustomFont_{font_size}"
        if not hasattr(self, '_reportlab_fonts_registered'):
            self._reportlab_fonts_registered = set()

        if font_name not in self._reportlab_fonts_registered:
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                self._reportlab_fonts_registered.add(font_name)
                print(f"✅ ReportLab font registered: {font_name}")
            except Exception as e:
                print(f"❌ ReportLab font registration failed: {e}, falling back to PIL")
                return self._generate_images_ultra_fast(text, font_path, font_size, max_width, max_height, padding)

        drawable_width = max_width - 2 * padding
        drawable_height = max_height - 2 * padding
        line_height = font_size + 2  # Tight line spacing for maximum density
        max_lines = drawable_height // line_height

        words = text.split()
        images = []

        while words:
            # Fast estimation: calculate based on average characters per word
            avg_chars_per_word = 5.5  # English average
            total_chars_available = drawable_width * max_lines // 8  # Rough char width
            estimated_words = total_chars_available // avg_chars_per_word

            # Take a reasonable chunk and let ReportLab handle exact fitting
            chunk_words = int(min(max(50, estimated_words), len(words)))
            if chunk_words == 0:
                chunk_words = int(min(100, len(words)))

            chunk_text = " ".join(words[:chunk_words])
            words = words[chunk_words:]

            # Generate PDF with ReportLab
            packet = BytesIO()
            c = canvas.Canvas(packet, pagesize=(max_width, max_height))
            c.setFont(font_name, font_size)

            # Add text with tight line spacing
            text_object = c.beginText(padding, max_height - padding - font_size)

            # Split chunk_text into lines that fit
            lines = self._split_text_into_lines(chunk_text, font_name, font_size, drawable_width)

            for line in lines[:max_lines]:
                text_object.textLine(line)

            c.drawText(text_object)
            c.save()

            # Render PDF to image with pypdfium2 (blazingly fast)
            packet.seek(0)
            try:
                pdf = pdfium.PdfDocument(packet)
                page = pdf[0]
                bitmap = page.render(scale=1, rotation=0)
                pil_image = bitmap.to_pil()

                # Convert to base64
                buffer = BytesIO()
                pil_image.save(buffer, format='PNG', optimize=True)
                img_bytes = buffer.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                images.append(f"data:image/png;base64,{img_b64}")

            except Exception as e:
                print(f"❌ PDF rendering failed: {e}, falling back to PIL")
                # Fallback to PIL for this chunk
                fallback_images = self._generate_images_ultra_fast(chunk_text, font_path, font_size, max_width, max_height, padding)
                images.extend(fallback_images)

        return images

    def _calculate_text_capacity(self, words, font_name, font_size, drawable_width, max_lines):
        """
        Calculate exactly how many words fit in the available space.
        """
        if not words:
            return 0

        # Start with a reasonable chunk and expand
        test_words = min(200, len(words))
        best_fit = 0

        # Binary search for optimal word count
        low, high = 1, len(words)
        while low <= high:
            mid = (low + high) // 2
            test_text = " ".join(words[:mid])
            lines = self._split_text_into_lines(test_text, font_name, font_size, drawable_width)

            if len(lines) <= max_lines:
                best_fit = mid
                low = mid + 1
            else:
                high = mid - 1

        return max(1, best_fit)

    def _split_text_into_lines(self, text, font_name, font_size, max_width):
        """
        Split text into lines that fit within max_width using ReportLab metrics.
        """
        # Create a temporary canvas to measure text
        temp_packet = BytesIO()
        temp_canvas = canvas.Canvas(temp_packet, pagesize=(max_width * 2, 100))

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            width = temp_canvas.stringWidth(test_line, font_name, font_size)

            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def _generate_images(self, text, font_path, font_size, max_width, max_height, padding):
        """
        Generate images using the fastest available method.
        """
        print(f"🎯 _generate_images called with {len(text)} chars")
        print(f"DEBUG: REPORTLAB_AVAILABLE={REPORTLAB_AVAILABLE}, PYPDFIUM_AVAILABLE={PYPDFIUM_AVAILABLE}")
        if REPORTLAB_AVAILABLE and PYPDFIUM_AVAILABLE:
            print(f"🚀 Using ReportLab + pypdfium2 for rendering ({len(text)} chars)")
            return self._generate_images_lightning_fast(text, font_path, font_size, max_width, max_height, padding)
        elif REPORTLAB_AVAILABLE:
            print(f"🔧 Using ReportLab for rendering ({len(text)} chars)")
            return self._generate_images_reportlab(text, font_path, font_size, max_width, max_height, padding)
        else:
            print(f"🖼️ Using PIL for rendering ({len(text)} chars)")
            return self._generate_images_ultra_fast(text, font_path, font_size, max_width, max_height, padding)

    @property
    def chat(self):
        return ChatWrapper(self)

    @property
    def responses(self):
        return ResponsesWrapper(self)


class ChatWrapper:
    def __init__(self, parent):
        self.parent = parent
        self.completions = CompletionsWrapper(parent)


class CompletionsWrapper:
    def __init__(self, parent):
        self.parent = parent

    def create(self, *args, **kwargs):
        return self.parent.chat_completions_create(*args, **kwargs)


class ResponsesWrapper:
    def __init__(self, parent):
        self.parent = parent

    def create(self, *args, compression=None, **kwargs):
        return self.parent.responses_create(*args, compression=compression, **kwargs)


class AsyncChatWrapper:
    def __init__(self, parent):
        self.parent = parent
        self.completions = AsyncCompletionsWrapper(parent)


class AsyncCompletionsWrapper:
    def __init__(self, parent):
        self.parent = parent

    async def create(self, *args, **kwargs):
        return await self.parent.chat_completions_create(*args, **kwargs)


class AsyncResponsesWrapper:
    def __init__(self, parent):
        self.parent = parent

    async def create(self, *args, compression=None, **kwargs):
        return await self.parent.responses_create(*args, compression=compression, **kwargs)

    @property
    def chat(self):
        return ChatWrapper(self)

    @property
    def responses(self):
        return ResponsesWrapper(self)


class AsyncUnLOCC:
    """
    Asynchronous wrapper for OpenAI client with optical compression support.
    """

    def __init__(self, *args, **kwargs):
        self.client = AsyncOpenAI(*args, **kwargs)
        # Default font path - users should provide their own font file
        # This is a fallback that may not work depending on installation
        self.default_compression = {
            'font_path': os.path.join(os.path.dirname(__file__), 'AtkinsonHyperlegible-Regular.ttf'),
            'font_size': 15,
            'max_width': 864,
            'max_height': 864,
            'padding': 10
        }
        # Cache for font objects and metrics
        self._font_cache = {}
        self._font_metrics_cache = {}

    def _get_font(self, font_path, font_size):
        """Cache and return font object."""
        cache_key = (font_path, font_size)
        if cache_key not in self._font_cache:
            if not os.path.exists(font_path):
                raise FileNotFoundError(f"Font file not found: {font_path}")
            self._font_cache[cache_key] = ImageFont.truetype(font_path, font_size)
        return self._font_cache[cache_key]

    def _get_font_metrics(self, font_path, font_size):
        """Cache and return font metrics."""
        cache_key = (font_path, font_size)
        if cache_key not in self._font_metrics_cache:
            font = self._get_font(font_path, font_size)
            # Get average character width using a sample string
            sample_bbox = font.getbbox("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
            avg_char_width = (sample_bbox[2] - sample_bbox[0]) / len("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

            # Get line height
            line_bbox = font.getbbox("Ag")  # Use chars that go from baseline to cap height
            line_height = (line_bbox[3] - line_bbox[1]) + (font_size // 4)

            # Get space width
            space_bbox = font.getbbox(" ")
            space_width = space_bbox[2] - space_bbox[0]

            self._font_metrics_cache[cache_key] = {
                'avg_char_width': avg_char_width,
                'line_height': line_height,
                'space_width': space_width,
                'font': font
            }
        return self._font_metrics_cache[cache_key]

    def _estimate_text_width(self, text, metrics):
        """Fast estimation of text width using cached metrics."""
        return len(text) * metrics['avg_char_width']

    def _wrap_text_fast(self, text, metrics, max_width):
        """Fast text wrapping using width estimation."""
        lines = []
        words = text.split()
        if not words:
            return []

        current_line = words[0]
        current_width = self._estimate_text_width(current_line, metrics)

        for word in words[1:]:
            word_width = self._estimate_text_width(word, metrics)
            space_width = metrics['space_width']
            new_width = current_width + space_width + word_width

            if new_width <= max_width:
                current_line += " " + word
                current_width = new_width
            else:
                lines.append(current_line)
                current_line = word
                current_width = word_width

        lines.append(current_line)
        return lines

    def _compress_content(self, content, compression_params):
        """
        Compress text content into images.
        Handles string content and lists of content parts.
        """
        if isinstance(content, str):
            print(f"📝 Compressing string content: {len(content)} chars")
            images = self._generate_images(content, **compression_params)
            return [{"type": "image_url", "image_url": {"url": img}} for img in images]
        elif isinstance(content, list):
            new_content = []
            for part in content:
                if part.get("type") == "text":
                    print(f"📝 Compressing text part: {len(part['text'])} chars")
                    images = self._generate_images(part["text"], **compression_params)
                    new_content.extend([{"type": "image_url", "image_url": {"url": img}} for img in images])
                else:
                    new_content.append(part)
            return new_content
        else:
            return content

    def _compress_messages(self, messages):
        """Compress text content in chat messages if 'compressed' is specified per message."""
        new_messages = []
        for msg in messages:
            new_msg = msg.copy()
            compressed = msg.get("compressed")
            if compressed and "content" in msg:
                if isinstance(compressed, bool):
                    params = self.default_compression.copy()
                else:
                    params = {**self.default_compression, **compressed}
                new_msg["content"] = self._compress_content(msg["content"], params)
                # Remove the 'compressed' key from the message
                new_msg.pop("compressed", None)
            new_messages.append(new_msg)
        return new_messages

    # Async chat completions
    async def chat_completions_create(self, messages=None, **kwargs):
        if messages is not None and 'messages' not in kwargs:
            kwargs['messages'] = messages
        compressed_messages = self._compress_messages(kwargs['messages'])
        kwargs['messages'] = compressed_messages
        return await self.client.chat.completions.create(**kwargs)

    # Async responses
    async def responses_create(self, input=None, compression=None, **kwargs):
        if input is not None and 'input' not in kwargs:
            kwargs['input'] = input
        if compression:
            if isinstance(compression, bool):
                params = self.default_compression.copy()
            else:
                params = {**self.default_compression, **compression}
            compressed_input = self._compress_content(kwargs['input'], params)
            kwargs['input'] = compressed_input
        return await self.client.responses.create(**kwargs)

    @property
    def chat(self):
        return AsyncChatWrapper(self)

    @property
    def responses(self):
        return AsyncResponsesWrapper(self)

    def _wrap_text_accurate(self, text, font, max_width):
        """Accurate text wrapping using actual font measurements."""
        lines = []
        words = text.split()
        if not words:
            return []
        current_line = words[0]
        for word in words[1:]:
            bbox = font.getbbox(current_line + " " + word)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines


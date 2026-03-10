
import json
from typing import Any, Generator, Iterable, Literal


class JsonArrayStreamer:
    """Streamer for JSON arrays.

    Use like:
    ```python
    streamer = JsonArrayStreamer()
    for text_fragment in text_generator:
        for json_obj in streamer.next_text(text_fragment):
            # Do something with the newly parsed JSON object(s)
        if streamer.finished:
            break
    streamer.finish()
    ```
    """
    def __init__(self) -> None:
        self.decoder = json.JSONDecoder()
        self.started = False
        self.finished = False
        self.buffer = ""

    def next_text(self, next_fragment: str) -> Generator[dict[str, Any], None, None]:
        self.buffer += next_fragment
        
        # 1. Handle the array opening '[' (and skip any preamble text)
        if not self.started:
            if '[' in self.buffer:
                self.buffer = self.buffer[self.buffer.find('[') + 1:] # Skip past '['
                self.started = True
            else:
                return
        
        # 2. Try to decode as many objects as possible from the current buffer
        while True:
            # Remove leading whitespace/commas between objects
            self.buffer = self.buffer.lstrip()
            
            # Check for end of array
            if self.buffer.startswith(']'):
                self.finished = True
                return
            
            # Skip comma separator if present
            if self.buffer.startswith(','):
                self.buffer = self.buffer[1:].lstrip()
            
            # If there is no closing brace, wait for more data
            if "}" not in self.buffer:
                return
            
            # Try to decode a JSON object
            try:
                # raw_decode returns (object, end_index)
                obj, idx = self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError:
                # We reached the end of the buffer but the object is incomplete.
                # Break the inner loop and wait for the next chunk from generator.
                return
            else:
                self.buffer = self.buffer[idx:]  # Advance buffer past this object
                yield obj
    
    def finish(self) -> None:
        if not self.started:
            raise ValueError("Missing start of array '['")
        if not self.finished:
            raise ValueError("Missing end of array ']'")


def process_json_array(text_generator: Iterable[str]) -> Generator[dict[str, Any], None, None]:
    streamer = JsonArrayStreamer()
    for text_fragment in text_generator:
        for json_obj in streamer.next_text(text_fragment):
            yield json_obj
    streamer.finish()


def process_json(text: str, bounds: Literal["{}", "[]"]) -> Any:
    try:
        start = text.index(bounds[0])
        end = text.rindex(bounds[1])
    except ValueError:
        print(f"Error finding JSON bounds in: {text!r}")
        raise
    decoder = json.JSONDecoder()
    try:
        return decoder.raw_decode(text[start:end + 1])[0]
    except json.JSONDecodeError:
        print(f"Error parsing JSON: {text!r}")
        raise

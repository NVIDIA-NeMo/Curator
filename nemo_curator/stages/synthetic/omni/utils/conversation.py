from dataclasses import dataclass
import dataclasses
from typing import Any, Literal, Optional


@dataclass(kw_only=True, slots=True)
class Media:
    """A media object in a conversation."""

    pass


@dataclass(kw_only=True, slots=True)
class ImageMedia(Media):
    """An image media object in a conversation."""

    # Relative path to the image file
    value: str


@dataclass(kw_only=True, slots=True)
class VideoMedia(Media):
    """A video media object in a conversation. May contain audio."""

    # Relative path to the video file
    value: str

    #: If set, the video needs to be trimmed to the given range in seconds.
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass(kw_only=True, slots=True)
class AudioMedia(Media):
    """An audio media object in a conversation."""

    # Relative path to the audio file
    value: str


@dataclass(kw_only=True, slots=True)
class Message:
    """A message in a conversation between a user and an assistant."""

    #: The sender of the message
    sender: Literal["user", "assistant", "system"]

    #: The message content
    fragments: list[Media | str]


@dataclass(kw_only=True, slots=True)
class ConversationSample:
    """Sample type for a conversation between a user and an assistant.

    Can include media of various types.
    """

    __MEDIA_TYPES__ = {
        "image": ImageMedia,
        "video": VideoMedia,
        "audio": AudioMedia,
    }
    __MEDIA_TYPES_REVERSE__ = {v: k for k, v in __MEDIA_TYPES__.items()}

    #: The messages in the conversation
    conversation: list[Message]

    def to_dict(self) -> dict:
        return dict(
            conversation=[
                dict(
                    sender=msg.sender,
                    fragments=[
                        frag
                        if isinstance(frag, str)
                        else dict(
                            t=ConversationSample.__MEDIA_TYPES_REVERSE__[type(frag)],
                            **dataclasses.asdict(frag),
                        )
                        for frag in msg.fragments
                    ],
                )
                for msg in self.conversation
            ],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSample":
        """Parse a serialized conversation sample.

        Expected format (as produced by :meth:`to_dict`):
        {
          "conversation": [
            {"sender": "user"|"assistant"|"system", "fragments": [str | {"t": "image"|"video"|"audio", ...}]},
            ...
          ]
        }
        """
        if not data:
            return cls(conversation=[])

        raw_conversation = data.get("conversation", [])
        if raw_conversation is None:
            return cls(conversation=[])
        if not isinstance(raw_conversation, list):
            raise TypeError(f"Expected 'conversation' to be a list, got {type(raw_conversation)}")

        conversation: list[Message] = []
        for msg in raw_conversation:
            if not isinstance(msg, dict):
                raise TypeError(f"Expected conversation message to be a dict, got {type(msg)}")
            sender = msg.get("sender")
            fragments_raw = msg.get("fragments", [])
            if not isinstance(fragments_raw, list):
                raise TypeError(f"Expected message 'fragments' to be a list, got {type(fragments_raw)}")

            fragments: list[Media | str] = []
            for frag in fragments_raw:
                if isinstance(frag, str):
                    fragments.append(frag)
                    continue
                if not isinstance(frag, dict):
                    raise TypeError(f"Expected fragment to be a str or dict, got {type(frag)}")

                # 't' is used by to_dict(); accept 'type' as a common alias.
                media_type = frag.get("t") or frag.get("type")
                if not isinstance(media_type, str):
                    raise TypeError(f"Expected media fragment to have a string 't', got {media_type!r}")
                media_cls = cls.__MEDIA_TYPES__.get(media_type)
                if media_cls is None:
                    raise ValueError(f"Unknown media type {media_type!r}; expected one of {sorted(cls.__MEDIA_TYPES__.keys())}")

                media_kwargs = {k: v for k, v in frag.items() if k not in {"t", "type"}}
                fragments.append(media_cls(**media_kwargs))

            conversation.append(Message(sender=sender, fragments=fragments))  # type: ignore[arg-type]

        return cls(conversation=conversation)

"""
Session state management for OpenFlipbook.
Keeps track of conversation history, last image, and generation state.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
import io
import base64


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    image_b64: Optional[str] = None  # base64 encoded image


@dataclass
class Session:
    id: str
    messages: list = field(default_factory=list)
    last_image_b64: Optional[str] = None
    last_image_description: Optional[str] = None
    last_prompt: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # Generation state
    is_generating: bool = False
    generation_mode: str = "image"  # "image" or "video"

    def add_user_message(self, content: str, image_b64: Optional[str] = None):
        self.messages.append(Message(role="user", content=content, image_b64=image_b64))
        self.last_active = time.time()

    def add_assistant_message(self, content: str, image_b64: Optional[str] = None):
        self.messages.append(Message(role="assistant", content=content, image_b64=image_b64))
        self.last_active = time.time()
        if image_b64:
            self.last_image_b64 = image_b64

    def get_history_text(self) -> str:
        """Get conversation history as formatted text."""
        history = []
        for msg in self.messages[-10:]:  # Last 10 messages
            role = "User" if msg.role == "user" else "Assistant"
            history.append(f"{role}: {msg.content}")
        return "\n".join(history)

    def to_dict(self):
        return {
            "id": self.id,
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self.messages
            ],
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "last_active": self.last_active,
            "is_generating": self.is_generating,
        }


class SessionManager:
    def __init__(self, max_sessions: int = 100):
        self.sessions: dict[str, Session] = {}
        self.max_sessions = max_sessions

    def create(self) -> Session:
        """Create a new session."""
        sid = str(uuid.uuid4())[:8]
        session = Session(id=sid)
        self.sessions[sid] = session
        self._cleanup()
        return session

    def get(self, sid: str) -> Optional[Session]:
        """Get session by ID."""
        return self.sessions.get(sid)

    def _cleanup(self):
        """Remove old sessions if over max."""
        if len(self.sessions) > self.max_sessions:
            # Remove oldest by last_active
            oldest = sorted(self.sessions.items(), key=lambda x: x[1].last_active)
            for sid, _ in oldest[:10]:
                del self.sessions[sid]


# Global session manager
manager = SessionManager()

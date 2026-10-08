"""
Contabilidad de tokens de OpenAI por petición.

`TrackedOpenAI` envuelve un cliente AsyncOpenAI y acumula el `usage` de cada
`chat.completions.create` (no streaming), sin tocar los agentes: la API lo usa
para registrar el consumo por organización (cuotas y facturación).
"""

from __future__ import annotations

from typing import Any


class _TrackedCompletions:
    def __init__(self, completions: Any, tracker: "TrackedOpenAI"):
        self._completions = completions
        self._tracker = tracker

    async def create(self, **kwargs: Any) -> Any:
        response = await self._completions.create(**kwargs)
        self._tracker.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._tracker.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._tracker.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _TrackedChat:
    def __init__(self, chat: Any, tracker: "TrackedOpenAI"):
        self.completions = _TrackedCompletions(chat.completions, tracker)


class TrackedOpenAI:
    """Proxy de AsyncOpenAI que suma los tokens de las llamadas de chat."""

    def __init__(self, client: Any):
        self._client = client
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.chat = _TrackedChat(client.chat, self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

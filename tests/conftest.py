import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeCompletions:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    """Sustituto mínimo de AsyncOpenAI: registra las llamadas a chat.completions.create."""

    def __init__(self, answer: str = "respuesta de prueba"):
        self.chat = SimpleNamespace(completions=FakeCompletions(answer))

    @property
    def calls(self):
        return self.chat.completions.calls


@pytest.fixture
def fake_openai():
    return FakeOpenAI()

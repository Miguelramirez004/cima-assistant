from cima_core import run_consulta, run_formulacion, run_prospecto
from cima_core.cima_rag import CIMARagAgent
from cima_core.formulacion import FormulationAgent
from cima_core.prospecto import ProspectoGenerator
from cima_core.service import extract_references


def test_extract_references_dedupes_and_builds_urls():
    text = ("Según [Ref 1: IBUPROFENO KERN 600 mg (Nº Registro: 12345)] y "
            "[Ref 2: DALSY (Nº Registro: 67890)](https://x) y de nuevo "
            "[Ref 1: IBUPROFENO KERN 600 mg (Nº Registro: 12345)]")
    refs = extract_references(text)
    assert [r.nregistro for r in refs] == ["12345", "67890"]
    assert refs[0].url == "https://cima.aemps.es/cima/dochtml/ft/12345/FichaTecnica.html"


async def test_formulacion_redirects_prospecto_without_calling_openai(fake_openai):
    result = await run_formulacion("Redactar un prospecto de ibuprofeno", openai_client=fake_openai)
    assert result.redirect == "prospecto"
    assert fake_openai.calls == []


async def test_formulacion_maps_result(monkeypatch, fake_openai):
    async def fake_answer(self, question):
        return {"answer": "Fórmula [Ref 1: X (Nº Registro: 111)]", "context": "ctx", "references": 1}

    monkeypatch.setattr(FormulationAgent, "answer_question", fake_answer)
    result = await run_formulacion("Suspensión oral de omeprazol 2 mg/ml", openai_client=fake_openai)
    assert result.success and result.redirect is None
    assert result.context == "ctx"
    assert result.references[0].nregistro == "111"


async def test_formulacion_error_is_reported(monkeypatch, fake_openai):
    async def boom(self, question):
        raise RuntimeError("OpenAI caído")

    monkeypatch.setattr(FormulationAgent, "answer_question", boom)
    result = await run_formulacion("Suspensión oral de omeprazol", openai_client=fake_openai)
    assert result.success is False


async def test_consulta_maps_references_and_history(monkeypatch, fake_openai):
    seen = {}

    async def fake_ask(self, question, history=None):
        seen["history"] = history
        return {
            "answer": "ok", "reasoning": "• paso", "success": True,
            "references": [{"title": "DALSY", "url": "https://cima.aemps.es/cima/dochtml/ft/67890/FichaTecnica.html"}],
        }

    monkeypatch.setattr(CIMARagAgent, "ask", fake_ask)
    result = await run_consulta("¿y la dosis?", [{"role": "user", "content": "ibuprofeno"}],
                                openai_client=fake_openai)
    assert seen["history"] == [{"role": "user", "content": "ibuprofeno"}]
    assert result.references[0].nregistro == "67890"
    assert result.reasoning == "• paso"


async def test_prospecto_maps_result(monkeypatch, fake_openai):
    async def fake_generate(self, query):
        return {"prospecto": "PROSPECTO", "context": "Nombre: DALSY\nNº Registro: 67890",
                "medication": "DALSY"}

    monkeypatch.setattr(ProspectoGenerator, "generate_prospecto", fake_generate)
    result = await run_prospecto("Prospecto de ibuprofeno", openai_client=fake_openai)
    assert result.success and result.content == "PROSPECTO"
    assert result.medication_name == "DALSY" and result.nregistro == "67890"

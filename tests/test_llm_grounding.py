"""Model-based claim checking and extraction, with scripted fake models."""

import json
import re
import threading
import time
from types import SimpleNamespace as NS

import pytest

from agent_evals.agents import FunctionAgent
from agent_evals.grounding import LLMChecker, LLMClaimExtractor, check_grounding
from agent_evals.llm import FunctionModel, OpenAIChatModel, TextModel
from agent_evals.models import AgentResult, Source, TestCase
from agent_evals.runner import run_eval_sync
from agent_evals.scorers import get_scorer

SOURCES = [
    Source(
        id="policy", text="Refunds are available within 30 days of purchase. Shipping costs are not refunded."
    ),
    Source(id="faq", text="Refunds are paid back to the original payment method."),
]


def judge(verdicts):
    """A fake checker model: looks up the claim in the prompt and replies with a scripted verdict."""

    def reply(prompt):
        claim = re.search(r"^Claim: (.*)$", prompt, re.MULTILINE).group(1)
        return verdicts[claim]

    return FunctionModel(reply)


def verdict(v, source_id=None, quote=None):
    return json.dumps({"verdict": v, "source_id": source_id, "quote": quote})


def test_paraphrase_supported_with_a_real_quote():
    model = judge(
        {
            "You get your money back within a month.": verdict(
                "supported", "policy", "Refunds are available within 30 days of purchase."
            )
        }
    )
    v = LLMChecker(model).check("You get your money back within a month.", SOURCES)
    assert v.supported and v.source_id == "policy"


@pytest.mark.parametrize(
    "reply, reason",
    [
        (verdict("contradicted", "policy", "Shipping costs are not refunded."), "contradicted by sources"),
        (verdict("not_found"), "not in sources"),
        (verdict("supported", "policy", "Shipping is always free."), "checker's quote isn't in the source"),
        (verdict("supported", "blog", "x"), "checker cited unknown source 'blog'"),
        ("I think it's fine!", "couldn't read the checker's reply"),
    ],
)
def test_unsupported_verdicts(reply, reason):
    v = LLMChecker(judge({"We refund shipping.": reply})).check("We refund shipping.", SOURCES)
    assert not v.supported and v.reason.startswith(reason)


def test_quotes_are_matched_loosely():
    model = judge({"c": verdict("supported", "faq", "refunds are paid back to the original payment method")})
    assert LLMChecker(model).check("c", SOURCES).supported


def test_extractor_splits_into_atomic_claims_and_falls_back():
    model = FunctionModel(lambda p: '["Refunds take 30 days.", "Refunds include shipping."]')
    assert LLMClaimExtractor(model)("Refunds take 30 days and include shipping.") == [
        "Refunds take 30 days.",
        "Refunds include shipping.",
    ]
    broken = FunctionModel(lambda p: "Here are the claims: refunds, shipping")
    assert LLMClaimExtractor(broken)("Refunds take 30 days. Shipping is free.") == [
        "Refunds take 30 days.",
        "Shipping is free.",
    ]


def test_scorer_with_model_end_to_end():
    claims = FunctionModel(lambda p: '["Refunds are available for 30 days.", "Shipping is refunded too."]')
    checker = judge(
        {
            "Refunds are available for 30 days.": verdict(
                "supported", "policy", "Refunds are available within 30 days of purchase."
            ),
            "Shipping is refunded too.": verdict(
                "contradicted", "policy", "Shipping costs are not refunded."
            ),
        }
    )

    class Both:
        def complete(self, prompt):
            return (claims if prompt.startswith("Break this answer") else checker).complete(prompt)

    assert isinstance(Both(), TextModel)
    scorer = get_scorer("groundedness", model=Both(), extract_with_model=True)
    case = TestCase(id="c", input="refunds?", sources=SOURCES)
    score = scorer(
        case, AgentResult(output="Refunds are available for 30 days and shipping is refunded too.")
    )
    assert score.value == 0.5 and not score.passed
    assert "'Shipping is refunded too.' (contradicted by sources)" in score.reason


def test_extract_with_model_needs_a_model():
    with pytest.raises(ValueError):
        get_scorer("groundedness", extract_with_model=True)


def test_openai_chat_model_adapter():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return NS(choices=[NS(message=NS(content="ok"))])

    model = OpenAIChatModel(NS(chat=NS(completions=NS(create=create))), "judge-model", system="Be strict.")
    assert model.complete("hi") == "ok"
    assert sent["model"] == "judge-model" and sent["temperature"] == 0.0
    assert sent["messages"] == [
        {"role": "system", "content": "Be strict."},
        {"role": "user", "content": "hi"},
    ]


def test_slow_sync_scorers_run_concurrently():
    running = 0
    peak = 0
    lock = threading.Lock()

    def slow_scorer(case, result):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return get_scorer("contains")(case, result)

    cases = [TestCase(id=str(i), input="q", expected_output="yes") for i in range(4)]
    run = run_eval_sync(FunctionAgent(lambda q: "yes"), cases, [slow_scorer], concurrency=4, commit="")
    assert run.pass_rate == 1.0 and peak > 1


def test_check_grounding_with_custom_extractor():
    report = check_grounding("anything", SOURCES, extract=lambda a: ["Refunds are available within 30 days."])
    assert report.groundedness == 1.0

from agent_evals.grounding import LexicalChecker, check_grounding, content_words, split_claims
from agent_evals.models import AgentResult, Source, Step, TestCase, ToolCall, Trace
from agent_evals.scorers import get_scorer

POLICY = Source(
    id="refunds",
    text=(
        "Refunds are available within 30 days of purchase. Items must be unused and in their original "
        "packaging. Shipping costs are not refunded. Store credit is offered after 30 days."
    ),
)


def test_split_claims_skips_filler_and_questions():
    answer = (
        "Sure! You can return it within 30 days.\n"
        "- It must be unused\n"
        "2) Keep the original packaging.\n"
        "Would you like the return label? I hope this helps."
    )
    assert split_claims(answer) == [
        "You can return it within 30 days.",
        "It must be unused",
        "Keep the original packaging.",
    ]


def test_content_words():
    assert content_words("The flights to Berlin's airports") == {"flight", "berlin", "airport"}


def test_supported_and_invented_claims():
    report = check_grounding(
        "Refunds are available within 30 days of purchase. Refunds take 5 business days. "
        "You can also get a refund through PayPal Express.",
        [POLICY],
    )
    supported = [v.supported for v in report.verdicts]
    assert supported == [True, False, False]
    assert report.groundedness == 1 / 3
    assert report.verdicts[0].source_id == "refunds"
    assert report.verdicts[1].reason == "number not in sources: 5"
    assert "paypal" in report.verdicts[2].reason


def test_wrong_number_is_unsupported_even_with_matching_words():
    v = LexicalChecker().check("Refunds are available within 60 days of purchase.", [POLICY])
    assert not v.supported and v.reason == "number not in sources: 60"


def test_claim_spanning_two_sentences():
    v = LexicalChecker().check("Items must be unused, and shipping costs are not refunded.", [POLICY])
    assert v.supported


def test_no_claims_is_fully_grounded():
    assert check_grounding("Happy to help! Anything else?", [POLICY]).groundedness == 1.0


def scored(answer, sources=(POLICY,), trace=None, **opts):
    case = TestCase(id="c", input="refund policy?", sources=list(sources))
    return get_scorer("groundedness", **opts)(case, AgentResult(output=answer, trace=trace or Trace()))


def test_groundedness_scorer():
    good = scored("Refunds are available within 30 days of purchase. Items must be unused.")
    assert good.passed and good.value == 1.0 and good.reason == "2 of 2 claims supported"

    bad = scored("Refunds are available within 30 days of purchase. Refunds take 5 business days.")
    assert not bad.passed and bad.value == 0.5
    assert "Unsupported: 'Refunds take 5 business days.' (number not in sources: 5)" in bad.reason

    lenient = scored(
        "Refunds are available within 30 days of purchase. Refunds take 5 business days.", threshold=0.5
    )
    assert lenient.passed


def test_tool_results_count_as_sources():
    trace = Trace(
        steps=[
            Step(
                type="tool",
                tool_call=ToolCall(
                    name="get_order", arguments={"id": "A1"}, result={"status": "shipped", "carrier": "DHL"}
                ),
            )
        ]
    )
    answer = "Your order status is shipped with carrier DHL."
    assert scored(answer, trace=trace).passed
    assert not scored(answer, trace=trace, include_tool_results=False).passed


def test_scorer_only_applies_with_sources():
    scorer = get_scorer("groundedness")
    assert not scorer.applies_to(TestCase(id="c", input="q"))
    assert scorer.applies_to(TestCase(id="c", input="q", sources=[POLICY]))

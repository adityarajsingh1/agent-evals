import json
from pathlib import Path

import pytest

from agent_evals.calibration import AgreementReport, LabelledExample, calibrate, load_labels
from agent_evals.llm import FunctionModel
from agent_evals.models import Score
from agent_evals.scorers import get_scorer

LABELS = Path(__file__).parent.parent / "examples" / "judge_labels.jsonl"


def naive_judge():
    """A deliberately naive fake judge: likes any answer that mentions '30 days'."""

    def complete(prompt):
        answer = prompt.split("Answer to grade:\n", 1)[1].split("\n\nScore each", 1)[0]
        s = 5 if "30 days" in answer else 2
        return json.dumps({"scores": {"correct": s, "complete": s}})

    return get_scorer("judge", model=FunctionModel(complete))


def test_load_example_labels():
    examples = load_labels(LABELS)
    assert len(examples) == 8
    assert examples[1].result.output == "You can return items within 60 days."
    assert examples[1].note == "wrong window"


def test_calibrate_naive_judge_on_examples():
    report = calibrate(naive_judge(), load_labels(LABELS))
    # passes r1, r4, r8 (mention "30 days"); people passed r1, r3, r5, r7, r8
    assert (report.both_pass, report.false_pass, report.false_fail, report.both_fail) == (2, 1, 3, 2)
    assert report.accuracy == 0.5
    # chance agreement: judge passes 3/8, humans 5/8 -> 3/8*5/8 + 5/8*3/8 = 0.46875
    assert report.kappa == pytest.approx((0.5 - 0.46875) / (1 - 0.46875))
    assert report.false_pass_rate == pytest.approx(1 / 3)
    assert report.false_fail_rate == pytest.approx(3 / 5)
    assert not report.trustworthy()
    assert [d.case_id for d in report.disagreements] == ["r3", "r4", "r5", "r7"]
    assert report.disagreements[1].note == "contradicts policy"
    assert report.mean_abs_error is not None
    assert report.summary().startswith("8 labelled answers: accuracy 50%, kappa 0.06")


def test_perfect_judge():
    def oracle(case, result):
        good = result.output in {"good"}
        return Score(name="oracle", value=float(good), passed=good)

    examples = [
        LabelledExample.model_validate(
            {"case": {"id": str(i), "input": "q"}, "output": out, "human_passed": out == "good"}
        )
        for i, out in enumerate(["good", "bad", "good", "bad"])
    ]
    report = calibrate(oracle, examples)
    assert report.accuracy == 1.0 and report.kappa == 1.0 and report.trustworthy()


def test_kappa_edge_cases():
    assert AgreementReport().kappa == 0.0
    everyone_passes = AgreementReport(both_pass=5)
    assert everyone_passes.kappa == 1.0 and everyone_passes.false_pass_rate is None


def test_label_file_errors(tmp_path):
    bad = tmp_path / "labels.jsonl"
    bad.write_text('{"case": {"id": "a", "input": "q"}, "output": "x"}\n')
    with pytest.raises(ValueError, match="labels.jsonl:1"):
        load_labels(bad)
    with pytest.raises(ValueError, match="either 'output' or 'result'"):
        LabelledExample.model_validate(
            {"case": {"id": "a", "input": "q"}, "output": "x", "result": {}, "human_passed": True}
        )

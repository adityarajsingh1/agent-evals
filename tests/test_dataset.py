from pathlib import Path

import pytest

from agent_evals.dataset import DatasetError, load_dataset

EXAMPLE = Path(__file__).parent.parent / "examples" / "weather.jsonl"


def test_example_dataset_loads():
    cases = load_dataset(EXAMPLE)
    assert [c.id for c in cases] == ["berlin-now", "compare", "no-tool"]
    assert cases[1].expected_tools[1].arguments == {"city": "Oslo"}


@pytest.mark.parametrize(
    "lines, message",
    [
        (['{"id": "a", "input": "hi"', ""], ":1: not valid JSON"),
        (['{"id": "a"}'], ":1: input: Field required"),
        (["", '{"id": "a", "input": "hi", "extra": 1}'], ":2: extra: Extra inputs are not permitted"),
        (
            ['{"id": "a", "input": "hi"}', '{"id": "a", "input": "again"}'],
            ":2: duplicate id 'a' (first used on line 1)",
        ),
        (["# only a comment", ""], "no test cases"),
    ],
)
def test_errors_point_at_the_line(tmp_path, lines, message):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines))
    with pytest.raises(DatasetError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load_dataset(path)

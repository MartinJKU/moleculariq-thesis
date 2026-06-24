import pytest

from miqthesis.training.rewards import (
    compare_json,
    extract_answer_block,
    format_reward,
    parse_json_answer,
    verifier_reward_value,
)


def test_answer_parsing_and_normalization():
    text = '<answer>{"atoms": [2, 1], "count": 2}</answer>'
    assert extract_answer_block(text) == '{"atoms": [2, 1], "count": 2}'
    assert parse_json_answer(text) == {"atoms": [2, 1], "count": 2}
    assert compare_json(parse_json_answer(text), {"count": 2, "atoms": [1, 2]})["exact"] == 1


def test_format_reward_is_bounded_and_requires_clean_suffix():
    assert format_reward('<answer>{"count": 2}</answer>') == pytest.approx(1.0)
    assert format_reward('<answer>{"count": 2}</answer> explanation') == pytest.approx(0.75)
    assert format_reward("not json") == pytest.approx(0.0)


def test_verifier_reward_orders_count_answers():
    target = {"ring_count": 3, "oxygen_count": 1}
    perfect = verifier_reward_value(
        '<answer>{"ring_count": 3, "oxygen_count": 1}</answer>', target, "count"
    )
    partial = verifier_reward_value(
        '<answer>{"ring_count": 3, "oxygen_count": 9}</answer>', target, "count"
    )
    wrong = verifier_reward_value(
        '<answer>{"other": 3}</answer>', target, "count"
    )
    invalid = verifier_reward_value("oops", target, "count")
    assert perfect > partial > wrong > invalid


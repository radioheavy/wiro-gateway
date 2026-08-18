import pytest

from wiro_gateway.extract import extract_assistant_text, _find_text_in_obj


def test_extract_from_debugoutput():
    detail = {"tasklist": [{"debugoutput": "Hello, world!", "status": "task_postprocess_end", "outputs": []}]}
    assert extract_assistant_text(detail) == "Hello, world!"


def test_extract_from_outputs_content():
    detail = {"tasklist": [{"debugoutput": "", "outputs": [{"content": "from content", "contenttype": "text/plain"}]}]}
    assert extract_assistant_text(detail) == "from content"


def test_extract_from_outputs_text():
    detail = {"tasklist": [{"debugoutput": "", "outputs": [{"text": "from text"}]}]}
    assert extract_assistant_text(detail) == "from text"


def test_extract_walks_json_content():
    detail = {"tasklist": [{"debugoutput": "", "outputs": [{"content": '{"text": "nested"}'}]}]}
    assert extract_assistant_text(detail) == "nested"


def test_extract_parameters_legacy():
    detail = {"tasklist": [{"debugoutput": "", "outputs": [], "parameters": {"result": "legacy"}}]}
    assert extract_assistant_text(detail) == "legacy"


def test_extract_raises_when_nothing_found():
    detail = {"tasklist": [{"debugoutput": "", "outputs": [], "parameters": {}}]}
    with pytest.raises(ValueError):
        extract_assistant_text(detail)


def test_extract_raises_empty_tasklist():
    with pytest.raises(ValueError):
        extract_assistant_text({"tasklist": []})


def test_find_text_in_obj():
    obj = {"foo": {"bar": [{"baz": "hi"}]}}
    assert _find_text_in_obj(obj) is None
    obj2 = {"foo": {"bar": [{"text": "found it"}]}}
    assert _find_text_in_obj(obj2) == "found it"


def test_extract_strips_answer_wrapper():
    detail = {"tasklist": [{"debugoutput": "<answer>Hello world</answer>", "outputs": []}]}
    assert extract_assistant_text(detail) == "Hello world"


def test_extract_keeps_mismatched_tags():
    detail = {"tasklist": [{"debugoutput": "<answer>oops</wrong>", "outputs": []}]}
    assert extract_assistant_text(detail) == "<answer>oops</wrong>"


def test_extract_keeps_midtext_tags():
    detail = {"tasklist": [{"debugoutput": "before <answer>middle</answer> after", "outputs": []}]}
    assert extract_assistant_text(detail) == "before <answer>middle</answer> after"

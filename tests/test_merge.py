from wiro_gateway.wiro_to_params import merge_messages


def test_basic_chat():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi!"},
    ]
    sys, prompt = merge_messages(msgs, None)
    assert "You are helpful." in sys
    assert "[User]" in prompt
    assert "Hi!" in prompt
    assert prompt.endswith("[Assistant]\n")


def test_multi_turn():
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ]
    sys, prompt = merge_messages(msgs, "Be brief.")
    assert sys == "Be brief."
    assert "[User]\nA" in prompt
    assert "[Assistant]\nB" in prompt
    assert "[User]\nC" in prompt


def test_empty_messages_gets_fallback():
    sys, prompt = merge_messages([], None)
    assert "Hello." in prompt
    assert prompt.endswith("[Assistant]\n")


def test_top_level_system_prompts_concatenated():
    msgs = [{"role": "system", "content": "A"}, {"role": "user", "content": "B"}]
    sys, _ = merge_messages(msgs, "Top level")
    assert "Top level" in sys
    assert "A" in sys

import types


def _mk_msg(role: str):
    # Minimal shape used by summarize_via_sliding_window.
    return types.SimpleNamespace(
        role=role,
        name=None,
        content=[],
        tool_calls=[],
        tool_returns=None,
        tool_call_id=None,
    )


def test_sliding_window_cutoff_progresses(monkeypatch):
    """Regression test: cutoff index should monotonically progress for small message counts.

    Previously, `round(eviction_percentage * n)` could repeat the same cutoff (e.g. 1)
    across many iterations, spamming logs and stalling convergence.
    """

    from letta.services.summarizer import summarizer_sliding_window as sw

    attempted_cutoffs = []

    async def fake_count_tokens(actor, llm_config, messages):
        # Return a value that decreases as we keep fewer messages (i.e. higher cutoff).
        # goal is (1 - 0.3) * 10000 = 7000, so we want to converge once we keep <= 3 messages.
        kept_count = len(messages)
        return kept_count * 2000

    def fake_find_safe_cutoff_index(*, messages, target_cutoff_index, **kwargs):
        # Pretend every index is safe.
        attempted_cutoffs.append(target_cutoff_index)
        return min(max(1, target_cutoff_index), len(messages) - 1)

    async def fake_simple_summary(**kwargs):
        return "ok"

    monkeypatch.setattr(sw, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(sw, "_find_safe_cutoff_index", fake_find_safe_cutoff_index)
    monkeypatch.setattr(sw, "simple_summary", fake_simple_summary)

    # Construct 1 system + 6 conversation messages.
    msgs = [_mk_msg("system")] + [_mk_msg("assistant") for _ in range(6)]

    llm_cfg = types.SimpleNamespace(model_endpoint_type="openai", model="x", context_window=10000)
    summ_cfg = types.SimpleNamespace(sliding_window_percentage=0.3, prompt_acknowledgement=False, prompt=None, clip_chars=None)

    # Run the async function.
    import asyncio

    summary, kept = asyncio.run(
        sw.summarize_via_sliding_window(
            actor=types.SimpleNamespace(),
            llm_config=llm_cfg,
            summarizer_config=summ_cfg,
            in_context_messages=msgs,
        )
    )

    assert isinstance(summary, str)
    assert kept and kept[0].role == "system"
    # Cutoffs should progress (no repeated 1,1,1... loops).
    assert attempted_cutoffs
    assert len(attempted_cutoffs) == len(set(attempted_cutoffs))

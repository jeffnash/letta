# Fork Merge Guide

This document describes the process for merging upstream `letta-ai/letta` changes into the `jeffnash/letta` fork, including known issues and pitfalls to watch for.

## Overview

This fork (`jeffnash/letta`) maintains custom modifications on top of the upstream `letta-ai/letta` repository. Periodically, we need to merge upstream changes to stay current while preserving our customizations.

## Prerequisites

Ensure your remotes are configured:
```bash
git remote -v
# Should show:
# origin    git@github.com:jeffnash/letta.git (fetch/push)
# upstream  git@github.com:letta-ai/letta.git (fetch/push)
```

If `upstream` is not configured:
```bash
git remote add upstream git@github.com:letta-ai/letta.git
```

## Merge Process

### 1. Fetch and Merge

```bash
git fetch upstream
git merge upstream/main --no-edit
```

### 2. Resolve Conflicts

When conflicts occur, keep these fork customizations:

| File/Area | Our Customization | Resolution |
|-----------|-------------------|------------|
| `README.md` | Fork notice at top | Keep our fork notice + upstream content |
| `letta/schemas/providers/__init__.py` | `CLIProxyProvider` import/export | Keep our additions |
| `letta/schemas/providers/base.py` | `CLIProxyProvider` in imports | Keep our additions |
| `letta/agents/letta_agent.py` | `step_id=step_id` in telemetry | Keep our additions |
| `letta/services/agent_manager.py` | `organization_id` in BlocksTags | Keep our version |
| `letta/services/streaming_service.py` | `completed_at` timestamp for runs | Keep our version |
| `letta/server/rest_api/routers/v1/messages.py` | Enhanced `to_letta_messages()` | Keep our version |
| `letta/server/rest_api/routers/v1/conversations.py` | `Request` import | Keep our version |

### 3. Post-Merge Audit (CRITICAL)

After resolving conflicts, run these audits to catch common merge issues:

#### A. Nested Function Shadowing

Check for nested functions that shadow module-level helpers:

```bash
# Find nested async function definitions
grep -rn "^[[:space:]]\+async def _" letta/ --include="*.py"

# Compare with module-level definitions
grep -rn "^async def _" letta/ --include="*.py"
```

**Known Issue**: `letta/services/summarizer/summarizer.py`

Upstream sometimes adds a nested `_run_summarizer_request` inside `simple_summary()` that shadows the module-level helper. The nested version has signature `(req_data, req_messages_obj)` while callers expect `(llm_client=, summarizer_llm_config=, request_data=, input_messages_obj=)`.

**Fix**: Delete the nested function, keep only the module-level one with telemetry methods.

#### B. Duplicate Method Definitions

Check for duplicate method definitions in the same class:

```bash
# Find duplicate function names in same file
for file in $(find letta/ -name "*.py" -type f); do
  funcs=$(grep -oP "(?<=def )[a-zA-Z_][a-zA-Z0-9_]*" "$file" 2>/dev/null | sort | uniq -d)
  if [ -n "$funcs" ]; then
    echo "=== $file ===" 
    echo "$funcs"
  fi
done
```

**Known Issue**: `letta/services/block_manager.py`

Upstream has duplicate `_move_block_to_sequence` method definitions. Delete the second occurrence.

#### C. Telemetry Method Consistency

Ensure summarization code uses telemetry-enabled methods:

```bash
grep -n "stream_async\|request_async" letta/services/summarizer/summarizer.py
```

Should use:
- `stream_async_with_telemetry()` (not `stream_async()`)
- `request_async_with_telemetry()` (not `request_async()`)
- `log_provider_trace_async()` after streaming operations

### 4. Verify No Conflict Markers

```bash
grep -rn "<<<<<<< HEAD" letta/ || echo "No conflict markers found"
```

### 5. Stage and Commit

```bash
git add -A
git commit -m "Merge upstream/main into jeffnash/main"
```

## Common Pitfalls

### 1. Nested Function Shadowing (CRITICAL)

**Symptom**: Runtime error like `got an unexpected keyword argument 'llm_client'`

**Cause**: A nested function inside another function has the same name as a module-level function but different signature. Python's scoping rules mean the nested function shadows the module-level one within that scope.

**Example**:
```python
# Module level - correct signature with telemetry
async def _run_summarizer_request(
    llm_client: LLMClient,
    summarizer_llm_config: LLMConfig,
    request_data: dict,
    input_messages_obj: list[Message],
) -> str:
    ...

async def simple_summary(...):
    # BAD: This shadows the module-level function!
    async def _run_summarizer_request(req_data: dict, req_messages_obj: list[Message]) -> str:
        ...
    
    # This call will fail because it hits the nested function
    result = await _run_summarizer_request(
        llm_client=llm_client,  # Error: unexpected keyword argument
        ...
    )
```

**Prevention**: After merges, search for nested function definitions and verify they don't shadow module-level helpers.

### 2. Missing Telemetry Methods

**Symptom**: Missing telemetry data, or subtle behavior differences

**Cause**: Upstream code might use non-telemetry methods (`stream_async`, `request_async`) while our fork expects telemetry variants.

**Fix**: Replace with `*_with_telemetry()` variants and add `log_provider_trace_async()` calls where needed.

### 3. Parameter Naming Inconsistencies

Watch for these common parameter name mismatches:
- `req_data` vs `request_data`
- `req_messages_obj` vs `input_messages_obj`
- `llm_config` vs `summarizer_llm_config`

## Testing After Merge

1. Run the test suite:
   ```bash
   uv run pytest -s tests
   ```

2. Start the server and verify basic functionality:
   ```bash
   uv run letta server
   ```

3. Test summarization specifically (common source of merge issues):
   ```bash
   # Trigger a summarization by filling context window
   ```

## Related Repositories

When merging this repo, you may also need to merge:
- `jeffnash/letta-code` (CLI client) - has similar upstream merge process
- `jeffnash/CLIProxyAPI` (proxy layer)

Coordinate merges across repos to ensure compatibility.

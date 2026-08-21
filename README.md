# omp-bash-classifier

Model-judged bash permission requests for OMP. Commands the host would run without asking anybody get classified first: trivial ones still run silently, risky ones raise a real permission request instead of executing unnoticed. Inspired by upstream OMP PR [#6263](https://github.com/can1357/oh-my-pi) ("approve-for-me approval mode with LLM reviewer", P3-rejected) and [#8683](https://github.com/can1357/oh-my-pi).

## What it does

Registers a `tool_call` handler for `bash`. The native bash tool is untouched — same description, same schema (including `async`), same approval declaration, same execution path. The plugin only sits in front of it and may block with a reason or ask the user first.

`tool_call` fires before the approval gate for model-issued calls, may await a human dialog (the runner pauses its handler budget across dialogs, so the human is not on a clock), and the runner fails closed if the handler throws or times out (`extensionHandlers.toolCallTimeoutMs`, 30s default).

Native approval precedence is **deny > CRITICAL > allow > prompt** (`tools/bash.ts:557-577`), so a critical-pattern hit outranks any `allow` or `prompt` rule that also matches. The plugin evaluates in that same order:

| Command state | Plugin behavior |
|---|---|
| `bash.patterns` `deny` rule, or `tools.approval.bash: deny` | untouched — native gate blocks it |
| Critical pattern (`CRITICAL_BASH_PATTERNS`) | permission request, in every approval mode, no model call |
| Caller-supplied `env` | permission request, before prompt/allow exemptions — env values can hold secrets, and `PATH`/`BASH_ENV`/`LD_PRELOAD`/`GIT_PAGER` decide which program actually runs |
| `bash.patterns` `prompt` rule | untouched — native force-prompts, in every mode including `yolo` |
| `bash.patterns` narrow `allow` rule | untouched after the env check — an explicit decision about the command string, no model call |
| Blanket `allow` rule (`*`, `**`, `* *` — any all-star pattern) | classified — a blanket "run everything" is what this plugin refines |
| Command longer than 2,000 chars | blocked — neither the classifier nor the dialog can review unseen content |
| No matching pattern rule | classified in every approval mode: SAFE passes to the native gate; UNSAFE/UNSURE raise a plugin request |
| `tools.approval.bash: prompt` with no matching pattern rule | untouched — native prompts; a matching pattern rule's decision takes precedence natively |

The interactive request is a **Run / Deny** confirmation. Its message — not an option description — contains the full (≤2,000-character) command plus native-resolved cwd, `env` keys, `pty`, timeout, and async state as an indented JSON code block. The verbatim block prevents command Markdown (`<!-- … -->`, emphasis, backticks) from disappearing or changing in the TUI. This also matters outside the TUI: ACP/RPC selectors forward option labels but omit descriptions, while confirmation messages are carried on every UI adapter.

## Why critical patterns need this

Native bash marks `CRITICAL_BASH_PATTERNS` hits `{ tier: "exec", override: true }` with **no explicit policy**. `resolveApproval` (`tools/approval.ts:156-171`) ignores `override` in `yolo` mode, so in `yolo` a critical command — `curl … | sh`, `rm -rf /`, `dd of=/dev/…`, `mkfs`, `kill -9 1`, `nc -e` — is auto-approved and just runs. Worse, because critical outranks the `prompt` branch natively, a `prompt` rule you wrote for exactly those commands (`rm -rf * → prompt`) never fires for the critical spelling of them. This plugin checks critical patterns before it honors any rule, in every mode, and turns them into a permission request.

## Safety model

1. The plugin can only **add** friction: it blocks, or asks. It never bypasses the native gate and never runs a command itself.
2. Static rules stay authoritative below the critical and env checks. `deny` and `prompt` pattern rules, and narrow `allow` rules, are otherwise honored untouched; the plugin reads `bash.patterns` with the same tokenizer (`tokenizeShellSegments`) and the same allow-rule shell-control guard as the builtin, so it agrees with the native gate about which rule matches. "Narrow" means breadth, not spelling: `**` and `* *` are blanket rules and are classified like `*`. Native pattern decisions outrank non-deny `tools.approval.bash` policies; the plugin mirrors that precedence.
3. Fail-closed: a command over 2,000 characters is blocked outright. An `env` override, classifier error/timeout, and malformed verdict raise a permission request when a UI exists and **block** when headless. An unexpected plugin throw always blocks. A command the gate could not judge is never silently auto-run.
4. Verdict parsing is anchored to the start of the reply, so a model that reasons aloud and mentions `SAFE` mid-answer cannot produce a SAFE verdict. Injection is addressed separately: command and cwd are JSON data between per-call random delimiters, and the prompt ends with a mechanical scan — text addressing the reviewer, naming a verdict, claiming a part is an inert example or already approved, or imitating the delimiter makes the verdict UNSAFE. That is a measured mitigation, not a guarantee: see the model table for how much of it survives per model.
5. Session-scoped cache keyed by (session, native-resolved working directory, `env`, `pty`, timeout, async, command) — every execution-affecting input. Classifier verdicts never cross sessions, directories, environments, or time/execution modes; only the current session's entries are dropped at its boundaries.
6. When settings cannot be read (an SDK or isolated session with no global settings singleton), the plugin honors no static rules and classifies everything rather than blocking every bash call.

## What it does not cover

- **Only the `bash` tool.** `eval`, `hub` (`op: "start"`) and every other exec-tier tool still auto-run under `yolo`, unclassified. An attacker who can choose the tool can choose one of those.
- **Another `tool_call` handler can revise input after this handler.** OMP gives every handler the original input and applies the last revision afterward; this API provides no post-revision gate. A later extension can therefore invalidate this classifier's judgment.
- **Internal-URL cwd values.** Native bash expands `skill://`, `agent://`, `artifact://`, `memory://`, `rule://`, and `local://` cwd values using session-only router state unavailable to extensions. The plugin blocks those cwd forms rather than mislabel the execution directory or fail open; use the resolved filesystem path instead.
- **The contents of what a command runs.** `npm test`, `make`, and dependency installs are judged as the routine commands they are; the classifier does not read `package.json` scripts or install hooks, so a hostile repository's own test script is not inspected.

## Install

```bash
git clone https://github.com/STRML/omp-bash-classifier.git
cd omp-bash-classifier && omp plugin install .
```

This symlinks the directory into `~/.omp/plugins/node_modules/omp-bash-classifier` and writes `omp-plugins.lock.json`. No build step, no runtime deps. Then start a new OMP session (plugins load at session start).

Uninstall: `omp plugin uninstall omp-bash-classifier`.

## Model used

The `@tiny` model role — the role core reserves for online title/memory/classifier work — falling back to the session model when `@tiny` resolves to nothing. Single turn, reasoning disabled, 15s timeout.

Assign the TINY role in `/models`, or set it in a `config.yml` layer — a list (or a comma-separated string) is an ordered preference, and the first entry that resolves to an available model wins:

```yaml
modelRoles:
  tiny:
    - anthropic/claude-haiku-4-5
    - openai-codex/gpt-5.4
    - openrouter/deepseek/deepseek-v4-flash
```

`omp config set modelRoles.tiny …` does not work — the CLI addresses `modelRoles` only as a whole record. Global scope is `<agentDir>/config.yml` (`omp config path`), project scope is `.omp/config.yml`. Unset, the role auto-resolves through the `smol` chain, which on a Claude-only setup is a Sonnet-class model.

That list is availability fallback, resolved once per call: it covers a provider you have no credentials for or a model that has left the catalog, not a request that fails midway. `retry.fallbackChains` does not apply here — it belongs to the agent loop's turn recovery, and this is a single `completeSimple` call — so a classifier timeout or provider error raises a permission request instead of trying the next model.

Pick the model on measured behavior, not size. Scoring candidates on the shipped prompt — eight routine commands, eight destructive, five that append text to the command telling the classifier to answer SAFE — at 5 reps and four concurrent calls, with no provider errors:

| model | judged an injected command SAFE | judged a destructive command SAFE | extra prompt on routine work | p50 |
|---|---|---|---|---|
| `anthropic/claude-haiku-4-5` | 0/25 | 0/40 | 0/40 | 1.1s |
| `openai-codex/gpt-5.4-mini` | 0/25 | 0/40 | 0/40 | 2.7s |
| `deepseek/deepseek-v4-flash` | 0/25 | 0/40 | 6/40 — calls `git add -A && git commit` UNSAFE | 1.7s |
| `anthropic/claude-sonnet-5` | 2/25 | 0/40 | 0/40 | 1.7s |

Every model catches every plainly destructive command; the difference is whether a command that argues for its own verdict can talk the classifier into SAFE, and that does not track model strength — the Sonnet-class model the `smol` chain lands on is the weakest of the four. Measure before switching: an earlier version of this prompt let claude-sonnet-5 through on 29/50 injection samples.

Cursor-provider models (`composer-*`, `gpt-5.4-nano-*`, `gemini-3.7-flash-*`) are unusable here. They answer as an agent, not a classifier: `composer-2.5-fast` returns thinking blocks and tool calls narrating how it will run the command, `gpt-5.4-nano-low` returns no content at all. Every reply parses as no-verdict, so every bash command raises a prompt.

One call per novel command, then cached for the session per (native-resolved cwd, `env`, `pty`, timeout, async, command).

## Configuration

It reads the existing `bash.patterns` and `tools.approval` settings, so your current guardrails keep working. To exempt a command shape from classification entirely, give it a narrow `allow` rule in `bash.patterns` — the plugin honors those after the critical-pattern check and never pays for a model call.

### Decision audit log

Off by default. When enabled, every gate decision is appended as one JSON object per line to `~/.omp/logs/bash-classifier-audit.jsonl`, so a run can be reviewed afterwards — by hand, or by handing the file to a more capable model — to sort the prompts that represented real risk from the ones flagged too eagerly.

Enable with either (env wins over config):

```bash
OMP_BASH_CLASSIFIER_AUDIT=1 omp                      # one process
omp plugin config set omp-bash-classifier audit true  # persistent
```

Path override: `OMP_BASH_CLASSIFIER_AUDIT_PATH`, or the `auditPath` setting. An explicit `OMP_BASH_CLASSIFIER_AUDIT=0` overrides an enabled setting.

Each record carries the evidence (`command` verbatim, native-resolved `cwd`, `envKeys`, `pty`, `timeoutSeconds`, `async`), the claim (`branch`, plus `verdict`/`reason`/`model`/`cached`/`latencyMs` when a model was asked, and `rule` when a static rule decided), and the effect (`outcome`). Quiet auto-allowed calls are logged too — over-flagging and under-flagging are only measurable together.

`branch` names which link in the precedence chain decided: `overlength`, `deny-rule`, `critical-pattern`, `env-override`, `prompt-rule`, `narrow-allow`, `user-policy-prompt`, `classified`, `unclassified`, `internal-url-cwd`, `plugin-error`. `outcome` is `ran`, `approved`, `denied`, `blocked`, or `host-decides`.

`criticalDowngraded: true` marks the one branch that weakens the strongest check: a recursive delete that matched a built-in critical pattern but was sent to classification by the temp-scope exemption. Grep it first when auditing — without the flag it is indistinguishable from a command that never matched a critical pattern at all.

`host-decides` means the plugin deliberately stayed out and the native gate owned the call — do not read it as "the plugin allowed it". Those get a second `branch: "host-approval"` record with the host's real answer, correlated on `toolCallId`:

```
prompt-rule/host-decides   toolu_01X2…  git reset --hard HEAD   rule: git reset --hard* → prompt
host-approval/denied       toolu_01X2…
```

#### Request context

A verdict cannot be judged "too eager" without knowing what was being attempted, so two context sources are recorded — both for the reviewer only, never as classifier input.

**Caller intent** (`branch: "intent"`) is emitted as its own row, joined on `toolCallId`, because the host exposes it on `tool_execution_start` — after the decision. It is metadata rather than a decision, so it carries no `outcome`. It is present whether or not the command ran:

```
classified/ran     toolu_01Ua…  git status --short
intent             toolu_01Ua…  "Check working tree status"
```

**The user turn** is off even when the log is on, because a prompt carries far more sensitive material than a command line and enabling a decision log should not silently start transcribing the conversation:

```bash
OMP_BASH_CLASSIFIER_AUDIT_PROMPT=1 omp
omp plugin config set omp-bash-classifier auditPrompt true
```

That adds `prompt` (capped at 2,000 chars), `promptLength`, and `promptTruncated` — the flag is explicit so a clipped prompt is never mistaken for the whole request. `input` fires in interactive mode only, so headless runs record `null`.

#### Why context never reaches the classifier

`CLASSIFIER_PROMPT` treats the command as untrusted DATA and scans it for steering — text addressing the reviewer, claims that something is a test or already approved, instructions to reinterpret the rules. Conversational context is reachable by anything that enters the transcript: tool output, file contents, repo instruction files, MCP results. Feeding it to the classifier would hand exactly that material the trusted channel the fence exists to deny, letting an injected "the user already approved this cleanup" justify a destructive command. The gate judges effects; stated purpose is for the human reading the log afterwards.

Two invariants: caller-supplied `env` **values** are never written (key names only), and auditing never changes gate behavior — every write is best-effort and swallows its own errors, so an unwritable path cannot decide whether a command runs.

Settings are read through the host module instance (`pi.pi.settings`). A plugin-local `import { settings }` resolves to a second copy of the singleton with no global instance and throws `Settings not initialized`, which would block every bash call. A session that legitimately has no global settings (SDK, isolated) is handled instead: the plugin logs one warning, honors no static rules, and classifies everything.

## Risks

- **Command privacy**: classified commands are sent, up to 2,000 characters plus the native-resolved working directory, to the `@tiny` model's provider. Command text can contain private paths, proprietary snippets, inline env assignments, or secrets in flags. The provider's logging and retention policies apply. Caller-supplied `env` values are never sent — that path asks the human instead.
- **Model misevaluation**: bounded by never being able to bypass the native gate, and by fail-closed handling of everything it cannot judge. It is not bounded against a repository whose own build/test scripts are hostile (see "What it does not cover").
- **Latency**: one small-model call per novel command that no static pattern rule decides, in every approval mode, cached per session. In `write`/`always-ask`, even a SAFE verdict still proceeds to the native prompt.
- **Extra prompts**: a false UNSAFE costs one plugin confirmation followed, in non-`yolo` modes, by the native confirmation. Classifier SAFE/UNSAFE verdicts are cached for the exact execution identity, but a human approval is deliberately one-time and never bypasses the native gate.

## Files

- `index.ts` — the plugin (single file: `tool_call` gate, static rule mirror, classifier, session cache).
- `package.json` — manifest (`omp.extensions: ["./index.ts"]`, dev-only deps for `bun run typecheck`).
- `LICENSE` — MIT.

## Development

```bash
bun install
ln -sfn "$(bun pm -g bin)/../install/global/node_modules/@oh-my-pi" node_modules/@oh-my-pi   # resolve host types
bun run typecheck
```

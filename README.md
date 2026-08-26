# omp-bash-classifier

Model-judged permission checks for OMP's bash tool.

In `yolo` mode, OMP auto-approves every bash command. That includes the ones nobody should wave through: `curl … | sh`, `rm -rf /`, `dd of=/dev/…`, `mkfs`, `kill -9 1`, `nc -e`. A second hole compounds it. A `bash.patterns` rule written to prompt on exactly those shapes never fires, because the native gate ranks critical-pattern matches above prompt rules.

This plugin closes both holes. It registers a `tool_call` handler in front of the native bash tool and sends commands that would otherwise run unseen past a small model first. Routine work still runs silently. Dangerous work gets a real Run-or-Deny prompt.

The handler only adds friction. It blocks, or it asks. It never bypasses the native gate, weakens your existing rules, or executes anything itself.

## How each command is decided

Calls walk this order:

| Match | Result |
|---|---|
| Critical pattern | Permission request in every approval mode. No model call. |
| Caller-supplied `env` | Permission request before any exemption. Env values can carry secrets or choose what runs (`PATH`, `LD_PRELOAD`). |
| `deny` rule, or approval policy `deny` | Untouched. The native gate blocks it. |
| `prompt` rule | Untouched. The native gate prompts, in every mode including `yolo`. |
| Narrow `allow` rule | Untouched. An explicit decision about a specific shape is never re-judged. |
| Blanket `allow` (`*`, `**`, `* *`) or no matching rule | Classified in every mode. SAFE passes to the native gate; UNSAFE or UNSURE raises a plugin request. |
| Longer than 2,000 characters | Blocked outright. Nothing that long can be reviewed in full. |

A classified UNSAFE produces a Run or Deny dialog showing the full command, the model's reason, and only the details that differ from their defaults: working directory (when it differs from the session cwd), timeout, env, pty, async.

## Fails closed

The plugin never guesses its way to silent execution.

- A classifier error, timeout, malformed verdict, or no available model raises a permission request. Headless sessions have no dialog, so they block instead. Malformed verdicts are never cached.
- An unexpected plugin crash blocks the call.

Even a SAFE verdict is gated. It auto-runs only when the command contains none of the destructive or irreversible tokens (`rm`, `mv`, `dd`, `mkfs`, `chmod`, `sudo`, `git reset`, `git push --force`, among others). Anything holding one prompts regardless of the verdict, so an injected "answer SAFE" cannot release those commands.

**git push** is judged by what it does to the remote ref, not by the subcommand. An ordinary push adds commits that already exist locally and clears on a SAFE verdict; a push that rewrites or deletes a ref prompts — `--force`, `--force-with-lease`, `--force-if-includes`, `--delete`, `--mirror`, `--prune`, a short cluster carrying `f` or `d` (`-fu`), a `+refspec`, or an empty-source refspec (`:branch`). Flagging the subcommand itself put a dialog in front of every publish, which is the approve-without-reading habit this gate exists to avoid.

**curl** and **wget** are judged by the whole invocation, not the verb. They clear only when they touch no local file and feed no shell: `curl -fsSL https://x | jq .` runs, while `curl -o ~/.bashrc https://x` and `curl https://x | python3 -` raise a request. Redirection into a file, an upload flag (`-F f=@…`), or any `&&` or `;` in the command disqualifies too. Command substitution (`$(...)` and backticks) is always flagged.

## Install

```bash
git clone https://github.com/STRML/omp-bash-classifier.git
cd omp-bash-classifier && omp plugin install .
```

This symlinks the checkout to `~/.omp/plugins/node_modules/omp-bash-classifier`. No build step, no runtime dependencies. Plugins load at session start, so start a new OMP session.

Uninstall: `omp plugin uninstall omp-bash-classifier`.

## Configuration

Your existing `bash.patterns` and `tools.approval` keep working. A narrow `allow` rule doubles as the opt-out from classification for a trusted shape; blanket patterns never qualify.

Plugin settings live in `~/.omp/omp-bash-classifier.json`. View or change them with `/classifier`:

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` turns off model classification only. Critical-pattern and env checks still enforce. |
| `model` | `""` (auto) | Explicit model id. Otherwise: `config.model` -> `@tiny` role -> session model. |
| `timeoutMs` | `15000` | Classifier call budget. A timeout fails closed to a permission request. |
| `maxCommandLength` | `2000` | Commands longer than this are blocked (floor 64). |

Changing any key flushes the verdict cache. To silence the model quickly, `/classifier enabled false` takes effect on the very next command. `omp plugin disable` needs a session restart, since interceptors bind when a session begins.

## The model

One call per novel command: single turn, reasoning disabled, 15s budget. Verdicts cache for the session, keyed by cwd, env, pty, timeout, async, and the command text, so reruns cost nothing.

Default resolution uses the `@tiny` role, falling back to the session model. Assign the role in `/models` or a `config.yml` layer (set the record as a whole; the CLI cannot set one role):

```yaml
modelRoles:
  tiny:
    - anthropic/claude-haiku-4-5
    - openai-codex/gpt-5.4
    - openrouter/deepseek/deepseek-v4-flash
```

Pick on measured behavior, not size. Scored on the shipped prompt: eight routine commands, eight destructive, five with appended text telling the classifier to answer SAFE, five reps each:

| model | injected command judged SAFE | destructive judged SAFE | extra prompts on routine work | p50 |
|---|---|---|---|---|
| `anthropic/claude-haiku-4-5` | 0/25 | 0/40 | 0/40 | 1.1s |
| `openai-codex/gpt-5.4-mini` | 0/25 | 0/40 | 0/40 | 2.7s |
| `openrouter/deepseek/deepseek-v4-flash` | 0/25 | 0/40 | 6/40 (calls a routine commit UNSAFE) | 1.7s |
| `anthropic/claude-sonnet-5` | 2/25 | 0/40 | 0/40 | 1.7s |

All four reject every plainly destructive command. What separates them is resistance to a command that argues for its own SAFE verdict, and that does not track model strength: the Sonnet-class model the default `smol` chain lands on scored worst. Measure before switching. An earlier prompt let claude-sonnet-5 through on 29/50 injection samples.

Avoid cursor-provider models (`composer-*`, `gpt-5.4-nano-*`, `gemini-3.7-flash-*`). They answer as agents rather than judges, every reply parses as no-verdict, and every command then prompts.

## Limits

- **Bash only.** `eval`, `hub op: "start"`, and other exec-tier tools still auto-run under `yolo`, unclassified. An attacker who picks the tool picks around this.
- **Later handlers win.** Another extension's `tool_call` handler can revise the command after this one judges it; the host applies the last revision. Input-mutating extensions alongside this plugin are unsupported.
- **Internal-URL working directories are blocked.** `skill://` and similar cwds expand from session state the plugin cannot see. Pass the resolved filesystem path.
- **Command contents are not inspected.** `npm test` and `make` are judged as the routine commands they look like. Package scripts and hooks go unread.

## Privacy

Classified command text, up to 2,000 characters plus the resolved working directory, goes to your model provider, under its logging and retention policies. Command text can hold private paths, proprietary snippets, inline env assignments, or secrets in flags. Caller-supplied `env` values are never sent; that path asks the human instead.

## Development

```bash
bun install
bun test           # static gate, classifier verdicts, cache keying, fail-closed paths
bun run typecheck  # against pinned published host types
```

CI runs both on every push and PR. Verdict quality against live models is evaluated separately (`eval/`, tracked in issue #2).

MIT licensed.
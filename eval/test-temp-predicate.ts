#!/usr/bin/env bun
/**
 * Tests for `isTempScopedRecursiveDelete`, the one place this plugin declines a
 * critical-pattern short-circuit.
 *
 * Kept as real assertions rather than a scored corpus because this is pure logic
 * with no model in the path: it either resolves a path correctly or it does not,
 * and a wrong answer here is a bypass of the strongest check in the gate. The
 * traversal rows are the reason the predicate exists — a `bash.patterns` glob
 * (`rm -rf /tmp/*`) accepts every one of them.
 *
 *   bun eval/test-temp-predicate.ts
 */
import { mkdirSync, rmSync, symlinkSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { isTempScopedRecursiveDelete } from "../index";

const CWD = "/Users/sean/sites/console-mode";

/** `[command, expected, why]` — expected `true` means "skip the critical short-circuit". */
const CASES: [string, boolean, string][] = [
	// Intended: provably inside a temp root.
	["rm -rf /tmp/console-mode-snapshots", true, "the real case from the audit log"],
	// The parent must EXIST for the physical check to answer. `/tmp/foo` is absent
	// here, so this is refused — and refusing costs nothing, because a delete whose
	// parent does not exist has nothing to remove. This row asserted `true` while
	// the predicate climbed to the nearest existing ancestor; that climb was the
	// round-2 glob bypass, so the behaviour changed deliberately.
	["rm -rf /tmp/foo/bar", false, "parent /tmp/foo does not exist — fail closed"],
	// Globs are refused outright — see the co-tenancy rationale in index.ts. Claude
	// Code's auto-mode blocks the same shape ("Shared Scratch Sweep") because /tmp is
	// shared with other agents whose live state a sweep destroys. This row asserted
	// `true` before that reasoning was adopted.
	["rm -rf /tmp/*", false, "pattern sweep of a shared scratch dir"],
	["rm -rf /tmp/bc-*", false, "prefix-glob sweep"],
	["rm -rf /tmp/a?c", false, "single-char glob"],
	["rm -rf /tmp/[ab]", false, "bracket expression"],
	["rm -rf -- /tmp/x", true, "operand separator"],
	["rm -fr /tmp/x", true, "flag order variant"],
	["rm -rf -v /tmp/x", true, "extra flags before the operand"],
	["rm -rf /tmp/a /tmp/b", true, "every operand inside a root"],

	// Traversal: the whole point. A glob allow rule accepts all of these.
	["rm -rf /tmp/../Users/sean/sites", false, "escapes via .."],
	["rm -rf /tmp/../..", false, "escapes to the filesystem root"],
	["rm -rf /tmp/a/../../Users/sean", false, "escapes from a nested path"],

	// Brace expansion — found by the review panel's pentester seat, 2026-08-20.
	// These are the nastiest cases in the table: `{`, `}` and `,` are NOT shell
	// control operators, so the command clears the shell-control guard as a single
	// token whose lexical resolve() lands inside /tmp. Bash then expands it into a
	// second word that leaves the root. Verified against bash:
	//   /tmp/{x,../../Users/sean}  ->  /tmp/x  /tmp/../../Users/sean
	["rm -rf /tmp/{x,../../Users/sean/sites}", false, "brace expansion escapes to /Users/sean/sites"],
	["rm -rf /tmp/{a,b}", false, "braces refused even when both branches stay in /tmp"],
	["rm -rf /tmp/x{,/../../../Users/sean}", false, "brace suffix form"],
	["rm -rf /tmp/{a,{b,../../etc}}", false, "nested braces"],

	// The root itself is not "inside" the root.
	["rm -rf /tmp", false, "wiping the whole temp dir stays critical"],
	["rm -rf /tmp/", false, "trailing slash resolves to the root"],

	// Plainly dangerous targets.
	["rm -rf /", false, "filesystem root"],
	["rm -rf /Users/sean", false, "home directory"],
	["rm -rf /tmp/a /Users/sean", false, "one bad operand disqualifies the whole command"],

	// Smuggling a second command past the check.
	["rm -rf /tmp/x && rm -rf /Users/sean", false, "&& chains a second delete"],
	["rm -rf /tmp/x; rm -rf /Users/sean", false, "; chains a second delete"],
	["rm -rf /tmp/x | tee /tmp/log", false, "pipe is shell control"],
	["rm -rf $(echo /tmp/x)", false, "command substitution"],

	// Unexpanded shell syntax: refused rather than guessed at.
	["rm -rf ~/tmp/x", false, "tilde is expanded by the shell, not here"],
	["rm -rf $HOME/../tmp/x", false, "variable is expanded by the shell, not here"],

	// Escalation and the coreutils override are never exempt.
	["sudo rm -rf /tmp/x", false, "sudo rm is its own critical pattern"],
	["rm -rf --no-preserve-root /tmp/x", false, "defeats coreutils' refusal to recurse on /"],

	// Not a delete at all.
	["dd if=/dev/zero of=/tmp/x", false, "not an rm"],
	["rm -rf", false, "no operand"],

	// Round 2, CRITICAL: backslash-escaped `..`. The predicate used a REJECT list
	// (`~ $ { }`) which did not name backslash. `resolve()` reads `\.\.` as
	// ordinary component names and collapses nothing, so the path stayed lexically
	// inside /tmp — while bash's quote removal handed rm `/tmp/../../…`. Verified:
	// that operand printf-expands to `/tmp/../../Users/sean/sites`. Needs no
	// filesystem setup, which is what made it the worst finding of either round.
	// The fix replaced the reject list with an allowlist.
	[String.raw`rm -rf /tmp/\.\./\.\./Users/sean/sites`, false, "backslash-escaped .. escapes via quote removal"],
	[String.raw`rm -rf /tmp/a\/../b`, false, "backslash anywhere in the operand"],
	["rm -rf '/tmp/a b'", false, "quote characters are not in the allowlist"],
	["rm -rf /tmp/a!b", false, "history-expansion character not allowlisted"],
];

/**
 * Symlink escape — round 1 (auditor, executor, both skeptics) and round 2
 * (pentester). Needs a real filesystem fixture: the predicate physically
 * resolves the operand's parent, so a hypothetical path cannot exercise it.
 * `/tmp` is world-writable and `ln -s` matches no critical pattern, so the gated
 * agent can plant this link itself with a command the classifier calls SAFE.
 */
const FIXTURE = "/tmp/bc-predicate-fixture";
const LINKED: [string, boolean, string][] = [
	[`rm -rf ${FIXTURE}/real`, true, "genuinely inside a temp root"],
	[`rm -rf ${FIXTURE}/escape/src`, false, "traverses a symlink pointing out of /tmp"],
	// Deleting the LINK is not deleting its target: rm(1) "removes symbolic links,
	// not the files referenced by the links", so this removes an entry inside /tmp
	// and nothing outside it. The parent is verified physically, which is what the
	// check actually needs to establish. The round-1 expectation of `false` here
	// described the old climb, not a real risk.
	[`rm -rf ${FIXTURE}/escape`, true, "removes the symlink itself, not its target"],
	[`rm -rf ${FIXTURE}/escape/*`, false, "glob expands THROUGH the symlink to real paths outside"],
	[`rm -rf ${FIXTURE}/missing`, true, "nonexistent leaf whose parent is inside /tmp"],
	// Round 2, CRITICAL: a glob in a NON-FINAL component. The previous fix climbed
	// to the nearest existing ancestor, and a glob component never exists
	// literally — so the climb skipped straight past the symlink it was meant to
	// catch and landed on /tmp, which is a root. The final-position row above gave
	// false assurance because only that position was ever tested.
	[`rm -rf ${FIXTURE}/esc*/src`, false, "glob in a non-final component hides the symlink"],
	[`rm -rf ${FIXTURE}/real/*`, false, "glob refused even with a verified parent"],
	[`rm -rf ${FIXTURE}/nope/deep/leaf`, false, "parent chain absent — fail closed, no climbing"],
];

let failures = 0;
const check = (command: string, expected: boolean, why: string): void => {
	const actual = isTempScopedRecursiveDelete(command, CWD);
	if (actual === expected) return;
	failures++;
	console.error(`FAIL  expected ${expected}, got ${actual}\n      ${command}\n      (${why})`);
};

for (const [command, expected, why] of CASES) check(command, expected, why);

// Build the symlink fixture, run the physical cases, tear it down either way.
mkdirSync(`${FIXTURE}/real`, { recursive: true });
rmSync(`${FIXTURE}/escape`, { force: true });
symlinkSync(join(homedir(), "sites"), `${FIXTURE}/escape`);
try {
	for (const [command, expected, why] of LINKED) check(command, expected, why);
} finally {
	rmSync(FIXTURE, { recursive: true, force: true });
}

const total = CASES.length + LINKED.length;
console.log(`${total - failures}/${total} passed`);
if (failures > 0) process.exit(1);

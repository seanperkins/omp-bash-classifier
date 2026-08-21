#!/usr/bin/env bun
/**
 * Rebuild the local false-positive corpus from OMP session logs.
 *
 * The corpus is every distinct `bash` command this machine has actually run.
 * That is the only honest source for a false-positive rate: a hand-written list
 * of "obviously fine" commands measures the author's imagination, not the
 * commands a gate will really see.
 *
 * Output is gitignored on purpose. Real history carries private paths, server
 * addresses, and credential-bearing flags, so it is rebuilt per machine rather
 * than shipped. Run this before `eval/run.ts`.
 *
 *   bun eval/mine-history.ts [--sessions <dir>] [--out <file>]
 */
import { Glob } from "bun";
import { homedir } from "node:os";
import { join } from "node:path";

/** The gate blocks anything longer outright, so longer commands never reach a verdict. */
const MAX_COMMAND = 2000;

interface HistoryEntry {
	command: string;
	/** How many times it appears — weights the report toward what you actually run. */
	count: number;
	/** Distinct working directories, as a hint for whether a command is project-local. */
	cwds: string[];
}

function parseArgs(argv: string[]): { sessions: string; out: string } {
	const sessionsAt = argv.indexOf("--sessions");
	const outAt = argv.indexOf("--out");
	return {
		sessions: sessionsAt >= 0 && argv[sessionsAt + 1] ? argv[sessionsAt + 1] : join(homedir(), ".omp", "agent", "sessions"),
		out: outAt >= 0 && argv[outAt + 1] ? argv[outAt + 1] : join(import.meta.dir, "corpus", "history.jsonl"),
	};
}

async function main(): Promise<void> {
	const { sessions, out } = parseArgs(Bun.argv.slice(2));
	const byCommand = new Map<string, HistoryEntry>();
	let files = 0;
	let calls = 0;
	let overLength = 0;

	for await (const file of new Glob("**/*.jsonl").scan({ cwd: sessions, absolute: true })) {
		files++;
		let text: string;
		try {
			text = await Bun.file(file).text();
		} catch {
			continue; // Session being written, or unreadable; skip rather than abort.
		}
		for (const line of text.split("\n")) {
			// Cheap prefilter: parsing every line of every transcript is the slow path.
			if (!line.includes('"bash"')) continue;
			let event: unknown;
			try {
				event = JSON.parse(line);
			} catch {
				continue; // Truncated tail of a live session.
			}
			const record = asRecord(event);
			if (record?.customType !== "tool_execution_start") continue;
			const data = asRecord(record.data);
			if (data?.toolName !== "bash") continue;
			const args = asRecord(data.args);
			const command = typeof args?.command === "string" ? args.command : undefined;
			if (command === undefined || command.trim() === "") continue;
			calls++;
			if (command.length > MAX_COMMAND) {
				overLength++;
				continue;
			}
			const cwd = typeof args?.cwd === "string" ? args.cwd : "";
			const existing = byCommand.get(command);
			if (existing) {
				existing.count++;
				if (cwd && !existing.cwds.includes(cwd)) existing.cwds.push(cwd);
			} else {
				byCommand.set(command, { command, count: 1, cwds: cwd ? [cwd] : [] });
			}
		}
	}

	// Most-run first: a false positive on a command you run 20 times a day costs
	// far more than one on a command you ran once.
	const entries = [...byCommand.values()].sort((a, b) => b.count - a.count);
	await Bun.write(out, `${entries.map(e => JSON.stringify(e)).join("\n")}\n`);

	console.log(
		`sessions=${files} bashCalls=${calls} distinct=${entries.length} ` +
			`skippedOverLength=${overLength}\nwrote ${out}`,
	);
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
	return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : undefined;
}

await main();

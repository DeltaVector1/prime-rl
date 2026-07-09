#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
import threading
from collections import Counter, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rollout Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d8dde5;
      --text: #1f2937;
      --muted: #5f6b7a;
      --accent: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
      --good: #047857;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; }
    header { height: 54px; display: flex; align-items: center; gap: 16px; padding: 0 18px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 3; }
    h1 { font-size: 16px; margin: 0; font-weight: 700; }
    .file { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .layout { display: grid; grid-template-columns: 420px 1fr; min-height: calc(100vh - 54px); }
    aside { border-right: 1px solid var(--line); background: var(--panel); min-width: 0; }
    main { min-width: 0; padding: 16px; }
    .filters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 12px; border-bottom: 1px solid var(--line); }
    .filters input, .filters select { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px 9px; background: #fff; color: var(--text); font: inherit; }
    .filters .wide { grid-column: 1 / -1; }
    button { border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--text); padding: 8px 10px; font: inherit; cursor: pointer; }
    button:hover { border-color: var(--accent); }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 12px; border-bottom: 1px solid var(--line); }
    .stat { border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fafbfc; }
    .stat b { display: block; font-size: 15px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .list { height: calc(100vh - 210px); overflow: auto; }
    .row { padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; display: grid; gap: 5px; }
    .row:hover, .row.active { background: #eef6f5; }
    .rowtop { display: flex; gap: 8px; align-items: center; min-width: 0; }
    .badge { display: inline-flex; align-items: center; border-radius: 999px; border: 1px solid var(--line); padding: 2px 7px; font-size: 12px; background: #fff; white-space: nowrap; }
    .badge.train { border-color: #99c2bd; color: #075e56; }
    .badge.eval { border-color: #b7c5e8; color: #254585; }
    .badge.bad { border-color: #f0b4b4; color: var(--bad); }
    .badge.good { border-color: #9bd3ba; color: var(--good); }
    .summary-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 650; }
    .muted { color: var(--muted); }
    .mono { font-family: var(--mono); }
    .detail-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
    .detail-title { font-size: 18px; font-weight: 750; }
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .tabs button.active { border-color: var(--accent); color: var(--accent); }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; overflow: auto; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .kv { border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fafbfc; min-width: 0; }
    .kv span { display: block; color: var(--muted); font-size: 12px; }
    .kv b { overflow-wrap: anywhere; }
    pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--mono); font-size: 12px; line-height: 1.45; }
    .judge { border: 1px solid var(--line); border-radius: 6px; padding: 10px; margin-bottom: 10px; background: #fbfcfd; }
    .judge h3 { margin: 0 0 8px 0; font-size: 13px; }
    .turn { border-bottom: 1px solid var(--line); padding: 10px 0; }
    .turn:last-child { border-bottom: 0; }
    .role { font-weight: 700; color: var(--accent); margin-bottom: 5px; }
    .empty { padding: 32px; text-align: center; color: var(--muted); }
    @media (max-width: 920px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .list { height: 360px; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Rollout Viewer</h1>
    <div id="file" class="file"></div>
  </header>
  <div class="layout">
    <aside>
      <div class="filters">
        <select id="kind"><option value="">kind: any</option><option value="train">train</option><option value="eval">eval</option></select>
        <select id="type"><option value="">type: any</option><option value="rollout">rollout</option><option value="train_batch_summary">train summary</option><option value="eval_batch_summary">eval summary</option></select>
        <input id="env" placeholder="env contains">
        <input id="step" placeholder="step">
        <input id="q" class="wide" placeholder="search labels, errors, stop condition">
        <button id="refresh" class="wide">Refresh</button>
      </div>
      <div class="stats" id="stats"></div>
      <div class="list" id="list"><div class="empty">Loading...</div></div>
    </aside>
    <main>
      <div id="detail" class="empty">Select a row.</div>
    </main>
  </div>
<script>
const state = { selectedLine: null, selectedTab: "overview", records: [] };
const $ = (id) => document.getElementById(id);
function fmt(v) {
  if (v === null || v === undefined || v === "") return "-";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return String(v);
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function qs() {
  const p = new URLSearchParams();
  for (const id of ["kind", "type", "env", "step", "q"]) {
    const value = $(id).value.trim();
    if (value) p.set(id, value);
  }
  p.set("limit", "250");
  return p.toString();
}
async function loadStats() {
  const res = await fetch("/api/stats");
  const data = await res.json();
  $("file").textContent = data.file;
  $("stats").innerHTML = `
    <div class="stat"><b>${fmt(data.total_lines)}</b><span>rows</span></div>
    <div class="stat"><b>${fmt(data.rollouts)}</b><span>rollouts</span></div>
    <div class="stat"><b>${fmt(data.summaries)}</b><span>summaries</span></div>`;
}
async function loadRecords() {
  const res = await fetch("/api/records?" + qs());
  const data = await res.json();
  state.records = data.records;
  renderList(data);
}
function renderList(data) {
  const list = $("list");
  if (!data.records.length) {
    list.innerHTML = `<div class="empty">No records matched ${esc(data.total_matched)} scanned matches.</div>`;
    return;
  }
  list.innerHTML = data.records.map(r => {
    const active = r.line === state.selectedLine ? " active" : "";
    const reward = r.reward === null || r.reward === undefined ? "" : `<span class="badge ${r.reward > 0 ? "good" : "bad"}">reward ${fmt(r.reward)}</span>`;
    const drop = r.drop_reason ? `<span class="badge bad">${esc(r.drop_reason)}</span>` : "";
    return `<div class="row${active}" data-line="${r.line}">
      <div class="rowtop">
        <span class="badge ${esc(r.kind || "")}">${esc(r.kind || "-")}</span>
        <span class="badge">${esc(r.record_type)}</span>
        ${reward}${drop}
      </div>
      <div class="summary-title">${esc(r.title)}</div>
      <div class="muted mono">line ${r.line} step ${fmt(r.step)} env ${esc(r.env_name || "-")} judges ${fmt(r.judge_logs_count)}</div>
    </div>`;
  }).join("");
  for (const row of list.querySelectorAll(".row")) {
    row.addEventListener("click", () => selectLine(Number(row.dataset.line)));
  }
}
async function selectLine(line) {
  state.selectedLine = line;
  const res = await fetch(`/api/record?line=${line}`);
  const data = await res.json();
  renderDetail(data.record);
  renderList({records: state.records, total_matched: state.records.length});
}
function tabButton(name, label) {
  const active = state.selectedTab === name ? "active" : "";
  return `<button class="${active}" data-tab="${name}">${label}</button>`;
}
function renderDetail(record) {
  const detail = $("detail");
  const labels = record.labels || {};
  const rollout = record.rollout || {};
  const title = record.record_type === "rollout"
    ? `${record.kind || ""} rollout ${labels.rollout_id || ""}`
    : `${record.record_type} step ${record.step}`;
  const tabs = [
    tabButton("overview", "Overview"),
    tabButton("reward", "Reward"),
    tabButton("judges", "Judge Logs"),
    tabButton("trajectory", "Trajectory"),
    tabButton("raw", "Raw JSON")
  ].join("");
  detail.innerHTML = `<div class="detail-head"><div class="detail-title">${esc(title)}</div><div class="muted mono">line ${record.line || ""}</div></div><div class="tabs">${tabs}</div><div class="panel" id="tabbody"></div>`;
  for (const btn of detail.querySelectorAll(".tabs button")) {
    btn.addEventListener("click", () => { state.selectedTab = btn.dataset.tab; renderDetail(record); });
  }
  renderTab(record);
}
function renderTab(record) {
  const body = $("tabbody");
  const labels = record.labels || {};
  const rollout = record.rollout || {};
  if (state.selectedTab === "overview") {
    const items = {
      record_type: record.record_type, kind: record.kind, step: record.step, env: labels.env_name || record.env_name,
      example_id: labels.example_id, group_id: labels.group_id, rollout_id: labels.rollout_id,
      reward: rollout.reward ?? record.reward, trainable: labels.is_trainable, drop_reason: labels.drop_reason,
      error_type: labels.error_type, truncated: labels.is_truncated, stop_condition: labels.stop_condition
    };
    body.innerHTML = `<div class="grid">${Object.entries(items).map(([k,v]) => `<div class="kv"><span>${esc(k)}</span><b>${esc(fmt(v))}</b></div>`).join("")}</div>`;
  } else if (state.selectedTab === "reward") {
    body.innerHTML = `<pre>${esc(JSON.stringify(record.reward_calculation || record.reward_mean_calculation || {}, null, 2))}</pre>`;
  } else if (state.selectedTab === "judges") {
    const logs = record.judge_logs || [];
    if (!logs.length) { body.innerHTML = `<div class="empty">No judge logs on this record.</div>`; return; }
    body.innerHTML = logs.map((j, i) => `<div class="judge">
      <h3>${i + 1}. ${esc(j.kind || "judge")} <span class="muted">${esc(j.model || "")}</span></h3>
      <div class="muted">error: ${esc(j.error || "-")} parsed: ${esc(fmt(j.parsed_score))}</div>
      <h3>Prompt</h3><pre>${esc(j.prompt || "")}</pre>
      <h3>Response</h3><pre>${esc(j.response || "")}</pre>
    </div>`).join("");
  } else if (state.selectedTab === "trajectory") {
    const traj = rollout.trajectory || [];
    if (!traj.length) { body.innerHTML = `<div class="empty">No trajectory on this record.</div>`; return; }
    body.innerHTML = traj.map((turn, i) => `<div class="turn"><div class="role">turn ${i + 1}</div><pre>${esc(JSON.stringify(turn, null, 2))}</pre></div>`).join("");
  } else {
    body.innerHTML = `<pre>${esc(JSON.stringify(record, null, 2))}</pre>`;
  }
}
$("refresh").addEventListener("click", () => { loadStats(); loadRecords(); });
for (const id of ["kind", "type", "env", "step", "q"]) {
  $(id).addEventListener("keydown", (ev) => { if (ev.key === "Enter") loadRecords(); });
  $(id).addEventListener("change", loadRecords);
}
loadStats();
loadRecords();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a JSONL rollout viewer.")
    parser.add_argument("--file", required=True, type=Path, help="Path to rollouts.jsonl")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", default=8787, type=int, help="Bind port")
    return parser.parse_args()


def get_path_value(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def record_env(record: dict[str, Any]) -> str | None:
    return (
        get_path_value(record, "labels.env_name")
        or record.get("env_name")
        or get_path_value(record, "rollout.env_name")
    )


def record_reward(record: dict[str, Any]) -> Any:
    return get_path_value(record, "rollout.reward") if record.get("record_type") == "rollout" else None


def record_judge_logs(record: dict[str, Any]) -> list[Any]:
    logs = record.get("judge_logs")
    if isinstance(logs, list):
        return logs
    rollout_logs = get_path_value(record, "rollout.judge_logs")
    if isinstance(rollout_logs, list):
        return rollout_logs
    return []


def summarize_record(line: int, record: dict[str, Any]) -> dict[str, Any]:
    labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
    rollout = record.get("rollout") if isinstance(record.get("rollout"), dict) else {}
    record_type = str(record.get("record_type", "unknown"))
    env_name = record_env(record)
    title_parts = [
        record_type,
        str(record.get("kind") or ""),
        f"env={env_name}" if env_name else "",
        f"example={labels.get('example_id')}" if labels.get("example_id") is not None else "",
        f"stop={labels.get('stop_condition')}" if labels.get("stop_condition") else "",
    ]
    if record_type.endswith("summary"):
        counts = record.get("counts") if isinstance(record.get("counts"), dict) else {}
        title_parts.append(" ".join(f"{k}={v}" for k, v in list(counts.items())[:4]))
    elif rollout.get("error"):
        title_parts.append(f"error={labels.get('error_type') or 'yes'}")
    summary = {
        "line": line,
        "record_type": record_type,
        "kind": record.get("kind"),
        "step": record.get("step"),
        "env_name": env_name,
        "title": " ".join(part for part in title_parts if part),
        "reward": record_reward(record),
        "drop_reason": labels.get("drop_reason"),
        "judge_logs_count": len(record_judge_logs(record)),
    }
    summary["_search"] = json.dumps(
        {
            "title": summary["title"],
            "record_type": summary["record_type"],
            "kind": summary["kind"],
            "env_name": summary["env_name"],
            "step": summary["step"],
            "labels": labels,
            "reward_calculation": record.get("reward_calculation"),
        },
        default=str,
    ).lower()
    return summary


def summary_matches(summary: dict[str, Any], filters: dict[str, str]) -> bool:
    if filters.get("kind") and str(summary.get("kind", "")) != filters["kind"]:
        return False
    if filters.get("type") and str(summary.get("record_type", "")) != filters["type"]:
        return False
    if filters.get("env") and filters["env"].lower() not in str(summary.get("env_name") or "").lower():
        return False
    if filters.get("step") and str(summary.get("step", "")) != filters["step"]:
        return False
    return not filters.get("q") or filters["q"].lower() in str(summary.get("_search") or "")


class RolloutStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.identity: tuple[int, int] | None = None
        self.position = 0
        self.summaries: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.envs: Counter[str] = Counter()

    def _reset(self, identity: tuple[int, int]) -> None:
        self.identity = identity
        self.position = 0
        self.summaries.clear()
        self.counts.clear()
        self.envs.clear()

    def refresh(self) -> None:
        with self.lock:
            if not self.path.exists():
                self.path = resolve_jsonl_path(self.path)
                if not self.path.exists():
                    return
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity != self.identity or stat.st_size < self.position:
                self._reset(identity)

            with open(self.path, "rb") as handle:
                handle.seek(self.position)
                while True:
                    offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        self.position = handle.tell()
                        break
                    if not raw_line.endswith(b"\n"):
                        self.position = offset
                        break

                    line_no = len(self.summaries) + 1
                    try:
                        record = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        record = {
                            "record_type": "parse_error",
                            "error": repr(exc),
                            "raw_line": raw_line[:2000].decode("utf-8", errors="replace"),
                        }
                    summary = summarize_record(line_no, record)
                    summary["_offset"] = offset
                    self.summaries.append(summary)
                    record_type = str(record.get("record_type", "unknown"))
                    self.counts[record_type] += 1
                    env_name = record_env(record)
                    if env_name:
                        self.envs[str(env_name)] += 1
                    self.position = handle.tell()

    def stats(self) -> dict[str, Any]:
        self.refresh()
        with self.lock:
            return {
                "file": str(self.path),
                "total_lines": len(self.summaries),
                "rollouts": self.counts.get("rollout", 0),
                "summaries": sum(value for key, value in self.counts.items() if key.endswith("summary")),
                "record_types": self.counts,
                "envs": self.envs,
            }

    def records(self, filters: dict[str, str], limit: int) -> dict[str, Any]:
        self.refresh()
        rows = deque(maxlen=max(1, min(limit, 2000)))
        matched = 0
        with self.lock:
            for summary in self.summaries:
                if not summary_matches(summary, filters):
                    continue
                matched += 1
                rows.append({key: value for key, value in summary.items() if not key.startswith("_")})
            return {
                "total_lines": len(self.summaries),
                "total_matched": matched,
                "records": list(rows),
            }

    def record(self, line: int) -> dict[str, Any] | None:
        self.refresh()
        with self.lock:
            if line <= 0 or line > len(self.summaries):
                return None
            offset = int(self.summaries[line - 1]["_offset"])
            with open(self.path, "rb") as handle:
                handle.seek(offset)
                record = json.loads(handle.readline())
            record["line"] = line
            if "judge_logs" not in record:
                record["judge_logs"] = record_judge_logs(record)
            return record


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "RolloutViewer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    @property
    def rollout_store(self) -> RolloutStore:
        return self.server.rollout_store  # type: ignore[attr-defined]

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html()
            return
        try:
            if parsed.path == "/api/stats":
                self.send_json(self.api_stats())
            elif parsed.path == "/api/records":
                self.send_json(self.api_records(parse_qs(parsed.query)))
            elif parsed.path == "/api/record":
                self.send_json(self.api_record(parse_qs(parsed.query)))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": repr(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def api_stats(self) -> dict[str, Any]:
        return self.rollout_store.stats()

    def api_records(self, query: dict[str, list[str]]) -> dict[str, Any]:
        limit = int(query.get("limit", ["250"])[0])
        filters = {key: values[0] for key, values in query.items() if values and key != "limit"}
        return self.rollout_store.records(filters, limit)

    def api_record(self, query: dict[str, list[str]]) -> dict[str, Any]:
        line = int(query.get("line", ["0"])[0])
        if line <= 0:
            return {"error": "line must be positive"}
        record = self.rollout_store.record(line)
        return {"record": record} if record is not None else {"error": f"line {line} not found"}


def resolve_jsonl_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() or resolved.name != "rollouts.jsonl":
        return resolved
    candidates = list(resolved.parent.parent.glob("*/rollouts/rollouts.jsonl"))
    return candidates[0].resolve() if len(candidates) == 1 else resolved


def main() -> int:
    args = parse_args()
    path = resolve_jsonl_path(args.file)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.rollout_store = RolloutStore(path)  # type: ignore[attr-defined]
    shown_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Serving {html.escape(str(path))} at http://{shown_host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

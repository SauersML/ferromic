#!/usr/bin/env python3
"""Review chimpanzee-vs-GRCh38 inversion figures in a local web app."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "web/figures-site/public/inversions/data.json"
IMAGE_DIR = REPO_ROOT / "web/figures-site/public/inversions/img"
INVERSION_PATH = REPO_ROOT / "inv_properties.tsv"
DEFAULT_RESPONSES_PATH = REPO_ROOT / "data/chimp_alignment_responses.json"
ALLOWED_CLASSIFICATIONS = {"direct", "inverted", "na"}
WRITE_LOCK = threading.Lock()


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Chimp × GRCh38 alignment review</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #080b10;
      --panel: #11161f;
      --panel-raised: #171e29;
      --line: #283241;
      --line-strong: #3a4658;
      --text: #f1f4f8;
      --muted: #98a4b4;
      --green: #4bd29a;
      --blue: #7597ff;
      --amber: #e6b95c;
      --red: #ff7d76;
      --shadow: 0 18px 60px rgb(0 0 0 / 35%);
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; background: var(--bg); color: var(--text); }
    body {
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: -.01em;
    }
    button, select, a { font: inherit; }
    button, a { -webkit-tap-highlight-color: transparent; }
    .shell { min-height: 100vh; display: grid; grid-template-rows: auto auto 1fr auto; }
    header {
      display: flex; align-items: center; gap: 18px; padding: 16px 22px;
      border-bottom: 1px solid var(--line); background: #0c1017;
    }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 240px; }
    .mark {
      width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
      background: linear-gradient(135deg, var(--green), var(--blue)); color: #07100d;
      font-weight: 900; letter-spacing: -.08em;
    }
    h1 { font-size: 15px; margin: 0; font-weight: 720; letter-spacing: 0; }
    .subtitle { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .progress-block { display: flex; align-items: center; gap: 12px; flex: 1; }
    .progress-track {
      height: 7px; border-radius: 99px; background: #242c38; overflow: hidden; flex: 1;
      min-width: 100px;
    }
    .progress-fill {
      height: 100%; width: 0; border-radius: inherit;
      background: linear-gradient(90deg, var(--green), var(--blue)); transition: width .22s ease;
    }
    .progress-text { color: var(--muted); font-size: 12px; white-space: nowrap; }
    .progress-text strong { color: var(--text); }
    .toolbar { display: flex; gap: 8px; align-items: center; }
    .tool, .download {
      min-height: 35px; padding: 0 12px; border-radius: 9px; border: 1px solid var(--line);
      background: var(--panel); color: var(--text); text-decoration: none; cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center; font-size: 12px;
    }
    .tool:hover, .download:hover { border-color: var(--line-strong); background: var(--panel-raised); }
    .context {
      display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 18px;
      padding: 13px 22px; border-bottom: 1px solid var(--line); background: var(--panel);
    }
    .position { font-size: 12px; color: var(--muted); }
    .position strong { color: var(--text); font-variant-numeric: tabular-nums; }
    .locus { text-align: center; }
    .locus-id {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: clamp(14px, 1.5vw, 18px); font-weight: 700; letter-spacing: -.025em;
    }
    .locus-detail { color: var(--muted); font-size: 11px; margin-top: 3px; }
    .save-state { justify-self: end; display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); }
    .save-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 4px rgb(75 210 154 / 10%); }
    .save-state.saving .save-dot { background: var(--amber); }
    .save-state.error { color: var(--red); }
    .save-state.error .save-dot { background: var(--red); }
    main { min-height: 0; display: grid; place-items: center; padding: clamp(14px, 2.7vw, 36px); }
    .figure-frame {
      position: relative; width: min(100%, 1540px); min-height: 260px; display: grid; place-items: center;
      border: 1px solid var(--line); border-radius: 18px; overflow: hidden; background: white;
      box-shadow: var(--shadow);
    }
    .figure-frame img { display: block; width: 100%; height: auto; max-height: calc(100vh - 300px); object-fit: contain; }
    .loading {
      position: absolute; inset: 0; display: grid; place-items: center; background: white; color: #667080;
      font-size: 13px; transition: opacity .12s;
    }
    .loading.hidden { opacity: 0; pointer-events: none; }
    footer {
      border-top: 1px solid var(--line); background: #0c1017; padding: 15px 22px 18px;
      position: sticky; bottom: 0;
    }
    .prompt { text-align: center; color: var(--muted); font-size: 12px; margin-bottom: 11px; }
    .answers { display: grid; grid-template-columns: repeat(3, minmax(140px, 230px)); justify-content: center; gap: 10px; }
    .answer {
      min-height: 58px; border: 1px solid var(--line-strong); border-radius: 12px; cursor: pointer;
      background: var(--panel); color: var(--text); font-size: 15px; font-weight: 720;
      transition: transform .1s, border-color .1s, background .1s;
    }
    .answer:hover { transform: translateY(-2px); background: var(--panel-raised); }
    .answer:focus-visible, .tool:focus-visible, .download:focus-visible {
      outline: 2px solid var(--blue); outline-offset: 2px;
    }
    .answer span { display: block; font-size: 10px; color: var(--muted); margin-top: 2px; font-weight: 500; }
    .answer.direct { border-bottom: 3px solid var(--green); }
    .answer.inverted { border-bottom: 3px solid var(--blue); }
    .answer.na { border-bottom: 3px solid #758092; }
    .answer.selected.direct { background: rgb(75 210 154 / 16%); border-color: var(--green); }
    .answer.selected.inverted { background: rgb(117 151 255 / 16%); border-color: var(--blue); }
    .answer.selected.na { background: rgb(152 164 180 / 13%); border-color: #98a4b4; }
    .nav-row { display: flex; justify-content: center; gap: 16px; margin-top: 11px; }
    .nav {
      color: var(--muted); background: transparent; border: 0; cursor: pointer; font-size: 12px; padding: 5px 8px;
    }
    .nav:hover { color: var(--text); }
    .nav:disabled { opacity: .28; cursor: default; }
    .error-panel { max-width: 640px; color: var(--red); padding: 28px; text-align: center; }
    @media (max-width: 780px) {
      header { flex-wrap: wrap; padding: 13px 14px; gap: 11px; }
      .brand { min-width: 0; flex: 1; }
      .progress-block { order: 3; flex-basis: 100%; }
      .download { display: none; }
      .context { grid-template-columns: 1fr auto; padding: 11px 14px; }
      .position { display: none; }
      .locus { text-align: left; min-width: 0; }
      .locus-id { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      main { padding: 10px; align-items: start; }
      .figure-frame { min-height: 160px; border-radius: 12px; }
      .figure-frame img { max-height: none; }
      footer { padding: 12px 10px 14px; }
      .answers { grid-template-columns: repeat(3, 1fr); gap: 7px; }
      .answer { min-width: 0; min-height: 54px; font-size: 13px; }
    }
  </style>
</head>
<body>
<div class="shell" id="app">
  <header>
    <div class="brand">
      <div class="mark">↔</div>
      <div><h1>Chimp × GRCh38</h1><div class="subtitle">Inversion alignment review</div></div>
    </div>
    <div class="progress-block">
      <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
      <div class="progress-text"><strong id="answeredCount">0</strong> / <span id="totalCount">0</span> saved</div>
    </div>
    <div class="toolbar">
      <a class="download" href="/api/export.csv" download>Export CSV</a>
      <a class="download" href="/api/export.json" download>Export JSON</a>
      <button class="tool" id="nextUnanswered">Next unanswered</button>
    </div>
  </header>

  <section class="context">
    <div class="position">Figure <strong id="position">—</strong></div>
    <div class="locus">
      <div class="locus-id" id="locusId">Loading…</div>
      <div class="locus-detail" id="locusDetail"></div>
    </div>
    <div class="save-state" id="saveState"><span class="save-dot"></span><span id="saveText">Ready</span></div>
  </section>

  <main id="main">
    <div class="figure-frame">
      <img id="figure" alt="">
      <div class="loading" id="loading">Loading alignment figure…</div>
    </div>
  </main>

  <footer>
    <div class="prompt">Orientation of the chimp alignment inside the red inversion boundaries</div>
    <div class="answers">
      <button class="answer direct" data-value="direct">Direct<span>D</span></button>
      <button class="answer inverted" data-value="inverted">Inverted<span>I</span></button>
      <button class="answer na" data-value="na">N/A<span>N</span></button>
    </div>
    <div class="nav-row">
      <button class="nav" id="previous">← Back</button>
      <button class="nav" id="next">Next →</button>
    </div>
  </footer>
</div>

<script>
  const state = { items: [], responses: {}, index: 0, saving: false };
  const $ = id => document.getElementById(id);
  const answerButtons = [...document.querySelectorAll(".answer")];

  function formatNumber(value) {
    return new Intl.NumberFormat("en-US").format(value);
  }

  function currentItem() {
    return state.items[state.index];
  }

  function responseFor(invId) {
    return state.responses[invId]?.classification || null;
  }

  function setSaveState(mode, text) {
    $("saveState").className = "save-state" + (mode === "ready" ? "" : ` ${mode}`);
    $("saveText").textContent = text;
  }

  function updateProgress() {
    const answered = Object.keys(state.responses).filter(id => state.items.some(item => item.inv_id === id)).length;
    $("answeredCount").textContent = answered;
    $("totalCount").textContent = state.items.length;
    $("progressFill").style.width = state.items.length ? `${answered / state.items.length * 100}%` : "0%";
  }

  function preloadAround() {
    [state.index - 1, state.index + 1].forEach(index => {
      const item = state.items[index];
      if (item) (new Image()).src = item.image_url;
    });
  }

  function render() {
    const item = currentItem();
    if (!item) return;
    $("position").textContent = `${state.index + 1} of ${state.items.length}`;
    $("locusId").textContent = item.inv_id;
    $("locusDetail").textContent =
      `${item.region} · ${formatNumber(item.size_bp)} bp · ${item.image_file}`;
    const figure = $("figure");
    $("loading").classList.remove("hidden");
    figure.alt = `Chimpanzee versus GRCh38 alignment for ${item.inv_id}, ${item.region}`;
    figure.src = item.image_url;
    answerButtons.forEach(button => {
      button.classList.toggle("selected", button.dataset.value === responseFor(item.inv_id));
    });
    $("previous").disabled = state.index === 0;
    $("next").disabled = state.index === state.items.length - 1;
    const response = responseFor(item.inv_id);
    setSaveState("ready", response ? `Saved: ${response === "na" ? "N/A" : response[0].toUpperCase() + response.slice(1)}` : "Not answered");
    history.replaceState(null, "", `#${encodeURIComponent(item.inv_id)}`);
    preloadAround();
  }

  function goTo(index) {
    if (state.saving || index < 0 || index >= state.items.length) return;
    state.index = index;
    render();
  }

  function nextUnanswered(startAfterCurrent = false) {
    if (!state.items.length) return;
    const start = startAfterCurrent ? state.index + 1 : state.index;
    for (let offset = 0; offset < state.items.length; offset++) {
      const index = (start + offset) % state.items.length;
      if (!responseFor(state.items[index].inv_id)) {
        goTo(index);
        return;
      }
    }
    setSaveState("ready", "All figures answered");
  }

  async function classify(classification) {
    if (state.saving) return;
    const item = currentItem();
    state.saving = true;
    answerButtons.forEach(button => button.disabled = true);
    setSaveState("saving", "Saving…");
    try {
      const response = await fetch("/api/responses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inv_id: item.inv_id, classification })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      state.responses[item.inv_id] = payload.response;
      updateProgress();
      answerButtons.forEach(button => {
        button.classList.toggle("selected", button.dataset.value === classification);
      });
      setSaveState("ready", "Saved");
      if (state.index < state.items.length - 1) {
        window.setTimeout(() => goTo(state.index + 1), 180);
      } else if (Object.keys(state.responses).length < state.items.length) {
        window.setTimeout(() => nextUnanswered(true), 180);
      } else {
        setSaveState("ready", "All figures answered");
      }
    } catch (error) {
      setSaveState("error", `Not saved: ${error.message}`);
    } finally {
      state.saving = false;
      answerButtons.forEach(button => button.disabled = false);
    }
  }

  async function boot() {
    try {
      const response = await fetch("/api/bootstrap", { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      state.items = payload.items;
      state.responses = Object.fromEntries(payload.responses.map(record => [record.inv_id, record]));
      const hashId = decodeURIComponent(location.hash.slice(1));
      const hashIndex = state.items.findIndex(item => item.inv_id === hashId);
      if (hashIndex >= 0) state.index = hashIndex;
      else {
        const firstUnanswered = state.items.findIndex(item => !responseFor(item.inv_id));
        state.index = firstUnanswered >= 0 ? firstUnanswered : 0;
      }
      updateProgress();
      render();
    } catch (error) {
      $("main").innerHTML = `<div class="error-panel">Could not start the reviewer: ${error.message}</div>`;
      setSaveState("error", "Unavailable");
    }
  }

  $("figure").addEventListener("load", () => $("loading").classList.add("hidden"));
  $("figure").addEventListener("error", () => {
    $("loading").textContent = "This alignment figure could not be loaded.";
    $("loading").classList.remove("hidden");
  });
  answerButtons.forEach(button => button.addEventListener("click", () => classify(button.dataset.value)));
  $("previous").addEventListener("click", () => goTo(state.index - 1));
  $("next").addEventListener("click", () => goTo(state.index + 1));
  $("nextUnanswered").addEventListener("click", () => nextUnanswered(true));
  document.addEventListener("keydown", event => {
    if (event.metaKey || event.ctrlKey || event.altKey || state.saving) return;
    const key = event.key.toLowerCase();
    if (key === "d") classify("direct");
    else if (key === "i") classify("inverted");
    else if (key === "n") classify("na");
    else if (event.key === "ArrowLeft") goTo(state.index - 1);
    else if (event.key === "ArrowRight") goTo(state.index + 1);
  });

  boot();
</script>
</body>
</html>
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_coordinates() -> dict[str, dict[str, object]]:
    with INVERSION_PATH.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle, delimiter="\t")
        header = next(rows)
        chrom_index = header.index("Chromosome")
        start_index = header.index("Start")
        end_index = header.index("End")
        id_index = header.index("OrigID")
        coordinates: dict[str, dict[str, object]] = {}
        for row in rows:
            if len(row) <= id_index:
                continue
            inv_id = row[id_index].strip()
            if not inv_id:
                continue
            start = int(row[start_index])
            end = int(row[end_index])
            chrom = row[chrom_index]
            coordinates[inv_id] = {
                "chrom": chrom,
                "start": start,
                "end": end,
                "region": f"{chrom}:{start:,}-{end:,}",
            }
    return coordinates


def load_items() -> list[dict[str, object]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    coordinates = load_coordinates()
    items: list[dict[str, object]] = []
    for record in manifest["records"]:
        inv_id = record["inv_id"]
        image_file = f"{inv_id}.chimp.webp"
        image_path = IMAGE_DIR / image_file
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing chimp alignment figure: {image_path}")
        if inv_id not in coordinates:
            raise KeyError(f"No inv_properties.tsv row for {inv_id}")
        item = {
            "inv_id": inv_id,
            **coordinates[inv_id],
            "size_bp": record["size_bp"],
            "image_file": image_file,
            "image_url": f"/images/{image_file}",
        }
        items.append(item)
    return items


def load_response_document(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "schema_version": 1,
            "dataset": "chimp_vs_hg38_inversion_alignments",
            "updated_at": None,
            "responses": [],
        }
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("responses"), list):
        raise ValueError(f"Unsupported response file format: {path}")
    return document


def save_response_document(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ReviewServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        items: list[dict[str, object]],
        responses_path: Path,
    ) -> None:
        super().__init__(address, ReviewHandler)
        self.items = items
        self.items_by_id = {item["inv_id"]: item for item in items}
        self.responses_path = responses_path


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[review] {self.address_string()} {format_string % args}")

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        download_name: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_bytes(body, "application/json; charset=utf-8", status)

    def response_records(self) -> list[dict[str, object]]:
        return load_response_document(self.server.responses_path)["responses"]  # type: ignore[return-value]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self.send_bytes(APP_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/bootstrap":
                self.send_json({"items": self.server.items, "responses": self.response_records()})
                return
            if path == "/api/export.json":
                document = load_response_document(self.server.responses_path)
                body = (json.dumps(document, indent=2) + "\n").encode("utf-8")
                self.send_bytes(
                    body,
                    "application/json; charset=utf-8",
                    download_name="chimp_alignment_responses.json",
                )
                return
            if path == "/api/export.csv":
                output = io.StringIO()
                fields = [
                    "inv_id",
                    "chrom",
                    "start",
                    "end",
                    "region",
                    "size_bp",
                    "image_file",
                    "classification",
                    "updated_at",
                ]
                writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.response_records())
                self.send_bytes(
                    output.getvalue().encode("utf-8"),
                    "text/csv; charset=utf-8",
                    download_name="chimp_alignment_responses.csv",
                )
                return
            if path.startswith("/images/"):
                image_name = unquote(path.removeprefix("/images/"))
                inv_id = image_name.removesuffix(".chimp.webp")
                item = self.server.items_by_id.get(inv_id)
                if not item or item["image_file"] != image_name:
                    self.send_json({"error": "Unknown alignment figure"}, HTTPStatus.NOT_FOUND)
                    return
                image = (IMAGE_DIR / image_name).read_bytes()
                self.send_bytes(image, "image/webp")
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/responses":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 16_384:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(content_length))
            inv_id = payload.get("inv_id")
            classification = payload.get("classification")
            if inv_id not in self.server.items_by_id:
                raise ValueError("Unknown inversion ID")
            if classification not in ALLOWED_CLASSIFICATIONS:
                raise ValueError("Classification must be direct, inverted, or na")
            timestamp = utc_now()
            item = self.server.items_by_id[inv_id]
            response = {
                "inv_id": inv_id,
                "chrom": item["chrom"],
                "start": item["start"],
                "end": item["end"],
                "region": item["region"],
                "size_bp": item["size_bp"],
                "image_file": item["image_file"],
                "classification": classification,
                "updated_at": timestamp,
            }
            with WRITE_LOCK:
                document = load_response_document(self.server.responses_path)
                by_id = {record["inv_id"]: record for record in document["responses"]}  # type: ignore[index]
                by_id[inv_id] = response
                document["updated_at"] = timestamp
                document["responses"] = [
                    by_id[item_record["inv_id"]]
                    for item_record in self.server.items
                    if item_record["inv_id"] in by_id
                ]
                save_response_document(self.server.responses_path, document)
            self.send_json({"saved": True, "response": response})
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except OSError as error:
            self.send_json({"error": f"Could not save responses: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a click-through reviewer for all chimp-vs-GRCh38 inversion figures."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument(
        "--responses",
        type=Path,
        default=DEFAULT_RESPONSES_PATH,
        help=f"Response JSON path (default: {DEFAULT_RESPONSES_PATH})",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open the app in a browser")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_items()
    responses_path = args.responses.expanduser().resolve()
    load_response_document(responses_path)
    server = ReviewServer((args.host, args.port), items, responses_path)
    host_for_url = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{host_for_url}:{server.server_port}"
    print(f"Reviewing {len(items)} chimp-vs-GRCh38 figures")
    print(f"Responses will be saved to {responses_path}")
    print(f"Open {url}")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping reviewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

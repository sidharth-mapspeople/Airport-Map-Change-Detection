"""Build a visual results page from an evaluation run.

For each case: the original map and the edited version (detections marked on both), the changes that were
introduced (before/after crops from Test_files_edited/previews), the changes the system detected (old/new crops
with confidence and verdict), per-case scores, and cost and time per agent.

Built automatically at the end of scripts/evaluate_edits.py; rebuild an older run with scripts/visual_report.py.
Writes eval_results/<timestamp>/visual/index.html (+ img/). No model calls.
"""

from __future__ import annotations

import json
import webbrowser
from functools import lru_cache
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mapdiff.agents.change_detection import IDENTITY, alignment_transforms
from mapdiff.agents.reporting import confidence_text
from mapdiff.evaluation import EDITS_DIR, EVAL_DATA_DIR, describe, load_manifest, pct, score_case
from mapdiff.ingest import load_document
from mapdiff.llm.usage import cost_of
from mapdiff.models import AgentUsage, Change, ChangeSet, Point
from mapdiff.store import MapStore

PREVIEWS = EDITS_DIR / "previews"
DISPLAY_PX, CROP_SRC_PX, CROP_OUT_PX, CROP_HALF = 1400, 4000, 300, 0.035
COLORS = {"found": (22, 150, 80), "false_positive": (214, 48, 49), "rejected": (130, 130, 130),
          "missed": (235, 140, 0)}
LABELS = {"found": "Correct", "false_positive": "False positive", "rejected": "Rejected by confidence check",
          "missed": "Missed"}


@lru_cache(maxsize=4)
def page_image(path: str, px: int) -> Image.Image:
    return load_document(path, px)[0].image.convert("RGB")


def font(size: int) -> ImageFont.ImageFont:
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def save(img: Image.Image, out: Path, name: str) -> str:
    (out / "img").mkdir(parents=True, exist_ok=True)
    img.save(out / "img" / name, "JPEG", quality=78, optimize=True)
    return f"img/{name}"


def overlay(img: Image.Image, marks: list[tuple[Point, str, str]]) -> Image.Image:
    """Numbered rings at normalised points. marks: (point, status, label)."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    r = max(14, img.width // 60)
    f = font(int(r * 1.1))
    for p, status, label in marks:
        x, y, col = p.x * img.width, p.y * img.height, COLORS[status]
        for w in range(4):
            d.ellipse([x - r - w, y - r - w, x + r + w, y + r + w], outline=col)
        tb = d.textbbox((0, 0), label, font=f)
        tx, ty = x + r + 4, y - r - (tb[3] - tb[1])
        d.rectangle([tx - 3, ty - 2, tx + tb[2] - tb[0] + 3, ty + tb[3] - tb[1] + 6], fill=col)
        d.text((tx - tb[0], ty - tb[1] + 2), label, font=f, fill="white")
    return img


def crop(path: str, p: Point, status: str) -> Image.Image:
    img = page_image(path, CROP_SRC_PX)
    h = CROP_HALF * max(img.size)
    cx, cy = p.x * img.width, p.y * img.height
    c = img.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h))).resize((CROP_OUT_PX, CROP_OUT_PX))
    d = ImageDraw.Draw(c)
    r = CROP_OUT_PX * 0.12
    m = CROP_OUT_PX / 2
    for w in range(3):
        d.ellipse([m - r - w, m - r - w, m + r + w, m + r + w], outline=COLORS[status])
    return c


def positions(c: Change, cs: ChangeSet) -> tuple[Point | None, Point | None]:
    """(point on the old map, point on the new map) for a change, mapped across the alignment."""
    al = next((a for a in cs.alignment if a.page == c.page), None)
    to_old, to_new = alignment_transforms(al) if al else IDENTITY
    if c.old and c.new:
        return c.old.anchor, c.new.anchor
    if c.old:
        return c.old.anchor, to_new(c.old.anchor)
    if c.new:
        return to_old(c.new.anchor), c.new.anchor
    return None, None


def build(results_dir: Path, data_dir: Path = EVAL_DATA_DIR) -> Path:
    """Write ``results_dir/visual/index.html`` for an evaluation run; returns its path."""
    summary = json.loads((results_dir / "summary.json").read_text())
    manifest = {c["id"]: c for c in load_manifest()}
    out = results_dir / "visual"
    out.mkdir(parents=True, exist_ok=True)
    sections, overview = [], []
    store = MapStore(data_dir / summary["name"])
    maps = {m["file"]: m for m in summary["maps"]}
    overview.append(run_card(summary))
    for ce in summary["cases"]:
        case = manifest[ce["case"]]
        cs = ChangeSet.model_validate_json(Path(ce["report"]).with_suffix(".json").read_text())
        old, new = store.load(cs.map_id, cs.from_version), store.load(cs.map_id, cs.to_version)
        old_src = str(store.source_path(cs.map_id, cs.from_version))
        new_src = str(store.source_path(cs.map_id, cs.to_version))
        score = score_case(case["id"], case["expected"], cs)
        sections.append((case["id"], case_section(case, ce, cs, score, old, new, old_src, new_src,
                                                  maps.get(case["variant"]), out)))
    (out / "index.html").write_text(page(summary, overview, sections))
    return out / "index.html"


def cost(usage: list[dict]) -> float:
    return cost_of([AgentUsage.model_validate(u) for u in usage]) or 0.0


def run_card(prof: dict) -> str:
    usage = prof["usage"]
    tot_cost = cost(usage)
    tot_time = sum(m["seconds"] for m in prof["maps"] if not m["cached"]) + sum(
        c["change_scoring_seconds"] for c in prof["cases"])
    fresh = [m for m in prof["maps"] if m["usage"]]
    per_map = sum(cost(m["usage"]) for m in fresh) / len(fresh) if fresh else 0
    rows = "".join(
        f"<tr><td>{escape(u['agent'])}</td><td>{escape(prof['agents'].get(u['agent'], ', '.join(u['models'])))}</td>"
        f"<td class=num>{u['calls']}</td><td class=num>{u['input_tokens']:,}</td><td class=num>{u['output_tokens']:,}</td>"
        f"<td class=num>${(u['cost_usd'] or 0):.3f}</td><td class=num>{u['wall_seconds']:.0f}s</td>"
        f"<td><div class=bar><span style='width:{100 * (u['cost_usd'] or 0) / tot_cost if tot_cost else 0:.0f}%'></span></div></td></tr>"
        for u in sorted(usage, key=lambda u: -(u["cost_usd"] or 0)))
    status = escape(prof["stopped"]) if prof["stopped"] else ""
    return f"""
<section class=card>
  <h2>Run: {escape(prof['name'])}</h2>
  {f'<p class=warn>{status}</p>' if status else ''}
  <div class=stats>
    <div><b>{pct(prof['recall'])}</b><span>Recall (edits found)</span></div>
    <div><b>{pct(prof['precision'])}</b><span>Precision</span></div>
    <div><b>{prof['false_positives']}</b><span>False positives</span></div>
    <div><b>${tot_cost:.2f}</b><span>Total model cost</span></div>
    <div><b>${per_map:.3f}</b><span>Cost per map</span></div>
    <div><b>{tot_time / 60:.1f} min</b><span>Total time</span></div>
  </div>
  <h3>Cost and time per agent</h3>
  <div class=scroll><table>
    <thead><tr><th>Agent</th><th>Model</th><th class=num>Calls</th><th class=num>Input tok</th>
      <th class=num>Output tok</th><th class=num>Cost</th><th class=num>Wall time</th><th>Share of cost</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</section>"""


def case_section(case, ce, cs, score, old, new, old_src, new_src, vmap, out) -> str:
    slug = case['id']
    status_of: dict[int, str] = {}
    for _, c, st in score.pairs:
        if c is not None:
            status_of[id(c)] = st
    for c in score.unmatched:
        status_of[id(c)] = "false_positive"
    detected = [c for c in cs.changes if id(c) in status_of or c.rejected]
    for c in detected:
        status_of.setdefault(id(c), "rejected")

    old_marks, new_marks, items = [], [], []
    for n, c in enumerate(detected, 1):
        st = status_of[id(c)]
        po, pn = positions(c, cs)
        if po:
            old_marks.append((po, st, str(n)))
        if pn:
            new_marks.append((pn, st, str(n)))
        crops = ""
        if po and pn:
            a = save(crop(old_src, po, st), out, f"{slug}-c{n}-old.jpg")
            b = save(crop(new_src, pn, st), out, f"{slug}-c{n}-new.jpg")
            crops = (f"<div class=pair><figure><img src='{a}' alt='old'><figcaption>Before</figcaption></figure>"
                     f"<figure><img src='{b}' alt='new'><figcaption>After</figcaption></figure></div>")
        note = c.verification_note or c.reason
        items.append(f"""<li class='det {st}'>
  <div class=dethead><span class=num-badge>{n}</span><span class='chip {st}'>{LABELS[st]}</span>
    <span class=conf>{escape(confidence_text(c))}{' · needs review' if c.needs_review else ''}</span></div>
  <p>{escape(c.summary)}</p>{f'<p class=note>{escape(note)}</p>' if note else ''}{crops}</li>""")

    # Missed expected changes with a known location are marked on both maps.
    for exp, _, st in score.pairs:
        if st != "found" and "at" in exp:
            p = Point(x=exp["at"][0], y=exp["at"][1])
            old_marks.append((p, "missed", "?"))
            new_marks.append((p, "missed", "?"))

    img_old = save(overlay(page_image(old_src, DISPLAY_PX), old_marks), out, f"{slug}-old.jpg")
    img_new = save(overlay(page_image(new_src, DISPLAY_PX), new_marks), out, f"{slug}-new.jpg")

    prefix = Path(case["variant"]).stem.rsplit("_", 1)[0] + ("_edit" if case["kind"] == "edit" else "_restyle")
    previews = sorted(PREVIEWS.glob(f"{prefix}*.png"))
    prev_html = "".join(f"<img src='{save(Image.open(p).convert('RGB'), out, f'{slug}-prev-{i}.jpg')}' "
                        f"alt='before and after'>" for i, p in enumerate(previews))
    exp_rows = "".join(
        f"<li><span class='chip {st}'>{LABELS[st] if st != 'found' else 'Found'}</span> {escape(describe(e))}"
        + (f" <span class=muted>→ detection #{detected.index(c) + 1}</span>" if c is not None and c in detected else "")
        + "</li>" for e, c, st in score.pairs) or "<li class=muted>None: content is identical, so any detection is a false positive.</li>"

    mo = old.quality.mapping_confidence if old.quality else None
    mn = new.quality.mapping_confidence if new.quality else None
    case_cost = cost((vmap or {}).get("usage", [])) + cost(ce["change_scoring_usage"])
    secs = (vmap or {}).get("seconds", 0) + ce["change_scoring_seconds"]
    return f"""
<section class='card case' id='{escape(slug)}'>
  <div class=casehead><h2>{escape(case['id'])}</h2><span class='kind {case['kind']}'>{case['kind']}</span></div>
  <p>{escape(case['description'])}</p>
  <div class=stats small>
    <div><b>{score.expected}</b><span>Changes introduced</span></div>
    <div><b>{len(score.true_positives)}</b><span>Found</span></div>
    <div><b>{len(score.missed)}</b><span>Missed</span></div>
    <div><b>{len(score.false_positives)}</b><span>False positives</span></div>
    <div><b>{pct(score.recall)}</b><span>Recall</span></div>
    <div><b>{'–' if mo is None else f'{mo:.2f}'} → {'–' if mn is None else f'{mn:.2f}'}</b><span>Mapping confidence</span></div>
    <div><b>${case_cost:.3f}</b><span>Cost (edited map + scoring)</span></div>
    <div><b>{secs:.0f}s</b><span>Time</span></div>
  </div>
  <div class=maps>
    <figure><img src='{img_old}' alt='original map'><figcaption>Original · <code>{escape(Path(case['baseline']).name)}</code></figcaption></figure>
    <figure><img src='{img_new}' alt='edited map'><figcaption>Edited · <code>{escape(Path(case['variant']).name)}</code></figcaption></figure>
  </div>
  <div class=cols>
    <div><h3>Changes introduced</h3><ul class=plain>{exp_rows}</ul><div class=previews>{prev_html}</div></div>
    <div><h3>Changes detected ({len(detected)})</h3>
      <ol class=plain>{''.join(items) or '<li class=muted>No changes reported.</li>'}</ol></div>
  </div>
</section>"""


def page(summary: dict, overview: list[str], sections: list[tuple[str, str]]) -> str:
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Change Detection Results</title>
<style>
:root {{ --bg:#f6f6f4; --card:#fff; --ink:#1c1e22; --muted:#606671; --line:#dcdfe4; --soft:#eef0f3;
  --found:#169650; --fp:#d63031; --rej:#828282; --miss:#eb8c00; --accent:#3a6fd8; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141619; --card:#1c1f24; --ink:#e7e9ed;
  --muted:#a0a6b0; --line:#343942; --soft:#262a31; --accent:#7ba3f2; }} }}
:root[data-theme="dark"] {{ --bg:#141619; --card:#1c1f24; --ink:#e7e9ed; --muted:#a0a6b0; --line:#343942;
  --soft:#262a31; --accent:#7ba3f2; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif }}
main {{ max-width:1200px; margin:0 auto; padding:28px 16px 60px }}
h1 {{ font-size:26px; margin:0 0 4px }} h2 {{ font-size:19px; margin:0 }} h3 {{ font-size:15px; margin:18px 0 8px }}
.lede,.muted {{ color:var(--muted) }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; margin:18px 0 }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:14px 0 }}
.stats div {{ background:var(--soft); border-radius:10px; padding:10px 12px }}
.stats b {{ display:block; font-size:20px }} .stats span {{ color:var(--muted); font-size:12.5px }}
.stats[small] b {{ font-size:17px }}
table {{ width:100%; border-collapse:collapse; font-size:13.5px }}
th,td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); white-space:nowrap }}
th {{ color:var(--muted); font-weight:600; font-size:12px }} .num {{ text-align:right }}
.scroll {{ overflow-x:auto }}
.bar {{ width:120px; height:8px; background:var(--soft); border-radius:4px }} .bar span {{ display:block; height:100%; background:var(--accent); border-radius:4px }}
.casehead {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap }}
.kind {{ font-size:12px; font-weight:700; padding:2px 8px; border-radius:20px; background:var(--soft); text-transform:uppercase; letter-spacing:.05em }}
.maps {{ display:grid; grid-template-columns:1fr 1fr; gap:12px }}
.maps img {{ width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; cursor:zoom-in }}
figure {{ margin:0 }} figcaption {{ color:var(--muted); font-size:12.5px; margin-top:4px }}
.cols {{ display:grid; grid-template-columns:1fr 1.3fr; gap:24px }}
ul.plain,ol.plain {{ list-style:none; padding:0; margin:0 }}
ul.plain li {{ margin:6px 0 }}
.previews img {{ max-width:100%; border:1px solid var(--line); border-radius:6px; margin-top:8px; display:block }}
.det {{ border:1px solid var(--line); border-left:4px solid var(--rej); border-radius:8px; padding:10px 12px; margin:0 0 10px }}
.det.found {{ border-left-color:var(--found) }} .det.false_positive {{ border-left-color:var(--fp) }}
.det p {{ margin:6px 0 0 }} .note {{ color:var(--muted); font-size:13px }}
.dethead {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap }}
.num-badge {{ font-weight:700; background:var(--ink); color:var(--card); border-radius:50%; width:22px; height:22px; display:inline-grid; place-items:center; font-size:12px }}
.chip {{ font-size:12px; font-weight:600; padding:1px 8px; border-radius:20px; color:#fff; background:var(--rej) }}
.chip.found {{ background:var(--found) }} .chip.false_positive {{ background:var(--fp) }} .chip.missed {{ background:var(--miss) }}
.conf {{ color:var(--muted); font-size:12.5px }}
.pair {{ display:flex; gap:8px; margin-top:8px }} .pair img {{ width:150px; height:150px; border-radius:6px; border:1px solid var(--line) }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:13px; color:var(--muted) }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:50%; border:3px solid; margin-right:5px; vertical-align:-2px }}
.warn {{ color:var(--miss); font-weight:600 }}
nav a {{ color:var(--accent); margin-right:10px; font-size:13.5px }}
@media (max-width:760px) {{ .maps,.cols {{ grid-template-columns:1fr }} }}
</style></head><body><main>
<h1>Change Detection Results</h1>
<p class=lede>Synthetic edits of real MapsPeople airport maps. Each case compares the original with an edited or
restyled copy. Edit cases measure what the system finds; restyle cases (same content, new colours or scale) should
produce no changes. Total model spend for this run: <b>${summary['spent_usd']:.2f}</b>.</p>
<div class=legend><span><i style="border-color:var(--found)"></i>Correct detection</span>
<span><i style="border-color:var(--fp)"></i>False positive</span><span><i style="border-color:var(--rej)"></i>Rejected by change-confidence check</span>
<span><i style="border-color:var(--miss)"></i>Missed (known location)</span></div>
{''.join(overview)}
<nav>{''.join(f"<a href='#{escape(slug)}'>{escape(slug)}</a>" for slug, _ in sections)}</nav>
{''.join(html for _, html in sections)}
<script>document.querySelectorAll('.maps img').forEach(i => i.onclick = () => window.open(i.src));</script>
</main></body></html>"""


def show(page_path: Path) -> None:
    """Open a built page in the default browser."""
    webbrowser.open(page_path.resolve().as_uri())

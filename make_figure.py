#!/usr/bin/env python3
"""Draw the figure at the top of README.md from the files in results/.

    python make_figure.py

Writes docs/images/decode-light.svg and docs/images/decode-dark.svg, and prints the sentence the figure's alt text uses.
"""
import json
import os
import re
import statistics
import sys

import make_table

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "images")
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "before": "#2a78d6", "after": "#eb6834"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "before": "#3987e5", "after": "#d95926"},
}
KEYS = [("before", "Build 26200.6899, before the update"), ("after", "Build 26200.9711, after the update")]
TITLE = "Gemma 4 E4B as a Foundry Local model: decode speed"
NOTE = "Tokens per second on one laptop, on two Windows 11 builds"
WIDTH, BAR, VALUE_ROOM = 860, 22, 60


def read(name):
    path = os.path.join(HERE, "results", name)
    if not os.path.exists(path):
        sys.exit(f"make_figure: results/{name} not found")
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def before(form, device):
    """Return the decode speed of a Foundry Local row of make_table.ROWS."""
    row = next((r for r in make_table.ROWS if r[0] == form and r[1] == "Foundry" and device in r[2]), None)
    if row is None:
        sys.exit(f"make_figure: no Foundry Local row for {form!r} in make_table.ROWS")
    speed = make_table.load(row[4])
    if speed is None or "decode_tok_s_median" not in speed:
        sys.exit(f"make_figure: no decode speed in results/{row[4]}")
    return speed["decode_tok_s_median"]


def after_gpu():
    """Return the median of the runs after the update with no limit on the daemon's CPU cores."""
    rows = re.findall(r'^\S+\s+(\{"label": "all2?",.*\})\s*$', read("r8_diag_after_update.txt"), re.M)
    if not rows:
        sys.exit("make_figure: no unlimited runs in results/r8_diag_after_update.txt")
    return statistics.median(json.loads(r)["decode_tok_s"] for r in rows)


def after_cpu():
    """Return the median decode speed of the CPU package after the update."""
    m = re.search(r"SUMMARY (\{.*\})", read("r8_cpu_after.txt"))
    if m is None:
        sys.exit("make_figure: no SUMMARY in results/r8_cpu_after.txt")
    return json.loads(m.group(1))["decode_tok_s_median"]


def speeds():
    """Return [(label, before, after, digits printed for after)]."""
    return [
        ("Int4 package, 8K context, RTX 5080 Laptop", before("int4 (RTN), 8K context", "RTX"), after_gpu(), 0),
        ("Int4 package, Core Ultra 9 386H CPU", before("int4 (RTN, CPU recipe)", "CPU"), after_cpu(), 1),
    ]


def text(x, y, body, size, fill, weight="400"):
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{fill}">{body}</text>'


def draw(rows, theme, desc):
    c = THEMES[theme]
    unit = (WIDTH - VALUE_ROOM) / max(max(b, a) for _, b, a, _ in rows)
    parts = [text(0, 28, TITLE, 20, c["text"], "600"), text(0, 52, NOTE, 14, c["muted"])]
    x = 0
    for key, label in KEYS:
        parts.append(f'<rect x="{x:g}" y="66" width="14" height="14" rx="4" fill="{c[key]}"/>')
        parts.append(text(x + 20, 78, label, 14, c["muted"]))
        x += 20 + 8.2 * len(label) + 20
    y = 116
    for label, b, a, digits in rows:
        parts.append(text(0, y, label, 15, c["text"]))
        for i, (key, value, shown) in enumerate((("before", b, f"{b:.1f}"), ("after", a, f"{a:.{digits}f}"))):
            top = y + 8 + i * (BAR + 4)
            w = value * unit
            parts.append(f'<rect x="0" y="{top:g}" width="{w:g}" height="{BAR}" rx="4" fill="{c[key]}"/>')
            parts.append(text(w + 8, top + 16, shown, 15, c["text"], "600"))
        y += 8 + 2 * BAR + 4 + 36
    height = y - 36 + 8
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" height="{height:g}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{TITLE}</title>\n<desc id="d">{desc}</desc>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def main():
    rows = speeds()
    desc = " ".join(f"{label}: {b:.1f} before, {a:.{digits}f} after." for label, b, a, digits in rows)
    os.makedirs(OUT, exist_ok=True)
    for theme in THEMES:
        with open(os.path.join(OUT, f"decode-{theme}.svg"), "w", encoding="utf-8") as f:
            f.write(draw(rows, theme, desc))
    print(desc)


if __name__ == "__main__":
    main()

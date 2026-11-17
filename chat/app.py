"""Chat with Gemma 4 E4B on this PC: a small window, the answer streams in token by token, and a badge shows how many
network connections the chat holds (read from Windows every second).

    python app.py [--device auto|cuda|webgpu|cpu|fake] [--adapter N] [--model-dir <folder>]
                  [--context 2048|4096|8192] [--maximized] [--log <file.jsonl>]

--log (chat.cmd -Log) adds one JSON line to the file each time an answer ends: the time, the question, the whole answer,
the line under the answer exactly as the window shows it, the token count, seconds, tokens/s, milliseconds from the
press to the first token on the screen, how it ended (finish: "end" = the model ended it, "full" = the chat filled up
during it, "stopped" = Stop, "full_before" = the chat was full before the question, "error"), the chat's token count
and turn, the device, the chat's seed (one per chat: chat_seed) and the network badge's text at that moment. An
unexpected error in the window adds a line with "error" instead. Without --log nothing is written.

chat.cmd starts it after the setup. The model runs in a child process of this window (engine.py --serve), connected by
pipes: no network, and the window keeps responding while the model loads or reads a long question. This process never
imports the model runtime.

The window's title tells what it is doing: "Gemma 4 E4B — loading model…" while the model loads, "Gemma 4 E4B" when it
is ready, "Gemma 4 E4B — answering…" while an answer streams, "Gemma 4 E4B — model not loaded" after a failure.
"""
import prelude  # noqa: F401  (first: telemetry off and power throttling off before anything else loads)

import argparse
import ctypes
import json
from datetime import datetime
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
import tkinter.font as tkfont
import unicodedata

import customtkinter as ctk

import engine
import netcount

HERE = os.path.dirname(os.path.abspath(__file__))
APP_TITLE = "Gemma 4 E4B"
TITLES = {"loading": APP_TITLE + " — loading model…", "ready": APP_TITLE,
          "answering": APP_TITLE + " — answering…", "failed": APP_TITLE + " — model not loaded"}
GUIDE = "Ask Gemma 4 anything. It runs on this PC."
# The suggestions on the first screen, sent to the model exactly as written. A click puts the text in the input box and
# sends it, the same way as typing it.
SUGGESTIONS = ["Explain why the sky is blue.",
               "Write a polite email asking my landlord to fix a leaking faucet.",
               "Give me a 3-day meal plan with a shopping list."]
PLACEHOLDER = "Ask anything"
HINT = "Enter to send · Shift+Enter for a new line · Ctrl + / Ctrl − text size"
MUTEX_NAME = "Local\\Gemma4E4BChatOnThisPC"
ZOOMS = [0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

# (light, dark) colors
C = {
    "bg": ("#FFFFFF", "#1E1F22"), "panel": ("#F5F6F8", "#25272B"), "line": ("#E1E4E8", "#34373D"),
    "text": ("#1F2328", "#E8E9EC"), "dim": ("#5F6670", "#A3A9B1"), "accent": ("#1A73E8", "#4C8DF6"),
    "accent_hover": ("#1557B0", "#3B79E0"), "bubble": ("#E8F0FE", "#2C3B57"), "chip": ("#F1F3F5", "#2B2E33"),
    "chip_hover": ("#E2E6EA", "#363A40"), "input": ("#FFFFFF", "#2A2C30"), "code": ("#F3F4F6", "#2A2C31"),
    "ok_bg": ("#E6F4EA", "#1D3A28"), "ok_text": ("#137333", "#86DBA4"), "warn_bg": ("#FCE8E6", "#4A2422"),
    "warn_text": ("#B3261E", "#F4B9B4"), "stop": ("#C5221F", "#E8625C"), "stop_hover": ("#A31A17", "#D14B45"),
    "on_accent": ("#FFFFFF", "#FFFFFF"),
}
PRESSED_MS = 200   # a pressed suggestion keeps its pressed color this long before the chat shows
STOP_GUARD_S = 0.4   # the Send button turns into Stop; a second click this soon after Send is not a Stop
STOP_TIMEOUT_MS = 10000   # after Stop, no token and no end for this long = the model process is ended
# font sizes in pixels at 100 % (Windows display scaling and the Ctrl +/- zoom multiply them)
UI_FONTS = {"title": (22, "bold"), "device": (13, "normal"), "badge": (13, "normal"), "button": (14, "normal"),
            "guide": (24, "bold"), "status": (14, "normal"), "chip": (15, "normal"), "input": (16, "normal"),
            "hint": (12, "normal"), "bubble": (16, "normal"), "bar": (14, "normal")}
TEXT_FONTS = {"p": (16, "normal", "roman"), "b": (16, "bold", "roman"), "i": (16, "normal", "italic"),
              "bi": (16, "bold", "italic"), "h1": (24, "bold", "roman"), "h2": (20, "bold", "roman"),
              "h3": (18, "bold", "roman"), "h4": (16, "bold", "roman"), "mono": (14, "normal", "roman"),
              "monob": (14, "bold", "roman"), "monoi": (14, "normal", "italic"), "stats": (14, "normal", "roman"),
              "statsb": (14, "bold", "roman"),
              "gap": (7, "normal", "roman"), "hr": (2, "normal", "roman")}
COLUMN = 860   # widest text column, in pixels at 100 %


def pick(color, mode):
    return color[1] if mode == "Dark" else color[0]


# --- text Tk cannot draw, and math markup ------------------------------------------------------------------------------
# Tk 8.6 draws no color emoji: characters above U+FFFF (most emoji, flags), and symbols asked for in emoji style (a
# U+FE0F after them) or emoji-style by default, are left out, with the space next to them.
EMOJI_BMP = set(map(ord, "⌚⌛⏩⏪⏫⏬⏰⏳◽◾☔☕♈♉♊"
                         "♋♌♍♎♏♐♑♒♓♿⚓⚡⚪⚫⚽"
                         "⚾⛄⛅⛎⛔⛪⛲⛳⛵⛺⛽✅✊✋✨"
                         "❌❎❓❔❕❗➕➖➗➰➿⬛⬜⭐⭕"))
JOINERS = {0x200D, 0xFE0E, 0xFE0F, 0x20E3}
GONE = "\x00"


def drop_emoji(text):
    out = []
    chars = list(text)
    for k, ch in enumerate(chars):
        cp = ord(ch)
        nxt = ord(chars[k + 1]) if k + 1 < len(chars) else 0
        if cp > 0xFFFF or 0xD800 <= cp <= 0xDFFF or cp in EMOJI_BMP or cp in JOINERS or \
                (nxt == 0xFE0F and not ch.isalnum() and cp > 0x7F):
            out.append(GONE)
        else:
            out.append(ch)
    s = "".join(out)
    if GONE not in s:
        return s

    def gap(m):
        inside = m.start() > 0 and m.end() < len(s)
        return " " if inside and m.group(1) and m.group(2) else ""

    return re.sub(r"([ \t]*)\x00+([ \t]*)", gap, s)


MATH = {"rightarrow": "→", "to": "→", "leftarrow": "←", "gets": "←", "Rightarrow": "⇒", "implies": "⇒",
        "Leftarrow": "⇐", "leftrightarrow": "↔", "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓",
        "approx": "≈", "sim": "~", "simeq": "≃", "propto": "∝", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥",
        "neq": "≠", "ne": "≠", "equiv": "≡", "circ": "°", "degree": "°", "infty": "∞", "partial": "∂",
        "nabla": "∇", "sum": "∑", "prod": "∏", "int": "∫", "ldots": "…", "cdots": "⋯", "dots": "…",
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
        "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
        "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ",
        "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
        "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
        "quad": " ", "qquad": "  ", "left": "", "right": "", "displaystyle": "", "big": "", "Big": ""}
MATH_PUNCT = {",": " ", ";": " ", ":": " ", "!": "", " ": " ", "%": "%", "$": "$", "&": "&", "#": "#", "_": "_",
              "{": "{", "}": "}", "\\": " "}
SUPERSCRIPT = dict(zip("0123456789+-−=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿⁱ"))
SUBSCRIPT = dict(zip("0123456789+-−=()aeoxhklmnpst", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ"))
MATH_SPAN = re.compile(r"\$(?=[^\s$])([^$\n]{1,120}?)(?<=[^\s$])\$")
DISPLAY_MATH = re.compile(r"\$\$(?=\S)([^$\n]{1,200}?)(?<=\S)\$\$")
CODE_SPAN = re.compile(r"`[^`\n]*`")


def latex_text(body):
    """The text of a short LaTeX formula: \\text{N}_2 -> N₂, 1/\\lambda^4 -> 1/λ⁴, \\propto -> ∝, 200^\\circ -> 200°; a
    command not in the MATH table keeps its name without the backslash."""
    # commands whose argument is the text: \text{N} -> N (also \vec{v} -> v, \hat{x} -> x)
    s = re.sub(r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|mathit|mathsf|mathcal|boldsymbol|operatorname|mbox|"
               r"vec|hat|bar|overline|underline|tilde)\s*\{([^{}]*)\}", r"\1", body)
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", lambda m: "√" + (m.group(1) if len(m.group(1)) == 1 else f"({m.group(1)})"), s)
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "°", s)
    s = re.sub(r"\\([A-Za-z]+)\s?|\\(.)",
               lambda m: MATH.get(m.group(1), m.group(1)) + (" " if m.group(0).endswith(" ") and
                                                             MATH.get(m.group(1), m.group(1)) else "")
               if m.group(1) else MATH_PUNCT.get(m.group(2), m.group(2)), s)

    def script(m, table, mark):
        text = m.group(1) if m.group(1) is not None else m.group(2)
        if text and all(ch in table for ch in text):
            return "".join(table[ch] for ch in text)
        return mark + (text if len(text) == 1 else f"({text})")

    s = re.sub(r"\^\{([^{}]*)\}|\^(\S)", lambda m: script(m, SUPERSCRIPT, "^"), s)
    s = re.sub(r"_\{([^{}]*)\}|_(\S)", lambda m: script(m, SUBSCRIPT, "_"), s)
    return s.replace("{", "").replace("}", "")


def replace_math(text):
    """$...$ and $$...$$ formulas become plain text (see latex_text); `code` is left as it is. A $...$ span counts as a
    formula when it holds \\ ^ _ { } or is 1 to 3 letters and does not start with "{", so prices such as "$5 and $10"
    and shell text such as ${HOME} stay as they are."""
    if "$" not in text:
        return text

    def span(m):
        body = m.group(1)
        if body.startswith("{") or (not re.search(r"[\\^_{}]", body) and not re.fullmatch(r"[A-Za-z]{1,3}", body)):
            return m.group(0)
        return latex_text(body)

    def part(chunk):
        chunk = DISPLAY_MATH.sub(lambda m: latex_text(m.group(1)), chunk)
        return MATH_SPAN.sub(span, chunk)

    out, pos = [], 0
    for m in CODE_SPAN.finditer(text):
        out.append(part(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(part(text[pos:]))
    return "".join(out)


def hold_open_math(line):
    """A line still streaming in may end inside a $... formula that is not closed yet: show the text before it until
    the closing $ arrives (a $ followed by plain text, such as a price, is shown at once)."""
    if "$" not in line:
        return line
    masked = CODE_SPAN.sub(lambda m: " " * len(m.group(0)), line)
    masked = DISPLAY_MATH.sub(lambda m: " " * len(m.group(0)), masked)
    masked = MATH_SPAN.sub(lambda m: " " * len(m.group(0)), masked)
    i = masked.find("$")
    while i >= 0:
        rest = masked[i + 1:].lstrip("$")
        if rest[:1] and not rest[:1].isspace() and re.search(r"[\\^_{}]", rest):
            return line[:i]
        i = masked.find("$", i + 1)
    return line


def display_text(text):
    return drop_emoji(replace_math(text))


def cell_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


# --- markdown, line by line, while the answer streams in -------------------------------------------------------------
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
RULE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
NUMBER = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
CHECKBOX = re.compile(r"\[([ xX])\]\s+")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_MIN_CHARS = 12   # a table whose columns cannot get this many characters each is drawn as blocks
TABLE_SEP = re.compile(r"^\s*\|?(\s*:?-{2,}:?\s*\|)+\s*(:?-{2,}:?\s*)?\|?\s*$")
INLINE = re.compile(r"\*\*\*(?=\S)(?P<bi>.+?)(?<=\S)\*\*\*"
                    r"|\*\*(?=\S)(?P<b>.+?)(?<=\S)\*\*"
                    r"|__(?=\S)(?P<b2>.+?)(?<=\S)__"
                    r"|(?<![\w*])\*(?=[^\s*])(?P<i>.+?)(?<=[^\s*])\*(?![\w*])"
                    r"|(?<![\w_])_(?=[^\s_])(?P<i2>.+?)(?<=[^\s_])_(?![\w_])"
                    r"|`(?P<code>[^`]+)`"
                    r"|\[(?P<link>[^\]]+)\]\([^)\s]*\)")
STYLE_OF = {"bi": "bi", "b": "b", "b2": "b", "i": "i", "i2": "i", "code": "icode", "link": "link"}


def inline_segments(text):
    """[(text, inline style or "")] for **bold**, *italic*, `code` and [links](...) (shown as their text)."""
    out, pos = [], 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], ""))
        name = next(k for k, v in m.groupdict().items() if v is not None)
        out.append((m.group(name), STYLE_OF[name]))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], ""))
    return out


def close_open_marks(line):
    """A line still streaming in may end inside **bold** or `code`: close the mark so it does not show as text. A mark
    that opens at the very end of the line (after a space, or alone), with none of its text yet, waits for its text."""
    opening = re.search(r"(?:^|(?<=\s))[*_`]+$", line)
    if opening:
        line = line[:opening.start()]
    if line.count("`") % 2:
        line += "`"
    if line.replace("***", "").count("**") % 2 and not line.rstrip().endswith("**"):
        line += "**"
    return line


def plain(text):
    return "".join(seg for seg, _ in inline_segments(text))


def depth_of(indent):
    n = len(indent.replace("\t", "    "))
    return 0 if n < 2 else 1 if n < 6 else 2


def split_cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def wrap_segments(segs, width):
    """Lines of at most `width` cells from [(text, style)], broken at spaces (a longer word is cut)."""
    lines, line, used = [], [], 0

    def push(text, style):
        nonlocal used
        if line and line[-1][1] == style:
            line[-1] = (line[-1][0] + text, style)
        else:
            line.append((text, style))
        used += cell_width(text)

    def end_line():
        nonlocal line, used
        while line and line[-1][0].endswith(" "):
            text, style = line[-1]
            line[-1] = (text.rstrip(" "), style)
            if not line[-1][0]:
                line.pop()
        lines.append(line)
        line, used = [], 0

    for text, style in segs:
        for part in re.findall(r"\S+|\s+", text):
            if part.isspace():
                if line and used < width:
                    push(" ", style)
                continue
            if used and used + cell_width(part) > width:
                end_line()
            while cell_width(part) > width:
                cut, w = "", 0
                for ch in part:
                    if w + cell_width(ch) > width:
                        break
                    cut, w = cut + ch, w + cell_width(ch)
                push(cut, style)
                end_line()
                part = part[len(cut):]
            if part:
                push(part, style)
    if line or not lines:
        end_line()
    return lines


class AnswerView:
    """One answer in the chat's Text widget. Complete lines are drawn once, for good; the line still coming in (and a
    table still growing) are drawn again on every update after the mark that separates them."""

    count = 0

    def __init__(self, chat):
        self.chat, self.t = chat, chat.text
        self.raw = ""
        self.footer = []
        self.finished = False
        self.reset()

    def reset(self):
        if getattr(self, "mark", None):
            self.t.mark_unset(self.mark)
        AnswerView.count += 1
        self.mark = f"answer{AnswerView.count}"
        self.done = 0          # characters of raw drawn for good
        self.in_code = False
        self.in_math = False   # inside a $$ ... $$ block of lines
        self.table = []
        self.blank = True      # the last line drawn for good was blank (or there is none yet)
        self.t.mark_set(self.mark, "end-1c")
        self.t.mark_gravity(self.mark, "left")

    def feed(self, text):
        self.raw += text

    def draw(self, footer, final=False):
        t = self.t
        self.footer = footer
        self.finished = final
        t.delete(self.mark, "end-1c")
        t.mark_gravity(self.mark, "right")
        while True:
            nl = self.raw.find("\n", self.done)
            if nl < 0:
                break
            self.final_line(self.raw[self.done:nl])
            self.done = nl + 1
        if final:
            if self.done < len(self.raw):
                self.final_line(self.raw[self.done:])
                self.done = len(self.raw)
            if self.table:
                self.draw_table(self.table, self.mark)
                self.table = []
        t.mark_gravity(self.mark, "left")
        if not final:
            tail = self.raw[self.done:]
            rows = list(self.table)
            if tail.lstrip().startswith("|") and not self.in_code:
                rows.append(close_open_marks(tail))   # a table row still coming in
                tail = ""
            if rows:
                self.draw_table(rows, "end-1c")
            if not self.in_code:
                if re.fullmatch(r"\s*[-*_]{1,2}\s*", tail):
                    tail = ""   # the start of a rule (or of bold text): wait for the rest of the line
                tail = close_open_marks(hold_open_math(tail)) if tail.strip() else tail
            if tail:
                self.draw_line(tail, "end-1c")
            elif not self.raw.strip():
                t.insert("end-1c", "…\n", ("p", "dim"))
        for text, tags in footer:
            t.insert("end-1c", text, tags)

    def final_line(self, line):
        if self.table and (self.in_code or not (TABLE_ROW.match(line) or TABLE_SEP.match(line))):
            self.draw_table(self.table, self.mark)
            self.table = []
            self.blank = False
        if not self.in_code and TABLE_ROW.match(line):
            self.table.append(line)
            return
        self.draw_line(line, self.mark, final=True)

    def draw_line(self, line, at, final=False):
        t = self.t
        if FENCE.match(line):
            if final:
                self.in_code = not self.in_code
                t.insert(at, "\n", ("codepad",))
                self.blank = False
            return
        if self.in_code:
            t.insert(at, drop_emoji(line) + "\n", ("code",))
            if final:
                self.blank = False
            return
        if line.strip() == "$$":
            if final:
                self.in_math = not self.in_math
            return
        if self.in_math:
            t.insert(at, drop_emoji(latex_text(line.strip())) + "\n", ("p",))
            if final:
                self.blank = False
            return
        line = display_text(line)
        if not line.strip():
            if not self.blank:
                t.insert(at, "\n", ("gap",))
            if final:
                self.blank = True
            return
        if final:
            self.blank = False
        m = HEADING.match(line)
        if m:
            level = min(len(m.group(1)), 4)
            t.insert(at, plain(m.group(2)) + "\n", (f"h{level}",))
            return
        if RULE.match(line):
            t.insert(at, "\n", ("gap",))
            t.insert(at, "\n", ("hr",))
            t.insert(at, "\n", ("gap",))
            return
        m = BULLET.match(line)
        if m:
            depth = depth_of(m.group(1))
            block = f"li{depth}"
            body = m.group(2)
            box = CHECKBOX.match(body)
            if box:
                mark = "☑" if box.group(1) in "xX" else "☐"
                body = body[box.end():]
            else:
                mark = "•" if depth == 0 else "◦"
            t.insert(at, mark + " ", (block, "bullet"))
            self.inline(body, block, at)
            return
        m = NUMBER.match(line)
        if m:
            depth = depth_of(m.group(1))
            block = f"ol{depth}"
            t.insert(at, m.group(2) + ". ", (block, "bullet"))
            self.inline(m.group(3), block, at)
            return
        m = QUOTE.match(line)
        if m:
            t.insert(at, "▍ ", ("quote", "bullet"))
            self.inline(m.group(1), "quote", at)
            return
        self.inline(line, "p", at)

    def inline(self, text, block, at):
        for seg, style in inline_segments(text):
            if seg:
                self.t.insert(at, seg, (block, style) if style else (block,))
        self.t.insert(at, "\n", (block,))

    def draw_table(self, rows, at):
        """A grid in the monospaced font: columns as wide as their widest cell, narrowed and wrapped inside their
        column when the text column is too narrow."""
        body = [r for r in rows if not TABLE_SEP.match(r)]
        if not body:
            return
        cells = [[inline_segments(display_text(c)) for c in split_cells(r)] for r in body]
        n = max(len(r) for r in cells)
        cells = [r + [[("", "")]] * (n - len(r)) for r in cells]
        natural = [max(3, max(cell_width("".join(s for s, _ in r[i])) for r in cells)) for i in range(n)]
        if sum(min(w, TABLE_MIN_CHARS) for w in natural) + 3 * (n - 1) > self.chat.table_chars():
            self.draw_table_blocks(cells, at)
            return
        room = max(n * 4, self.chat.table_chars() - 3 * (n - 1))
        widths = list(natural)
        while sum(widths) > room and max(widths) > 4:
            widths[widths.index(max(widths))] -= 1
        t = self.t
        style_tag = {"b": "tableb", "bi": "tableb", "i": "tablei", "icode": "tablecode", "link": "tablelink"}
        for k, row in enumerate(cells):
            wrapped = [wrap_segments(cell, widths[i]) for i, cell in enumerate(row)]
            for j in range(max(len(w) for w in wrapped)):
                for i, w in enumerate(wrapped):
                    segs = w[j] if j < len(w) else []
                    used = 0
                    for s, st in segs:
                        tags = ("table", "tableb") if k == 0 else ("table", style_tag[st]) if st else ("table",)
                        t.insert(at, s, tags)
                        used += cell_width(s)
                    pad = widths[i] - used
                    sep = " │ " if i < n - 1 else ""
                    t.insert(at, " " * max(pad, 0) + sep if sep else "", ("table",))
                t.insert(at, "\n", ("table",))
            if k == 0:
                t.insert(at, "─┼─".join("─" * w for w in widths) + "\n", ("table", "dim"))

    def draw_table_blocks(self, cells, at):
        """A table too wide for the text column: one block per row, its first cell in bold, then one
        "header: cell" line per column (no added colon when the header ends in ?, : or .), a thin line between the
        blocks."""
        t = self.t
        header = ["".join(seg for seg, _ in c) for c in cells[0]]
        for k, row in enumerate(cells[1:]):
            if k:
                t.insert(at, "\n", ("gap",))
                t.insert(at, "\n", ("hr",))
                t.insert(at, "\n", ("gap",))
            for seg, style in row[0]:
                t.insert(at, seg, ("p", "b"))
            t.insert(at, "\n", ("p",))
            for i in range(1, len(row)):
                if not any(seg.strip() for seg, _ in row[i]):
                    continue
                if i < len(header) and header[i].strip():
                    label = header[i].rstrip()
                    t.insert(at, label + (" " if label.endswith(("?", ":", ".")) else ": "), ("p", "b"))
                for seg, style in row[i]:
                    t.insert(at, seg, ("p", style) if style else ("p",))
                t.insert(at, "\n", ("p",))


class ChatView:
    """The conversation: one Text widget. Questions are bubbles on the right (embedded labels), answers are text."""

    def __init__(self, app, parent):
        self.app = app
        self.frame = ctk.CTkFrame(parent, corner_radius=0, fg_color=C["bg"])
        self.text = tk.Text(self.frame, wrap="word", borderwidth=0, highlightthickness=0, padx=0, pady=0,
                            undo=False, cursor="arrow", state="disabled", takefocus=0)
        self.scroll = ctk.CTkScrollbar(self.frame, command=self.scrolled)
        self.text.configure(yscrollcommand=self.scroll.set)
        # new text is followed until the reader scrolls away from the end (a question turns it on again)
        self.follow = True
        self.text.bind("<MouseWheel>", self.after_scroll, add="+")
        self.text.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.items = []        # ("q", text, label) / ("a", AnswerView) / ("note", text)
        self.width = 0
        self.redo = None
        self.text.bind("<Configure>", self.on_resize)

    # layout ----------------------------------------------------------------------------------------------------
    def column(self):
        w = max(self.text.winfo_width(), 200)
        col = min(w - 2 * self.app.px(24), self.app.px(COLUMN))
        left = max((w - col) // 2, self.app.px(16))
        return left, max(w - left - col, 0), col

    def table_chars(self):
        return max(24, self.column()[2] // max(self.app.text_fonts["mono"].measure("0"), 1))

    def on_resize(self, event=None):
        if self.text.winfo_width() != self.width:
            self.width = self.text.winfo_width()
            self.layout()
            self.app.fit_box()   # the input box follows the text column from the first layout on
            if self.redo:
                self.app.after_cancel(self.redo)
            self.redo = self.app.after(150, self.rerender)

    def layout(self):
        """Margins, spacing and colors of every tag (after a resize, a zoom or a light / dark switch)."""
        app, t = self.app, self.text
        mode = app.mode
        left, right, col = self.column()
        f = app.text_fonts
        em = f["p"].measure("m")
        dim, text, code = pick(C["dim"], mode), pick(C["text"], mode), pick(C["code"], mode)
        t.configure(bg=pick(C["bg"], mode), fg=text, selectbackground=pick(C["accent"], mode),
                    selectforeground="#FFFFFF", inactiveselectbackground=pick(C["line"], mode),
                    insertbackground=text)
        para = dict(lmargin1=left, lmargin2=left, rmargin=right, spacing1=0, spacing2=app.px(3), spacing3=app.px(4))
        t.tag_configure("p", font=f["p"], foreground=text, **para)
        for level, top in (("h1", 18), ("h2", 14), ("h3", 12), ("h4", 10)):
            t.tag_configure(level, font=f[level], foreground=text, lmargin1=left, lmargin2=left, rmargin=right,
                            spacing1=app.px(top), spacing3=app.px(6))
        bullet_w = f["p"].measure("• ")
        number_w = f["p"].measure("10. ")
        for depth in range(3):
            indent = left + depth * 2 * em
            t.tag_configure(f"li{depth}", font=f["p"], foreground=text, lmargin1=indent,
                            lmargin2=indent + bullet_w, rmargin=right, spacing2=app.px(3), spacing3=app.px(4))
            t.tag_configure(f"ol{depth}", font=f["p"], foreground=text, lmargin1=indent,
                            lmargin2=indent + number_w, rmargin=right, spacing2=app.px(3), spacing3=app.px(4))
        t.tag_configure("quote", font=f["i"], foreground=dim, lmargin1=left, lmargin2=left + f["p"].measure("▍ "),
                        rmargin=right, spacing3=app.px(4))
        pad = app.px(14)
        t.tag_configure("code", font=f["mono"], foreground=text, background=code, lmargin1=left + pad,
                        lmargin2=left + pad, rmargin=right, lmargincolor=code, rmargincolor=pick(C["bg"], mode))
        t.tag_configure("codepad", font=f["gap"], background=code, lmargin1=left, lmargin2=left, rmargin=right)
        t.tag_configure("table", font=f["mono"], foreground=text, lmargin1=left, lmargin2=left, rmargin=right,
                        spacing2=app.px(2), spacing3=app.px(3), wrap="none")
        t.tag_configure("gap", font=f["gap"])
        bg = pick(C["bg"], mode)
        t.tag_configure("hr", font=f["hr"], background=pick(C["line"], mode), lmargin1=left, lmargin2=left,
                        rmargin=right, lmargincolor=bg, rmargincolor=bg, spacing1=0, spacing2=0, spacing3=0)
        t.tag_configure("stats", font=f["stats"], foreground=text, lmargin1=left, lmargin2=left, rmargin=right,
                        spacing1=app.px(8), spacing3=app.px(22))
        t.tag_configure("note", font=f["p"], foreground=pick(C["warn_text"], mode), lmargin1=left, lmargin2=left,
                        rmargin=right, spacing1=app.px(6), spacing3=app.px(14))
        t.tag_configure("qline", justify="right", lmargin1=left, lmargin2=left, rmargin=right,
                        spacing1=app.px(10), spacing3=app.px(14))
        # inline tags come last: they win over the block tags for the font
        t.tag_configure("b", font=f["b"])
        t.tag_configure("i", font=f["i"])
        t.tag_configure("bi", font=f["bi"])
        t.tag_configure("icode", font=f["mono"], background=code)
        t.tag_configure("link", foreground=pick(C["accent"], mode), underline=True)
        t.tag_configure("bullet", foreground=dim)
        t.tag_configure("dim", foreground=dim)
        t.tag_configure("statsb", font=f["statsb"])
        t.tag_configure("tableb", font=f["monob"])
        t.tag_configure("tablei", font=f["monoi"])
        t.tag_configure("tablecode", background=code)
        t.tag_configure("tablelink", foreground=pick(C["accent"], mode))
        for tag in ("b", "i", "bi", "icode", "link", "bullet", "dim", "statsb", "tableb", "tablei", "tablecode",
                    "tablelink"):
            t.tag_raise(tag)
        for item in self.items:
            if item[0] == "q":
                item[2].configure(wraplength=int(col * 0.72 / app.widget_scaling()))

    # content ---------------------------------------------------------------------------------------------------
    def edit(self):
        self.text.configure(state="normal")

    def done_editing(self):
        self.text.configure(state="disabled")

    def at_bottom(self):
        return self.text.yview()[1] >= 0.999

    def scrolled(self, *args):
        self.text.yview(*args)
        self.after_scroll()

    def after_scroll(self, event=None):
        self.text.after_idle(self.update_follow)

    def update_follow(self):
        """After the reader scrolled: follow new text only when the view is at the end."""
        try:
            self.text.tk.call(self.text._w, "sync")   # line heights up to date before the view is read
        except tk.TclError:
            pass
        self.follow = self.at_bottom()

    def bubble(self, text):
        app = self.app
        col = self.column()[2]
        label = ctk.CTkLabel(self.text, text=display_text(text), font=app.ui_fonts["bubble"], justify="left",
                             anchor="w", fg_color=C["bubble"], text_color=C["text"], bg_color=C["bg"],
                             corner_radius=16, wraplength=int(col * 0.72 / app.widget_scaling()))
        label.configure(padx=14, pady=10)
        label.bind("<MouseWheel>", lambda e: (self.text.yview_scroll(int(-e.delta / 120), "units"), self.after_scroll()))
        start = self.text.index("end-1c")
        self.text.window_create("end-1c", window=label)
        self.text.insert("end-1c", "\n", ("qline",))
        self.text.tag_add("qline", start, "end-1c")
        return label

    def add_question(self, text):
        self.edit()
        self.items.append(("q", text, self.bubble(text)))
        self.done_editing()
        self.follow = True
        self.text.see("end")

    def add_answer(self):
        self.edit()
        view = AnswerView(self)
        self.items.append(("a", view))
        return view

    def note(self, text):
        self.items.append(("note", text))
        self.edit()
        self.text.insert("end-1c", text + "\n", ("note",))
        self.done_editing()
        self.text.see("end")

    def clear(self):
        self.follow = True
        self.edit()
        self.text.delete("1.0", "end")
        self.done_editing()
        for item in self.items:
            if item[0] == "q":
                item[2].destroy()
        self.items = []

    def rerender(self):
        """Draw the whole chat again (tables are wrapped to the new width)."""
        self.redo = None
        if not self.items:
            return
        follow = self.follow
        top = self.text.yview()[0]
        items = self.items
        self.edit()
        self.text.delete("1.0", "end")
        for item in items:
            if item[0] == "q":
                item[2].destroy()
        self.items = []
        for item in items:
            if item[0] == "q":
                self.items.append(("q", item[1], self.bubble(item[1])))
            elif item[0] == "a":
                view = item[1]
                view.reset()
                view.done = 0
                view.draw(view.footer, final=view.finished)
                self.items.append(item)
            else:
                self.text.insert("end-1c", item[1] + "\n", ("note",))
                self.items.append(item)
        self.done_editing()
        if follow:
            self.text.see("end")
        else:
            self.text.yview_moveto(top)


def windows_build():
    """The Windows build number as Windows itself reports it (kernel32.dll's version); None off Windows."""
    if sys.platform != "win32":
        return None
    v = sys.getwindowsversion()
    return getattr(v, "platform_version", (v.major, v.minor, v.build))[2]


def os_name(build=None):
    """The header's "Windows 11" or "Windows 10", from the build number (22000 and later = Windows 11). Not
    platform.release(): some Python versions report 10 on Windows 11. Empty off Windows."""
    if build is None:
        build = windows_build()
    if build is None:
        return ""
    return "Windows 11" if build >= 22000 else "Windows 10"


def rate_of(tokens, t_first, t_last):
    """tokens/s of an answer: (tokens - 1) / (t_last - t_first), or None when that span is under 0.2 s."""
    if tokens < 2 or t_first is None or t_last is None or t_last - t_first < 0.2:
        return None
    return (tokens - 1) / (t_last - t_first)


# --- the model in a child process ----------------------------------------------------------------------------------
def kill_children_with_this_process():
    """Put this process in a Job Object that ends every process in it when this process ends (crash included): the
    model process started later is in the job too, so it never outlives the window."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return f"CreateJobObject failed ({ctypes.get_last_error()})"
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        return f"SetInformationJobObject failed ({ctypes.get_last_error()})"
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        return f"AssignProcessToJobObject failed ({ctypes.get_last_error()})"
    globals()["_JOB"] = job   # held until this process ends
    return "on"


class ChildEngine:
    """engine.py --serve in a child process; JSON lines over its stdin / stdout. Events go to the window's queue."""

    def __init__(self, args, events):
        self.events = events
        exe = sys.executable
        if os.path.basename(exe).lower() == "pythonw.exe":
            exe = os.path.join(os.path.dirname(exe), "python.exe")   # the child needs real stdin / stdout pipes
        cmd = [exe, os.path.join(HERE, "engine.py"), "--serve", "--device", args.device, "--context",
               str(args.context), "--parent-pid", str(os.getpid())]
        if args.model_dir:
            cmd += ["--model-dir", args.model_dir]
        if args.device == "webgpu":
            cmd += ["--adapter", str(args.adapter)]
        if args.auto_picked:
            cmd += ["--auto-picked"]
        if args.device == "fake":
            cmd += ["--fake-load-s", str(args.fake_load_s), "--fake-rate", str(args.fake_rate),
                    "--fake-first-token-s", str(args.fake_first_token_s)]
            if args.fake_fail:
                cmd += ["--fake-fail", args.fake_fail]
            if args.fake_text:
                cmd += ["--fake-text", args.fake_text]
        flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", ORT_DISABLE_TELEMETRY="1")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  cwd=HERE, env=env, creationflags=flags)
        self.stderr_tail = []
        self.to_child = queue.Queue()   # written by a thread of its own: the window never waits on the pipe
        self.write_ok = True
        threading.Thread(target=self.read_out, daemon=True).start()
        threading.Thread(target=self.read_err, daemon=True).start()
        self.writer = threading.Thread(target=self.write_in, daemon=True)
        self.writer.start()

    def read_out(self):
        for raw in self.p.stdout:
            try:
                self.events.put(json.loads(raw.decode("utf-8")))
            except ValueError:
                continue
        code = self.p.wait()
        time.sleep(0.2)   # let the stderr reader catch the last lines
        self.events.put({"ev": "engine_exit", "code": code, "stderr": self.stderr_tail[-6:], "child": self})

    def read_err(self):
        for raw in self.p.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self.stderr_tail = (self.stderr_tail + [line])[-40:]

    def write_in(self):
        while True:
            obj = self.to_child.get()
            try:
                if obj is None:
                    self.p.stdin.close()
                    return
                self.p.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                self.p.stdin.flush()
            except (OSError, ValueError):
                self.write_ok = False
                return

    def send(self, obj):
        """Queue one line for the child; False when the child or its pipe is gone."""
        if not self.write_ok or self.p.poll() is not None:
            return False
        self.to_child.put(obj)
        return True

    def kill(self):
        if self.p.poll() is None:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(self.p.pid), "/T", "/F"], capture_output=True,
                               creationflags=0x08000000)
            else:
                self.p.kill()

    def close(self, wait=4.0):
        self.send({"op": "quit"})
        self.to_child.put(None)   # then the writer closes the child's stdin
        try:
            self.p.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            self.kill()


# --- the window -----------------------------------------------------------------------------------------------------
class Pill(ctk.CTkFrame):
    """A rounded button made of a frame and a label: its text wraps, it shows a pressed color, and it runs its command
    when the press is released over it (a CTkButton also needs a pointer-enter event before the release)."""

    def __init__(self, master, text, font, command, colors, radius=20, padx=20, pady=8, hold_ms=150):
        super().__init__(master, corner_radius=radius, fg_color=colors["bg"], border_width=1,
                         border_color=colors["border"], cursor="hand2")
        self.text, self.command, self.colors, self.hold_ms = text, command, colors, hold_ms
        self.enabled, self.pressed, self.hover, self.down = True, False, False, False
        self.label = ctk.CTkLabel(self, text=text, font=font, text_color=colors["text"], fg_color="transparent",
                                  justify="center", cursor="hand2")
        self.label.pack(padx=padx, pady=pady)
        for w in (self, self.label):
            w.bind("<Enter>", lambda e: self.set_hover(True))
            w.bind("<Leave>", lambda e: self.set_hover(False))
            w.bind("<ButtonPress-1>", self.on_press)
            w.bind("<ButtonRelease-1>", self.on_release)

    def paint(self):
        c = self.colors
        if self.pressed or self.down:
            bg, fg = c["pressed"], c["pressed_text"]
        else:
            bg = c["hover"] if self.hover and self.enabled else c["bg"]
            fg = c["text"] if self.enabled else c["dim"]
        self.configure(fg_color=bg)
        self.label.configure(text_color=fg)

    def set_hover(self, on):
        self.hover = on
        self.paint()

    def set_enabled(self, on):
        self.enabled = on
        self.paint()

    def on_press(self, event):
        if self.enabled:
            self.down = True
            self.paint()

    def on_release(self, event):
        if not self.down:
            return
        self.down = False
        x0, y0 = self.winfo_rootx(), self.winfo_rooty()
        if self.enabled and x0 <= event.x_root < x0 + self.winfo_width() and y0 <= event.y_root < y0 + self.winfo_height():
            self.pressed = True
            self.paint()
            self.after(self.hold_ms, self.unpress)
            self.command(self)
        else:
            self.paint()

    def unpress(self):
        self.pressed = False
        self.paint()


CHIP_COLORS = {"bg": C["chip"], "hover": C["chip_hover"], "text": C["text"], "dim": C["dim"], "border": C["line"],
               "pressed": C["accent"], "pressed_text": C["on_accent"]}
NEW_CHAT_COLORS = dict(CHIP_COLORS, bg="transparent")


class ChatApp(ctk.CTk):
    def __init__(self, args):
        ctk.set_appearance_mode("system")
        super().__init__()
        self.args = args
        self.label = engine.device_label(args.device, args.adapter)
        self.header_device = " · ".join(part for part in (os_name(), self.label) if part)   # Windows 11 · the device
        self.trace_file = open(args.trace, "a", encoding="utf-8") if args.trace else None   # test runs only
        self.log_failed = False
        self.events = queue.Queue()
        self.mode = ctk.get_appearance_mode()
        self.zoom_i = ZOOMS.index(1.0)
        self.state_ = "loading"      # loading | ready | failed
        self.turn = None             # the answer being written
        self.turn_n = 0
        self.sent_at = 0.0           # perf_counter of the last Send: the button's Stop waits STOP_GUARD_S after it
        self.net_last = None
        self.closing = False
        self.runner = None
        self.chat_shown = False
        self.header_two_rows = None
        self.chip_wrap = None
        self.last_layout = None
        self.poll_n = 0
        self.job = kill_children_with_this_process()

        self.title(TITLES["loading"])
        self.set_icon()
        self.ui_fonts = {k: ctk.CTkFont(family="Segoe UI", size=s, weight=w) for k, (s, w) in UI_FONTS.items()}
        families = set(tkfont.families(self))
        self.mono_family = next((f for f in ("Cascadia Mono", "Consolas", "Courier New") if f in families), "Courier")
        self.text_fonts = {k: tkfont.Font(self, family=self.mono_family if k.startswith("mono") else "Segoe UI",
                                          size=-1, weight=w, slant=s) for k, (_, w, s) in TEXT_FONTS.items()}
        self.size_text_fonts()
        self.build()
        self.place_window()
        ctk.AppearanceModeTracker.add(self.on_mode, self)
        ctk.ScalingTracker.add_widget(self.on_scaling, self)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.trace("start", device=args.device, label=self.label, header_device=self.header_device,
                   windows_build=windows_build(), args=vars(args), python=sys.version.split()[0],
                   power_throttling=prelude.POWER_THROTTLING, telemetry=os.environ.get("ORT_DISABLE_TELEMETRY"),
                   job=self.job, pid=os.getpid())
        self.after(30, self.poll)
        self.after(150, self.start_engine)
        self.after(300, self.refresh_net)

    # helpers ---------------------------------------------------------------------------------------------------
    def widget_scaling(self):
        try:
            return ctk.ScalingTracker.get_widget_scaling(self)
        except Exception:
            return 1.0

    def px(self, n):
        """n pixels at 100 % in this screen's pixels (Windows display scaling and the zoom)."""
        return max(1, int(round(n * self.widget_scaling() * ZOOMS[self.zoom_i])))

    def trace(self, kind, **kw):
        """Events for test runs (--trace, not shown in the help)."""
        if self.trace_file:
            row = {"t": time.time(), "perf": time.perf_counter(), "ev": kind}
            row.update(kw)
            self.trace_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self.trace_file.flush()

    def report_callback_exception(self, exc, val, tb):
        """An unexpected error in a window callback: to --trace and --log (pythonw.exe has no console to show it)."""
        text = "".join(traceback.format_exception(exc, val, tb))
        try:
            self.trace("callback_error", error=text[-3000:])
            self.record({"time": datetime.now().astimezone().isoformat(timespec="seconds"),
                         "error": f"{exc.__name__}: {val}", "traceback": text[-3000:]})
        finally:
            if sys.stderr:
                sys.stderr.write(text)

    def record(self, row):
        """One line in the --log file for an answer that ended (opened and closed each time)."""
        if not self.args.log or self.log_failed:
            return
        try:
            with open(self.args.log, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as error:
            self.log_failed = True
            self.chat.note(f"Answers are no longer written to {self.args.log}: {error.strerror or error}.")

    def set_icon(self):
        """A plain chat-bubble icon drawn here (no icon file)."""
        try:
            img = tk.PhotoImage(master=self, width=32, height=32)
            blue, white = "#1A73E8", "#FFFFFF"
            rows = []
            for y in range(32):
                row = []
                for x in range(32):
                    inside = (x - 15.5) ** 2 / 15.5 ** 2 + (y - 15.5) ** 2 / 15.5 ** 2 <= 1
                    bubble = 8 <= x <= 23 and 9 <= y <= 19 or (11 <= x <= 14 and 20 <= y <= 23 - (x - 11))
                    row.append(white if inside and bubble else blue if inside else "{}")
                rows.append("{" + " ".join(row) + "}")
            img.put(" ".join(rows))
            self.iconbitmap()   # keep customtkinter from setting its own icon
            self.iconphoto(True, img)
            self._icon = img
        except tk.TclError:
            pass

    def place_window(self):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        s = self.widget_scaling()
        w, h = min(1180, int(sw / s * 0.8)), min(860, int(sh / s * 0.85))
        x, y = int((sw - w * s) / 2), int((sh - h * s) / 2.4)
        self.geometry(f"{w}x{h}+{max(x, 0)}+{max(y, 0)}")
        self.minsize(560, 460)
        if self.args.maximized:
            self.after(50, lambda: self.state("zoomed"))

    def size_text_fonts(self):
        z, s = ZOOMS[self.zoom_i], self.widget_scaling()
        for k, (size, _, _) in TEXT_FONTS.items():
            self.text_fonts[k].configure(size=-max(1, int(round(size * z * s))))

    def set_title(self):
        key = "failed" if self.state_ == "failed" else "loading" if self.state_ == "loading" else \
            "answering" if self.turn is not None else "ready"
        if self.title() != TITLES[key]:
            self.title(TITLES[key])
            self.trace("title", title=TITLES[key])

    # building --------------------------------------------------------------------------------------------------
    def build(self):
        self.configure(fg_color=C["bg"])
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        F = self.ui_fonts

        head = ctk.CTkFrame(self, corner_radius=0, fg_color=C["panel"])
        head.grid(row=0, column=0, sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        self.head = head
        head.bind("<Configure>", lambda event: self.arrange_header())   # its width is known only after it is laid out
        names = ctk.CTkFrame(head, fg_color="transparent")
        names.grid(row=0, column=0, sticky="w", padx=(20, 8), pady=(10, 10))
        self.names = names
        ctk.CTkLabel(names, text=APP_TITLE, font=F["title"], text_color=C["text"]).pack(anchor="w")
        self.device_line = ctk.CTkLabel(names, text=self.header_device, font=F["device"], text_color=C["dim"])
        self.device_line.pack(anchor="w")
        self.badge = ctk.CTkLabel(head, text="  On this PC · 0 network connections  ", font=F["badge"],
                                  fg_color=C["ok_bg"], text_color=C["ok_text"], corner_radius=12, height=28)
        self.badge.grid(row=0, column=1, padx=8)
        self.new_btn = Pill(head, "New chat", F["button"], lambda pill: self.new_chat(), NEW_CHAT_COLORS, radius=16,
                            padx=16, pady=3)
        self.new_btn.grid(row=0, column=2, padx=(4, 20))
        ctk.CTkFrame(self, height=1, corner_radius=0, fg_color=C["line"]).grid(row=1, column=0, sticky="ew")

        body = ctk.CTkFrame(self, corner_radius=0, fg_color=C["bg"])
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self.welcome = ctk.CTkFrame(body, corner_radius=0, fg_color=C["bg"])
        self.welcome.grid(row=0, column=0, sticky="nsew")
        inner = ctk.CTkFrame(self.welcome, fg_color="transparent")
        inner.place(relx=0.5, rely=0.42, anchor="center")
        self.guide = ctk.CTkLabel(inner, text=GUIDE, font=F["guide"], text_color=C["text"], justify="center")
        self.guide.pack(pady=(0, 8))
        self.status = ctk.CTkLabel(inner, text=f"Loading the model on {self.label}…", font=F["status"],
                                   text_color=C["dim"], wraplength=640, justify="center")
        self.status.pack(pady=(0, 22))
        self.chips_frame = ctk.CTkFrame(inner, fg_color="transparent")
        self.chips_frame.pack()
        self.chips = [Pill(self.chips_frame, s, F["chip"], self.suggest, CHIP_COLORS, hold_ms=PRESSED_MS + 50)
                      for s in SUGGESTIONS]
        for chip in self.chips:
            chip.pack(side="top", pady=6)
        self.arrange_chips()
        self.chat = ChatView(self, body)
        self.chat.frame.grid(row=0, column=0, sticky="nsew")
        self.show_welcome()

        foot = ctk.CTkFrame(self, corner_radius=0, fg_color=C["bg"])
        foot.grid(row=3, column=0, sticky="ew")
        foot.grid_columnconfigure(0, weight=1)
        self.foot = foot
        self.bar_pad = (40, 40)
        # the bar that tells why the model is not there, with the way to load it again
        self.bar = ctk.CTkFrame(foot, corner_radius=12, fg_color=C["warn_bg"])
        self.bar.grid_columnconfigure(0, weight=1)
        self.bar_text = ctk.CTkLabel(self.bar, text="", font=F["bar"], text_color=C["warn_text"], justify="left",
                                     anchor="w", wraplength=760)
        self.bar_text.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=8)
        self.reload_btn = ctk.CTkButton(self.bar, text="Load the model again", font=F["button"], height=32,
                                        corner_radius=16, fg_color=C["accent"], hover_color=C["accent_hover"],
                                        command=self.reload)
        self.reload_btn.grid(row=0, column=1, padx=(0, 10), pady=8)
        self.box = ctk.CTkFrame(foot, corner_radius=20, fg_color=C["input"], border_width=1, border_color=C["line"])
        self.box.grid(row=1, column=0, sticky="ew", padx=40, pady=(10, 4))
        self.box.grid_columnconfigure(0, weight=1)
        self.input = ctk.CTkTextbox(self.box, height=self.input_height(), wrap="word", font=F["input"],
                                    fg_color=C["input"], text_color=C["text"], border_width=0, corner_radius=0,
                                    activate_scrollbars=True)
        self.input.grid(row=0, column=0, sticky="ew", padx=(16, 6), pady=8)
        self.send_btn = ctk.CTkButton(self.box, text="Send", font=F["button"], width=84, height=36, corner_radius=18,
                                      fg_color=C["accent"], hover_color=C["accent_hover"], command=self.send_or_stop)
        self.send_btn.grid(row=0, column=1, padx=(0, 10), pady=8, sticky="s")
        self.placeholder = ctk.CTkLabel(self.box, text=PLACEHOLDER, font=F["input"], text_color=C["dim"],
                                        fg_color=C["input"])
        self.placeholder.place(x=22, y=14)
        self.placeholder.bind("<Button-1>", lambda e: self.input.focus_set())
        self.hint = ctk.CTkLabel(foot, text=HINT, font=F["hint"], text_color=C["dim"])
        self.hint.grid(row=2, column=0, pady=(0, 10))

        self.input.bind("<Return>", self.on_enter)
        self.input.bind("<Shift-Return>", self.on_shift_enter)
        self.input.bind("<KeyRelease>", lambda e: self.update_placeholder())
        self.input.bind("<FocusIn>", lambda e: self.update_placeholder())
        for seq in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>", "<Control-semicolon>"):
            self.bind_all(seq, lambda e: self.zoom(+1))
        for seq in ("<Control-minus>", "<Control-KP_Subtract>"):
            self.bind_all(seq, lambda e: self.zoom(-1))
        for seq in ("<Control-0>", "<Control-KP_0>", "<Control-KP_Insert>"):
            self.bind_all(seq, lambda e: self.zoom(0))
        self.bind_all("<Control-MouseWheel>", lambda e: self.zoom(+1 if e.delta > 0 else -1))
        self.bind_all("<Escape>", lambda e: self.stop())
        self.bind("<Configure>", self.on_window_resize, add="+")
        self.after(100, self.input.focus_set)

    def input_height(self):
        return int(UI_FONTS["input"][0] * ZOOMS[self.zoom_i] * 1.5 * 3 + 8)

    def arrange_chips(self):
        """The suggestions one under the other, centered; a long one wraps inside its pill. The guide and the status
        line wrap to the window too."""
        width = self.winfo_width() / self.widget_scaling() if self.winfo_width() > 1 else 800
        wrap = int(max(200, min(width - 140, 760)))
        if wrap != self.chip_wrap:
            self.chip_wrap = wrap
            for chip in self.chips:
                chip.label.configure(wraplength=wrap)
            self.guide.configure(wraplength=wrap + 60)
            self.status.configure(wraplength=wrap + 60)

    def arrange_header(self):
        """Title and device on the left, the badge and New chat on the right; on a narrow window the badge and New chat
        go to a second row."""
        width = self.head.winfo_width()
        if width <= 1:
            return
        need = (self.names.winfo_reqwidth() + self.badge.winfo_reqwidth() + self.new_btn.winfo_reqwidth() +
                self.px(60))
        two = need > width
        if two == self.header_two_rows:
            return
        self.header_two_rows = two
        if two:
            self.names.grid(row=0, column=0, columnspan=3, sticky="w", padx=(20, 8), pady=(10, 4))
            self.badge.grid(row=1, column=0, sticky="w", padx=(20, 8), pady=(0, 10))
            self.new_btn.grid(row=1, column=2, sticky="e", padx=(4, 20), pady=(0, 10))
        else:
            self.names.grid(row=0, column=0, columnspan=1, sticky="w", padx=(20, 8), pady=(10, 10))
            self.badge.grid(row=0, column=1, sticky="", padx=8, pady=0)
            self.new_btn.grid(row=0, column=2, sticky="", padx=(4, 20), pady=0)

    def on_window_resize(self, event):
        if event.widget is self:
            self.arrange_chips()
            self.arrange_header()
            self.fit_box()

    def fit_box(self):
        """The input box under the text column."""
        if not self.chat.width or not hasattr(self, "box"):
            return
        left, right, col = self.chat.column()
        s = self.widget_scaling()
        pad = max(int(left / s), 16)
        self.bar_pad = (pad, max(int(right / s), 16) + 12)
        # customtkinter scales the arguments of .grid(), not of .grid_configure()
        self.box.grid(row=1, column=0, sticky="ew", padx=self.bar_pad, pady=(10, 4))
        if self.bar.winfo_manager():   # only while it is shown: .grid() on the hidden bar would show it
            self.bar.grid(row=0, column=0, sticky="ew", padx=self.bar_pad, pady=(10, 0))
        self.bar_text.configure(wraplength=max(int(col / s) - 220, 200))

    # light / dark and display scaling --------------------------------------------------------------------------
    def on_mode(self, mode):
        self.mode = mode
        self.chat.layout()

    def on_scaling(self, widget_scaling, window_scaling):
        self.size_text_fonts()
        self.chat.layout()
        self.chat.rerender()

    # zoom (Ctrl + / Ctrl - / Ctrl 0) ---------------------------------------------------------------------------
    def zoom(self, step):
        before = self.zoom_i
        if step == 0:
            self.zoom_i = ZOOMS.index(1.0)
        else:
            self.zoom_i = min(max(self.zoom_i + step, 0), len(ZOOMS) - 1)
        if self.zoom_i == before:
            return "break"
        z = ZOOMS[self.zoom_i]
        for k, (size, _) in UI_FONTS.items():
            self.ui_fonts[k].configure(size=max(6, int(round(size * z))))
        self.size_text_fonts()
        self.input.configure(height=self.input_height())
        self.chat.layout()
        self.chat.rerender()
        self.chip_wrap = None
        self.arrange_chips()
        self.header_two_rows = None
        self.after(50, self.arrange_header)
        self.fit_box()
        self.hint.configure(text=f"Text size {int(round(z * 100))} %")
        self.after(1600, lambda: self.hint.configure(text=HINT))
        self.trace("zoom", zoom=z)
        return "break"

    # the input box ---------------------------------------------------------------------------------------------
    def update_placeholder(self):
        """Show "Ask anything" while the box is empty; placed or removed only when that changes."""
        shown = self.placeholder.winfo_manager() == "place"
        if self.input.get("1.0", "end-1c"):
            if shown:
                self.placeholder.place_forget()
        elif not shown:
            self.placeholder.place(x=22, y=14)

    def on_enter(self, event):
        self.send()
        return "break"

    def on_shift_enter(self, event):
        self.input.insert("insert", "\n")
        self.update_placeholder()
        return "break"

    def suggest(self, chip):
        """A suggestion is a shortcut for typing it and pressing Send: the question goes to the model at once, without
        passing through the input box (so the box keeps "Ask anything"); the pressed pill stays in view for
        PRESSED_MS, then the chat shows."""
        self.send(reveal_ms=PRESSED_MS, text=chip.text)

    def show_chat(self):
        self.chat.frame.tkraise()
        self.chat_shown = True

    def show_welcome(self):
        self.welcome.tkraise()
        self.chat_shown = False

    def send_or_stop(self):
        if self.turn is not None:
            if time.perf_counter() - self.sent_at >= STOP_GUARD_S:   # not the second click of a double-click on Send
                self.stop()
        else:
            self.send()

    def set_busy(self, busy):
        failed = self.state_ == "failed"
        if busy:
            self.send_btn.configure(text="Stop", fg_color=C["stop"], hover_color=C["stop_hover"], state="normal")
        else:
            self.send_btn.configure(text="Send", fg_color=C["accent"], hover_color=C["accent_hover"],
                                    state="disabled" if failed else "normal")
        for chip in self.chips:
            chip.set_enabled(not (busy or failed))
        self.set_title()

    # asking ----------------------------------------------------------------------------------------------------
    def send(self, reveal_ms=0, text=None):
        text = (self.input.get("1.0", "end-1c") if text is None else text).strip()
        if not text or self.turn is not None or self.state_ == "failed":
            return
        t_press = time.perf_counter()
        self.sent_at = t_press
        self.input.delete("1.0", "end")
        self.update_placeholder()
        if reveal_ms:
            self.after(reveal_ms, self.show_chat)
        else:
            self.show_chat()
        self.chat.add_question(text)
        self.turn_n += 1
        view = self.chat.add_answer()
        self.turn = {"id": self.turn_n, "view": view, "t_press": t_press, "app_first": None, "tokens": 0,
                     "t_first": None, "t_last": None, "stopping": False, "dirty": True, "question": text, "events": 0}
        view.draw(self.footer())
        self.chat.done_editing()
        self.chat.text.see("end")
        self.set_busy(True)
        self.trace("ask", id=self.turn_n, chars=len(text), state=self.state_, question=text[:200])
        if self.runner is None or not self.runner.send({"op": "ask", "id": self.turn_n, "text": text}):
            self.finish({"finish": "error"}, note="The model process is not running. Load the model again.")

    def stop(self):
        if self.turn is None or self.turn["stopping"]:
            return
        self.turn["stopping"] = True
        self.send_btn.configure(text="Stopping…")
        self.trace("stop", id=self.turn["id"])
        if self.runner is not None:
            self.runner.send({"op": "stop", "id": self.turn["id"]})
        self.after(STOP_TIMEOUT_MS, lambda tid=self.turn["id"], seen=self.turn["events"]: self.stop_check(tid, seen))

    def stop_check(self, tid, seen):
        """STOP_TIMEOUT_MS after Stop: no token and no end since then = the model process is stuck; it is ended."""
        tr = self.turn
        if tr is None or tr["id"] != tid or self.closing:
            return
        if tr["events"] != seen:   # still answering: look again later
            now_seen = tr["events"]
            self.after(STOP_TIMEOUT_MS, lambda: self.stop_check(tid, now_seen))
            return
        self.trace("stop_timeout", id=tid)
        if self.runner is not None:
            self.runner.kill()
        self.fail(f"The model process did not stop within {STOP_TIMEOUT_MS // 1000} s and was ended.")

    def new_chat(self):
        if self.turn is not None:
            self.stop()
            self.after(100, self.new_chat)   # once the answer has stopped
            return
        if self.runner:
            self.runner.send({"op": "new_chat"})
        self.chat.clear()
        self.show_welcome()
        self.input.focus_set()
        self.trace("new_chat")

    def numbers(self):
        """The line under an answer, from the model process's clock, as (count and seconds, rate): seconds = from the
        moment the question's line reached the model process to the answer's last token; tokens/s = (the answer's
        tokens - 1) / the seconds from its first token to its last (shown once that is 0.2 s or more)."""
        tr = self.turn
        n = tr["tokens"]
        if n == 0 or tr["t_last"] is None:
            return "", ""
        rate = rate_of(n, tr["t_first"], tr["t_last"])
        return (f"{n} token{'s' if n != 1 else ''} in {tr['t_last']:.1f} s",
                f"{rate:.0f} tokens/s" if rate is not None else "")

    def stats_parts(self, final=False):
        """[(text, bold)] of the line under an answer as the window shows it ([] while there is none)."""
        head, rate = self.numbers()
        finish = self.turn.get("finish") if final else None
        if not head:
            return [("Stopped before the answer began.", False)] if finish == "stopped" else []
        parts = [(head, False)]
        if rate:
            parts += [(" · ", False), (rate, True)]
        if finish == "stopped":
            parts.append((" · stopped", False))
        elif finish == "full":
            parts.append((" · chat is full", False))
        return parts

    def stats_line(self, final=False):
        return "".join(text for text, _ in self.stats_parts(final))

    def footer(self, final=False, note=None):
        parts = []
        stats = self.stats_parts(final)
        for text, bold in stats:
            parts.append((text, ("stats", "statsb") if bold else ("stats",)))
        if stats:
            parts.append(("\n", ("stats",)))
        if note:
            parts.append((note + "\n", ("note",)))
        if final and not parts:
            parts.append(("\n", ("gap",)))
        return parts

    def finish(self, stats, note=None):
        tr = self.turn
        tr["finish"] = stats.get("finish")
        for key in ("tokens", "t_first", "t_last"):
            if stats.get(key) is not None:
                tr[key] = stats[key]
        if tr["finish"] == "full" and not note:
            note = "This chat is full. Start a new chat."
        if stats.get("chat_reset") and not note:
            note = "Stopped while the question was going in: your next question starts a new chat."
        follow = self.chat.follow
        self.chat.edit()
        tr["view"].draw(self.footer(final=True, note=note), final=True)
        self.chat.done_editing()
        if follow:
            self.chat.text.see("end")
        shown = self.stats_line(final=True)
        first_s = (tr["app_first"] - tr["t_press"]) if tr["app_first"] else None
        rate = rate_of(tr["tokens"], tr["t_first"], tr["t_last"])
        start = tr.get("start") or {}
        seconds = stats.get("seconds", tr["t_last"])
        self.record({
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "question": tr["question"],
            "answer": tr["view"].raw,
            "shown": shown,                    # the line under the answer, as drawn
            "note": note or "",                # the red line under it, if any
            "tokens": tr["tokens"],
            "seconds": round(seconds, 3) if seconds is not None else None,   # question received -> last token
            "tokens_per_s": round(rate, 2) if rate is not None else None,   # as on the screen
            "first_token_ms": round(first_s * 1000) if first_s is not None else None,   # press -> on the screen
            "finish": tr["finish"],
            "chat_tokens": stats.get("context_used"),
            "chat_max_tokens": stats.get("max_length", start.get("max_length")),
            "chat_turn": start.get("turn"),
            "device": self.args.device,
            "device_name": self.label,
            "chat_seed": start.get("chat_seed"),   # one seed per chat (drawn at its first question)
            "badge": self.badge.cget("text").strip(),
        })
        self.trace("done", id=tr["id"], finish=tr["finish"], tokens=tr["tokens"], press_to_first_token_s=first_s,
                   tokens_per_s=rate, shown=shown, note=note, stats=stats)
        self.turn = None
        self.set_busy(False)

    def trace_layout(self):
        """Where the buttons and the input box are on the screen, for test drivers (--trace only)."""
        def box(w):
            try:
                x, y = w.winfo_rootx(), w.winfo_rooty()
                return [x, y, x + w.winfo_width(), y + w.winfo_height()] if w.winfo_ismapped() else None
            except tk.TclError:
                return None

        rows = {"view": "chat" if self.chat_shown else "welcome", "chips": [box(c) for c in self.chips],
                "new_chat": box(self.new_btn), "input": box(self.input), "send": box(self.send_btn),
                "zoom": ZOOMS[self.zoom_i], "header_rows": 2 if self.header_two_rows else 1}
        body, stats = self.text_fonts["p"], self.text_fonts["stats"]
        rows["body_font"] = {"px": abs(int(body.cget("size"))), "ascent": body.metrics("ascent"),
                             "descent": body.metrics("descent"), "linespace": body.metrics("linespace")}
        rows["stats_px"] = abs(int(stats.cget("size")))
        if rows != self.last_layout:
            self.last_layout = rows
            self.trace("layout", **rows)

    # events from the model -------------------------------------------------------------------------------------
    def start_engine(self):
        self.state_ = "loading"
        self.set_title()
        try:
            self.runner = ChildEngine(self.args, self.events)
        except Exception as error:   # noqa: BLE001 - one line in the window
            self.runner = None
            self.fail(f"The model process could not start ({type(error).__name__}: {str(error)[:160]}). Run chat.cmd "
                      f"again to repair the setup.")
            return
        self.trace("engine_started", pid=self.runner.p.pid)

    def reload(self):
        if self.state_ != "failed":
            return
        old = self.runner
        if old is not None and old.p.poll() is None:
            old.close(wait=2.0)
        self.bar.grid_forget()
        self.status.configure(text=f"Loading the model on {self.label}…", text_color=C["dim"])
        self.status.pack(pady=(0, 22), before=self.chips_frame)
        if self.chat.items:
            self.chat.note("The model was loaded again: the next question starts a new chat.")
        self.trace("reload")
        self.start_engine()
        self.set_busy(False)

    def poll(self):
        if self.closing:
            return
        try:
            try:
                while True:
                    self.on_event(self.events.get_nowait())
            except queue.Empty:
                pass
            self.poll_n += 1
            if self.trace_file and self.poll_n % 8 == 0:
                self.trace_layout()
            tr = self.turn
            if tr is not None and tr["dirty"]:
                tr["dirty"] = False
                follow = self.chat.follow
                self.chat.edit()
                tr["view"].draw(self.footer())
                self.chat.done_editing()
                if follow:
                    self.chat.text.see("end")
        finally:   # an error is reported (report_callback_exception) and the loop goes on
            if not self.closing:
                self.after(30, self.poll)

    def on_event(self, ev):
        kind = ev.get("ev")
        tr = self.turn
        current = tr is not None and ev.get("id") == tr["id"]
        if current and kind in ("token", "start", "done", "full", "error"):
            tr["events"] += 1
        if kind == "token":
            if not current:
                return
            if tr["app_first"] is None:
                tr["app_first"] = time.perf_counter()
                self.trace("first_token", id=tr["id"], press_to_first_token_s=tr["app_first"] - tr["t_press"],
                         engine_t=ev.get("t"))
            if tr["t_first"] is None:
                tr["t_first"] = ev["t"]
            tr["t_last"] = ev["t"]
            tr["tokens"] = ev["n"]
            tr["view"].feed(ev["text"])
            tr["dirty"] = True
        elif kind == "start":
            if current:
                tr["start"] = {k: v for k, v in ev.items() if k not in ("ev", "id")}
                self.trace("engine_start", **{k: v for k, v in ev.items() if k != "ev"})
        elif kind == "done":
            if current:
                self.finish(ev["stats"])
        elif kind == "full":
            if current:
                if not self.input.get("1.0", "end-1c").strip():
                    self.input.insert("1.0", tr["question"])   # ready to send again in a new chat
                    self.update_placeholder()
                self.finish({"finish": "full_before"}, note=ev.get("error"))
        elif kind == "error":
            if current:
                self.finish({"finish": "error"}, note=ev.get("error"))
        elif kind == "hello":
            self.trace("engine_hello", **{k: v for k, v in ev.items() if k != "ev"})
        elif kind == "loaded":
            self.state_ = "ready"
            self.trace("loaded", **(ev.get("info") or {}))
            self.status.configure(text="")
            self.status.pack_forget()
            self.set_title()
        elif kind == "load_failed":
            self.fail(ev.get("error") or "The model did not load.")
        elif kind == "engine_exit":
            if self.closing or ev.get("child") is not self.runner:
                return
            self.trace("engine_exit", code=ev.get("code"), stderr=ev.get("stderr"))
            if self.state_ == "failed":
                return
            last = next((line for line in reversed(ev.get("stderr") or []) if line.strip()), "")[:160]
            msg = f"The model process stopped (exit code {ev.get('code')})."
            if last:
                msg += f" Its last message: {last}"
            self.fail(msg)

    def fail(self, message):
        self.state_ = "failed"
        self.trace("failed", error=message)
        self.status.configure(text="")
        self.status.pack_forget()
        self.bar_text.configure(text=message)
        self.bar.grid(row=0, column=0, sticky="ew", padx=self.bar_pad, pady=(10, 0))
        self.fit_box()
        if self.turn is not None:
            self.finish({"finish": "error"}, note=message)
        self.set_busy(False)

    # the network badge -----------------------------------------------------------------------------------------
    def refresh_net(self):
        if self.closing:
            return
        try:
            self.show_net()
        finally:
            if not self.closing:
                self.after(1000, self.refresh_net)

    def show_net(self):
        try:
            n = netcount.count()
            total = n["total"] if "error" not in n else None
        except Exception as error:   # noqa: BLE001 - not measured = "unknown", never an old 0
            n, total = {"error": f"{type(error).__name__}: {error}"}, None
        if total != self.net_last:
            self.net_last = total
            if total == 0:
                self.badge.configure(text="  On this PC · 0 network connections  ", fg_color=C["ok_bg"],
                                     text_color=C["ok_text"])
            elif total is None:
                self.badge.configure(text="  On this PC · network connections: unknown  ", fg_color=C["warn_bg"],
                                     text_color=C["warn_text"])
            else:
                self.badge.configure(text=f"  On this PC · {total} network connection{'s' if total != 1 else ''}  ",
                                     fg_color=C["warn_bg"], text_color=C["warn_text"])
            self.trace("net", total=total, tcp=n.get("tcp"), udp=n.get("udp"), pids=n.get("pids"), rows=n.get("rows"),
                     error=n.get("error"))

    def close(self):
        self.closing = True
        self.trace("close", runtime_imported="onnxruntime_genai" in sys.modules)
        try:
            if self.runner is not None:
                if self.turn is not None:
                    self.runner.send({"op": "stop", "id": self.turn["id"]})
                self.runner.close()
                self.trace("engine_closed", code=self.runner.p.poll())
        finally:
            if self.trace_file:
                self.trace_file.close()
                self.trace_file = None
            self.destroy()


# --- one window at a time -------------------------------------------------------------------------------------------
def bring_to_front():
    """Bring the open chat window (a Tk window whose title starts with the app's name) to the front."""
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [proc, wintypes.LPARAM]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    found = []

    def each(hwnd, _):
        cls, title = ctypes.create_unicode_buffer(64), ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 64)
        user32.GetWindowTextW(hwnd, title, 256)
        if cls.value == "TkTopLevel" and title.value.startswith(APP_TITLE):
            found.append(hwnd)
        return True

    user32.EnumWindows(proc(each), 0)
    for hwnd in found[:1]:
        user32.ShowWindow(hwnd, 9)     # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
    return bool(found)


def already_open():
    """True when another chat window holds the named mutex (it is then brought to the front)."""
    if sys.platform != "win32":
        return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    handle = k32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle or ctypes.get_last_error() != 183:   # ERROR_ALREADY_EXISTS
        globals()["_MUTEX"] = handle                    # held until this process ends
        return False
    bring_to_front()
    return True


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Chat with Gemma 4 E4B on this PC.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "webgpu", "cpu", "fake"])
    ap.add_argument("--model-dir", default="", help="a model folder (default: models\\<package> next to this file)")
    ap.add_argument("--adapter", type=int, default=0, help="with --device webgpu: the GPU by its WebGPU number "
                                                            "(default 0)")
    ap.add_argument("--auto-picked", action="store_true",
                    help="the device came from auto (chat.cmd resolves auto itself and says so with this)")
    ap.add_argument("--context", type=int, default=engine.DEFAULT_CONTEXT, choices=engine.CONTEXTS,
                    help="tokens one chat can hold (default %(default)s)")
    ap.add_argument("--maximized", action="store_true")
    ap.add_argument("--log", default="", help="add one JSON line per answer to this file (see the top of app.py)")
    ap.add_argument("--trace", default="", help=argparse.SUPPRESS)
    ap.add_argument("--fake-load-s", type=float, default=1.5, help=argparse.SUPPRESS)
    ap.add_argument("--fake-rate", type=float, default=60.0, help=argparse.SUPPRESS)
    ap.add_argument("--fake-first-token-s", type=float, default=0.3, help=argparse.SUPPRESS)
    ap.add_argument("--fake-fail", default="", choices=["", "load", "answer", "crash"], help=argparse.SUPPRESS)
    ap.add_argument("--fake-text", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.auto_picked = args.auto_picked or args.device == "auto"
    args.device = engine.resolve_device(args.device)
    return args


def main():
    args = parse_args()
    if already_open():
        print("Gemma 4 chat is already open: its window was brought to the front.", file=sys.stderr)
        return 3
    app = ChatApp(args)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

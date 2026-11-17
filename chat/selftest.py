"""Checks of the chat without its window, on Windows. Run it with the Python environment chat.cmd made, e.g.

    .venv-cuda\\Scripts\\python.exe selftest.py                 everything, the model on the device chat.cmd set up
    .venv-cuda\\Scripts\\python.exe selftest.py --skip-model    everything but the model (no GPU work)
    .venv-webgpu\\Scripts\\python.exe selftest.py --model-only --adapter 1    the model alone, on WebGPU's GPU 1

1. the process setup (prelude.py): ORT_DISABLE_TELEMETRY=1 and power throttling off
2. the network count (netcount.py): 0 for this process, at least 1 while it holds a UDP socket on 127.0.0.1, then 0;
   the same with the socket held by a child process, counted the way the window counts (this process and its children)
3. the text the window draws: emoji left out, formulas as text, rules and check boxes, table cells wrapped,
   open marks closed
4. the model process with fixed test answers (engine.py --serve --device fake): an answer, Stop, New chat, a full chat,
   an error in an answer, a crash, a failed load, its end when the window's process ends, only JSON lines on its stdout
5. the model process with the model: it loads, answers one short question with 0 network endpoints, writes only JSON
   lines on its stdout, and quits
Prints one PASS or FAIL line per check; the exit code is 0 only when every check passed.
"""
import prelude  # noqa: F401  (first, as in app.py and engine.py)

import argparse
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import engine
import netcount

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_FAST = ["--device", "fake", "--fake-load-s", "0.2", "--fake-first-token-s", "0.05", "--fake-rate", "4000"]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""), flush=True)
    return ok


class Child:
    """engine.py --serve the way the window starts it: pipes, no console window, JSON lines both ways."""

    def __init__(self, extra, parent_pid=None):
        exe = os.path.join(os.path.dirname(sys.executable), "python.exe")
        cmd = [exe if os.path.isfile(exe) else sys.executable, os.path.join(HERE, "engine.py"), "--serve",
               "--parent-pid", str(parent_pid or os.getpid())] + extra
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  cwd=HERE, env=env, creationflags=0x08000000 if sys.platform == "win32" else 0)
        self.events, self.bad, self.err, self.pid = queue.Queue(), [], [], None
        threading.Thread(target=self._out, daemon=True).start()
        threading.Thread(target=self._err, daemon=True).start()

    def _out(self):
        for raw in self.p.stdout:
            try:
                ev = json.loads(raw.decode("utf-8"))
            except ValueError:
                self.bad.append(raw[:120])
                continue
            if ev.get("ev") == "hello":
                self.pid = ev.get("pid")
            self.events.put(ev)
        self.events.put({"ev": "eof"})

    def _err(self):
        for raw in self.p.stderr:
            self.err.append(raw.decode("utf-8", "replace").rstrip())

    def send(self, obj):
        self.p.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.p.stdin.flush()

    def until(self, kinds, timeout):
        """Events up to and including the first one of `kinds` (or None at the end)."""
        seen, end = [], time.time() + timeout
        while time.time() < end:
            try:
                ev = self.events.get(timeout=max(0.05, end - time.time()))
            except queue.Empty:
                break
            seen.append(ev)
            if ev.get("ev") in kinds:
                return ev, seen
        return None, seen

    def answer(self, turn, text, timeout=60, stop_after=None):
        self.send({"op": "ask", "id": turn, "text": text})
        got, end, tokens, last = "", time.time() + timeout, 0, None
        while time.time() < end:
            ev, _ = self.until({"token", "done", "full", "error", "eof"}, end - time.time())
            if ev is None or ev["ev"] != "token":
                return ev, got, tokens
            got += ev["text"]
            tokens = ev["n"]
            last = ev
            if stop_after is not None and tokens >= stop_after:
                self.send({"op": "stop", "id": turn})
                stop_after = None
        return last, got, tokens

    def pids(self):
        return [p for p in (self.p.pid, self.pid) if p]

    def quit(self, timeout=30):
        try:
            self.send({"op": "quit"})
            self.p.stdin.close()
        except OSError:
            pass
        try:
            return self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(self.p.pid), "/T", "/F"], capture_output=True)
            return None


def part_setup():
    check("telemetry off", os.environ.get("ORT_DISABLE_TELEMETRY") == "1", os.environ.get("ORT_DISABLE_TELEMETRY"))
    check("power throttling off", prelude.POWER_THROTTLING == "off", prelude.POWER_THROTTLING)
    child_env = subprocess.run([sys.executable, "-c", "import prelude, os; print(os.environ.get('ORT_DISABLE_TELEMETRY'))"],
                               cwd=HERE, capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                                               if k != "ORT_DISABLE_TELEMETRY"})
    check("telemetry off in a process that imports prelude", child_env.stdout.strip() == "1", child_env.stdout.strip())


def part_netcount():
    before = netcount.count()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    during = netcount.count()
    s.close()
    after = netcount.count()
    check("network count control", before["total"] == 0 and during["total"] >= 1 and after["total"] == 0,
          f"before {before['total']}, with one UDP socket {during['total']}, after {after['total']}")
    # the path the window's badge depends on: a socket held by a child process, counted with the default process set
    # (this process and everything under it; a venv's python.exe starts the real interpreter as its own child)
    child = subprocess.Popen([sys.executable, "-c", "import socket, sys; s = socket.socket(socket.AF_INET, "
                              "socket.SOCK_DGRAM); s.bind(('127.0.0.1', 0)); sys.stdin.read()"],
                             stdin=subprocess.PIPE, creationflags=0x08000000)
    seen, end = None, time.time() + 15
    while time.time() < end:
        seen = netcount.count()
        if seen["total"]:
            break
        time.sleep(0.2)
    child.stdin.close()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
    gone = netcount.count()
    check("network count control in a child process", seen is not None and (seen["total"] or 0) >= 1
          and gone["total"] == 0, f"with the child's UDP socket {seen and seen['total']} (pids {seen and seen['pids']}), "
          f"after it ended {gone['total']}")


def part_text():
    import app

    cases = [
        (app.drop_emoji("### \U0001f5fc Day 1"), "### Day 1"),
        (app.drop_emoji("**Sunsets \U0001f305:**"), "**Sunsets:**"),
        (app.drop_emoji("\U0001f1ef\U0001f1f5 7 days in Japan \u2708\ufe0f"), "7 days in Japan"),
        (app.drop_emoji("Hello\U0001f389!"), "Hello!"),
        (app.drop_emoji("\u26e9\ufe0f Day 4 and \u23f1\ufe0f time"), "Day 4 and time"),
        (app.drop_emoji("### \U0001f4a1 Quick Tips"), "### Quick Tips"),
        (app.drop_emoji("**\U0001f4cc Quick Tips Before Sending:**"), "**Quick Tips Before Sending:**"),
        (app.drop_emoji("\U0001f469\u200d\U0001f373 Chef and \U0001f37d\ufe0f Dinner"), "Chef and Dinner"),
        (app.drop_emoji("Lunch \U0001f37d\ufe0f time"), "Lunch time"),
        (app.drop_emoji("Tokyo \u2192 Kyoto, 10\u00b0C \u2264 20\u00b0C, \u00a512,000"),
         "Tokyo \u2192 Kyoto, 10\u00b0C \u2264 20\u00b0C, \u00a512,000"),
        (app.replace_math("nitrogen ($\\text{N}_2$) and oxygen ($\\text{O}_2$)"), "nitrogen (N\u2082) and oxygen (O\u2082)"),
        (app.replace_math("($\\text{1}/\\lambda^4$)"), "(1/\u03bb\u2074)"),
        (app.replace_math("($\\text{Scattering} \\propto 1/\\lambda^4$)"), "(Scattering \u221d 1/\u03bb\u2074)"),
        (app.replace_math("Tokyo $\\rightarrow$ Kyoto"), "Tokyo \u2192 Kyoto"),
        (app.replace_math("$200^\\circ$C and $\\approx 450$ nm, 5 $\\times$ more, $\\le$ 3, $\\ge$ 4, $\\leftarrow$"),
         "200\u00b0C and \u2248 450 nm, 5 \u00d7 more, \u2264 3, \u2265 4, \u2190"),
        (app.replace_math("$10^{-3}$, $x_{12}$, $a \\cdot b \\neq c \\pm d$, $\\frac{1}{2}$, $\\sqrt{x}$"),
         "10\u207b\u00b3, x\u2081\u2082, a \u00b7 b \u2260 c \u00b1 d, 1/2, \u221ax"),
        (app.replace_math("it costs $5 and $10"), "it costs $5 and $10"),
        (app.replace_math("$x$ and $\\hbar$ and $\\vec{v}$"), "x and hbar and v"),
        (app.display_text("400\u00b0F (200\u00b0C), it\u2019s [Landlord's Name]"),
         "400\u00b0F (200\u00b0C), it\u2019s [Landlord's Name]"),
        (app.close_open_marks("**Day 1 \u2013 Arr"), "**Day 1 \u2013 Arr**"),
        (app.close_open_marks("`sumi"), "`sumi`"),
        (app.close_open_marks("a phenomenon called **"), "a phenomenon called "),
        (app.close_open_marks("run `"), "run "),
        (app.close_open_marks("**"), ""),
        (app.close_open_marks("| **Blue** | 450 nm | **"), "| **Blue** | 450 nm | "),
        (app.close_open_marks("called **Rayleigh scattering**"), "called **Rayleigh scattering**"),
        (app.close_open_marks("snake_case_"), "snake_case_"),
        (app.inline_segments("**Day 1 (Tokyo)** and *more*"), [("Day 1 (Tokyo)", "b"), (" and ", ""), ("more", "i")]),
        (app.inline_segments("[Kitchen/Bathroom - specify which one] and [a link](https://example.com)"),
         [("[Kitchen/Bathroom - specify which one] and ", ""), ("a link", "link")]),
        (bool(app.RULE.match("***")) and bool(app.RULE.match("---")) and not app.BULLET.match("***"), True),
        (app.BULLET.match("*   **Breakfast:** oats").group(2), "**Breakfast:** oats"),
        (app.CHECKBOX.match("[ ] Eggs").group(1) + app.CHECKBOX.match("[x] Milk").group(1), " x"),
        (app.replace_math("Add `export PATH=${HOME}/bin:${PATH}` to .bashrc"),
         "Add `export PATH=${HOME}/bin:${PATH}` to .bashrc"),
        (app.replace_math("Energy: $$E = mc^2$$ here"), "Energy: E = mc\u00b2 here"),
        (app.replace_math("shell ${a}${b} text"), "shell ${a}${b} text"),
        (app.hold_open_math("about 450 nm ($\\approx"), "about 450 nm ("),
        (app.hold_open_math("it costs $5 and"), "it costs $5 and"),
        (app.hold_open_math("($\\text{N}_2$) and $\\text{O"), "($\\text{N}_2$) and "),
    ]
    bad = [(got, want) for got, want in cases if got != want]
    left = [c for got, _ in cases for c in str(got) if c in "\u200d\ufe0f\x00"]
    check("text: emoji, math, rules, check boxes, open marks", not bad and not left,
          f"{len(cases) - len(bad)} of {len(cases)}, invisible characters left {len(left)}" +
          (f"; first wrong: {bad[0][0]!r} != {bad[0][1]!r}" if bad else ""))
    lines = app.wrap_segments([("Oatmeal with ", ""), ("berries", "b"), (" and honey", "")], 10)
    widths = [sum(app.cell_width(t) for t, _ in line) for line in lines]
    words = " ".join("".join(t for t, _ in line) for line in lines).split()
    check("text: table cells wrap", max(widths) <= 10 and words == "Oatmeal with berries and honey".split(),
          f"{len(lines)} lines, widths {widths}")
    names = [app.os_name(build) for build in (19045, 21999, 22000, 22631, 26100)]
    check("header: the Windows name from the build (22000 and later = Windows 11)",
          names == ["Windows 10", "Windows 10", "Windows 11", "Windows 11", "Windows 11"], ", ".join(names))
    here = app.os_name()
    check("header: this PC's Windows name", here in ("Windows 10", "Windows 11") if sys.platform == "win32" else here == "",
          f"{here!r} (build {app.windows_build()})")


def part_fake():
    c = Child(FAKE_FAST)
    hello, _ = c.until({"hello"}, 20)
    loaded, _ = c.until({"loaded", "load_failed"}, 20)
    check("fake: starts and loads", hello is not None and loaded is not None and loaded["ev"] == "loaded",
          f"hello pid {hello and hello.get('pid')}, telemetry off {hello and hello.get('telemetry_off')}")
    done, text, n = c.answer(1, "Explain why the sky is blue.")
    st = (done or {}).get("stats", {})
    check("fake: an answer streams to its end", done is not None and done["ev"] == "done" and st.get("finish") == "end"
          and st.get("tokens") == n and n > 50 and st["t_first"] <= st["t_last"] == st["seconds"],
          f"{n} tokens, finish {st.get('finish')}, {len(text)} characters")
    count = netcount.count(c.pids())
    check("fake: the model process holds no network endpoint", count["total"] == 0, f"pids {c.pids()}: {count['total']}")
    done, text, n = c.answer(2, "What is a good name for a cat?")
    check("fake: a second question in the same chat", done is not None and done["ev"] == "done", f"{n} tokens")
    c.send({"op": "new_chat"})
    done, _, n = c.answer(3, "Give me a 3-day meal plan with a shopping list.")
    check("fake: New chat, then a question", done is not None and done["ev"] == "done", f"{n} tokens")
    code = c.quit()
    check("fake: quits", code == 0, f"exit code {code}")
    check("fake: only JSON lines on the protocol pipe", not c.bad, f"{len(c.bad)} other lines")

    c = Child(["--device", "fake", "--fake-load-s", "0.1", "--fake-first-token-s", "0.05", "--fake-rate", "60"])
    c.until({"loaded"}, 20)
    done, text, n = c.answer(1, "Explain why the sky is blue.", stop_after=10)
    st = (done or {}).get("stats", {})
    check("fake: Stop ends the answer", done is not None and st.get("finish") == "stopped" and n < 40,
          f"finish {st.get('finish')} after {n} tokens")
    done, _, n = c.answer(2, "And why are sunsets red?", timeout=60)
    check("fake: a question after Stop", done is not None and done["ev"] == "done", f"{n} tokens")
    c.quit()

    c = Child(FAKE_FAST + ["--context", "2048"])
    c.until({"loaded"}, 20)
    finishes = []
    for turn in range(1, 12):
        done, _, n = c.answer(turn, "Plan a 7-day trip to Japan.")
        if done is None:
            finishes.append("no answer")
            break
        finishes.append("answer:" + done["stats"]["finish"] if done["ev"] == "done" else "event:" + done["ev"])
        if done["ev"] != "done":
            break
    check("fake: a chat gets full", finishes[-1] == "event:full", " ".join(finishes))
    full_line = (done or {}).get("error", "")
    check("fake: the full chat says what to do", full_line == "This chat is full. Start a new chat.", full_line)
    c.send({"op": "new_chat"})
    done, _, n = c.answer(99, "Explain why the sky is blue.")
    check("fake: New chat after a full chat", done is not None and done["ev"] == "done", f"{n} tokens")
    c.quit()

    c = Child(FAKE_FAST + ["--fake-fail", "answer"])
    c.until({"loaded"}, 20)
    done, _, n = c.answer(1, "Explain why the sky is blue.")
    check("fake: an error in an answer is one line", done is not None and done["ev"] == "error" and
          done.get("new_chat") is True, (done or {}).get("error", ""))
    c.quit()

    c = Child(FAKE_FAST + ["--fake-fail", "crash"])
    c.until({"loaded"}, 20)
    done, _, n = c.answer(1, "Explain why the sky is blue.")
    code = c.quit(timeout=20) if c.p.poll() is None else c.p.returncode
    check("fake: a crash ends the stream", done is not None and done["ev"] == "eof" and code == 9,
          f"last event {done and done['ev']}, exit code {code}")

    c = Child(FAKE_FAST + ["--fake-fail", "load"])
    ev, _ = c.until({"loaded", "load_failed"}, 20)
    try:
        code = c.p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        code = c.quit(timeout=5)
    check("fake: a failed load is one line and exit code 2", ev is not None and ev["ev"] == "load_failed" and code == 2,
          f"{(ev or {}).get('error', '')} (exit {code})")

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"], creationflags=0x08000000)
    c = Child(FAKE_FAST, parent_pid=holder.pid)
    c.until({"loaded"}, 20)
    holder.wait()
    t0 = time.time()
    try:
        code = c.p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        code = None
        c.quit()
    check("fake: ends when the window's process ends", code == 3, f"exit code {code} {time.time() - t0:.1f} s after")


def part_model(args):
    extra = ["--device", args.device, "--context", str(args.context)]
    if args.model_dir:
        extra += ["--model-dir", args.model_dir]
    if args.device == "webgpu":
        extra += ["--adapter", str(args.adapter)]
    c = Child(extra)
    t0 = time.time()
    ev, seen = c.until({"loaded", "load_failed"}, 600)
    info = (ev or {}).get("info", {})
    if not check(f"model: loads on {args.device}", ev is not None and ev["ev"] == "loaded",
                 f"{info.get('label')}, device_type {info.get('device_type')}, {info.get('load_s')} s, "
                 f"max_length {info.get('max_length')}" if ev and ev["ev"] == "loaded" else
                 (ev or {}).get("error", f"no answer in {time.time() - t0:.0f} s; stderr: {c.err[-3:]}")):
        c.quit()
        return
    counts = []
    c.send({"op": "ask", "id": 1, "text": "In one short sentence: what is the capital of France?"})
    text, done, end = "", None, time.time() + 120
    while time.time() < end:
        ev, _ = c.until({"token", "done", "full", "error", "eof"}, end - time.time())
        if ev is None or ev["ev"] != "token":
            done = ev
            break
        text += ev["text"]
        counts.append(netcount.count(c.pids() + [os.getpid()])["total"])
    st = (done or {}).get("stats", {})
    rate = ((st["tokens"] - 1) / (st["t_last"] - st["t_first"]) if st.get("tokens", 0) > 1 and st["t_last"] > st["t_first"]
            else None)
    check("model: answers one question", done is not None and done["ev"] == "done" and st.get("finish") == "end"
          and "paris" in text.lower(), f"{text.strip()!r}: {st.get('tokens')} tokens, first token at "
          f"{st.get('t_first')} s, {rate and round(rate, 1)} tokens/s")
    after = netcount.count(c.pids() + [os.getpid()])["total"]
    check("model: no network endpoint while it answers", counts and max(counts) == 0 and after == 0,
          f"{len(counts)} reads, most {max(counts) if counts else None}, after {after}")
    code = c.quit(timeout=60)
    check("model: quits", code == 0, f"exit code {code}")
    check("model: only JSON lines on the protocol pipe (the runtime's own printing goes to stderr)", not c.bad,
          f"{len(c.bad)} other lines: {c.bad[:2]}")


def main():
    ap = argparse.ArgumentParser(description="Checks of the chat without its window.")
    ap.add_argument("--skip-model", action="store_true", help="leave out part 5 (the model)")
    ap.add_argument("--model-only", action="store_true", help="part 5 (the model) alone")
    ap.add_argument("--device", default="", help="the device for part 5 (default: the one this environment is for)")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--adapter", type=int, default=0, help="WebGPU only: the GPU number (as chat.cmd -Adapter)")
    ap.add_argument("--context", type=int, default=engine.DEFAULT_CONTEXT, choices=engine.CONTEXTS)
    args = ap.parse_args()
    if not args.device:
        venv = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))).lower()
        args.device = venv[len(".venv-"):] if venv.startswith(".venv-") else engine.resolve_device("auto")
    sys.stdout.reconfigure(encoding="utf-8")
    parts = [] if args.model_only else [("setup", part_setup, ()), ("network count", part_netcount, ()),
                                         ("text", part_text, ()), ("fake", part_fake, ())]
    if not args.skip_model:
        parts.append(("model", part_model, (args,)))
    for name, part, part_args in parts:
        try:
            part(*part_args)
        except Exception as error:   # noqa: BLE001 - a check that crashes is a FAIL line, and the next part still runs
            check(f"{name}: ran to its end", False, f"{type(error).__name__}: {error}")
    print(f"selftest: {sum(RESULTS)} PASS, {len(RESULTS) - sum(RESULTS)} FAIL", flush=True)
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

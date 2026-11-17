"""Gemma 4 E4B for the chat window: ONNX Runtime GenAI in a child process of the window.

    python engine.py --serve --device cuda|webgpu|cpu|fake [--adapter N] [--model-dir <folder>] [--context 4096]
                     [--parent-pid N]
    python engine.py --resolve-device auto       prints what "auto" picks on this PC: cuda, webgpu or cpu
    python engine.py --check-model-dir <folder>  exit 0 when the folder holds every file of its package (by size)
    python engine.py --label <device>            prints the name the window shows for that device

The window (app.py) starts `engine.py --serve` and talks to it over the child's stdin and stdout, one JSON object per
line. Nothing here opens a network connection: the model files come from the folder chat.ps1 downloaded.

    window -> engine   {"op": "ask", "id": 3, "text": "..."}     a new question in the current chat
                       {"op": "stop", "id": 3}                   stop that answer
                       {"op": "new_chat"}                        forget the chat; the next question starts a new one
                       {"op": "quit"}
    engine -> window   {"ev": "hello", "pid": ..., "device": ...}
                       {"ev": "loaded", "info": {...}}  or  {"ev": "load_failed", "error": "one line"}
                       {"ev": "start", "id": 3, ...}             the question went into the model
                       {"ev": "token", "id": 3, "text": "...", "n": 12, "t": 0.84}
                       {"ev": "done", "id": 3, "stats": {...}}
                       {"ev": "full", "id": 3, "error": "This chat is full. Start a new chat."}
                       {"ev": "error", "id": 3, "error": "one line", "new_chat": true}

The numbers under an answer come from this process's clock. For one answer:
    t        = seconds from the moment the question's line arrived on this process's stdin
    seconds  = t of the answer's last token
    tokens/s = (the answer's tokens - 1) / (t of the last token - t of the first token)
The end-of-turn token that ends an answer is not counted.

One conversation = one GenAI Generator with a fixed max_length (the --context value) and one random seed (drawn when the
chat's first question comes; reported as chat_seed). Each new question appends only the tokens of the new turn, made by
the model's own chat template; the model keeps everything before it. Stop while a question is still going into the model
ends the chat: the next question starts a new one.
"""
import prelude  # noqa: F401  (first: telemetry off and power throttling off before the runtime loads)

import argparse
import json
import os
import queue
import re
import secrets
import sys
import sysconfig
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Gemma 4 E4B"
MANIFEST = os.path.join(HERE, "models.json")

# --- settings in one place -------------------------------------------------------------------------------------------
CONTEXTS = (2048, 4096, 8192)         # tokens one chat can hold (questions and answers together)
DEFAULT_CONTEXT = 4096
MIN_ANSWER_ROOM = 128                 # a question goes in only when this much room is left for its answer
PIECE = 256                           # a long question goes into the model in pieces of this many tokens
# "auto" = CUDA on an NVIDIA GPU with at least this much memory (MiB), else WebGPU. On a 16 GB GPU (RTX 5080 Laptop) a
# 4,096-token chat took the whole GPU's used memory up to about 12 GB; smaller GPUs were not measured.
AUTO_CUDA_MIN_MIB = 15000
# Free GPU memory asked before loading when "auto" picked CUDA (MiB); a named -Device cuda loads without this check and
# reports a real out-of-memory failure. On that GPU loading took 8,459-10,187 MiB (its used memory right after loading,
# minus the 300 MiB in use before).
GPU_FREE_MIB = 9500
CUDA_OPTIONS = {"enable_cuda_graph": "1"}   # CUDA execution provider options, set at load (genai_config.json untouched)
WEBGPU_OPTIONS = {}                   # WebGPU execution provider options; the GPU = "adapterIndex" (--adapter)
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_k": 64, "top_p": 0.95}   # the model's generation_config.json
END_IDS = (1, 106, 50)                # <eos>, <turn|> and id 50: the model's generation_config.json eos_token_id list
PLACEHOLDER = "QQ_PREVIOUS_ANSWER_QQ"
SESSIONS = ("decoder", "embedding", "vision", "speech")


class EngineError(Exception):
    """A failure the window shows as one line: what happened and what to do next."""


class ChatFull(EngineError):
    def __init__(self, text="This chat is full. Start a new chat."):
        super().__init__(text)


# --- the model files ------------------------------------------------------------------------------------------------
def read_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def package_for(device):
    return read_manifest()["devices"][device]


def default_model_dir(device):
    return os.path.join(HERE, "models", package_for(device))


def check_model_dir(model_dir):
    """Raise EngineError when the folder is missing, or files of a known package are missing or the wrong size."""
    name = os.path.basename(os.path.normpath(model_dir))
    own = os.path.normcase(os.path.abspath(model_dir)) == os.path.normcase(os.path.join(HERE, "models", name))
    if not os.path.isdir(model_dir):
        if own:
            raise EngineError(f"The model folder is missing: {model_dir}. Run chat.cmd to download it.")
        raise EngineError(f"The model folder {model_dir} does not exist.")
    try:
        packages = read_manifest()["packages"]
    except (OSError, ValueError, KeyError):
        packages = {}
    files = packages.get(name, {}).get("files")
    if files:
        bad = [f["path"] for f in files
               if not os.path.isfile(os.path.join(model_dir, f["path"]))
               or os.path.getsize(os.path.join(model_dir, f["path"])) != f["size"]]
        if bad:
            advice = ("Run chat.cmd again: it downloads what is missing." if own else
                      "Copy the missing files into that folder, or run chat.cmd without -ModelDir to download the "
                      "model.")
            raise EngineError(f"{len(bad)} of {len(files)} model files are missing or incomplete in {model_dir} "
                              f"(first: {bad[0]}). {advice}")
        return
    config = os.path.join(model_dir, "genai_config.json")
    if not os.path.isfile(config):
        raise EngineError(f"{model_dir} is not a model folder: genai_config.json is missing.")
    with open(config, encoding="utf-8") as f:
        model = json.load(f)["model"]
    for session in SESSIONS:
        filename = model.get(session, {}).get("filename")
        if filename and not os.path.isfile(os.path.join(model_dir, filename)):
            raise EngineError(f"The model file {filename} is missing in {model_dir}.")


# --- what this PC has (read locally: the registry and the NVIDIA driver's nvml.dll) ---------------------------------
class Nvml:
    """The first NVIDIA GPU through the driver's nvml.dll: name, total / free / used memory in MiB."""

    def __init__(self):
        import ctypes

        self.ok = False
        if sys.platform != "win32":
            return
        path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvml.dll")
        try:
            self.lib = ctypes.WinDLL(path)
            if self.lib.nvmlInit_v2() != 0:
                return
            self.handle = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
                return
        except (OSError, AttributeError):
            return

        class Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        self.memory = Memory()
        self.ctypes = ctypes
        self.ok = True

    def name(self):
        if not self.ok:
            return ""
        buf = self.ctypes.create_string_buffer(96)
        if self.lib.nvmlDeviceGetName(self.handle, buf, 96) != 0:
            return ""
        return buf.value.decode("utf-8", "replace")

    def mib(self):
        """{"total", "free", "used"} in MiB, or None."""
        if not self.ok or self.lib.nvmlDeviceGetMemoryInfo(self.handle, self.ctypes.byref(self.memory)) != 0:
            return None
        m = self.memory
        return {"total": m.total // 2**20, "free": m.free // 2**20, "used": m.used // 2**20}


def _registry_value(path, value):
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            return winreg.QueryValueEx(key, value)[0]
    except (ImportError, OSError):
        return None


def display_adapters():
    """[(name, memory in bytes or 0)] of the display adapters Windows lists (the registry's display device class)."""
    out = []
    try:
        import winreg

        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as key:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(key, i)
                except OSError:
                    break
                if not sub.isdigit():
                    continue
                name = _registry_value(base + "\\" + sub, "DriverDesc")
                if not name or "Microsoft Basic" in name or any(n == name for n, _ in out):
                    continue
                size = _registry_value(base + "\\" + sub, "HardwareInformation.qwMemorySize")
                out.append((str(name), int(size) if isinstance(size, int) else 0))
    except (ImportError, OSError):
        pass
    return out


def clean_name(name):
    return re.sub(r"\s+", " ", re.sub(r"\((R|TM|tm|r)\)", "", name)).strip()


def processor_name():
    return clean_name(str(_registry_value(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                                          "ProcessorNameString") or "")) or "this PC's processor"


def nvidia_adapter():
    """(name, memory in MiB) of the NVIDIA adapter with the most memory, or None. The memory comes from the registry,
    or from the driver when the registry does not say."""
    found = [(n, b) for n, b in display_adapters() if "NVIDIA" in n.upper()]
    if not found:
        return None
    name, size = max(found, key=lambda x: x[1])
    if not size:
        mem = Nvml().mib()
        size = mem["total"] * 2**20 if mem else 0
    return clean_name(name), size // 2**20


def resolve_device(device):
    """auto = CUDA on an NVIDIA GPU with AUTO_CUDA_MIN_MIB of memory or more, else WebGPU."""
    if device != "auto":
        return device
    gpu = nvidia_adapter()
    if gpu and gpu[1] >= AUTO_CUDA_MIN_MIB:
        return "cuda"
    return "webgpu"


def device_label(device, adapter=0):
    """What the window shows as the device: only what the app knows. cuda = the NVIDIA GPU as Windows names it; webgpu =
    the WebGPU adapter number (which GPU that number is, the WebGPU runtime decides; the app does not read its name);
    cpu = the processor's name."""
    if device == "cuda":
        gpu = nvidia_adapter()
        return gpu[0] if gpu else "NVIDIA GPU"
    if device == "webgpu":
        return f"WebGPU · GPU {int(adapter)}"
    if device == "fake":
        return "Test mode: fixed answers, no model"
    return processor_name()


# --- the model ------------------------------------------------------------------------------------------------------
def _add_cuda_dlls():
    """The CUDA 13 runtime and cuDNN from pip live in the venv's site-packages; the CUDA provider finds them on PATH."""
    site = sysconfig.get_paths()["purelib"]
    dirs = [os.path.join(site, "nvidia", "cu13", "bin", "x86_64"), os.path.join(site, "nvidia", "cu13", "bin"),
            os.path.join(site, "nvidia", "cudnn", "bin")]
    found = [d for d in dirs if os.path.isdir(d)]
    os.environ["PATH"] = os.pathsep.join(found + [os.environ.get("PATH", "")])
    return [os.add_dll_directory(d) for d in found] if hasattr(os, "add_dll_directory") else []


def first_line(error):
    text = str(error)
    return next((line.strip() for line in text.splitlines() if line.strip()), type(error).__name__)


def webgpu_config(config, options):
    """The package's genai_config.json (a dict) with every part of the model on WebGPU with these options."""
    config = json.loads(json.dumps(config))
    for part in SESSIONS:
        if part in config["model"]:
            config["model"][part].setdefault("session_options", {})["provider_options"] = [{"WebGPU": dict(options)}]
    return config


def webgpu_folder(model_dir, options, root=None):
    """The folder WebGPU loads the model from: <chat folder>/models/<package>.webgpu with a hard link to each file of
    the package (the same file on disk: no copy, no extra space) and a genai_config.json that puts every part of the
    model (decoder, embedding, vision, speech) on WebGPU. The package's own genai_config.json puts every part on CUDA;
    onnxruntime-genai 0.19.0 changes the provider of the decoder only and reads model files only from inside the folder
    of the genai_config.json it loads, so WebGPU needs a config of its own. The package's files are not changed.
    Made again at each start: links to replaced or removed files are renewed or removed."""
    src = os.path.abspath(model_dir)
    dst = os.path.join(root or os.path.join(HERE, "models"), os.path.basename(os.path.normpath(src)) + ".webgpu")
    try:
        with open(os.path.join(src, "genai_config.json"), encoding="utf-8") as f:
            config = webgpu_config(json.load(f), options)
        wanted = set()
        for base, _dirs, files in os.walk(src):
            for name in files:
                rel = os.path.normpath(os.path.relpath(os.path.join(base, name), src))
                if rel == "genai_config.json":
                    continue
                wanted.add(os.path.normcase(rel))
                source, link = os.path.join(src, rel), os.path.join(dst, rel)
                if os.path.isfile(link) and os.path.samefile(source, link):
                    continue
                os.makedirs(os.path.dirname(link), exist_ok=True)
                if os.path.lexists(link):
                    os.remove(link)
                os.link(source, link)
        for base, _dirs, files in os.walk(dst):
            for name in files:
                rel = os.path.normpath(os.path.relpath(os.path.join(base, name), dst))
                if rel != "genai_config.json" and os.path.normcase(rel) not in wanted:
                    os.remove(os.path.join(base, name))
        os.makedirs(dst, exist_ok=True)
        target = os.path.join(dst, "genai_config.json")
        with open(target + ".tmp", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        os.replace(target + ".tmp", target)
    except OSError as error:
        raise EngineError(f"WebGPU could not set up its model folder {dst} ({error.strerror or error}). WebGPU reads "
                          f"the model through hard links in that folder, which work only when the model folder is on "
                          f"the same NTFS drive as the chat folder.") from error
    return dst


def explain_load_error(error, device, label):
    first = first_line(error)
    low = str(error).lower()
    other = "chat.cmd -Device cpu" if device != "cpu" else "chat.cmd again to repair the setup"
    if device == "webgpu" and "creating execution provider" in low:   # e.g. -Adapter with a number this PC lacks
        return (f"{label} could not be opened on this PC. Pick another GPU with -Adapter (0 is the first), or start "
                f"the chat with {other}.")
    if any(w in low for w in ("out of memory", "outofmemory", "memoryallocation", "failed to allocate", "bad_alloc")):
        return (f"Not enough free memory on {label} to load {MODEL_NAME}. Close other apps that use it and open the "
                f"chat again, or start it with {other}.")
    if any(w in low for w in ("loadlibrary", "error 126", "cudnn", "cublas", ".dll")):
        return (f"The runtime files for {label} could not be loaded ({first[:160]}). Run chat.cmd again to repair "
                f"the setup, or start it with {other}.")
    return f"{MODEL_NAME} did not load on {label}: {first[:200]}. Start it with {other}."


class GenAIEngine:
    def __init__(self, device, model_dir=None, context=DEFAULT_CONTEXT, adapter=0, free_check=False):
        self.device = device
        self.free_check = free_check   # True when "auto" picked this device: the GPU memory check before loading
        self.model_dir = os.path.abspath(model_dir or default_model_dir(device))
        self.context = int(context)
        self.adapter = int(adapter)
        self.webgpu_options = dict(WEBGPU_OPTIONS, adapterIndex=str(self.adapter))
        self.label = device_label(device, self.adapter)
        self.og = self.model = self.tokenizer = self.generator = None
        self.max_length = 0
        self.end_ids = set(END_IDS)
        self.nvml = None
        self.turns = 0
        self.seed = None
        self.info = {}

    def gpu_mib(self):
        return self.nvml.mib() if self.nvml else None

    def load(self):
        check_model_dir(self.model_dir)
        with open(os.path.join(self.model_dir, "genai_config.json"), encoding="utf-8") as f:
            config_json = json.load(f)
        package_context = int(config_json["model"]["context_length"])
        self.max_length = min(self.context, package_context)
        eos = config_json["model"].get("eos_token_id", [])
        self.end_ids.update(eos if isinstance(eos, list) else [eos])
        if self.device == "cuda":
            self.nvml = Nvml()
            if not self.nvml.ok:
                raise EngineError(f"No NVIDIA GPU driver answered on this PC. Start the chat with chat.cmd -Device "
                                  f"webgpu or chat.cmd -Device cpu.")
            mem = self.nvml.mib()
            if self.free_check and mem and mem["free"] < GPU_FREE_MIB:
                raise EngineError(f"Not enough free memory on {self.label}: {mem['free'] / 1024:.1f} GB free, about "
                                  f"{GPU_FREE_MIB / 1024:.1f} GB needed. Close other apps that use the GPU and open "
                                  f"the chat again, or start it with chat.cmd -Device cuda to try anyway, or "
                                  f"chat.cmd -Device cpu.")
            self._dll_handles = _add_cuda_dlls()
        t0 = time.perf_counter()
        try:
            import numpy  # noqa: F401  (the tokenizer returns numpy arrays: imported while loading, not at a question)
            import onnxruntime_genai as og
        except ImportError as error:
            raise EngineError(f"The model runtime is not installed ({first_line(error)[:120]}). Run chat.cmd again "
                              f"to finish the setup.")
        self.og = og
        try:
            config = og.Config(self.model_dir)
            if self.device == "cuda":
                config.clear_providers()
                config.append_provider("cuda")
                for key, value in CUDA_OPTIONS.items():
                    config.set_provider_option("cuda", key, str(value))
            elif self.device == "webgpu":
                import onnxruntime_ep_webgpu as webgpu_ep

                og.register_execution_provider_library(webgpu_ep.get_ep_name(), webgpu_ep.get_library_path())
                config = og.Config(webgpu_folder(self.model_dir, self.webgpu_options))   # every part on WebGPU
                config.clear_providers()
                config.append_provider("WebGPU")
                for key, value in self.webgpu_options.items():
                    config.set_provider_option("WebGPU", key, str(value))
            elif self.device == "cpu":
                config.clear_providers()
            else:
                raise EngineError(f"Unknown device {self.device}.")
            # max_length is fixed before the model is built (as onnxruntime-genai's examples/python/common.py does)
            config.overlay(json.dumps({"search": {"max_length": self.max_length}}))
            self.model = og.Model(config)
            self.tokenizer = og.Tokenizer(self.model)
        except EngineError:
            raise
        except Exception as error:
            raise EngineError(explain_load_error(error, self.device, self.label)) from error
        self.info = {"device": self.device, "label": self.label, "load_s": round(time.perf_counter() - t0, 3),
                     "device_type": getattr(self.model, "device_type", ""), "model_dir": self.model_dir,
                     "max_length": self.max_length, "package_context": package_context,
                     "provider_options": CUDA_OPTIONS if self.device == "cuda" else
                     self.webgpu_options if self.device == "webgpu" else {},
                     "genai": getattr(og, "__version__", ""), "gpu_mib": self.gpu_mib(),
                     "end_ids": sorted(self.end_ids)}
        return self.info

    def new_chat(self):
        if self.generator is not None:
            del self.generator
            self.generator = None
        self.turns = 0

    def _new_generator(self):
        params = self.og.GeneratorParams(self.model)
        self.seed = secrets.randbelow(2**31 - 1)
        params.set_search_options(max_length=self.max_length, random_seed=self.seed, **SAMPLING)
        self.generator = self.og.Generator(self.model, params)

    def _render(self, messages):
        return self.tokenizer.apply_chat_template(json.dumps(messages, ensure_ascii=False), add_generation_prompt=True)

    def _turn_ids(self, text):
        """The tokens of the new turn: the whole prompt for a chat's first question, else the template's text after the
        previous answer (it starts with <turn|>: the generator does not keep the end token it stopped on)."""
        if self.turns == 0:
            return self.tokenizer.encode(self._render([{"role": "user", "content": text}]))
        bridge = self._render([{"role": "user", "content": "Q"}, {"role": "assistant", "content": PLACEHOLDER},
                               {"role": "user", "content": text}])
        return self.tokenizer.encode(bridge[bridge.index(PLACEHOLDER) + len(PLACEHOLDER):])

    def ask(self, text, on_start, on_token, should_stop):
        """Answer `text` in the current chat. on_token(text, n, perf_counter) runs for each answer token that adds
        text; should_stop() is read between prefill pieces and before each token. Returns the answer's numbers."""
        ids = self._turn_ids(text)
        used = self.generator.token_count() if self.generator is not None else 0
        room = self.max_length - used
        if len(ids) + MIN_ANSWER_ROOM > room:
            if self.turns == 0:
                more = "" if self.max_length >= max(CONTEXTS) else f", or start the chat with -Context {max(CONTEXTS)}"
                raise ChatFull(f"This question is too long for one chat: {len(ids)} tokens, and a chat holds "
                               f"{self.max_length}. Shorten it{more}.")
            raise ChatFull()
        if self.generator is None:
            self._new_generator()
        gpu_before = self.gpu_mib()
        on_start({"prompt_tokens": len(ids), "context_used": used, "max_length": self.max_length,
                  "turn": self.turns + 1, "chat_seed": self.seed})
        t_prefill = time.perf_counter()
        for i in range(0, len(ids), PIECE):
            if should_stop():
                # nothing went in yet: the chat is as it was; else the chat ends (the next question starts a new one)
                reset = i > 0 and used > 0
                if i > 0:
                    self.new_chat()
                return {"tokens": 0, "finish": "stopped", "t_first": None, "t_last": None, "prompt_tokens": len(ids),
                        "chat_reset": reset}
            self.generator.append_tokens(ids[i:i + PIECE])
        self.turns += 1
        t_prefill_end = time.perf_counter()
        stream = self.tokenizer.create_stream()
        n, t_first, t_last, finish = 0, None, None, None
        while True:
            if should_stop():
                finish = "stopped"
                break
            if self.generator.is_done():
                finish = "full"   # the chat reached max_length in the middle of this answer
                break
            self.generator.generate_next_token()
            token = int(self.generator.get_next_tokens()[0])
            now = time.perf_counter()
            if token in self.end_ids:
                finish = "end"
                break
            n += 1
            t_first = t_first or now
            t_last = now
            piece = stream.decode(token)
            if piece:
                on_token(piece, n, now)
        return {"tokens": n, "finish": finish, "t_first": t_first, "t_last": t_last, "prompt_tokens": len(ids),
                "prefill_s": round(t_prefill_end - t_prefill, 4), "context_used": self.generator.token_count(),
                "max_length": self.max_length, "gpu_mib_before": gpu_before, "gpu_mib_after": self.gpu_mib()}

    def close(self):
        self.new_chat()


# --- a stand-in for screen tests: fixed answers, no model -----------------------------------------------------------
# Fixed answers for the test mode (--device fake): the markdown, math, emoji and table forms seen in the model's answers;
# the Japanese words test CJK text.
FAKE_ANSWERS = {
    "sky": """### ☀️ Why the sky is blue

Sunlight looks white, but it is a mix of every color. In the air it meets nitrogen ($\\text{N}_2$) and oxygen ($\\text{O}_2$) molecules, which are much smaller than the light's wavelength.

* **Rayleigh scattering:** small molecules scatter short wavelengths much more than long ones ($\\text{Scattering} \\propto 1/\\lambda^4$), so blue light ($\\approx 450$ nm) scatters about 5.5 $\\times$ more than red light ($\\approx 700$ nm).
* **Light from every direction:** wherever you look in the sky, you see blue light that was bounced toward you.
    * Violet scatters even more, but sunlight has less of it and our eyes are less sensitive to it.
    * At noon the light crosses the least air; at sunset the most.
* **Sunsets 🌅:** the light crosses much more air at sunset, so most of the blue is scattered away before it reaches you (*the reds and oranges remain*).

---

#### 💡 A quick check
| Color | Wavelength | Scattering (vs red) |
| :--- | :--- | :--- |
| **Blue** | 450 nm | about 5.9 $\\times$ |
| **Green** | 550 nm | about 2.6 $\\times$ |
| **Red (reference)** | 700 nm | 1 $\\times$ |

```python
# relative scattering, blue vs red
blue, red = 450e-9, 700e-9
print((red / blue) ** 4)   # about 5.9
```

***

### Summary table

| Time of day | Light's path | Colors that reach you | What you see | Why |
| :--- | :--- | :--- | :--- | :--- |
| **Midday** | Short, straight down through the air | Mostly blue and violet, scattered toward you from every part of the sky | **A blue sky** | Small molecules scatter the short wavelengths in all directions much more than the long ones ($1/\\lambda^4$). |
| **Sunset** | Long, close to the horizon through much more air | Reds, oranges and yellows, after most of the blue has been scattered away on the way | **A red and orange sky** | Over the long path the blue light is scattered *away* from your line of sight before it can reach your eyes. |
""",
    "email": """Here is a polite email you can adapt:

**Subject:** Request to repair the leaking faucet in [Kitchen/Bathroom - specify which one]

Dear [Landlord's Name],

I hope you’re well. I’m writing to let you know that the faucet in my apartment (Unit 4B) has started to leak. Water drips steadily even when the handle is fully closed, and it’s getting worse.

Could you please arrange for a plumber to take a look? I’m usually home after 6 p.m. on weekdays and anytime on Saturday, but I’m happy to adjust to the plumber’s schedule.

Thank you for your help. Please let me know if you need any photos or more details.

Best regards,
[Your Name]
Unit 4B

***

**📌 Before you send:**
1. Say **when the leak started (and how bad it is)**.
2. Offer times when you are available ⏱️.
3. Keep a copy of the email for your records.
""",
    "meal": """## 🥗 A 3-day meal plan

This plan keeps cooking simple: **one big batch on Day 1** feeds two dinners. 👩‍🍳 Cook once, eat twice.

### 🗓️ Day 1
*   **Breakfast:** oatmeal with banana and peanut butter
*   **Lunch:** chicken and vegetable wraps
*   **Dinner:** a big pot of chili (half goes in the fridge for Day 3)

### 🗓️ Day 2
*   **Breakfast:** scrambled eggs on toast
*   **Lunch:** leftover wraps $\\rightarrow$ a quick salad bowl
*   **Dinner:** salmon, rice and broccoli (*25 minutes*), oven at 400°F (200°C)

### 🍽️ Day 3
| Meal | What | Time | Note |
| :--- | :--- | :--- | :--- |
| **Breakfast** | Yogurt with berries and granola | 5 min | No cooking |
| **Lunch** | Tuna sandwiches with a side salad | 10 min | Pack it the night before |
| **Dinner** | The chili from Day 1, with baked potatoes | 40 min | Bake the potatoes at $200^\\circ$C |

***

#### 🛒 Shopping list
* [ ] Bananas, onions, peppers, broccoli, potatoes, berries
* [ ] Chicken breast (500 g), salmon (2 fillets), eggs (6), tuna (1 can)
* [ ] Oats, rice, beans (2 cans), tomatoes (2 cans), tortillas
* [x] Yogurt, cheese, milk (already in the fridge)

Total cooking time: $\\approx 2$ hours over three days.
""",
    "trip": """### 🇯🇵 7 days in Japan ✈️

A first trip works best with **two bases**: Tokyo, then Kyoto. Tokyo $\\rightarrow$ Kyoto by *shinkansen* 🚄 takes about 2 h 15 min.

#### 🗼 Day 1–3: Tokyo
* **Day 1:** arrive, walk around Shibuya at night
* **Day 2:** Senso-ji in Asakusa ⛩️, then Akihabara
    * Go early: the temple is quiet before 8 a.m.
* **Day 3:** a day trip to Hakone (**hot springs (onsen)** and a view of Mt. Fuji 🗺️)

#### ⛩️ Day 4–7: Kyoto and Osaka
* **Day 4:** Fushimi Inari in the late afternoon
* **Day 5:** Arashiyama bamboo grove and Gion
* **Day 6:** Nara ($\\approx 45$ min by train)
* **Day 7:** Osaka $\\rightarrow$ the airport

---

| Item | Per day | Notes |
| :--- | :--- | :--- |
| **Hotel** | ¥12,000 | business hotels near stations |
| **Food** | ¥5,000 | ramen, sushi, convenience stores |
| **Transport** | ¥3,000 | an IC card (Suica or ICOCA) 💡 |

Useful phrases: `sumimasen` (excuse me) and `arigatou gozaimasu` (thank you), ありがとうございます. Temperatures in spring: $10^\\circ$C $\\le$ t $\\le 20^\\circ$C.
""",
    "other": """Sure. Here is a short overview with a few kinds of text, so you can see how answers look.

## A heading

Plain text with **bold**, *italic*, `inline code` and a [link](https://example.com).

- First point
- Second point with a longer sentence that wraps onto the next line when the window is narrow enough to need it.
  - A nested point

1. Step one
2. Step two

| Size | Speed | Note |
|---|---|---|
| small | fast | fits on a laptop |
| large | slower | needs more memory |

Emoji: 😀 🎉 ✅ ☀️ ✈️ — and Japanese: こんにちは、世界。
""",
}


def fake_answer_for(question):
    q = question.lower()
    if "japan" in q or "trip" in q:
        return FAKE_ANSWERS["trip"]
    if "sky" in q:
        return FAKE_ANSWERS["sky"]
    if "email" in q or "landlord" in q:
        return FAKE_ANSWERS["email"]
    if "meal" in q or "shopping" in q:
        return FAKE_ANSWERS["meal"]
    return FAKE_ANSWERS["other"]


class FakeEngine:
    """Streams fixed answers at a set speed, so the window can be tested without a model. It keeps a token count per
    chat like the real one (one piece of text = one token). fail="load": loading fails; fail="answer": an error after
    40 tokens; fail="crash": the process ends after 40 tokens (exit code 9)."""

    def __init__(self, load_s=1.5, first_token_s=0.3, rate=60.0, fail="", context=DEFAULT_CONTEXT, text_file=""):
        self.device, self.label = "fake", device_label("fake")
        self.load_s, self.first_token_s, self.rate = load_s, first_token_s, rate
        self.fail, self.max_length = fail, int(context)
        self.model_dir, self.info, self.used, self.turns = "", {}, 0, 0
        self.text = open(text_file, encoding="utf-8").read() if text_file else ""   # one answer for every question

    @staticmethod
    def pieces(text):
        return re.findall(r"\s*\S{1,4}|\s+", text)

    def load(self):
        time.sleep(self.load_s)
        if self.fail == "load":
            raise EngineError(f"{MODEL_NAME} did not load on {self.label}: a test failure (fail=load). "
                              f"Start it with chat.cmd -Device cpu.")
        self.info = {"device": "fake", "label": self.label, "load_s": self.load_s, "max_length": self.max_length}
        return self.info

    def new_chat(self):
        self.used, self.turns = 0, 0

    def ask(self, text, on_start, on_token, should_stop):
        n_prompt = len(self.pieces(text)) + 8
        if n_prompt + MIN_ANSWER_ROOM > self.max_length - self.used:
            if self.turns == 0:
                raise ChatFull(f"This question is too long for one chat: {n_prompt} tokens, and a chat holds "
                               f"{self.max_length}. Shorten it.")
            raise ChatFull()
        on_start({"prompt_tokens": n_prompt, "context_used": self.used, "max_length": self.max_length,
                  "turn": self.turns + 1, "chat_seed": None})
        self.used += n_prompt
        self.turns += 1
        parts = self.pieces(self.text or fake_answer_for(text))
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < self.first_token_s:
            if should_stop():
                return {"tokens": 0, "finish": "stopped", "t_first": None, "t_last": None, "prompt_tokens": n_prompt}
            time.sleep(0.01)
        n, finish, t_first, t_last, t1 = 0, "end", None, None, time.perf_counter()
        for piece in parts:
            if should_stop():
                finish = "stopped"
                break
            if self.used >= self.max_length:
                finish = "full"
                break
            if n == 40 and self.fail == "answer":
                raise RuntimeError("a test failure in the middle of an answer (fail=answer)")
            if n == 40 and self.fail == "crash":
                sys.stderr.write("fake engine: ending the process on purpose (fail=crash)\n")
                sys.stderr.flush()
                leave(9)
            n += 1
            self.used += 1
            now = time.perf_counter()
            t_first = t_first or now
            t_last = now
            on_token(piece, n, now)
            delay = t1 + n / self.rate - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        self.used += 1   # the end-of-turn token
        return {"tokens": n, "finish": finish, "t_first": t_first, "t_last": t_last, "prompt_tokens": n_prompt,
                "context_used": self.used, "max_length": self.max_length}

    def close(self):
        pass


def create(device, model_dir=None, context=DEFAULT_CONTEXT, fake=None, adapter=0, free_check=None):
    """free_check: the GPU memory check before loading (default: when "auto" is asked here)."""
    if free_check is None:
        free_check = device == "auto"
    device = resolve_device(device)
    if device == "fake":
        return FakeEngine(context=context, **(fake or {}))
    return GenAIEngine(device, model_dir, context, adapter, free_check)


# --- the child process ----------------------------------------------------------------------------------------------
def take_stdout():
    """Keep the pipe to the window for the JSON lines, and point fd 1 (and Windows' standard output handle) at stderr,
    so text that a native library prints cannot break a line of the protocol."""
    sys.stdout.flush()
    fd = os.dup(1)
    os.dup2(2, 1)
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32")
        k32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
        k32.SetStdHandle(wintypes.DWORD(-11 & 0xFFFFFFFF), wintypes.HANDLE(msvcrt.get_osfhandle(2)))
    sys.stdout = sys.stderr
    return os.fdopen(fd, "w", encoding="utf-8", newline="\n")


def leave(code):
    """End this process at once. On Windows without the loaded DLLs' own teardown (TerminateProcess): with the model on
    the GPU, that teardown ended in an access violation after the window's quit (seen with CUDA on this PC)."""
    try:
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.TerminateProcess(k32.GetCurrentProcess(), code)
    os._exit(code)


def stdin_lines():
    """The window's lines on standard input. On Windows the pipe is read only when bytes are waiting (PeekNamedPipe):
    a read left pending on standard input holds up every other call about that handle in this process, such as the
    C runtime start-up of a DLL loaded later (numpy's first import, during a question, waited until the window's next
    line arrived)."""
    if sys.platform != "win32":
        yield from sys.stdin.buffer
        return
    import ctypes
    import msvcrt
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32")
    k32.PeekNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, wintypes.LPDWORD,
                                  wintypes.LPDWORD, wintypes.LPDWORD]
    k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, wintypes.LPDWORD, wintypes.LPVOID]
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(sys.stdin.fileno()))
    waiting, got = wintypes.DWORD(), wintypes.DWORD()
    if not k32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(waiting), None):
        yield from sys.stdin.buffer   # not a pipe (a console or a file): plain reads
        return
    chunk = ctypes.create_string_buffer(65536)
    rest = b""
    while True:
        if not k32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(waiting), None):
            break   # the window closed its end of the pipe
        if not waiting.value:
            time.sleep(0.01)
            continue
        if not k32.ReadFile(handle, chunk, min(waiting.value, len(chunk)), ctypes.byref(got), None) or not got.value:
            break
        rest += chunk.raw[:got.value]
        while b"\n" in rest:
            line, rest = rest.split(b"\n", 1)
            yield line + b"\n"
    if rest:
        yield rest


def watch_parent(pid):
    """End this process when the window's process ends (with the window's Job Object, nothing is left behind)."""
    if sys.platform != "win32" or not pid:
        return
    import ctypes

    k32 = ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    handle = k32.OpenProcess(0x00100000, False, int(pid))   # SYNCHRONIZE
    if not handle:
        leave(3)   # the window is already gone

    def wait():
        k32.WaitForSingleObject(ctypes.c_void_p(handle), 0xFFFFFFFF)
        leave(3)

    threading.Thread(target=wait, daemon=True).start()


def serve(args):
    out = take_stdout()
    lock = threading.Lock()

    def send(obj):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with lock:
            try:
                out.write(line)
                out.flush()
            except (OSError, ValueError):
                leave(0)   # the window is gone: nobody to answer

    watch_parent(args.parent_pid)
    ops, stop_ids, ended = queue.Queue(), set(), threading.Event()

    def read_ops():
        for raw in stdin_lines():
            t_line = time.perf_counter()   # the question's clock starts when its line arrives
            try:
                msg = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            msg["_t"] = t_line
            if msg.get("op") == "stop":
                stop_ids.add(msg.get("id"))
            else:
                ops.put(msg)
        ended.set()          # the window closed its end of the pipe
        ops.put({"op": "quit"})

    threading.Thread(target=read_ops, daemon=True).start()
    fake = None
    if args.device == "fake":
        fake = {"load_s": args.fake_load_s, "first_token_s": args.fake_first_token_s, "rate": args.fake_rate,
                "fail": args.fake_fail, "text_file": args.fake_text}
    engine = create(args.device, args.model_dir or None, args.context, fake, args.adapter,
                    free_check=args.device == "auto" or args.auto_picked)
    send({"ev": "hello", "pid": os.getpid(), "device": engine.device, "label": engine.label,
          "telemetry_off": os.environ.get("ORT_DISABLE_TELEMETRY") == "1", "power_throttling": prelude.POWER_THROTTLING})
    try:
        send({"ev": "loaded", "info": engine.load()})
    except EngineError as error:
        send({"ev": "load_failed", "error": str(error)})
        out.close()
        leave(2)
    except Exception as error:   # noqa: BLE001 - shown to the user as one line
        send({"ev": "load_failed", "error": f"{MODEL_NAME} did not load: {type(error).__name__}: "
                                            f"{first_line(error)[:200]}"})
        out.close()
        leave(2)
    while True:
        msg = ops.get()
        op = msg.get("op")
        if op == "quit":
            break
        if op == "new_chat":
            engine.new_chat()
            continue
        if op != "ask":
            continue
        turn = msg.get("id")
        t_recv = msg.get("_t") or time.perf_counter()

        def started(info, turn=turn):
            send({"ev": "start", "id": turn, **info})

        def token(text, n, now, turn=turn, t0=t_recv):
            send({"ev": "token", "id": turn, "text": text, "n": n, "t": round(now - t0, 4)})

        try:
            stats = engine.ask(msg.get("text", ""), started, token, lambda turn=turn: turn in stop_ids or ended.is_set())
        except ChatFull as error:
            send({"ev": "full", "id": turn, "error": str(error)})
            continue
        except Exception as error:   # noqa: BLE001 - shown to the user, and the chat starts again
            engine.new_chat()
            low = str(error).lower()
            if any(w in low for w in ("out of memory", "outofmemory", "failed to allocate", "bad_alloc")):
                text = (f"{engine.label} ran out of memory in this answer. Your next question starts a new chat; "
                        f"close other apps that use it if this happens again.")
            else:
                text = (f"The answer stopped with an error ({type(error).__name__}: {first_line(error)[:160]}). "
                        f"Your next question starts a new chat.")
            send({"ev": "error", "id": turn, "new_chat": True, "error": text})
            continue
        for key in ("t_first", "t_last"):
            if stats.get(key) is not None:
                stats[key] = round(stats[key] - t_recv, 4)
        stats["seconds"] = stats["t_last"] if stats.get("t_last") is not None else round(time.perf_counter() - t_recv, 4)
        send({"ev": "done", "id": turn, "stats": stats})
    engine.close()
    out.close()
    leave(0)   # the WebGPU plugin 0.5.0 can keep a finished process alive, and CUDA's teardown crashed


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Gemma 4 E4B for the chat window (see the top of this file).")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--resolve-device", metavar="DEVICE")
    ap.add_argument("--check-model-dir", metavar="FOLDER")
    ap.add_argument("--label", metavar="DEVICE")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "webgpu", "cpu", "fake"])
    ap.add_argument("--adapter", type=int, default=0, help="with --device webgpu: the WebGPU adapter number")
    ap.add_argument("--auto-picked", action="store_true",
                    help="the device came from auto: check the free GPU memory before loading on CUDA")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT, choices=CONTEXTS)
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument("--fake-load-s", type=float, default=1.5, help=argparse.SUPPRESS)
    ap.add_argument("--fake-rate", type=float, default=60.0, help=argparse.SUPPRESS)
    ap.add_argument("--fake-first-token-s", type=float, default=0.3, help=argparse.SUPPRESS)
    ap.add_argument("--fake-fail", default="", choices=["", "load", "answer", "crash"], help=argparse.SUPPRESS)
    ap.add_argument("--fake-text", default="", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    if args.resolve_device:
        print(resolve_device(args.resolve_device))
        return 0
    if args.label:
        print(device_label(resolve_device(args.label), args.adapter))
        return 0
    if args.check_model_dir:
        try:
            check_model_dir(args.check_model_dir)
        except EngineError as error:
            print(error)
            return 1
        print("ok")
        return 0
    if args.serve:
        serve(args)
        return 0
    print("Run the chat with chat.cmd, or see the top of this file.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

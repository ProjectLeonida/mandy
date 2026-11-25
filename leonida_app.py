"""Leonida Builder: a small desktop window around make_game.py (standard library only, needs tkinter).

Pick the game folder and output folder, tick the stages you want, press Build. The build runs as a child
process (`make_game.py --no-tui ...`) and its log streams into the window; Stop ends it.
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
STAGES = ["check", "shaders", "assets", "runtime", "pipelines"]


def default_out():
    docs = Path.home() / "Documents"
    return str((docs if docs.is_dir() else Path.home()) / "Leonida")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Leonida - Game Builder")
        self.geometry("820x600")
        ico = HERE / "assets" / "logo.png"
        if ico.is_file():
            try:
                self._icon = tk.PhotoImage(file=str(ico)).subsample(8)
                self.iconphoto(True, self._icon)
            except tk.TclError:
                pass
        self.proc = None
        self.q = queue.Queue()
        self.game = tk.StringVar()
        self.out = tk.StringVar(value=default_out())
        self.jobs = tk.IntVar(value=max(1, (os.cpu_count() or 2) // 2))
        self.stage_vars = {s: tk.BooleanVar(value=True) for s in STAGES}
        self._ui()
        self.after(100, self._pump)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _ui(self):
        f = ttk.Frame(self, padding=10)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(4, weight=1)
        ttk.Label(f, text="Game folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.game).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="Browse...", command=lambda: self._pick(self.game)).grid(row=0, column=2)
        ttk.Label(f, text="(leave empty to find it automatically)").grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(f, text="Output folder").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(f, textvariable=self.out).grid(row=2, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(f, text="Browse...", command=lambda: self._pick(self.out)).grid(row=2, column=2, pady=(6, 0))
        opts = ttk.LabelFrame(f, text="Stages", padding=6)
        opts.grid(row=3, column=0, columnspan=3, sticky="ew", pady=8)
        for s in STAGES:
            ttk.Checkbutton(opts, text=s, variable=self.stage_vars[s]).pack(side="left", padx=6)
        ttk.Label(opts, text="Jobs").pack(side="left", padx=(20, 4))
        ttk.Spinbox(opts, from_=1, to=128, width=4, textvariable=self.jobs).pack(side="left")
        self.log = tk.Text(f, wrap="none", state="disabled", height=16)
        self.log.grid(row=4, column=0, columnspan=3, sticky="nsew")
        sb = ttk.Scrollbar(f, command=self.log.yview)
        sb.grid(row=4, column=3, sticky="ns")
        self.log["yscrollcommand"] = sb.set
        bar = ttk.Frame(f)
        bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.status = ttk.Label(bar, text="Ready")
        self.status.pack(side="left")
        self.btn_build = ttk.Button(bar, text="Build", command=lambda: self._run([]))
        self.btn_check = ttk.Button(bar, text="Check game files", command=lambda: self._run(["--check"]))
        self.btn_stop = ttk.Button(bar, text="Stop", command=self._stop, state="disabled")
        for b in (self.btn_stop, self.btn_check, self.btn_build):
            b.pack(side="right", padx=3)

    def _pick(self, var):
        d = filedialog.askdirectory(initialdir=var.get() or str(Path.home()))
        if d:
            var.set(d)

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _run(self, extra):
        if self.proc:
            return
        stages = [s for s in STAGES if self.stage_vars[s].get()]
        if not stages and "--check" not in extra:
            messagebox.showerror("Leonida", "Tick at least one stage.")
            return
        cmd = [sys.executable, "-I", str(HERE / "make_game.py"), "--no-tui", "--jobs", str(self.jobs.get())]
        if self.game.get().strip():
            cmd += ["--game", self.game.get().strip()]
        if self.out.get().strip():
            cmd += ["--out", self.out.get().strip()]
        if "--check" in extra:
            cmd += extra
        elif len(stages) != len(STAGES):
            cmd += ["--stages", ",".join(stages)]
        self._append("$ " + " ".join(cmd) + "\n")
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace")
        except OSError as e:
            self._append(f"Could not start: {e}\n")
            self.proc = None
            return
        self._busy(True)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, p):
        for line in p.stdout:
            self.q.put(line)
        self.q.put(("done", p.wait()))

    def _pump(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    self.proc = None
                    self._busy(False)
                    ok = item[1] == 0
                    self.status.configure(text="Done." if ok else f"Stopped (exit code {item[1]}).")
                    self._append("\nDone.\n" if ok else f"\nThe build stopped with exit code {item[1]}.\n")
                else:
                    self._append(item)
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def _busy(self, on):
        self.status.configure(text="Working..." if on else "Ready")
        self.btn_build.configure(state="disabled" if on else "normal")
        self.btn_check.configure(state="disabled" if on else "normal")
        self.btn_stop.configure(state="normal" if on else "disabled")

    def _stop(self):
        if self.proc:
            self.proc.terminate()

    def _close(self):
        if self.proc and not messagebox.askyesno("Leonida", "A build is running. Stop it and quit?"):
            return
        self._stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()

#!/usr/bin/env python3
"""
resume_agent_gui_final.py
Chat-style Tkinter GUI (left chat, right results table) for resume filtering.

Features:
- Load resume folder (PDF/DOCX)
- Chat-like natural language input to specify filters ("3+ years Java", etc.)
- Optional OpenAI integration (new client) for parsing/filter interpretation and skill extraction
- Heuristic fallbacks when OpenAI is not configured
- Strict skill matching: if user requests Java, only resumes with Java are eligible
- Reset button: clears filters/results but keeps loaded resumes (Option 2)
- Preview resume text on row double-click
- Export CSV of results; open results folder
"""

import os
import re
import json
import shutil
import pdfplumber
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from docx import Document
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont
import pandas as pd

# Try new OpenAI client (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    ai_client = OpenAI()  # will read OPENAI_API_KEY from environment
except Exception:
    OPENAI_AVAILABLE = False
    ai_client = None

# Config
COMMON_SKILLS = [
    "java","spring","spring boot","hibernate","python",".net","c#","c++",
    "javascript","typescript","react","angular","node","sql","mysql","postgres",
    "aws","azure","gcp","docker","kubernetes","jenkins","git","ci/cd","terraform",
    "microservices","rest","api","linux"
]
MODEL_NAME = "gpt-4o"  # change if you prefer

# -----------------------
# Text extraction helpers (pdfplumber + python-docx)
# -----------------------
def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        return ""
    return text

def extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(str(path))
    elif ext == ".docx":
        return extract_text_from_docx(str(path))
    return ""

# -----------------------
# Experience parsing (date ranges + numeric fallback)
# -----------------------
def parse_experience_from_text(text: str) -> float:
    text_single = text.replace("\n", " ")
    months_map = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
    }
    range_pattern = re.compile(
        r'([A-Za-z]{3,9})\.?\s+(\d{4})\s*(?:–|—|-|to)\s*(Present|present|Now|now|Current|current|[A-Za-z]{3,9})\.?\s*(\d{4})?'
    )
    intervals = []
    now = datetime.now()
    for m in range_pattern.finditer(text_single):
        s_mon, s_year, e_mon, e_year = m.groups()
        try:
            s_m = months_map.get(s_mon[:3].lower(), 1)
            s_y = int(s_year)
            if e_mon.lower() in ("present","now","current"):
                e_m, e_y = now.month, now.year
            else:
                e_m = months_map.get(e_mon[:3].lower(), 1)
                e_y = int(e_year) if e_year else s_y
            start_idx = s_y * 12 + s_m - 1
            end_idx = e_y * 12 + e_m - 1
            if end_idx >= start_idx:
                intervals.append((start_idx, end_idx))
        except Exception:
            continue
    total_months = 0
    if intervals:
        intervals.sort()
        merged = []
        cur_s, cur_e = intervals[0]
        for s,e in intervals[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        for s,e in merged:
            total_months += (e - s + 1)
        years = round(total_months / 12.0, 2)
        if years > 0:
            return years
    # fallback numeric mentions like "3 years", "3+ years"
    numeric_matches = re.findall(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)', text_single.lower())
    if numeric_matches:
        try:
            return round(float(max((float(x) for x in numeric_matches))), 2)
        except Exception:
            pass
    return 0.0

# -----------------------
# Heuristic: detect required skills from natural-language query
# -----------------------
def extract_required_skills_from_query(nl: str) -> List[str]:
    q = nl.lower()
    found = []
    for s in COMMON_SKILLS:
        if re.search(rf'\b{re.escape(s)}\b', q):
            found.append(s)
    return sorted(set(found))

# -----------------------
# Interpret filter (AI or heuristic)
# -----------------------
def interpret_filter_with_ai(nl_filter: str) -> Dict[str, Any]:
    # fallback heuristic
    def heuristic(nl: str) -> Dict[str, Any]:
        m = re.search(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b', nl.lower())
        min_exp = float(m.group(1)) if m else None
        required = extract_required_skills_from_query(nl)
        return {"min_experience": min_exp, "required_skills": required, "optional_skills": [], "must_have_phrases": []}

    if not OPENAI_AVAILABLE or ai_client is None:
        return heuristic(nl_filter)

    system = (
        "You are a JSON extractor. Convert the user's single-line job requirement into valid JSON "
        "with keys: min_experience (number or null), required_skills (list), optional_skills (list), must_have_phrases (list). "
        "Return JSON only."
    )
    user = f"Requirement: {nl_filter}\nReturn JSON."
    try:
        resp = ai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=300
        )
        text = resp.choices[0].message["content"]
        m = re.search(r'\{.*\}', text, flags=re.DOTALL)
        jtext = m.group(0) if m else text
        parsed = json.loads(jtext)
        return {
            "min_experience": float(parsed.get("min_experience")) if parsed.get("min_experience") not in (None,"","null") else None,
            "required_skills": parsed.get("required_skills") or [],
            "optional_skills": parsed.get("optional_skills") or [],
            "must_have_phrases": parsed.get("must_have_phrases") or []
        }
    except Exception:
        return heuristic(nl_filter)

# -----------------------
# Skill extraction (AI + heuristic fallback)
# -----------------------
def extract_skills_from_text(text: str) -> List[str]:
    snippet = text[:3500]
    if OPENAI_AVAILABLE and ai_client:
        prompt = (
            "Extract technical skills from this resume and return ONLY a JSON array of lowercase skill strings.\n\n"
            "Resume:\n" + snippet + "\n\nReturn JSON array only."
        )
        try:
            resp = ai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=300
            )
            content = resp.choices[0].message["content"].strip()
            m = re.search(r'\[.*\]', content, flags=re.DOTALL)
            arr_text = m.group(0) if m else content
            skills = json.loads(arr_text)
            # sanitize
            return sorted({s.strip().lower() for s in skills if isinstance(s, str) and s.strip()})
        except Exception:
            pass
    # fallback heuristic: check COMMON_SKILLS presence
    found = set()
    lower = text.lower()
    for sk in COMMON_SKILLS:
        if re.search(rf'\b{re.escape(sk)}\b', lower):
            found.add(sk)
    return sorted(found)

# -----------------------
# Evaluate resumes
# -----------------------
def evaluate_resumes(paths: List[Path], filter_struct: Dict[str, Any]) -> List[dict]:
    results = []
    for p in paths:
        text = extract_text(p)
        exp = parse_experience_from_text(text)
        skills = extract_skills_from_text(text)
        required = [r.lower() for r in filter_struct.get("required_skills", []) if r]
        # strict skill match: required skills must appear in resume skills
        if required:
            matched = [s for s in skills if any(req == s or req in s for req in required)]
            eligible_skills = all(any(req == s or req in s for s in skills) for req in required)
        else:
            matched = skills
            eligible_skills = True
        min_exp = filter_struct.get("min_experience")
        eligible_exp = True if (min_exp is None) else (exp >= min_exp)
        status = "✅ Eligible" if (eligible_skills and eligible_exp) else "❌ Not Eligible"
        results.append({
            "Candidate": p.stem,
            "Path": str(p),
            "Experience": round(exp,2),
            "Matched_Skills": ", ".join(matched),
            "All_Skills": ", ".join(skills),
            "Status": status,
            "Details": {"skills_ok": eligible_skills, "exp_ok": eligible_exp}
        })
    # optionally sort eligible first, then by experience
    results.sort(key=lambda r: (r["Status"].startswith("✅")*-1, r["Experience"]), reverse=True)
    return results

# -----------------------
# GUI: Chat (left) + Results table (right)
# -----------------------
class ResumeAgentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI-Agent Resume Filter — GUI")
        self.geometry("1100x760")
        self.configure(bg="#f4f7fb")
        self.resumes: List[Path] = []
        self.results: List[dict] = []

        # fonts & style
        self.header_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.small_font = tkfont.Font(family="Segoe UI", size=10)
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("Header.TLabel", background="#2E6AA6", foreground="white", font=self.header_font, padding=10)
        style.configure("Card.TFrame", background="white", relief="flat")

        header = ttk.Label(self, text="AI-Agent Resume Filter", style="Header.TLabel", anchor="center")
        header.pack(fill="x", padx=8, pady=(8,6))

        main = ttk.Frame(self, padding=10, style="Card.TFrame")
        main.pack(fill="both", expand=True, padx=8, pady=6)

        # Top controls
        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=(0,8))
        ttk.Button(controls, text="Select Folder", command=self.select_folder).pack(side="left")
        ttk.Button(controls, text="Run Agent", command=self.run_agent).pack(side="left", padx=(8,0))
        ttk.Button(controls, text="Reset", command=self.reset_filters_keep_resumes).pack(side="left", padx=8)
        ttk.Button(controls, text="Export CSV", command=self.export_csv).pack(side="left", padx=8)
        ttk.Button(controls, text="Open Results Folder", command=self.open_results_folder).pack(side="left", padx=8)

        self.folder_label = ttk.Label(controls, text="No folder selected", font=self.small_font)
        self.folder_label.pack(side="left", padx=12)

        # Layout frames: left chat, right results
        content = ttk.Frame(main)
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content)
        left.pack(side="left", fill="both", expand=True, padx=(0,8))
        right = ttk.Frame(content, width=460)
        right.pack(side="right", fill="both", expand=False)

        # Chat area (left)
        ttk.Label(left, text="Chat (type natural-language filter):", font=self.small_font).pack(anchor="w")
        self.chat_box = tk.Text(left, height=28, wrap="word", state="disabled", bg="#ffffff", padx=8, pady=6)
        self.chat_box.pack(fill="both", expand=True)
        entry_frame = ttk.Frame(left)
        entry_frame.pack(fill="x", pady=(6,0))
        self.nl_var = tk.StringVar()
        self.nl_entry = ttk.Entry(entry_frame, textvariable=self.nl_var, width=80)
        self.nl_entry.pack(side="left", padx=(0,8), fill="x", expand=True)
        self.nl_entry.bind("<Return>", self.on_nl_enter)
        ttk.Button(entry_frame, text="Send", command=self.on_send).pack(side="left")

        # Status label
        self.status_label = ttk.Label(left, text="Load a folder first, then type a request (e.g., '3+ years Java').", font=self.small_font)
        self.status_label.pack(anchor="w", pady=(6,0))

        # Results area (right)
        ttk.Label(right, text="Results", font=self.small_font).pack(anchor="w")
        cols = ("Candidate", "Experience", "Matched_Skills", "Status")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=20)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=120, minwidth=100)
        self.tree.pack(fill="both", expand=True)
        # scrollbars
        vsb = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(right, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        # tags and style
        self.tree.tag_configure('evenrow', background='#f7fbff')
        self.tree.tag_configure('oddrow', background='#ffffff')
        self.tree.tag_configure('eligible', foreground='#0b7a3b')
        self.tree.tag_configure('noteligible', foreground='#b02a2a')
        self.tree.bind("<Double-1>", self.on_row_double_click)

    # -----------------------
    # Folder selection
    # -----------------------
    def select_folder(self):
        folder = filedialog.askdirectory(title="Select resume folder (PDF/DOCX)")
        if not folder:
            return
        p = Path(folder)
        self.resumes = [p / f for f in os.listdir(p) if f.lower().endswith((".pdf", ".docx"))]
        self.folder_label.config(text=f"{len(self.resumes)} resumes loaded from: {folder}")
        self.append_chat("System", f"Loaded {len(self.resumes)} resumes from {folder}")
        self.status_label.config(text="Ready — enter your filter and press Send or Run Agent")

    # -----------------------
    # Chat send/enter handlers
    # -----------------------
    def on_nl_enter(self, event=None):
        self.on_send()

    def on_send(self):
        nl = self.nl_var.get().strip()
        if not nl:
            return
        if nl.lower() == "reset":
            self.reset_filters_keep_resumes()
            self.nl_var.set("")
            return
        self.append_chat("You", nl)
        # treat as immediate run (same as Run Agent)
        self.run_agent(nl)
        self.nl_var.set("")

    # -----------------------
    # Run agent: accept optional nl (if called from Send) else read entry
    # -----------------------
    def run_agent(self, nl_override: Optional[str]=None):
        if not self.resumes:
            messagebox.showwarning("No resumes", "Please select a folder with resumes first.")
            return
        nl = nl_override if nl_override is not None else self.nl_var.get().strip()
        if not nl:
            messagebox.showwarning("No input", "Please enter a natural-language requirement.")
            return

        self.status_label.config(text="Interpreting requirement and evaluating resumes...")
        self.update()

        # Interpret filter
        filter_struct = interpret_filter_with_ai(nl)
        # Ensure explicit skills are captured
        if not filter_struct.get("required_skills"):
            explicit = extract_required_skills_from_query(nl)
            if explicit:
                filter_struct["required_skills"] = explicit

        # Evaluate resumes
        results = evaluate_resumes(self.resumes, filter_struct)
        self.results = results

        # display only final table (no extra logs)
        self.populate_table(results)

        # show small summary in chat
        total = len(results)
        elig = sum(1 for r in results if r["Status"].startswith("✅"))
        self.append_chat("Agent", f"Filter applied. {elig}/{total} eligible. (min_experience={filter_struct.get('min_experience')}, required_skills={filter_struct.get('required_skills')})")
        self.status_label.config(text=f"Evaluation done — {elig}/{total} eligible")

    # -----------------------
    # Populate results table
    # -----------------------
    def populate_table(self, results: List[dict]):
        # clear
        for r in self.tree.get_children():
            self.tree.delete(r)
        for i, r in enumerate(results):
            tag = "evenrow" if i % 2 == 0 else "oddrow"
            status_tag = "eligible" if r["Status"].startswith("✅") else "noteligible"
            self.tree.insert("", "end", values=(r["Candidate"], r["Experience"], r["Matched_Skills"], r["Status"]), tags=(tag, status_tag))

    # -----------------------
    # Reset (Option 2): keep resumes, clear filters/results/chat
    # -----------------------
    def reset_filters_keep_resumes(self):
        self.nl_var.set("")
        for r in self.tree.get_children():
            self.tree.delete(r)
        self.results = []
        # clear chat
        self.chat_box.config(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.config(state="disabled")
        self.status_label.config(text="✔ Cleared filters/results. Resumes still loaded. Enter new requirement.")
        self.append_chat("Agent", "Cleared filters/results — resumes remain loaded. Enter new instructions.")

    # -----------------------
    # Chat append (left panel)
    # -----------------------
    def append_chat(self, who: str, text: str):
        self.chat_box.config(state="normal")
        self.chat_box.insert("end", f"{who}: {text}\n\n")
        self.chat_box.see("end")
        self.chat_box.config(state="disabled")

    # -----------------------
    # Export CSV of last results
    # -----------------------
    def export_csv(self):
        if not self.results:
            messagebox.showinfo("No data", "No results to export.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not save_path:
            return
        try:
            df = pd.DataFrame(self.results)
            # keep relevant cols
            keep = ["Candidate","Experience","Matched_Skills","All_Skills","Status"]
            df_out = df[keep] if set(keep).issubset(df.columns) else df
            df_out.to_csv(save_path, index=False)
            messagebox.showinfo("Exported", f"Results exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # -----------------------
    # Open latest results folder
    # -----------------------
    def open_results_folder(self):
        base = Path("results")
        if not base.exists():
            messagebox.showinfo("No results", "No results folder found yet.")
            return
        latest = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
        if not latest:
            messagebox.showinfo("No results", "No results folder found yet.")
            return
        path_to_open = latest[0]
        try:
            if os.name == "nt":
                os.startfile(path_to_open)
            elif os.name == "posix":
                os.system(f'xdg-open "{path_to_open}"')
            else:
                messagebox.showinfo("Path", str(path_to_open))
        except Exception:
            messagebox.showinfo("Path", str(path_to_open))

    # -----------------------
    # Row double-click preview
    # -----------------------
    def on_row_double_click(self, event):
        item = self.tree.selection()
        if not item:
            return
        vals = self.tree.item(item[0])["values"]
        candidate = vals[0]
        # find result by candidate
        r = next((x for x in self.results if x["Candidate"] == candidate), None)
        if not r:
            messagebox.showinfo("Preview", "No data available for this candidate.")
            return
        # show preview window with extracted text
        path = Path(r["Path"])
        try:
            text = extract_text(path)
        except Exception:
            text = ""
        preview = tk.Toplevel(self)
        preview.title(f"Preview — {candidate}")
        preview.geometry("800x600")
        txt = tk.Text(preview, wrap="word")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", text or "(No extracted text)")
        txt.config(state="disabled")

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    app = ResumeAgentGUI()
    app.mainloop()

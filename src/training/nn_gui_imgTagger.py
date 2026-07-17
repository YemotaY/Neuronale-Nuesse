# -*- coding: utf-8 -*-
#
#  nn_gui_imgTagger.py
#
#  Created by YemotaY on 2026-07-17.
#  MIT Licensed, 2026. All rights reserved.
#
#   b59190308@gmail.com
#
# pylint: disable-msg=F0401

import tkinter as tk
from tkinter import ttk, messagebox
from os import path, listdir, walk

from PIL import Image, ImageTk

'''
Ein Bild-Tagger auf Basis von Tkinter + Pillow.

Links: gruppierte, ein-/ausklappbare, scrollbare Übersicht der Bild-Ordner.
Rechts: Vorschau des gewählten Bildes, darunter ein Tag-Feld + Speichern.
Beim Speichern werden die Tags in die EXIF-Metadaten geschrieben
(XPKeywords 0x9C9E + ImageDescription 0x010E).

Abhängigkeiten:
pillow-12.3.0
'''

# EXIF-Tag-IDs
_EXIF_XPKEYWORDS = 0x9C9E        # Windows-Keywords (UTF-16-LE)
_EXIF_IMAGEDESCRIPTION = 0x010E  # ASCII/UTF-8 Beschreibung

_IMG_EXTENSIONS = (".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Wurzelordner der Bilder: .../src/img
_IMG_ROOT = path.normpath(path.join(path.dirname(path.abspath(__file__)), "..", "img"))

_PREVIEW_MAX = (640, 640)


class CollapsibleSection(tk.Frame):
    '''Ein ein-/ausklappbarer Abschnitt mit Kopfzeile und Inhaltsbereich.'''

    def __init__(self, parent, title, indent=0, expanded=False):
        super().__init__(parent, bg="#f0f0f0")
        self._expanded = expanded

        self.header = tk.Label(
            self,
            text=self._label(title),
            anchor="w",
            bg="#dfe6ee",
            fg="#1b2733",
            padx=6 + indent * 14,
            pady=4,
            font=("TkDefaultFont", 9, "bold"),
            cursor="hand2",
        )
        self.header.pack(fill="x")
        self.header.bind("<Button-1>", lambda _e: self.toggle())

        self._title = title
        self.body = tk.Frame(self, bg="#f0f0f0")
        if self._expanded:
            self.body.pack(fill="x")

    def _label(self, title):
        arrow = "▼" if self._expanded else "▶"
        return f"{arrow}  {title}"

    def toggle(self):
        self._expanded = not self._expanded
        self.header.config(text=self._label(self._title))
        if self._expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()


class ImgTaggerApp:
    '''Hauptfenster des Bild-Taggers.'''

    def __init__(self, root, img_root=_IMG_ROOT):
        self.root = root
        self.img_root = img_root
        self.current_path = None       # Pfad des aktuell geladenen Bildes
        self._preview_imgtk = None     # Referenz halten, sonst GC
        self._selected_label = None    # aktuell markierter Datei-Eintrag

        self.root.title("Neuronale Nüsse – Image Tagger")
        self.root.geometry("1100x720")
        self.root.minsize(820, 520)

        self._build_layout()
        self._populate_tree()

    # ------------------------------------------------------------------ Layout
    def _build_layout(self):
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # --- linke Seite: scrollbare Ordner-/Datei-Übersicht ---
        left = tk.Frame(paned, width=320, bg="#f0f0f0")
        paned.add(left, weight=0)

        tk.Label(
            left, text="Ordner", anchor="w", bg="#c7d2dd",
            fg="#1b2733", padx=8, pady=6, font=("TkDefaultFont", 10, "bold"),
        ).pack(fill="x")

        canvas_wrap = tk.Frame(left, bg="#f0f0f0")
        canvas_wrap.pack(fill="both", expand=True)

        self.tree_canvas = tk.Canvas(canvas_wrap, bg="#f0f0f0", highlightthickness=0, width=300)
        vbar = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.tree_canvas.yview)
        self.tree_canvas.configure(yscrollcommand=vbar.set)

        vbar.pack(side="right", fill="y")
        self.tree_canvas.pack(side="left", fill="both", expand=True)

        self.tree_inner = tk.Frame(self.tree_canvas, bg="#f0f0f0")
        self._tree_window = self.tree_canvas.create_window((0, 0), window=self.tree_inner, anchor="nw")

        self.tree_inner.bind(
            "<Configure>",
            lambda _e: self.tree_canvas.configure(scrollregion=self.tree_canvas.bbox("all")),
        )
        self.tree_canvas.bind(
            "<Configure>",
            lambda e: self.tree_canvas.itemconfig(self._tree_window, width=e.width),
        )
        self._bind_mousewheel(self.tree_canvas)

        # --- rechte Seite: Vorschau + Tag-Editor ---
        right = tk.Frame(paned, bg="#ffffff")
        paned.add(right, weight=1)

        self.preview_label = tk.Label(right, bg="#20262e", fg="#8a97a5",
                                      text="Kein Bild ausgewählt")
        self.preview_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.info_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.info_var, anchor="w", bg="#ffffff",
                 fg="#556") .pack(fill="x", padx=12)

        editor = tk.Frame(right, bg="#ffffff")
        editor.pack(fill="x", padx=12, pady=(6, 12))

        tk.Label(editor, text="Tags (mit Komma oder Semikolon getrennt):",
                 anchor="w", bg="#ffffff").pack(fill="x")

        self.tag_entry = tk.Entry(editor, font=("TkDefaultFont", 11))
        self.tag_entry.pack(fill="x", pady=(2, 8))
        self.tag_entry.bind("<Return>", lambda _e: self.save_tags())

        btn_row = tk.Frame(editor, bg="#ffffff")
        btn_row.pack(fill="x")

        self.save_btn = tk.Button(btn_row, text="Speichern", command=self.save_tags,
                                  state="disabled", width=14)
        self.save_btn.pack(side="right")

        self.status_var = tk.StringVar(value="Bereit.")
        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 bg="#eceff2", fg="#333", padx=8, pady=3).pack(fill="x", side="bottom")

    def _bind_mousewheel(self, widget):
        # Windows / macOS
        widget.bind_all("<MouseWheel>", self._on_mousewheel)
        # Linux
        widget.bind_all("<Button-4>", self._on_mousewheel)
        widget.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.tree_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.tree_canvas.yview_scroll(1, "units")
        else:
            self.tree_canvas.yview_scroll(int(-event.delta / 120), "units")

    # ------------------------------------------------------------ Baum füllen
    def _populate_tree(self):
        for child in self.tree_inner.winfo_children():
            child.destroy()

        if not path.isdir(self.img_root):
            tk.Label(self.tree_inner, text=f"Ordner nicht gefunden:\n{self.img_root}",
                     bg="#f0f0f0", fg="#a33", justify="left").pack(fill="x", padx=6, pady=6)
            return

        groups = sorted(d for d in listdir(self.img_root)
                        if path.isdir(path.join(self.img_root, d)))

        if not groups:
            # Flacher Ordner ohne Untergruppen
            self._build_folder_section(self.tree_inner, self.img_root, "Bilder", indent=0)
            return

        for group in groups:
            group_path = path.join(self.img_root, group)
            section = CollapsibleSection(self.tree_inner, group, indent=0, expanded=False)
            section.pack(fill="x")

            subdirs = sorted(d for d in listdir(group_path)
                             if path.isdir(path.join(group_path, d)))

            if subdirs:
                for sub in subdirs:
                    self._build_folder_section(section.body, path.join(group_path, sub),
                                               sub, indent=1)
            else:
                self._add_files(section.body, group_path, indent=1)

    def _build_folder_section(self, parent, folder_path, title, indent):
        sub = CollapsibleSection(parent, title, indent=indent, expanded=False)
        sub.pack(fill="x")
        self._add_files(sub.body, folder_path, indent=indent + 1)

    def _add_files(self, parent, folder_path, indent):
        files = sorted(f for f in listdir(folder_path)
                       if path.isfile(path.join(folder_path, f))
                       and f.lower().endswith(_IMG_EXTENSIONS))

        if not files:
            tk.Label(parent, text="(keine Bilder)", anchor="w", bg="#f0f0f0",
                     fg="#889", padx=6 + indent * 14, pady=2).pack(fill="x")
            return

        for fname in files:
            full = path.join(folder_path, fname)
            lbl = tk.Label(parent, text=fname, anchor="w", bg="#f0f0f0",
                           fg="#1b2733", padx=6 + indent * 14, pady=2, cursor="hand2")
            lbl.pack(fill="x")
            lbl.bind("<Button-1>", lambda _e, p=full, w=lbl: self.load_image(p, w))
            lbl.bind("<Enter>", lambda _e, w=lbl: w.config(bg="#e4ebf1")
                     if w is not self._selected_label else None)
            lbl.bind("<Leave>", lambda _e, w=lbl: w.config(bg="#f0f0f0")
                     if w is not self._selected_label else None)

    # --------------------------------------------------------------- Aktionen
    def load_image(self, img_path, label_widget=None):
        try:
            with Image.open(img_path) as img:
                existing = self._read_tags(img)
                preview = img.copy()
                preview.thumbnail(_PREVIEW_MAX, Image.LANCZOS)
                self._preview_imgtk = ImageTk.PhotoImage(preview)
                size = img.size
                fmt = img.format
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden:\n{exc}")
            self.status_var.set(f"Fehler beim Laden: {exc}")
            return

        self.current_path = img_path
        self.preview_label.config(image=self._preview_imgtk, text="")
        self.info_var.set(f"{path.basename(img_path)}   |   {fmt}   |   {size[0]}×{size[1]} px")

        self.tag_entry.delete(0, "end")
        self.tag_entry.insert(0, existing)
        self.save_btn.config(state="normal")
        self.status_var.set(f"Geladen: {img_path}")

        # Markierung in der Liste aktualisieren
        if self._selected_label is not None and self._selected_label.winfo_exists():
            self._selected_label.config(bg="#f0f0f0", fg="#1b2733")
        if label_widget is not None:
            label_widget.config(bg="#3d7bd6", fg="#ffffff")
        self._selected_label = label_widget

    def save_tags(self):
        if not self.current_path:
            return

        raw = self.tag_entry.get().strip()
        # In einheitliche Tag-Liste normalisieren
        parts = [t.strip() for chunk in raw.split(";") for t in chunk.split(",")]
        tags = [t for t in parts if t]
        keywords = ";".join(tags)

        try:
            with Image.open(self.current_path) as img:
                exif = img.getexif()
                # XPKeywords: UTF-16-LE, null-terminiert (Windows-Konvention)
                exif[_EXIF_XPKEYWORDS] = keywords.encode("utf-16-le") + b"\x00\x00"
                # ImageDescription: portable Textbeschreibung
                exif[_EXIF_IMAGEDESCRIPTION] = keywords

                save_kwargs = {"exif": exif.tobytes()}
                fmt = img.format
                if fmt in ("JPEG", "JPG", "JPEG2000"):
                    save_kwargs["quality"] = "keep"
                img.save(self.current_path, **save_kwargs)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Fehler", f"Tags konnten nicht gespeichert werden:\n{exc}")
            self.status_var.set(f"Speicherfehler: {exc}")
            return

        self.status_var.set(f"Gespeichert ({len(tags)} Tag(s)): {self.current_path}")

    # ---------------------------------------------------------------- Helpers
    def _read_tags(self, img):
        '''Liest vorhandene Keywords aus den EXIF-Daten und gibt sie als String zurück.'''
        try:
            exif = img.getexif()
        except Exception:  # noqa: BLE001
            return ""

        # Zuerst XPKeywords (UTF-16-LE) versuchen
        val = exif.get(_EXIF_XPKEYWORDS)
        if val:
            try:
                if isinstance(val, (tuple, list)):
                    val = bytes(val)
                if isinstance(val, bytes):
                    text = val.decode("utf-16-le", errors="ignore")
                    return text.replace("\x00", "").strip()
            except Exception:  # noqa: BLE001
                pass

        # Fallback: ImageDescription
        val = exif.get(_EXIF_IMAGEDESCRIPTION)
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="ignore")
        if isinstance(val, str):
            return val.strip()
        return ""


def main():
    root = tk.Tk()
    ImgTaggerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

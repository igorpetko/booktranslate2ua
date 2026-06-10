#!/usr/bin/env python3
"""
translate_book.py — перекладає FB2/EPUB/TXT з EN/RU на UK через Ollama.
Зберігає прогрес, не пропускає фрагменти, красивий вивід у консолі.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ── Конфіг ──────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
MODEL = "uamarchuan/lapa-v0.1-instruct:Q4_K_M"
CHUNK_CHARS = 900          # символів у фрагменті (менше = точніше, але повільніше)
MAX_RETRIES = 4
RETRY_DELAY = 3            # сек між повторами
TIMEOUT = 120              # сек на один запит

SYSTEM_PROMPT_BASE = (
    "Ти — професійний літературний перекладач. "
    "Перекладай текст на українську мову точно, зберігаючи стиль, пунктуацію та форматування оригіналу. "
    "Не додавай пояснень, коментарів чи зайвих слів — лише переклад."
)

SYSTEM_PROMPT_GUIDE = (
    "Ти — професійний літературний перекладач. "
    "Перекладай текст на українську мову точно, зберігаючи стиль, пунктуацію та форматування оригіналу. "
    "Не додавай пояснень, коментарів чи зайвих слів — лише переклад.\n\n"
    "КОНТЕКСТНИЙ ГАЙД ДЛЯ ЦІЄЇ КНИГИ (дотримуйся суворо):\n"
    "{guide}"
)

# Активний системний промпт (може бути замінений якщо є гід)
SYSTEM_PROMPT = SYSTEM_PROMPT_BASE

TRANSLATE_PROMPT = (
    "Переклади наступний текст на українську мову. "
    "Збережи абзаци, розриви рядків та форматування. "
    "Відповідай ТІЛЬКИ перекладеним текстом:\n\n{text}"
)

# Максимальна кількість символів гіду що вставляється в промпт
GUIDE_MAX_CHARS = 3_000

# ── Утиліти ──────────────────────────────────────────────────────────────────

def chunk_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def split_into_chunks(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Розбиває текст на фрагменти по абзацах, не перевищуючи max_chars."""
    paragraphs = re.split(r'\n{2,}', text.strip())
    chunks, current = [], ""
    for para in paragraphs:
        if not para.strip():
            continue
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).lstrip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def detect_source_lang(text_sample: str) -> str:
    """Груба евристика визначення мови."""
    cyrillic_ru = len(re.findall(r'[ёъыЁЪЫ]', text_sample))
    cyrillic_uk = len(re.findall(r'[іїєґІЇЄҐ]', text_sample))
    latin = len(re.findall(r'[a-zA-Z]', text_sample))
    if latin > len(text_sample) * 0.4:
        return "англійська"
    if cyrillic_uk > cyrillic_ru * 0.5:
        return "і без того схожа на українську (суржик, штолє?)"
    return "свино-собача"

# ── Контекстний гід ──────────────────────────────────────────────────────────

def load_guide(book_path: Path, guide_path: Optional[Path] = None) -> Optional[str]:
    """Шукає .guide.md поруч з книгою або за вказаним шляхом."""
    candidates = []
    if guide_path:
        candidates.append(guide_path)
    # <book>.guide.md
    candidates.append(book_path.with_suffix(".guide.md"))
    # <book_stem>.guide.md в тій самій папці
    candidates.append(book_path.parent / (book_path.stem + ".guide.md"))

    for p in candidates:
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            # Обрізаємо якщо задовгий
            if len(text) > GUIDE_MAX_CHARS:
                text = text[:GUIDE_MAX_CHARS] + "\n\n[...гід скорочено для контексту...]"

            return text, p
    return None, None


def activate_guide(guide_text: str):
    """Оновлює глобальний SYSTEM_PROMPT з вмістом гіду."""
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = SYSTEM_PROMPT_GUIDE.format(guide=guide_text)


# ── Зчитування форматів ──────────────────────────────────────────────────────

def read_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"id": "txt", "title": None, "content": text}]


def _p_not_in_title(p_tag) -> bool:
    """Повертає True якщо <p> не є нащадком <title>."""
    for parent in p_tag.parents:
        if parent.name == "title":
            return False
    return True


def _p_not_in_description(p_tag) -> bool:
    """Повертає True якщо <p> не є нащадком <description>."""
    for parent in p_tag.parents:
        if parent.name == "description":
            return False
    return True


def read_fb2(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_bytes(), "xml")
    sections = []

    for body in soup.find_all("body"):
        for sec in body.find_all(["section", "body"], recursive=False):
            title_tag = sec.find("title")
            title = title_tag.get_text(" ", strip=True) if title_tag else None

            # Тільки <p> що НЕ всередині <title>
            paras = []
            for p in sec.find_all("p"):
                if not _p_not_in_title(p):
                    continue
                t = p.get_text(" ", strip=True)
                if t:
                    paras.append(t)

            content = "\n\n".join(paras)
            if content.strip():
                sections.append({
                    "id": str(len(sections)),
                    "title": title,
                    "content": content,
                })

    if not sections:
        # fallback: всі <p> не в title і не в description
        paras = []
        for p in soup.find_all("p"):
            if _p_not_in_title(p) and _p_not_in_description(p):
                t = p.get_text(" ", strip=True)
                if t:
                    paras.append(t)
        sections = [{"id": "fb2", "title": None, "content": "\n\n".join(paras)}]
    return sections


def read_epub(path: Path) -> tuple[list[dict], dict]:
    """Повертає (sections, meta) де meta містить заголовок/автора для FB2."""
    import ebooklib
    from ebooklib import epub
    book = epub.read_epub(str(path), options={"ignore_ncx": True})

    meta = {
        "title":  book.title or path.stem,
        "author": ", ".join(
            str(a) for a in (book.metadata.get("DC", {}).get("creator", []) or [])
        ) or "",
        "lang": "uk",
        "src_lang": "en",
    }

    # Spine — правильний порядок
    spine_ids = [sid for sid, _ in book.spine]
    items_by_id = {item.get_id(): item
                   for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}
    ordered = ([items_by_id[sid] for sid in spine_ids if sid in items_by_id]
               or list(items_by_id.values()))

    sections = []
    for item in ordered:
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        h = soup.find(re.compile(r'^h[1-6]$'))
        title = h.get_text(strip=True) if h else None
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")
                 if p.get_text(strip=True)]
        content = "\n\n".join(paras)
        if content.strip():
            sections.append({"id": item.get_id(), "title": title, "content": content})
    return sections, meta

# ── Ollama ────────────────────────────────────────────────────────────────────

def translate_chunk(text: str, attempt_info: str = "") -> Optional[str]:
    prompt = TRANSLATE_PROMPT.format(text=text)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat",
                              json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            result = data.get("message", {}).get("content", "").strip()
            if result:
                return result
            console.print(f"[yellow]  ⚠ Порожня відповідь, спроба {attempt}/{MAX_RETRIES}[/]")
        except requests.exceptions.Timeout:
            console.print(f"[yellow]  ⚠ Timeout, спроба {attempt}/{MAX_RETRIES}[/]")
        except Exception as e:
            console.print(f"[red]  ✗ {e}, спроба {attempt}/{MAX_RETRIES}[/]")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)
    return None


def check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        for m in models:
            if "lapa" in m.lower():
                global MODEL
                MODEL = m
                return True
        console.print(f"[yellow]Доступні моделі: {models}[/]")
        console.print(f"[red]Модель lapa не знайдена![/]")
        return False
    except Exception as e:
        console.print(f"[red]Ollama недоступна: {e}[/]")
        return False

# ── Прогрес ───────────────────────────────────────────────────────────────────

class Progress_Store:
    def __init__(self, cache_path: Path):
        self.path = cache_path
        self.data: dict = {}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                self.data = {}

    def save(self):
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str):
        self.data[key] = value
        self.save()

    def count_done(self) -> int:
        return len(self.data)

# ── Збереження результату ─────────────────────────────────────────────────────

def save_txt(sections: list[dict], out_path: Path, store: Progress_Store):
    lines = []
    for sec in sections:
        if sec["title"]:
            lines.append(f"\n{'='*60}\n{sec['title']}\n{'='*60}\n")
        for chunk_text in sec["chunks"]:
            cid = chunk_id(chunk_text)
            translated = store.get(cid) or chunk_text
            lines.append(translated + "\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_fb2(original_path: Path, sections: list[dict],
             out_path: Path, store: Progress_Store):
    """Зберігає перекладений FB2, замінюючи текст <p> тегів напряму."""
    soup = BeautifulSoup(original_path.read_bytes(), "xml")

    # Перебудовуємо soup з тими самими p_tags що вказують на вузли
    # Але soup перечитаний заново — треба знайти <p> по позиції.
    # Збираємо всі параграфи в порядку документа
    all_doc_p = soup.find_all("p")

    # Будуємо плоский список: (оригінальний текст параграфу, перекладений текст)
    # з усіх chunks всіх секцій
    translated_paras = []
    for sec in sections:
        # Розкладаємо chunks назад на параграфи
        for chunk_text in sec["chunks"]:
            cid = chunk_id(chunk_text)
            translated = store.get(cid) or chunk_text
            # chunk містить кілька параграфів розділених \n\n
            orig_paras = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
            trans_paras = [p.strip() for p in translated.split("\n\n") if p.strip()]
            # Якщо кількість параграфів збігається — маппінг 1:1
            # Якщо ні — розподіляємо перекладений текст рівномірно
            if len(trans_paras) == len(orig_paras):
                for op, tp in zip(orig_paras, trans_paras):
                    translated_paras.append((op, tp))
            else:
                # Fallback: перший оригінальний параграф отримує весь переклад,
                # решта маркуються як вже покриті
                for i, op in enumerate(orig_paras):
                    tp = trans_paras[i] if i < len(trans_paras) else ""
                    translated_paras.append((op, tp))

    # Будуємо словник orig_text -> translated_text
    trans_map = {}
    for orig, trans in translated_paras:
        if orig and trans:
            trans_map[orig] = trans

    # Замінюємо текст в <p> тегах документа
    replaced = 0
    for p_tag in all_doc_p:
        orig_text = p_tag.get_text(" ", strip=True)
        if orig_text in trans_map:
            # Зберігаємо структуру тегу, замінюємо тільки текст
            p_tag.clear()
            p_tag.string = trans_map[orig_text]
            replaced += 1

    console.print(f"[dim]  FB2: замінено {replaced}/{len(all_doc_p)} параграфів[/dim]")
    out_path.write_bytes(soup.encode("utf-8"))


def save_epub_as_fb2(epub_path: Path, sections: list[dict],
                     out_path: Path, store: Progress_Store, meta: dict):
    """Конвертує перекладений EPUB у FB2."""
    # Будуємо trans_map
    trans_map = {}
    for sec in sections:
        for chunk_text in sec["chunks"]:
            cid = chunk_id(chunk_text)
            translated = store.get(cid) or chunk_text
            orig_paras  = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
            trans_paras = [p.strip() for p in translated.split("\n\n") if p.strip()]
            if len(trans_paras) == len(orig_paras):
                for op, tp in zip(orig_paras, trans_paras):
                    if op and tp:
                        trans_map[op] = tp
            else:
                for i, op in enumerate(orig_paras):
                    tp = trans_paras[i] if i < len(trans_paras) else op
                    if op and tp:
                        trans_map[op] = tp

    # Будуємо FB2 XML
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"'
        ' xmlns:l="http://www.w3.org/1999/xlink">',
        '<description>',
        '<title-info>',
    ]

    title  = meta.get("title", epub_path.stem)
    author = meta.get("author", "")
    if author:
        parts = author.split()
        first = " ".join(parts[:-1]) if len(parts) > 1 else author
        last  = parts[-1] if len(parts) > 1 else ""
        lines += [
            f'<author><first-name>{first}</first-name>'
            f'<last-name>{last}</last-name></author>',
        ]
    lines += [
        f'<book-title>{title} (переклад)</book-title>',
        '<lang>uk</lang>',
        f'<src-lang>{meta.get("src_lang","en")}</src-lang>',
        '</title-info>',
        '</description>',
        '<body>',
        f'<title><p>{title}</p></title>',
    ]

    for sec in sections:
        lines.append('<section>')
        if sec.get("title"):
            sec_title = sec["title"].replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')
            lines.append(f'<title><p>{sec_title}</p></title>')
        for chunk_text in sec["chunks"]:
            cid = chunk_id(chunk_text)
            translated = store.get(cid) or chunk_text
            orig_paras  = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
            trans_paras = [p.strip() for p in translated.split("\n\n") if p.strip()]
            # Якщо збіг — беремо пара-пара, інакше весь переклад одним блоком
            if len(trans_paras) == len(orig_paras):
                paras = trans_paras
            else:
                paras = trans_paras if trans_paras else orig_paras
            for para in paras:
                escaped = para.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
                lines.append(f'<p>{escaped}</p>')
        lines.append('</section>')

    lines += ['</body>', '</FictionBook>']

    fb2_out = out_path.with_suffix('.fb2')
    fb2_out.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[dim]  EPUB→FB2: {len(sections)} секцій, {len(trans_map)} параграфів замінено[/dim]")
    return fb2_out


def save_result(fmt: str, original_path: Path, sections: list[dict],
                out_path: Path, store: Progress_Store):
    if fmt == "epub":
        meta = sections[0].get("_epub_meta", {}) if sections else {}
        save_epub_as_fb2(original_path, sections, out_path, store, meta)
        return
    elif fmt == "fb2":
        try:
            save_fb2(original_path, sections, out_path, store)
        except Exception as e:
            console.print(f"[yellow]FB2 save error ({e}), зберігаю як TXT[/]")
            save_txt(sections, out_path.with_suffix(".txt"), store)
    else:
        save_txt(sections, out_path, store)

# ── Головна логіка ─────────────────────────────────────────────────────────────

def make_header(src_file: str, fmt: str, lang: str, total_chunks: int, done: int, guide_info: str = ""):
    table = Table(box=box.ROUNDED, show_header=False, border_style="cyan",
                  padding=(0, 1), expand=True)
    table.add_column(justify="right", style="bold cyan", min_width=18)
    table.add_column(style="white")
    table.add_row("📂 Файл:", src_file)
    table.add_row("📖 Формат:", fmt.upper())
    table.add_row("🌐 Мова джерела:", lang)
    table.add_row("🔧 Модель:", MODEL)
    table.add_row("📦 Фрагменти:", f"{done}/{total_chunks} (вже перекладено: {done})")
    if guide_info:
        table.add_row("📋 Гід:", guide_info)
    return Panel(table, title="[bold yellow]📚 Book Translator UA[/]",
                 border_style="yellow", padding=(0, 1))


def translate_book(input_path: Path, output_path: Optional[Path], force: bool, guide_path: Optional[Path] = None):
    fmt = input_path.suffix.lower().lstrip(".")
    if fmt not in ("txt", "fb2", "epub"):
        console.print(f"[red]Непідтримуваний формат: {fmt}[/]")
        sys.exit(1)

    # Завантажуємо гід (якщо є)
    guide_text, guide_found = load_guide(input_path, guide_path)
    if guide_text:
        activate_guide(guide_text)

    # Читаємо
    console.print(f"\n[cyan]⏳ Читаємо {fmt.upper()}...[/]")
    if fmt == "txt":
        sections = read_txt(input_path)
    elif fmt == "fb2":
        sections = read_fb2(input_path)
    else:
        sections, epub_meta = read_epub(input_path)
        # Зберігаємо meta в першій секції для передачі в save_epub_as_fb2
        if sections:
            sections[0]["_epub_meta"] = epub_meta

    # Розбиваємо на чанки
    for sec in sections:
        sec["chunks"] = split_into_chunks(sec["content"])

    all_chunks = [(sec, ch) for sec in sections for ch in sec["chunks"]]
    total = len(all_chunks)

    if total == 0:
        console.print("[red]Текст не знайдено у файлі![/]")
        sys.exit(1)

    # Визначаємо мову
    sample = " ".join(c for _, c in all_chunks[:10])[:2000]
    lang = detect_source_lang(sample)

    # Прогрес-файл
    cache_path = input_path.with_suffix(".translate_cache.json")
    store = Progress_Store(cache_path)
    already_done = sum(1 for _, ch in all_chunks if store.get(chunk_id(ch)))

    # Вихідний файл
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_ua")

    guide_info = f"[green]{guide_found.name}[/] ({len(guide_text):,} символів)" if guide_text else "[dim]не знайдено[/dim]"
    console.print(make_header(input_path.name, fmt, lang, total, already_done, guide_info))
    console.print()

    if already_done == total and not force:
        console.print("[green]✓ Всі фрагменти вже перекладено! Збираємо файл...[/]")
        save_result(fmt, input_path, sections, output_path, store)
        console.print(f"[bold green]✓ Збережено:[/] {output_path}")
        return

    # Прогрес-бар
    progress = Progress(
        SpinnerColumn(spinner_name="dots", style="bold cyan"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]ETA:"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    )

    task = progress.add_task("Переклад", total=total, completed=already_done)

    stats = {"ok": already_done, "fail": 0, "skip": 0}
    start_ts = time.time()

    with Live(progress, console=console, refresh_per_second=4):
        for idx, (sec, chunk_text) in enumerate(all_chunks):
            cid = chunk_id(chunk_text)

            # Вже є в кеші
            if store.get(cid) and not force:
                stats["skip"] += 1
                progress.advance(task)
                continue

            # Показуємо прев'ю
            preview = chunk_text[:80].replace("\n", " ")
            sec_title = sec.get("title") or f"секція {sec['id']}"
            progress.update(task,
                description=f"[cyan]{sec_title[:30]}[/] [dim]{preview}…[/dim]")

            translated = translate_chunk(chunk_text, f"{idx+1}/{total}")

            if translated:
                store.set(cid, translated)
                stats["ok"] += 1
            else:
                # Залишаємо оригінал
                store.set(cid, chunk_text)
                stats["fail"] += 1
                progress.print(f"[red]  ✗ Помилка на фрагменті {idx+1}, залишено оригінал[/]")

            progress.advance(task)

            # Автозбереження файлу кожні 20 фрагментів
            if (idx + 1) % 20 == 0:
                save_result(fmt, input_path, sections, output_path, store)

    # Фінальне збереження
    console.print()
    with console.status("[bold green]Зберігаємо файл..."):
        save_result(fmt, input_path, sections, output_path, store)

    elapsed = time.time() - start_ts

    # Підсумок
    result_table = Table(box=box.DOUBLE_EDGE, border_style="green",
                         show_header=False, padding=(0, 2), expand=False)
    result_table.add_column(justify="right", style="bold", min_width=22)
    result_table.add_column()
    result_table.add_row("✅ Перекладено:", f"[green]{stats['ok']}[/] фрагментів")
    result_table.add_row("⏭  З кешу:", f"[cyan]{stats['skip']}[/] фрагментів")
    result_table.add_row("❌ Помилок:", f"[red]{stats['fail']}[/] (збережено оригінал)")
    result_table.add_row("⏱  Час:", f"{elapsed/60:.1f} хв ({elapsed:.0f} с)")
    result_table.add_row("📄 Результат:", str(output_path))
    result_table.add_row("💾 Кеш:", str(cache_path))

    console.print(Panel(result_table, title="[bold green]✓ Готово![/]",
                        border_style="green"))

    if stats["fail"] > 0:
        console.print(f"\n[yellow]💡 Є {stats['fail']} помилок. Запустіть знову — вони будуть перероблені.[/]")
    console.print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Перекладає FB2/EPUB/TXT на українську через Ollama (lapa).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  python translate_book.py book.fb2
  python translate_book.py book.epub -o book_ua.txt
  python translate_book.py book.txt --force
  python translate_book.py book.fb2 --chunk-size 600
""")
    parser.add_argument("input", help="Вхідний файл (fb2/epub/txt)")
    parser.add_argument("-o", "--output", help="Вихідний файл (за замовчуванням: <name>_ua.<ext>)")
    parser.add_argument("--guide", default=None,
                        help="Шлях до .guide.md (за замовч. шукає <book>.guide.md поруч з файлом)")
    parser.add_argument("--force", action="store_true",
                        help="Перекладати заново навіть якщо є кеш")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Розмір фрагменту в символах (за замовч. 900)")
    parser.add_argument("--model", default=None,
                        help="Назва моделі Ollama (за замовч. автовизначення lapa)")
    parser.add_argument("--ollama-url", default=None,
                        help="URL Ollama (за замовч. http://localhost:11434)")
    args = parser.parse_args()

    global CHUNK_CHARS, OLLAMA_URL, MODEL
    if args.chunk_size is not None:
        CHUNK_CHARS = args.chunk_size
    if args.ollama_url is not None:
        OLLAMA_URL = args.ollama_url
    if args.model:
        MODEL = args.model

    console.rule("[bold yellow]📚 Book Translator UA[/]")

    # Перевірка Ollama
    with console.status("[cyan]Перевіряємо Ollama..."):
        ok = check_ollama()
    if not ok:
        console.print("[red]Запустіть Ollama та переконайтесь, що модель lapa завантажена:[/]")
        console.print("  ollama pull uamarchuan/lapa-v0.1-instruct:Q4_K_M")
        sys.exit(1)
    console.print(f"[green]✓ Ollama OK, модель: {MODEL}[/]")

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Файл не знайдено: {input_path}[/]")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    guide_path = Path(args.guide) if args.guide else None
    translate_book(input_path, output_path, args.force, guide_path)


if __name__ == "__main__":
    main()

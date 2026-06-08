#!/usr/bin/env python3
"""
build_guide.py — будує контекстний гід перекладача (.md) для FB2/EPUB/TXT.

Два етапи:
  1. Статистичний аналіз: власні назви, частотність, мова (без LLM).
  2. LLM-аналіз (Ollama): витягує персонажів, терміни, стиль з вибірки тексту.

Результат: <назва_книги>.guide.md — вставляється в системний промпт translate_book.py.

Моделі:
  --translate-model  — модель для перекладу (lapa, за замовч. автовизначення)
  --analysis-model   — модель для аналізу/JSON (qwen2.5, mistral тощо)
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

OLLAMA_URL       = "http://localhost:11434"
TRANSLATE_MODEL  = "uamarchuan/lapa-v0.1.2-instruct:Q4_K_M"  # для translate_book.py
ANALYSIS_MODEL   = "qwen2.5:7b-instruct-q4_K_M"              # для аналізу тут
TIMEOUT          = 180

SAMPLE_CHARS     = 12_000   # символів вибірки для LLM
GUIDE_MAX_CHARS  = 3_000    # ліміт гіду що йде в контекст перекладача

# ── Зчитування ────────────────────────────────────────────────────────────────

def read_text(path: Path) -> tuple[str, str]:
    fmt = path.suffix.lower().lstrip(".")
    if fmt == "txt":
        return path.read_text(encoding="utf-8", errors="replace"), "txt"
    if fmt == "fb2":
        soup = BeautifulSoup(path.read_bytes(), "xml")
        return soup.get_text("\n\n"), "fb2"
    if fmt == "epub":
        import ebooklib
        from ebooklib import epub
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            parts.append(soup.get_text("\n\n"))
        return "\n\n".join(parts), "epub"
    raise ValueError(f"Непідтримуваний формат: {fmt}")


# ── Статистичний аналіз ───────────────────────────────────────────────────────

def detect_lang(text: str) -> str:
    ru  = len(re.findall(r'[ёъыЁЪЫ]', text))
    uk  = len(re.findall(r'[іїєґІЇЄҐ]', text))
    lat = len(re.findall(r'[a-zA-Z]', text))
    if lat / max(len(text), 1) > 0.35:
        return "English"
    if uk > ru * 0.4:
        return "Ukrainian (mixed)"
    return "Russian"


def extract_proper_nouns(text: str, lang: str) -> list[tuple[str, int]]:
    if lang == "English":
        sentences = re.split(r'(?<=[.!?])\s+', text)
        words: list[str] = []
        for sent in sentences:
            tokens = sent.strip().split()
            if len(tokens) > 1:
                words.extend(tokens[1:])
        candidates = [w.strip('",.:;!?()[]—–-«»') for w in words
                      if w and w[0].isupper() and len(w) > 2]
    else:
        candidates = re.findall(r'\b[А-ЯЇІЄҐЁA-Z][а-яїієґёa-z]{2,}\b', text)

    stopwords = {
        "The","This","That","These","Those","There","Their","He","She","It",
        "We","You","They","His","Her","Its","But","And","Or","Not","For",
        "With","From","Into","When","Where","What","Who","How","All","One",
        "Two","Was","Had","Has","Have","Been","Were","Are","Did","Does",
        "Он","Она","Они","Его","Её","Это","Как","Но","Что","Все","Уже",
        "Ещё","Тут","Там","Вот","Нет","Да","Его","При","Без","Над",
    }
    counter = Counter(candidates)
    return [(w, c) for w, c in counter.most_common(60)
            if c >= 2 and w not in stopwords]


def sample_text(text: str, chars: int = SAMPLE_CHARS) -> str:
    """Рівномірна вибірка з початку, середини і кінця."""
    n = len(text)
    if n <= chars:
        return text
    chunk = chars // 3
    start = text[:chunk]
    mid   = text[n//2 - chunk//2 : n//2 + chunk//2]
    end   = text[n - chunk:]
    return f"{start}\n\n[...середина книги...]\n\n{mid}\n\n[...кінець книги...]\n\n{end}"


# ── Ollama ────────────────────────────────────────────────────────────────────

def list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def resolve_model(requested: str, models: list[str], keyword: str) -> Optional[str]:
    """Шукає модель за точним ім'ям, потім за keyword-підрядком."""
    # Точний збіг
    if requested in models:
        return requested
    # Префіксний збіг (без тегу)
    for m in models:
        if m.startswith(requested.split(":")[0]):
            return m
    # Fallback по keyword
    for m in models:
        if keyword in m.lower():
            return m
    return None


def check_ollama_models() -> tuple[Optional[str], Optional[str]]:
    """
    Перевіряє Ollama і повертає (analysis_model, translate_model).
    Друкує зрозумілий звіт.
    """
    models = list_models()
    if not models:
        console.print("[red]✗ Ollama недоступна або немає моделей[/]")
        return None, None

    ana  = resolve_model(ANALYSIS_MODEL,  models, "qwen")
    tra  = resolve_model(TRANSLATE_MODEL, models, "lapa")

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", padding=(0,1))
    t.add_column("Роль")
    t.add_column("Запит")
    t.add_column("Знайдено")
    t.add_column("Статус")

    t.add_row(
        "🔍 Аналіз",
        ANALYSIS_MODEL,
        ana or "—",
        "[green]✓[/]" if ana else "[red]✗ не знайдено[/]",
    )
    t.add_row(
        "🌐 Переклад",
        TRANSLATE_MODEL,
        tra or "—",
        "[green]✓[/]" if tra else "[yellow]⚠ не знайдено (не критично)[/]",
    )
    console.print(t)

    if not ana:
        console.print(f"\n[yellow]Доступні моделі: {models}[/]")
        console.print(f"[yellow]Завантажте: ollama pull {ANALYSIS_MODEL}[/]")

    return ana, tra


def llm_call(model: str, system: str, user: str, json_mode: bool = False) -> Optional[str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.05, "num_predict": 3000},
    }
    if json_mode:
        payload["format"] = "json"  # Ollama structured output (якщо модель підтримує)
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        console.print(f"[red]LLM помилка ({model}): {e}[/]")
        return None


# ── LLM-аналіз ───────────────────────────────────────────────────────────────

EXTRACT_SYSTEM = (
    "You are a literary analyst. "
    "Respond ONLY with valid JSON. No explanations, no markdown fences, no comments. "
    "Use double quotes for all keys and string values."
)

# Використовуємо конкатенацію рядків щоб уникнути проблем з .format() і дужками JSON
EXTRACT_PROMPT_TEMPLATE = (
    "Analyze the book excerpts below and extract:\n\n"
    "1. characters — main and important secondary characters.\n"
    "   For each: name (original), gender (male/female/unknown),\n"
    "   role (protagonist/antagonist/secondary),\n"
    "   speech (formal/colloquial/archaic/rude/unknown),\n"
    "   address_to_others (ти/ви/mixed/unknown),\n"
    "   address_from_others (ти/ви/mixed/unknown),\n"
    "   note (short description, 1 sentence, in Ukrainian).\n\n"
    "2. glossary — unique terms, place names, organizations, magic systems, technologies.\n"
    "   For each: original, suggested_ua (Ukrainian translation or transliteration),\n"
    "   explanation (1 sentence, in Ukrainian).\n\n"
    "3. style — author style: tone (epic/lyrical/noir/humorous/neutral/etc),\n"
    "   pov (first/third/mixed), era (modern/medieval/future/etc),\n"
    "   notes (style notes, 1-2 sentences, in Ukrainian).\n\n"
    "4. translation_challenges — potential translation difficulties.\n"
    "   Each item: original, challenge (description in Ukrainian).\n\n"
    "Return ONLY this JSON structure:\n"
    '{"characters":[...],"glossary":[...],"style":{...},"translation_challenges":[...]}\n\n'
    "TEXT TO ANALYZE:\n"
    "{sample}"
)


def parse_llm_json(raw: str) -> Optional[dict]:
    """Намагається витягнути JSON з відповіді моделі кількома способами."""
    if not raw:
        return None

    # 1. Пряме парсування
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Прибираємо markdown-огорожі
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```\s*$', '', cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Шукаємо перший JSON-об'єкт у тексті
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Показуємо що отримали для діагностики
    console.print(f"[yellow]⚠ Не вдалося розпарсити JSON. Перші 600 символів відповіді:[/]")
    console.print(f"[dim]{raw[:600]}[/dim]")
    return None


def analyze_with_llm(model: str, sample: str) -> Optional[dict]:
    prompt = EXTRACT_PROMPT_TEMPLATE.format(sample=sample[:SAMPLE_CHARS])

    console.print(f"  [dim]Відправляємо {len(prompt):,} символів у {model}...[/dim]")

    with console.status(f"[cyan]🤖 {model} аналізує текст... (1-3 хв)[/]"):
        # Спробуємо з json_mode
        raw = llm_call(model, EXTRACT_SYSTEM, prompt, json_mode=True)

    result = parse_llm_json(raw)
    if result:
        return result

    # Якщо json_mode не допоміг — спробуємо без нього
    console.print("[yellow]  Повторна спроба без json_mode...[/]")
    with console.status(f"[cyan]🤖 Повторна спроба...[/]"):
        raw2 = llm_call(model, EXTRACT_SYSTEM, prompt, json_mode=False)
    return parse_llm_json(raw2)


# ── Інтерактивний режим ───────────────────────────────────────────────────────

def interactive_fill(data: dict, book_title: str) -> dict:
    console.print()
    console.rule("[bold yellow]✏️  Перевірка та доповнення[/]")

    console.print("\n[bold cyan]── Book Overview ──[/]")
    data.setdefault("overview", {})
    ov = data["overview"]
    ov["title"]  = Prompt.ask("  Назва книги",          default=ov.get("title", book_title))
    ov["author"] = Prompt.ask("  Автор",                default=ov.get("author", ""))
    ov["genre"]  = Prompt.ask("  Жанр",                 default=ov.get("genre", ""))
    ov["era"]    = Prompt.ask("  Епоха/сетинг",         default=ov.get("era", ""))
    ov["lang"]   = Prompt.ask("  Мова оригіналу (EN/RU)", default=ov.get("lang", "EN"))

    style_data = data.get("style", {})
    if style_data:
        console.print(f"\n[bold cyan]── Стиль (від LLM) ──[/]")
        console.print(
            f"  Тон: [green]{style_data.get('tone','?')}[/]  "
            f"POV: [green]{style_data.get('pov','?')}[/]  "
            f"Епоха: [green]{style_data.get('era','?')}[/]"
        )
        console.print(f"  Нотатки: [dim]{style_data.get('notes','')}[/dim]")

    chars = data.get("characters", [])
    if chars:
        console.print(f"\n[bold cyan]── Персонажі ({len(chars)}) ──[/]")
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        t.add_column("Ім'я",      style="cyan")
        t.add_column("Стать")
        t.add_column("Роль")
        t.add_column("Мовлення")
        t.add_column("→ інших")
        t.add_column("← від")
        t.add_column("Нотатка", max_width=38, style="dim")
        for c in chars:
            t.add_row(
                c.get("name",""), c.get("gender","?"), c.get("role","?"),
                c.get("speech","?"), c.get("address_to_others","?"),
                c.get("address_from_others","?"), c.get("note",""),
            )
        console.print(t)

    if Confirm.ask("  Додати персонажа вручну?", default=False):
        while True:
            name = Prompt.ask("  Ім'я (Enter — завершити)", default="")
            if not name:
                break
            chars.append({
                "name":               name,
                "gender":             Prompt.ask("  Стать", default="unknown"),
                "role":               Prompt.ask("  Роль (protagonist/antagonist/secondary)", default="secondary"),
                "speech":             Prompt.ask("  Мовлення (formal/colloquial/archaic/rude)", default="colloquial"),
                "address_to_others":  Prompt.ask("  Звертається (ти/ви/mixed)", default="ти"),
                "address_from_others":Prompt.ask("  До нього/неї (ти/ви/mixed)", default="ти"),
                "note":               Prompt.ask("  Нотатка", default=""),
            })

    glossary = data.get("glossary", [])
    if glossary:
        console.print(f"\n[bold cyan]── Глосарій ({len(glossary)} термінів) ──[/]")
        t2 = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        t2.add_column("Оригінал",   style="cyan")
        t2.add_column("Українська")
        t2.add_column("Пояснення",  max_width=42, style="dim")
        for g in glossary[:25]:
            t2.add_row(g.get("original",""), g.get("suggested_ua",""), g.get("explanation",""))
        if len(glossary) > 25:
            console.print(f"  [dim]... і ще {len(glossary)-25} термінів[/dim]")
        console.print(t2)

    if Confirm.ask("  Додати термін вручну?", default=False):
        while True:
            orig = Prompt.ask("  Оригінал (Enter — завершити)", default="")
            if not orig:
                break
            glossary.append({
                "original":     orig,
                "suggested_ua": Prompt.ask("  Українська"),
                "explanation":  Prompt.ask("  Пояснення", default=""),
            })

    return data


# ── Генерація .md ─────────────────────────────────────────────────────────────

def build_md(data: dict, proper_nouns: list[tuple[str, int]],
             analysis_model: str, translate_model: str) -> str:
    ov    = data.get("overview", {})
    st    = data.get("style", {})
    chars = data.get("characters", [])
    glos  = data.get("glossary", [])
    chals = data.get("translation_challenges", [])

    title = ov.get("title", "Невідома книга")
    lines = [
        f"# Translation Guide: {title}",
        "",
        "> Автоматично підхоплюється `translate_book.py` як контекст перекладача.",
        f"> Аналіз: `{analysis_model}` | Переклад: `{translate_model}`",
        "> Відредагуйте вручну для кращої якості.",
        "",
        "## 1. Book Overview", "",
        f"- **Title:** {title}",
    ]
    for k, label in [("author","Author"),("genre","Genre"),("era","Era/Setting")]:
        if ov.get(k): lines.append(f"- **{label}:** {ov[k]}")
    if st.get("tone"): lines.append(f"- **Tone:** {st['tone']}")
    if st.get("pov"):  lines.append(f"- **Narrative POV:** {st['pov']}")
    if ov.get("lang"): lines.append(f"- **Source language:** {ov['lang']}")
    if st.get("notes"):lines.append(f"- **Style notes:** {st['notes']}")
    lines += ["", "## 2. Character Profiles", ""]

    if chars:
        for c in chars:
            lines += [
                f"### {c.get('name','?')}",
                f"- **Gender:** {c.get('gender','?')}",
                f"- **Role:** {c.get('role','?')}",
                f"- **Speech style:** {c.get('speech','?')}",
                f"- **Address to others:** {c.get('address_to_others','?')}",
                f"- **Others address them:** {c.get('address_from_others','?')}",
            ]
            if c.get("note"): lines.append(f"- **Notes:** {c['note']}")
            lines.append("")
    else:
        lines += ["*(персонажів не виявлено — заповніть вручну)*", ""]

    lines += ["## 3. Glossary", ""]
    if glos:
        lines += ["| Original | Ukrainian | Explanation |",
                  "|----------|-----------|-------------|"]
        for g in glos:
            o = g.get("original","").replace("|","\\|")
            u = g.get("suggested_ua","").replace("|","\\|")
            e = g.get("explanation","").replace("|","\\|")
            lines.append(f"| {o} | {u} | {e} |")
    else:
        lines += ["*(глосарій порожній — топ власних назв зі статистики:)*", "",
                  "| Original | Ukrainian | Explanation |",
                  "|----------|-----------|-------------|"]
        for word, count in proper_nouns[:30]:
            lines.append(f"| {word} | | ({count}×) |")
    lines += ["", "## 4. Translation Notes", ""]

    if chals:
        lines += ["### Potential challenges", ""]
        for ch in chals:
            lines.append(f"- **`{ch.get('original','')}`** — {ch.get('challenge','')}")
        lines.append("")

    lines += [
        "### Do NOT translate",
        "*(власні назви, бренди, магічні слова що лишаються як є)*",
        "",
        "### Recurring patterns",
        "*(особливості авторського стилю)*",
        "",
    ]

    if proper_nouns:
        lines += ["## 5. Frequent Proper Nouns (auto-detected)", "",
                  "| Term | Count |", "|------|-------|"]
        for word, count in proper_nouns[:40]:
            lines.append(f"| {word} | {count} |")
        lines.append("")

    result = "\n".join(lines)
    if len(result) > GUIDE_MAX_CHARS * 2:
        console.print(
            f"[yellow]⚠ Гід великий ({len(result):,} симв). "
            f"Рекомендується скоротити до ~{GUIDE_MAX_CHARS} для контексту перекладача.[/]"
        )
    return result


# ── Головна логіка ────────────────────────────────────────────────────────────

def build_guide(input_path: Path, output_path: Optional[Path],
                skip_llm: bool, skip_interactive: bool):

    console.rule("[bold yellow]📖 Build Guide[/]")

    # Заголовок з моделями
    info = Table(box=box.SIMPLE, show_header=False, padding=(0,1))
    info.add_column(style="bold cyan", min_width=18)
    info.add_column()
    info.add_row("🔍 Аналіз (LLM):", ANALYSIS_MODEL)
    info.add_row("🌐 Переклад:", TRANSLATE_MODEL)
    console.print(info)
    console.print()

    # Читаємо
    with console.status(f"[cyan]Читаємо {input_path.name}...[/]"):
        try:
            text, fmt = read_text(input_path)
        except Exception as e:
            console.print(f"[red]Помилка читання: {e}[/]")
            sys.exit(1)

    console.print(f"[green]✓[/] Прочитано: [cyan]{len(text):,}[/] символів, формат [cyan]{fmt.upper()}[/]")

    # Статистика
    with console.status("[cyan]Статистичний аналіз...[/]"):
        lang         = detect_lang(text)
        proper_nouns = extract_proper_nouns(text, lang)
        sample       = sample_text(text)

    console.print(
        f"[green]✓[/] Мова: [cyan]{lang}[/]  |  "
        f"Власних назв: [cyan]{len(proper_nouns)}[/]  |  "
        f"Вибірка: [cyan]{len(sample):,}[/] символів"
    )

    data: dict = {
        "overview": {"lang": lang},
        "style": {}, "characters": [],
        "glossary": [], "translation_challenges": [],
    }

    # LLM аналіз
    if not skip_llm:
        console.print("\n[bold cyan]── Перевірка моделей Ollama ──[/]")
        analysis_model, _ = check_ollama_models()

        if not analysis_model:
            console.print("[yellow]⚠ Модель для аналізу не знайдена, пропускаємо LLM.[/]")
        else:
            console.print()
            result = analyze_with_llm(analysis_model, sample)
            if result:
                # Зберігаємо overview окремо щоб не перетерти
                overview_backup = data["overview"]
                data.update(result)
                data["overview"] = {**result.get("overview", {}), **overview_backup}
                console.print(
                    f"[green]✓[/] LLM виявив: "
                    f"[cyan]{len(data.get('characters',[]))}[/] персонажів, "
                    f"[cyan]{len(data.get('glossary',[]))}[/] термінів, "
                    f"[cyan]{len(data.get('translation_challenges',[]))}[/] складнощів"
                )
            else:
                console.print("[yellow]⚠ LLM не повернув результат, продовжуємо зі статистикою.[/]")
    else:
        console.print("[dim]LLM-аналіз пропущено (--no-llm)[/dim]")

    # Інтерактив
    if not skip_interactive:
        data = interactive_fill(data, input_path.stem)

    # Генеруємо .md
    md_content = build_md(data, proper_nouns, ANALYSIS_MODEL, TRANSLATE_MODEL)

    if output_path is None:
        output_path = input_path.with_suffix(".guide.md")

    output_path.write_text(md_content, encoding="utf-8")

    # Підсумок
    rt = Table(box=box.DOUBLE_EDGE, border_style="green", show_header=False, padding=(0,2))
    rt.add_column(justify="right", style="bold", min_width=20)
    rt.add_column()
    rt.add_row("📄 Гід збережено:", str(output_path))
    rt.add_row("📏 Розмір:",        f"{len(md_content):,} символів")
    rt.add_row("👥 Персонажів:",    str(len(data.get("characters", []))))
    rt.add_row("📚 Термінів:",      str(len(data.get("glossary", []))))
    rt.add_row("🔤 Власних назв:",  str(len(proper_nouns)))
    rt.add_row("", "")
    rt.add_row("▶ Далі:",           f"mcedit {output_path.name}  →  ./translate book.fb2")
    console.print()
    console.print(Panel(rt, title="[bold green]✓ Гід готовий![/]", border_style="green"))
    console.print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Будує контекстний гід перекладача (.guide.md) для FB2/EPUB/TXT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  python build_guide.py book.fb2
  python build_guide.py book.epub -o myguide.md
  python build_guide.py book.fb2 --analysis-model qwen2.5:7b-instruct-q4_K_M
  python build_guide.py book.txt --no-llm
  python build_guide.py book.fb2 --no-interactive
""")
    parser.add_argument("input",
        help="Вхідний файл (fb2/epub/txt)")
    parser.add_argument("-o", "--output",
        help="Вихідний .md (за замовч.: <name>.guide.md)")
    parser.add_argument("--analysis-model", default=None,
        metavar="MODEL",
        help=f"Модель для аналізу/JSON (за замовч.: {ANALYSIS_MODEL})")
    parser.add_argument("--translate-model", default=None,
        metavar="MODEL",
        help=f"Модель для перекладу — записується в гід (за замовч.: {TRANSLATE_MODEL})")
    parser.add_argument("--no-llm", action="store_true",
        help="Пропустити LLM-аналіз (тільки статистика)")
    parser.add_argument("--no-interactive", action="store_true",
        help="Пропустити інтерактивне заповнення")
    parser.add_argument("--ollama-url", default=None,
        help=f"URL Ollama (за замовч.: {OLLAMA_URL})")
    args = parser.parse_args()

    global OLLAMA_URL, ANALYSIS_MODEL, TRANSLATE_MODEL
    if args.ollama_url:      OLLAMA_URL      = args.ollama_url
    if args.analysis_model:  ANALYSIS_MODEL  = args.analysis_model
    if args.translate_model: TRANSLATE_MODEL = args.translate_model

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Файл не знайдено: {input_path}[/]")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None
    build_guide(input_path, output_path, args.no_llm, args.no_interactive)


if __name__ == "__main__":
    main()

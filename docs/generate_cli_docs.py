#!/usr/bin/env python3
"""Generate Sphinx RST pages for VINE's command-line tools from the
.help_src files in src/progs/. These files are the source of truth for
each program's --help text, so this keeps the CLI reference in sync
with actual program behavior instead of duplicating it by hand.

Usage: python3 docs/generate_cli_docs.py
"""
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGS = ROOT / "src" / "progs"
OUT = ROOT / "docs" / "source" / "cli"

SECTION_RE = re.compile(r'^[A-Z][A-Z0-9 /\-]*:$')
OPTION_RE = re.compile(r'^ {4}(-{1,2}\S.*)$')


def escape_rst(text):
    # source text is plain English, not RST -- escape characters that
    # would otherwise be parsed as (unintended) inline markup.
    return text.replace("*", "\\*")


def parse_help_src(text):
    lines = text.split("\n")
    i = 0
    assert lines[0].startswith("PROGRAM:")
    title = lines[0][len("PROGRAM:"):].strip()
    i = 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    assert lines[i].startswith("USAGE:")
    usage = [lines[i][len("USAGE:"):].strip()]
    i += 1
    while i < len(lines) and lines[i].strip() != "":
        usage.append(lines[i].strip())
        i += 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    assert lines[i].startswith("DESCRIPTION:")
    desc = [lines[i][len("DESCRIPTION:"):].strip()]
    i += 1
    while i < len(lines) and not SECTION_RE.match(lines[i]):
        desc.append(lines[i])
        i += 1
    while desc and desc[-1].strip() == "":
        desc.pop()

    sections = []
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        m = SECTION_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name = lines[i][:-1].strip()
        i += 1
        body = []
        while i < len(lines) and not SECTION_RE.match(lines[i]):
            body.append(lines[i])
            i += 1
        while body and body[-1].strip() == "":
            body.pop()
        while body and body[0].strip() == "":
            body.pop(0)
        sections.append((name, body))
    return title, usage, desc, sections


def render_options(body_lines):
    out = []
    i, n = 0, len(body_lines)
    while i < n:
        line = body_lines[i]
        m = OPTION_RE.match(line)
        if m:
            header = m.group(1).strip()
            i += 1
            desc = []
            while i < n and body_lines[i].strip() != "" and not OPTION_RE.match(body_lines[i]):
                desc.append(body_lines[i].strip())
                i += 1
            desc_text = escape_rst(" ".join(desc)) if desc else "(no description)"
            out.append(f"``{header}``\n    {desc_text}\n")
        else:
            i += 1
    return "\n".join(out)


def render_commands(body_lines):
    text = "\n".join(body_lines)
    text = textwrap.dedent(text)
    indented = "\n".join(("    " + l if l.strip() else "") for l in text.split("\n"))
    return ".. code-block:: console\n\n" + indented + "\n"


def render_page(name, title, usage, desc, sections):
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append(".. code-block:: console")
    lines.append("")
    for u in usage:
        lines.append("    " + u)
    lines.append("")
    lines.append(escape_rst("\n".join(desc)))
    lines.append("")
    for sec_name, body in sections:
        lines.append(sec_name.title())
        lines.append("-" * len(sec_name.title()))
        lines.append("")
        if sec_name.upper() == "TYPICAL COMMANDS":
            lines.append(render_commands(body))
        else:
            lines.append(render_options(body))
        lines.append("")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generated = []
    for help_src in sorted(PROGS.glob("*.help_src")):
        name = help_src.stem
        title, usage, desc, sections = parse_help_src(help_src.read_text())
        rst = render_page(name, title, usage, desc, sections)
        (OUT / f"{name}.rst").write_text(rst)
        generated.append(name)
        print(f"wrote {OUT / (name + '.rst')}")

    # vine is the primary tool; list it first, then the rest alphabetically
    generated.sort(key=lambda n: (n != "vine", n))

    # Some programs (e.g. crisprLnl) have no .help_src and are documented
    # by a hand-maintained .rst file directly in cli/ -- pick those up too
    # so the index stays complete without needing to be hand-edited.
    hand_maintained = sorted(
        p.stem for p in OUT.glob("*.rst")
        if p.stem not in generated and p.stem != "index"
    )
    pages = generated + hand_maintained

    index = ["Command-Line Reference", "=======================", "",
             "VINE installs the following command-line programs.", "",
             ".. toctree::", "   :maxdepth: 1", ""]
    for p in pages:
        index.append(f"   {p}")
    (OUT / "index.rst").write_text("\n".join(index) + "\n")
    print(f"wrote {OUT / 'index.rst'}")


if __name__ == "__main__":
    main()

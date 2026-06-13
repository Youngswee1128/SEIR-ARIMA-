from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}

FIG_REF_RE = re.compile(r"(?:图|Figure|Fig\.?)\s*[0-9A-Za-z]+(?:[-\.][0-9A-Za-z]+)?", re.I)
CAPTION_RE = re.compile(r"^\s*(?:图|Figure|Fig\.?)\s*([0-9A-Za-z]+(?:[-\.][0-9A-Za-z]+)?)", re.I)


def qn(prefix: str, name: str) -> str:
    return f"{{{NS[prefix]}}}{name}"


def paragraph_text(p: ET.Element) -> str:
    parts: list[str] = []
    for node in p.iter():
        if node.tag == qn("w", "t") and node.text:
            parts.append(node.text)
        elif node.tag == qn("w", "tab"):
            parts.append("\t")
        elif node.tag == qn("w", "br"):
            parts.append("\n")
    return "".join(parts)


def paragraph_style(p: ET.Element) -> str:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        return ""
    pstyle = ppr.find("w:pStyle", NS)
    if pstyle is None:
        return ""
    return pstyle.attrib.get(qn("w", "val"), "")


def drawing_rels(p: ET.Element) -> list[str]:
    rels: list[str] = []
    for blip in p.findall(".//a:blip", NS):
        rid = blip.attrib.get(qn("r", "embed")) or blip.attrib.get(qn("r", "link"))
        if rid:
            rels.append(rid)
    return rels


def load_rels(zf: zipfile.ZipFile) -> dict[str, str]:
    rels_path = "word/_rels/document.xml.rels"
    if rels_path not in zf.namelist():
        return {}
    root = ET.fromstring(zf.read(rels_path))
    out: dict[str, str] = {}
    for rel in root:
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rid and target:
            out[rid] = target
    return out


def inspect_docx(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
        rels = load_rels(zf)
        body = root.find("w:body", NS)
        paragraphs = []
        figures = []
        captions = []
        refs = []
        if body is None:
            return {"path": str(path), "paragraphs": []}
        for idx, p in enumerate(body.findall("w:p", NS), start=1):
            text = paragraph_text(p).strip()
            style = paragraph_style(p)
            rel_ids = drawing_rels(p)
            media = [rels.get(rid, rid) for rid in rel_ids]
            item = {
                "idx": idx,
                "style": style,
                "text": text,
                "has_drawing": bool(rel_ids),
                "media": media,
            }
            paragraphs.append(item)
            if rel_ids:
                figures.append(item)
            cap = CAPTION_RE.match(text)
            if cap:
                captions.append({**item, "number": cap.group(1)})
            matches = FIG_REF_RE.findall(text)
            if matches:
                refs.append({**item, "refs": matches})
    return {
        "path": str(path),
        "paragraph_count": len(paragraphs),
        "figure_paragraph_count": len(figures),
        "caption_count": len(captions),
        "paragraphs": paragraphs,
        "figures": figures,
        "captions": captions,
        "refs": refs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    reports = [inspect_docx(path) for path in args.docx]
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    for report in reports:
        print(f"FILE: {report['path']}")
        print(
            f"paragraphs={report['paragraph_count']} "
            f"figure_paragraphs={report['figure_paragraph_count']} "
            f"captions={report['caption_count']}"
        )
        print("FIGURE PARAGRAPHS")
        for fig in report["figures"]:
            print(f"  P{fig['idx']}: media={fig['media']} text={fig['text'][:100]}")
        print("CAPTIONS")
        for cap in report["captions"]:
            print(f"  P{cap['idx']}: {cap['text']}")
        print("TEXT REFERENCES")
        for ref in report["refs"]:
            print(f"  P{ref['idx']}: {ref['refs']} | {ref['text'][:160]}")
        if args.full:
            print("PARAGRAPHS")
            for p in report["paragraphs"]:
                marker = " [IMG]" if p["has_drawing"] else ""
                print(f"  P{p['idx']} {p['style']}{marker}: {p['text']}")


if __name__ == "__main__":
    main()

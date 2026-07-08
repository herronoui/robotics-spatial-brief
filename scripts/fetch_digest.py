#!/usr/bin/env python3
"""Fetch and categorize robotics + spatial intelligence news worldwide."""

from __future__ import annotations

import json
import re
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "digest.json"

USER_AGENT = "robotics-spatial-brief/1.0 (github.com/herronoui/robotics-spatial-brief)"
MAX_PER_TOPIC = 15
SUMMARY_LEN = 200

SPATIAL_KEYWORDS = (
    "spatial", "3d", "world model", "scene", "nerf", "gaussian splat",
    "slam", "depth", "point cloud", "embodied", "vision-language", "vla",
    "vlm", "grounding", "navigation", "mapping", "reconstruction",
    "sim-to-real", "sim2real", "video prediction", "occupancy",
)

ROBOTICS_KEYWORDS = (
    "robot", "robotics", "humanoid", "manipul", "locomotion", "gripper",
    "actuator", "drone", "uav", "autonomous", "haptic", "soft robot",
    "warehouse", "factory", "surgical",
)

REGION_HINTS = {
    "United States": ("mit", "stanford", "berkeley", "carnegie", "boston dynamics", "tesla", "figure ai", "nvidia", "google", "darpa", "nasa"),
    "Europe": ("abb", "kuka", "germany", "france", "oxford", "eth zurich", "epfl"),
    "China & Asia": ("china", "chinese", "beijing", "shanghai", "unitree", "dji", "tsinghua", "korea", "korean", "hyundai"),
    "Japan & Korea": ("japan", "japanese", "tokyo", "sony", "honda", "toyota", "softbank"),
}

SOURCES = [
    {"name": "arXiv Robotics", "voice": "Global Research", "url": "https://rss.arxiv.org/rss/cs.RO", "default_topic": "robotics"},
    {"name": "arXiv Computer Vision", "voice": "Global Research", "url": "https://rss.arxiv.org/rss/cs.CV", "default_topic": "spatial", "keywords": list(SPATIAL_KEYWORDS)},
    {"name": "arXiv AI", "voice": "Global Research", "url": "https://rss.arxiv.org/rss/cs.AI", "default_topic": "spatial", "keywords": list(SPATIAL_KEYWORDS + ROBOTICS_KEYWORDS)},
    {"name": "IEEE Spectrum", "voice": "IEEE", "url": "https://spectrum.ieee.org/rss/fulltext", "keywords": list(ROBOTICS_KEYWORDS + SPATIAL_KEYWORDS)},
    {"name": "MIT News", "voice": "MIT", "url": "https://news.mit.edu/rss/research", "keywords": list(ROBOTICS_KEYWORDS + SPATIAL_KEYWORDS)},
    {"name": "The Robot Report", "voice": "Industry", "url": "https://www.therobotreport.com/feed/", "default_topic": "robotics"},
    {"name": "TechCrunch Robotics", "voice": "Startup Scene", "url": "https://techcrunch.com/category/robotics/feed/", "default_topic": "robotics"},
    {"name": "Robohub", "voice": "Global Community", "url": "https://robohub.org/feed/"},
]


def fetch_xml(url: str) -> ET.Element | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for ctx in (ssl.create_default_context(), ssl._create_unverified_context()):
        try:
            with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
                return ET.fromstring(resp.read())
        except Exception:
            continue
    print(f"  skip {url}")
    return None


def strip_html(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def summarize(text: str, title: str) -> str:
    text = strip_html(text)
    if not text:
        return f"Latest update: {title}."
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in parts:
        candidate = (out + " " + s).strip()
        if len(candidate) > SUMMARY_LEN and out:
            break
        out = candidate
        if len(out) >= 100:
            break
    if not out:
        out = text[:SUMMARY_LEN]
    if len(out) > SUMMARY_LEN:
        out = out[: SUMMARY_LEN - 1].rsplit(" ", 1)[0] + "…"
    return out


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value[:19], fmt[:19])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def find_child(parent: ET.Element, name: str) -> ET.Element | None:
    for child in parent:
        if local_tag(child.tag) == name:
            return child
    return None


def find_text(parent: ET.Element, name: str) -> str:
    node = find_child(parent, name)
    return (node.text or "").strip() if node is not None else ""


def classify_topic(title: str, desc: str, default: str | None) -> str:
    blob = f"{title} {desc}".lower()
    spatial = sum(1 for k in SPATIAL_KEYWORDS if k in blob)
    robotics = sum(1 for k in ROBOTICS_KEYWORDS if k in blob)
    if spatial > robotics:
        return "spatial"
    if robotics > spatial:
        return "robotics"
    return default or "robotics"


def infer_region(title: str, desc: str, source: str) -> str:
    blob = f"{title} {desc} {source}".lower()
    for region, hints in REGION_HINTS.items():
        if any(h in blob for h in hints):
            return region
    return "Global"


def blob_matches_keywords(blob: str, keywords: list[str] | None) -> bool:
    if not keywords:
        return True
    b = blob.lower()
    return any(k.lower() in b for k in keywords)


def parse_items(root: ET.Element, source: dict) -> list[dict]:
    items: list[dict] = []
    channel = root.find("channel")
    if channel is not None:
        entries = channel.findall("item")
    else:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns) or root.findall("entry")

    for entry in entries:
        if local_tag(entry.tag) == "item":
            title = find_text(entry, "title")
            link = find_text(entry, "link")
            if not link:
                link_node = find_child(entry, "link")
                if link_node is not None:
                    link = link_node.text or link_node.get("href") or ""
            desc = find_text(entry, "description") or find_text(entry, "content")
            pub = find_text(entry, "pubDate") or find_text(entry, "date")
        else:
            title = find_text(entry, "title")
            link = entry.get("href") or ""
            link_node = find_child(entry, "link")
            if link_node is not None:
                link = link_node.get("href") or link_node.text or ""
            summary_node = find_child(entry, "summary")
            if summary_node is None:
                summary_node = find_child(entry, "content")
            desc = summary_node.text if summary_node is not None else ""
            pub = find_text(entry, "published") or find_text(entry, "updated")

        title = strip_html(title)
        link = link.strip()
        if not title or not link:
            continue

        blob = f"{title} {desc}"
        if not blob_matches_keywords(blob, source.get("keywords")):
            continue

        topic = classify_topic(title, desc, source.get("default_topic"))
        region = infer_region(title, desc, source["name"])
        published = parse_date(pub)

        items.append({
            "title": title,
            "summary": summarize(desc, title),
            "topic": topic,
            "region": region,
            "source": source["name"],
            "voice": source["voice"],
            "url": link,
            "published": published.isoformat() if published else None,
        })
    return items


def dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        key = re.sub(r"\W+", "", it["title"].lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def main() -> None:
    all_items: list[dict] = []
    for src in SOURCES:
        print(f"Fetching {src['name']}…")
        root = fetch_xml(src["url"])
        if root is None:
            continue
        batch = parse_items(root, src)
        print(f"  {len(batch)} items")
        all_items.extend(batch)

    all_items = dedupe(all_items)
    all_items.sort(key=lambda x: x.get("published") or "", reverse=True)

    robotics = [i for i in all_items if i["topic"] == "robotics"][:MAX_PER_TOPIC]
    spatial = [i for i in all_items if i["topic"] == "spatial"][:MAX_PER_TOPIC]

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "robotics": robotics,
        "spatial": spatial,
        "all_count": len(all_items),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(robotics)} robotics + {len(spatial)} spatial items")


if __name__ == "__main__":
    main()

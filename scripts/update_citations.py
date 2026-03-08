"""
Fetch papers citing DeepSpot from Semantic Scholar and update README.md.

Uses the Semantic Scholar API (no key required for basic usage).
DeepSpot DOI: 10.1101/2025.02.09.25321567
"""

import json
import re
import urllib.request
import urllib.error
from pathlib import Path

DEEPSPOT_DOI = "10.1101/2025.02.09.25321567"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
README_PATH = Path(__file__).parent.parent / "README.md"

SECTION_START = "<!-- CITATIONS:START -->"
SECTION_END = "<!-- CITATIONS:END -->"


def fetch_citing_papers():
    """Fetch papers that cite DeepSpot from Semantic Scholar."""
    url = (
        f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{DEEPSPOT_DOI}/citations"
        f"?fields=title,authors,year,externalIds,url,venue"
        f"&limit=500"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "DeepSpot-Citation-Bot"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"API error: {e.code} {e.reason}")
        return []

    papers = []
    for item in data.get("data", []):
        paper = item.get("citingPaper", {})
        if not paper.get("title"):
            continue

        authors = paper.get("authors", [])
        author_str = authors[0]["name"] if authors else "Unknown"
        if len(authors) > 1:
            author_str += " et al."

        doi = paper.get("externalIds", {}).get("DOI", "")
        link = f"https://doi.org/{doi}" if doi else paper.get("url", "")
        year = paper.get("year", "")
        venue = paper.get("venue", "")

        papers.append({
            "title": paper["title"],
            "authors": author_str,
            "year": year,
            "venue": venue,
            "link": link,
        })

    # Sort by year descending
    papers.sort(key=lambda p: p.get("year") or 0, reverse=True)
    return papers


def format_citations(papers):
    """Format papers as a markdown list."""
    if not papers:
        return "No citations found yet. Check back soon!\n"

    lines = []
    for p in papers:
        entry = f"- **{p['title']}**"
        meta_parts = [p["authors"]]
        if p["venue"]:
            meta_parts.append(p["venue"])
        if p["year"]:
            meta_parts.append(str(p["year"]))
        entry += f" — {', '.join(meta_parts)}."
        if p["link"]:
            entry += f" [Link]({p['link']})"
        lines.append(entry)

    return "\n".join(lines) + "\n"


def update_readme(papers):
    """Update the citations section in README.md."""
    readme = README_PATH.read_text()

    citations_md = format_citations(papers)
    new_section = f"{SECTION_START}\n{citations_md}{SECTION_END}"

    if SECTION_START in readme and SECTION_END in readme:
        pattern = re.escape(SECTION_START) + r".*?" + re.escape(SECTION_END)
        readme = re.sub(pattern, new_section, readme, flags=re.DOTALL)
    else:
        print("Citation markers not found in README.md. Please add them manually.")
        return False

    README_PATH.write_text(readme)
    print(f"Updated README with {len(papers)} citing paper(s).")
    return True


if __name__ == "__main__":
    papers = fetch_citing_papers()
    update_readme(papers)

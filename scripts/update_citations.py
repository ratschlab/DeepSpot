"""
Fetch papers citing DeepSpot and update README.md.

Combines results from Semantic Scholar and OpenCitations APIs to maximize
coverage. Uses Semantic Scholar for metadata and OpenCitations to discover
additional citing DOIs.

DeepSpot DOI: 10.1101/2025.02.09.25321567
"""

import json
import re
import urllib.request
import urllib.error
from pathlib import Path

DEEPSPOT_DOI = "10.1101/2025.02.09.25321567"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
OPENCITATIONS_API = "https://opencitations.net/index/api/v2"
README_PATH = Path(__file__).parent.parent / "README.md"

SECTION_START = "<!-- CITATIONS:START -->"
SECTION_END = "<!-- CITATIONS:END -->"

SS_FIELDS = "title,authors,year,externalIds,url,venue,journal,publicationDate"


def _api_get(url):
    """Make a GET request and return parsed JSON, or None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "DeepSpot-Citation-Bot"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"API error for {url}: {e}")
        return None


def fetch_ss_citations():
    """Fetch citing papers from Semantic Scholar."""
    url = (
        f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{DEEPSPOT_DOI}/citations"
        f"?fields={SS_FIELDS}&limit=500"
    )
    data = _api_get(url)
    if not data:
        return {}

    papers = {}
    for item in data.get("data", []):
        paper = item.get("citingPaper", {})
        doi = paper.get("externalIds", {}).get("DOI", "")
        if not paper.get("title"):
            continue
        key = doi.lower() if doi else paper.get("paperId", "")
        papers[key] = _parse_ss_paper(paper)
    return papers


def fetch_opencitations_dois():
    """Fetch citing DOIs from OpenCitations (may find papers Semantic Scholar misses)."""
    url = f"{OPENCITATIONS_API}/citations/doi:{DEEPSPOT_DOI}"
    data = _api_get(url)
    if not data:
        return set()

    dois = set()
    for item in data:
        citing = item.get("citing", "")
        for part in citing.split():
            if part.startswith("doi:"):
                dois.add(part[4:])
    return dois


def fetch_ss_paper_by_doi(doi):
    """Fetch a single paper's metadata from Semantic Scholar by DOI."""
    url = f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{doi}?fields={SS_FIELDS}"
    data = _api_get(url)
    if not data or not data.get("title"):
        return None
    return _parse_ss_paper(data)


def _parse_ss_paper(paper):
    """Parse a Semantic Scholar paper object into our format."""
    authors = [a["name"] for a in paper.get("authors", []) if a.get("name")]
    doi = paper.get("externalIds", {}).get("DOI", "")
    link = f"https://doi.org/{doi}" if doi else paper.get("url", "")
    venue = paper.get("venue", "")
    journal = paper.get("journal", {})
    if journal and journal.get("name") and not venue:
        venue = journal["name"]

    return {
        "title": paper["title"],
        "authors": authors,
        "year": paper.get("year", ""),
        "venue": venue,
        "link": link,
        "date": paper.get("publicationDate", ""),
    }


def fetch_all_citing_papers():
    """Combine Semantic Scholar and OpenCitations to find all citing papers."""
    # Fetch from Semantic Scholar (primary source with full metadata)
    papers = fetch_ss_citations()
    print(f"Semantic Scholar: {len(papers)} citing paper(s)")

    # Fetch citing DOIs from OpenCitations and fill in any missing ones
    oc_dois = fetch_opencitations_dois()
    print(f"OpenCitations: {len(oc_dois)} citing DOI(s)")

    missing_dois = [doi for doi in oc_dois if doi.lower() not in papers]
    for doi in missing_dois:
        paper = fetch_ss_paper_by_doi(doi)
        if paper:
            papers[doi.lower()] = paper
            print(f"  Added from OpenCitations: {paper['title']}")

    all_papers = list(papers.values())
    # Sort by publication date descending, then year
    all_papers.sort(
        key=lambda p: p.get("date") or str(p.get("year", "")),
        reverse=True,
    )
    print(f"Total: {len(all_papers)} unique citing paper(s)")
    return all_papers


def format_authors(authors):
    """Format author list: 'First Author, Second Author, ..., and Last Author'."""
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + ", and " + authors[-1]


def format_citations(papers):
    """Format papers as a numbered markdown list in academic citation style."""
    if not papers:
        return "No citations found yet. Check back soon!\n"

    lines = []
    for i, p in enumerate(papers, 1):
        author_str = format_authors(p["authors"])
        parts = [f"{i}. {author_str}"]
        parts.append(f'"{p["title"]}."')
        if p["venue"]:
            parts.append(f'*{p["venue"]}*')
        if p["year"]:
            parts.append(f'({p["year"]}).')
        else:
            parts[-1] += "."
        entry = " ".join(parts)
        if p["link"]:
            entry += f" [DOI]({p['link']})"
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
    papers = fetch_all_citing_papers()
    update_readme(papers)

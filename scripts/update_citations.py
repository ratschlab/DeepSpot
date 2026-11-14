"""
Fetch papers citing DeepSpot and update README.md.

Sources, combined and deduplicated (by DOI, then by normalised title):
  - Semantic Scholar: citations of each DeepSpot DOI (all pages; optional
    SEMANTIC_SCHOLAR_API_KEY raises the rate limit)
  - OpenAlex: works that cite DeepSpot (all pages)
  - OpenCitations: citing DOIs; metadata for DOIs the other two miss comes
    from OpenAlex, else Semantic Scholar

Every paper ever found is kept in scripts/citations.json and is never dropped
because a source failed or missed it (a Semantic Scholar outage once shrank
the list from 21 papers to 1 for a week). Requests are retried on rate limits
and server errors. If every source fails, nothing is written and the script
exits with status 1. To remove a paper, delete it from citations.json.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Every DOI under which DeepSpot can be cited (add the journal DOI once published).
DEEPSPOT_DOIS = ["10.1101/2025.02.09.25321567"]
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
OPENALEX_API = "https://api.openalex.org"
OPENCITATIONS_API = "https://opencitations.net/index/api/v2"
README_PATH = Path(__file__).resolve().parent.parent / "README.md"
STORE_PATH = Path(__file__).resolve().parent / "citations.json"

SECTION_START = "<!-- CITATIONS:START -->"
SECTION_END = "<!-- CITATIONS:END -->"

SS_FIELDS = "title,authors,year,externalIds,url,venue,journal,publicationDate"
USER_AGENT = "DeepSpot-Citation-Bot (https://github.com/ratschlab/DeepSpot)"
RETRY_STATUS = {429, 500, 502, 503, 504}


def _api_get(url, headers=None, attempts=6):
    """GET and parse JSON, retrying rate limits, server and network errors; None if it still fails."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS:
                print(f"API error {e.code} for {url}")
                return None
            retry_after = e.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else 2 ** attempt
            reason = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError, ValueError) as e:  # network errors, timeouts, bad JSON
            wait, reason = 2 ** attempt, type(e).__name__
        if attempt < attempts:
            print(f"  {reason}, retry {attempt}/{attempts - 1} in {min(wait, 60)} s: {url}")
            time.sleep(min(wait, 60))
    print(f"API failed after {attempts} attempts: {url}")
    return None


def _doi_of(link):
    match = re.search(r"doi\.org/(.+)$", link or "", re.IGNORECASE)
    return match.group(1).strip().lower() if match else ""


def _norm_title(title):
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


# ---------------------------------------------------------------- sources

def _parse_ss_paper(paper):
    """Semantic Scholar paper object -> our record."""
    doi = (paper.get("externalIds") or {}).get("DOI", "")
    venue = paper.get("venue") or ((paper.get("journal") or {}).get("name") or "")
    return {
        "title": paper["title"],
        "authors": [a["name"] for a in paper.get("authors", []) if a.get("name")],
        "year": paper.get("year") or "",
        "venue": venue,
        "link": f"https://doi.org/{doi}" if doi else paper.get("url", ""),
        "date": paper.get("publicationDate") or "",
    }


def _parse_openalex_work(work):
    """OpenAlex work object -> our record."""
    doi = _doi_of(work.get("doi") or "")
    source = (work.get("primary_location") or {}).get("source") or {}
    return {
        "title": work.get("display_name") or work.get("title") or "",
        "authors": [a["author"]["display_name"] for a in work.get("authorships", [])
                    if (a.get("author") or {}).get("display_name")],
        "year": work.get("publication_year") or "",
        "venue": source.get("display_name") or "",
        "link": f"https://doi.org/{doi}" if doi else work.get("id", ""),
        "date": work.get("publication_date") or "",
    }


def fetch_semantic_scholar(doi):
    """Citing papers from Semantic Scholar (all pages), or None if the API fails."""
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"x-api-key": key} if key else {}
    papers, offset = [], 0
    while offset is not None:
        data = _api_get(f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{doi}/citations"
                        f"?fields={SS_FIELDS}&limit=1000&offset={offset}", headers)
        if data is None:
            return None
        papers += [_parse_ss_paper(item["citingPaper"]) for item in data.get("data", [])
                   if (item.get("citingPaper") or {}).get("title")]
        offset = data.get("next")
    return papers


def fetch_openalex(doi):
    """Citing works from OpenAlex (all pages), or None if the API fails."""
    work = _api_get(f"{OPENALEX_API}/works/doi:{doi}")
    if work is None:
        return None
    work_id = work["id"].rsplit("/", 1)[-1]
    papers, cursor = [], "*"
    while cursor:
        data = _api_get(f"{OPENALEX_API}/works?filter=cites:{work_id}&per-page=200"
                        f"&cursor={urllib.parse.quote(cursor)}")
        if data is None:
            return None
        papers += [_parse_openalex_work(w) for w in data.get("results", []) if w.get("display_name") or w.get("title")]
        cursor = (data.get("meta") or {}).get("next_cursor")
    return papers


def fetch_opencitations_dois(doi):
    """Citing DOIs from OpenCitations, or None if the API fails."""
    data = _api_get(f"{OPENCITATIONS_API}/citations/doi:{doi}")
    if data is None:
        return None
    return {part[4:].lower() for item in data for part in item.get("citing", "").split() if part.startswith("doi:")}


def fetch_metadata_by_doi(doi):
    """One paper's record by DOI: OpenAlex first, then Semantic Scholar."""
    work = _api_get(f"{OPENALEX_API}/works/doi:{doi}")
    if work and (work.get("display_name") or work.get("title")):
        return _parse_openalex_work(work)
    paper = _api_get(f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{doi}?fields={SS_FIELDS}")
    return _parse_ss_paper(paper) if paper and paper.get("title") else None


def fetch_all_citing_papers():
    """Records from all sources, or None if every source failed."""
    found, ok_sources = [], 0
    for doi in DEEPSPOT_DOIS:
        for name, fetch in (("Semantic Scholar", fetch_semantic_scholar), ("OpenAlex", fetch_openalex)):
            papers = fetch(doi)
            print(f"{name}: {'FAILED' if papers is None else len(papers)} citing paper(s) of {doi}")
            if papers is not None:
                ok_sources += 1
                found += papers
        oc_dois = fetch_opencitations_dois(doi)
        print(f"OpenCitations: {'FAILED' if oc_dois is None else len(oc_dois)} citing DOI(s) of {doi}")
        if oc_dois is not None:
            ok_sources += 1
            known = {_doi_of(p["link"]) for p in found}
            for citing in sorted(oc_dois - known):
                paper = fetch_metadata_by_doi(citing)
                if paper:
                    found.append(paper)
                else:
                    print(f"  no metadata for OpenCitations DOI {citing}")
    return found if ok_sources else None


# ---------------------------------------------------------------- store

ENTRY_RE = re.compile(r'^\d+\.\s+(?P<authors>.+?)\s+"(?P<title>.+?)\.?"\.?'
                      r'(?:\s+\*(?P<venue>.+?)\*\.?)?(?:\s+\((?P<year>\d{4})\)\.)?'
                      r'(?:\s+\[DOI\]\((?P<link>[^)\s]+)\))?\s*$')


def seed_from_readme(readme):
    """Records parsed from the README list (used once, when citations.json does not exist yet)."""
    if SECTION_START not in readme or SECTION_END not in readme:
        return []
    records = []
    for line in readme.split(SECTION_START, 1)[1].split(SECTION_END, 1)[0].splitlines():
        m = ENTRY_RE.match(line.strip())
        if m:
            records.append({"title": m["title"], "authors": m["authors"], "venue": m["venue"] or "",
                            "year": int(m["year"]) if m["year"] else "", "link": m["link"] or "", "date": ""})
        elif line.strip() and not line.startswith("No citations"):
            sys.exit(f"cannot parse README citation line: {line!r}")
    return records


def format_authors(authors):
    """Format author list: 'First Author, Second Author, ..., and Last Author'."""
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + ", and " + authors[-1]


def merge(store, papers):
    """Add new papers to the store; for known ones only fill empty fields or upgrade the link to a DOI.

    The author string of a known paper is kept as stored: Semantic Scholar and OpenAlex spell names
    differently from run to run, and manual fixes in citations.json must survive.
    """
    own = {d.lower() for d in DEEPSPOT_DOIS}
    by_doi = {_doi_of(r["link"]): r for r in store if _doi_of(r["link"])}
    by_title = {_norm_title(r["title"]): r for r in store}
    added = []
    for p in papers:
        doi = _doi_of(p["link"])
        if doi in own or not _norm_title(p["title"]):
            continue
        rec = by_doi.get(doi) if doi else None
        rec = rec or by_title.get(_norm_title(p["title"]))
        if rec is None:
            rec = {"title": p["title"].strip(), "authors": format_authors(p["authors"]), "venue": p["venue"] or "",
                   "year": p["year"] or "", "link": p["link"] or "", "date": p["date"] or ""}
            store.append(rec)
            added.append(rec)
        else:
            for field in ("venue", "year", "date"):
                if not rec.get(field) and p.get(field):
                    rec[field] = p[field]
            if doi and not _doi_of(rec["link"]):
                rec["link"] = p["link"]
        if _doi_of(rec["link"]):
            by_doi[_doi_of(rec["link"])] = rec
        by_title[_norm_title(rec["title"])] = rec
    return added


def sort_key(r):
    # Newest first, title as a stable tiebreaker so papers sharing a date don't reorder between runs.
    return (r.get("date") or str(r.get("year", "")), r.get("title", ""))


def format_citations(records):
    """Format records as a numbered markdown list in academic citation style."""
    if not records:
        return "No citations found yet. Check back soon!\n"
    lines = []
    for i, r in enumerate(records, 1):
        parts = [f"{i}. {r['authors']}", f'"{r["title"].rstrip(". ")}."']
        if r["venue"]:
            parts.append(f'*{r["venue"]}*')
        if r["year"]:
            parts.append(f'({r["year"]}).')
        else:
            parts[-1] += "."
        entry = " ".join(parts)
        if r["link"]:
            entry += f" [DOI]({r['link']})"
        lines.append(entry)
    return "\n".join(lines) + "\n"


def main():
    readme = README_PATH.read_text()
    if SECTION_START not in readme or SECTION_END not in readme:
        sys.exit("Citation markers not found in README.md. Please add them manually.")
    store = json.loads(STORE_PATH.read_text()) if STORE_PATH.exists() else seed_from_readme(readme)
    n_before = len(store)

    papers = fetch_all_citing_papers()
    if papers is None:
        sys.exit("ERROR: every citation source failed; README.md and citations.json left unchanged.")
    for rec in merge(store, papers):
        print(f"  new: {rec['title']}")

    store.sort(key=sort_key, reverse=True)
    STORE_PATH.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n")
    new_section = f"{SECTION_START}\n{format_citations(store)}{SECTION_END}"
    pattern = re.escape(SECTION_START) + r".*?" + re.escape(SECTION_END)
    README_PATH.write_text(re.sub(pattern, lambda _: new_section, readme, flags=re.DOTALL))
    print(f"Citing papers: {n_before} known before, {len(store)} now.")


if __name__ == "__main__":
    main()

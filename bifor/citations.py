"""Check each index citation against Crossref: first author and year must match."""
from __future__ import annotations

import re
import time
from pathlib import Path

import httpx
import pandas as pd

from . import indices

API = "https://api.crossref.org/works"
CITE = re.compile(r"([A-Z][A-Za-z\-']+)[^0-9]{0,60}?((?:19|20)\d\d)")


def parse(citation: str) -> tuple[str, int] | None:
    m = CITE.search(citation or "")
    return (m.group(1), int(m.group(2))) if m else None


def check(citation: str, client: httpx.Client) -> dict:
    p = parse(citation)
    if not p:
        return {"status": "unparsed"}
    surname, year = p
    r = _get(client, {"query.bibliographic": citation, "rows": 8,
                      "select": "DOI,title,author,issued,container-title"})
    offtopic = None
    for it in r.json()["message"]["items"]:
        fam = [a.get("family", "") for a in it.get("author", [])[:1]]
        yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        if fam and fam[0].lower() == surname.lower() and yr is not None and abs(yr - year) <= 1:
            title, journal = (it.get("title") or [""])[0], (it.get("container-title") or [""])[0]
            hit = {"doi": it["DOI"], "found_title": title, "found_year": yr, "journal": journal}
            if TOPIC.search(f"{title} {journal}"):
                return {"status": "verified", **hit}
            offtopic = offtopic or hit
    if offtopic:
        return {"status": "author_year_only", **offtopic}
    return {"status": "not_in_crossref", "first_author": surname, "year": year}


TOPIC = re.compile(r"vegetat|reflect|spectr|chlorophyll|leaf|leaves|canopy|remote sens|sentinel|red.?edge|plant|"
                   r"photosynth|biomass|satellite|hyperspectral|imagery|stress|forest|radiation|crop|wheat|"
                   r"photochem|photobiol|grassland|nitrogen|LAI|band", re.I)


def _get(client: httpx.Client, params: dict, tries: int = 5) -> httpx.Response:
    for k in range(tries):
        r = client.get(API, params=params)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        time.sleep(2 ** (k + 1))
    r.raise_for_status()
    return r


def run() -> pd.DataFrame:
    rows = []
    with httpx.Client(timeout=30, headers={"User-Agent": "bifor-pipeline/0.1 (citation check)"}) as c:
        for name in indices.REGISTRY:
            backing = indices.catalog_entry(name).get("backing", "")
            cite = backing.split(";")[0].strip()
            row = {"index": name, "citation": cite}
            if cite:
                try:
                    row |= check(cite, c)
                except httpx.HTTPError as e:
                    row |= {"status": "error", "error": str(e)}
                time.sleep(1.0)
            else:
                row["status"] = "no_citation"
            rows.append(row)
    out = pd.DataFrame(rows)
    # Hand-checked results for citations the automatic query misses (targeted title
    # queries, or works that have no DOI at all). These replace the automatic row.
    ov = pd.read_csv(Path(__file__).with_name("citation_overrides.csv"))
    out = out.set_index("index")
    for r in ov.itertuples():
        for col in ("status", "doi", "found_title", "note"):
            out.loc[r.index, col] = getattr(r, col)
    return out.reset_index()

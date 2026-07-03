"""Recherche documentaire sur internet — le systeme va CHERCHER quand il ne sait pas.

Zero cle API, zero dependance nouvelle : recherche via l'endpoint HTML de DuckDuckGo
(httpx, deja requis par le serveur), extraction de texte lisible en stdlib pure.

Contrat (tout est best-effort et hors-ligne-safe) :
  search(query, k)  -> [{"title", "url", "snippet"}]     ([] si reseau indisponible)
  fetch(url)        -> texte lisible de la page          ("" si echec)
  research(query)   -> [{"title", "url", "text"}]        (search + fetch des top pages)

`http_get` est injectable partout (tests deterministes sans reseau, cf. smoke_learn).
"""

from __future__ import annotations

import html as _html
import re
import urllib.parse

_UA = {"User-Agent": "Mozilla/5.0 (compatible; LLML-research/0.1)"}
_SEARCH_URL = "https://html.duckduckgo.com/html/"
_MAX_PAGE_CHARS = 20_000  # cap : une page de doc suffit rarement a plus, et le RAG chunke
_TIMEOUT = 15.0

# <a class="result__a" href="//duckduckgo.com/l/?uddg=<url-encodee>&rut=...">Titre</a>
_RESULT_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_SNIPPET_RE = re.compile(
    r'class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_DROP_BLOCKS_RE = re.compile(
    r"<(script|style|nav|header|footer|aside|noscript|svg|form)\b.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _default_http_get(url: str, params: dict | None = None) -> str:
    """GET texte via httpx (follow_redirects). Leve en cas d'echec — les appelants
    attrapent et degradent en resultat vide."""
    import httpx

    resp = httpx.get(url, params=params, headers=_UA, timeout=_TIMEOUT, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _strip_tags(fragment: str) -> str:
    return _html.unescape(_TAG_RE.sub("", fragment or "")).strip()


def _decode_ddg_href(href: str) -> str:
    """Les liens DDG sont des redirections //duckduckgo.com/l/?uddg=<url> : on decode."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com"):
        qs = urllib.parse.parse_qs(parsed.query)
        target = (qs.get("uddg") or [""])[0]
        if target:
            return target
    return href


def search(query: str, k: int = 5, http_get=None) -> list[dict]:
    """Recherche web. Retourne jusqu'a k resultats {"title","url","snippet"} — [] si echec."""
    get = http_get or _default_http_get
    try:
        page = get(_SEARCH_URL, {"q": query}) or ""
    except Exception:  # noqa: BLE001 — reseau coupe/bloque : on degrade, on ne plante pas
        return []
    snippets = [_strip_tags(m.group("snippet")) for m in _SNIPPET_RE.finditer(page)]
    out: list[dict] = []
    for i, m in enumerate(_RESULT_RE.finditer(page)):
        url = _decode_ddg_href(m.group("href"))
        if not url.startswith(("http://", "https://")):
            continue
        out.append({
            "title": _strip_tags(m.group("title")),
            "url": url,
            "snippet": snippets[i] if i < len(snippets) else "",
        })
        if len(out) >= k:
            break
    return out


def fetch(url: str, http_get=None) -> str:
    """Telecharge une page et la reduit en texte lisible (cap _MAX_PAGE_CHARS)."""
    if not url.startswith(("http://", "https://")):
        return ""
    get = http_get or _default_http_get
    try:
        raw = get(url) or ""
    except Exception:  # noqa: BLE001
        return ""
    body = _DROP_BLOCKS_RE.sub(" ", raw)
    # Conserver la structure : titres et paragraphes deviennent des sauts de ligne.
    body = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|pre|section|article)\s*>", "\n", body)
    body = re.sub(r"(?i)<br\s*/?>", "\n", body)
    text = _html.unescape(_TAG_RE.sub(" ", body))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()[:_MAX_PAGE_CHARS]


def research(query: str, k_pages: int = 3, http_get=None) -> list[dict]:
    """Recherche + lecture : retourne jusqu'a k_pages {"title","url","text"} non vides."""
    pages: list[dict] = []
    for hit in search(query, k=max(k_pages * 2, k_pages), http_get=http_get):
        text = fetch(hit["url"], http_get=http_get)
        if len(text) < 200:  # page vide / paywall / JS-only : inutile pour etudier
            continue
        pages.append({"title": hit["title"], "url": hit["url"], "text": text})
        if len(pages) >= k_pages:
            break
    return pages

"""Competitor research via the Graph API ``business_discovery`` edge.

Added 2026-09-11 for the Choiz gateway: a content agent needs to read what the
competition posts, not just our own account. ``business_discovery`` is the
official endpoint for exactly that — no browser, no scraping, no login wall.

Shape of the call::

    GET /{ig_user_id}?fields=business_discovery.username(<target>){<subfields>}

``ig_user_id`` is OUR Instagram business account (the querying identity, from
``INSTAGRAM_BUSINESS_ACCOUNT_ID``); ``<target>`` is the competitor's handle.

What this edge DOES return
--------------------------
Profile: username, name, biography, website, followers_count, follows_count,
media_count, profile_picture_url. Media: caption, media_type,
media_product_type, permalink, timestamp, like_count, comments_count and
``children`` — the individual slides of a carousel, which is the whole point
for creative analysis.

What it does NOT return (do not build on these)
-----------------------------------------------
* **Lookup by post URL.** The edge is keyed by USERNAME. There is no official
  way to resolve ``instagram.com/p/<shortcode>`` to an account. ``find_post``
  below therefore takes the handle AND the URL, and pages that account's feed
  until the shortcode matches. The oEmbed endpoint does expose ``author_name``,
  but Meta's docs explicitly prohibit repurposing its metadata for analytics,
  so it is deliberately not used here.
* **Non-professional or private accounts.** Target must be a public Business or
  Creator account. Age-gated accounts are excluded by Meta.
* **Reach / impressions / saves / shares.** Those are owner-only insights.
  Likes and comment COUNTS are all the engagement this edge exposes.
* **Comment text, Stories, followers lists.**

Required permissions on the access token: ``instagram_basic``,
``instagram_manage_insights``, ``pages_read_engagement`` (plus ``ads_read`` or
``ads_management`` if the Page role was granted through Business Manager).

CDN URLs
--------
``media_url`` values are signed scontent.cdninstagram.com links: **~600 chars
each** (measured 2026-09-11, not estimated — an 8-slide carousel is ~4.8 KB of
URLs alone) and they expire within hours. They are omitted by default — callers that
actually need to look at the creative pass ``include_media_urls=True``. Note
this module builds PLAIN DICTS and never routes media through
``models.instagram_models``, whose validators strip query strings: stripping the
signature makes the URL 403, which is fine for the owner-account tools (they
only needed the payload smaller) but useless when the point is fetching the
image.

Payload discipline
------------------
Large tool results have historically been unreliable through claude.ai. The
~2-3 KB "ceiling" turned out not to be size-driven (see the gateway's
``project_claudeai_payload_ceiling`` note — the real cause is
anthropics/claude-ai-mcp#211), but small results are still the cheap
mitigation, so the defaults here are deliberately conservative: 5 posts,
captions truncated to 400 chars, CDN URLs off. Every one of those is a
parameter the caller raises explicitly. Measured sizes are in the constants
block below.

The one deliberate exception is ``find_post``: one post, full caption, every
slide. The caller already narrowed it to a single post, so the whole value of
the call is completeness — which on a 20-slide carousel with CDN URLs on means
a response in the 10 KB range. That is the intended trade, not an oversight.

STATUS — VERIFIED LIVE 2026-09-11
---------------------------------
Probed from the gateway EC2 with the production Timeless token against
``nike``. Confirmed working, no App Review needed:

* Permissions on the existing token are sufficient — no OAuthException.
* ``id``, ``caption``, ``media_type``, ``media_product_type``, ``permalink``,
  ``timestamp``, ``like_count``, ``comments_count``, ``children{id,media_type}``
  and ``media_url`` (on both parent and children) all return data.
* ``view_count`` is ACCEPTED by the edge — Graph simply omits it where it does
  not apply (a photo carousel has no views) rather than erroring. So
  ``_view_count_supported`` is expected to stay True in practice; the fallback
  below is belt-and-braces for a future API change, not a known failure.
* A CAROUSEL_ALBUM parent DOES carry its own ``media_url`` (the first slide).
  ``shape_post`` prefers the children list anyway, so this changes nothing.
* Reels come back with ``media_product_type: "REELS"`` and a ``/reel/<code>/``
  permalink, which ``PERMALINK_RE`` already matches.

Control flow — paging, cursor exhaustion, date windows, sorting, shortcode
matching, error mapping, the view_count fallback, payload budgets — is covered
by an offline harness against a scripted fake Graph, plus a round-trip through
the real MCP lowlevel Server.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import structlog

from .instagram_client import InstagramAPIError, InstagramClient

logger = structlog.get_logger(__name__)

# Instagram handles: letters, digits, period, underscore, max 30.
USERNAME_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")
SHORTCODE_RE = re.compile(r"^[A-Za-z0-9_-]{5,30}$")
# /p/ = feed post, /reel/ + /reels/ = reel, /tv/ = legacy IGTV.
PERMALINK_RE = re.compile(r"instagram\.com/(?:p|reel|reels|tv)/([A-Za-z0-9_-]{5,30})")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PROFILE_FIELDS = (
    "id,username,name,biography,website,"
    "followers_count,follows_count,media_count,profile_picture_url"
)

PAGE_SIZE = 25
MAX_PAGES_CAP = 20
# Measured worst case (2000-char captions, 3-slide carousels, offline harness):
#   limit=8 chars=600 -> 7.5 KB      limit=6 chars=400 -> 4.5 KB
#   limit=5 chars=400 -> 3.2 KB      limit=5 chars=300 -> 2.7 KB
# Per-post overhead beyond the caption is ~260 bytes. The defaults below land a
# plain call around 3 KB; a caller that wants more asks for it explicitly.
POSTS_DEFAULT = 5
CAPTION_CHARS_DEFAULT = 400
CAPTION_CHARS_CAP = 4000
MAX_USERNAMES = 5

MEDIA_URL_NOTE = (
    "media_urls are signed Instagram CDN links and expire within hours — "
    "fetch them now or re-request."
)

# Flipped to False the first time Graph rejects view_count on this edge, so we
# stop asking for the rest of the process lifetime. Module-level state is safe:
# stateless_http only makes MCP SESSIONS ephemeral, the uvicorn process is not.
_view_count_supported = True


class CompetitorLookupError(Exception):
    """A business_discovery lookup failed in a way worth explaining to the model."""

    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint


# --- validation -----------------------------------------------------------


def clean_username(value: str) -> str:
    """Normalise and validate a handle.

    Strict validation is not cosmetic: the handle is interpolated into the
    ``business_discovery.username(...)`` field expression, so an unvalidated
    value could inject arbitrary field grammar into the request.
    """
    u = (value or "").strip().lstrip("@")
    if u.startswith("http"):
        # Tolerate a pasted profile URL: instagram.com/<handle>/
        m = re.search(r"instagram\.com/([A-Za-z0-9._]{1,30})", u)
        if m:
            u = m.group(1)
    if not USERNAME_RE.match(u):
        raise CompetitorLookupError(
            f"'{value}' is not a valid Instagram handle.",
            hint="Pass the handle only, e.g. 'nike' — not a URL, not an @.",
        )
    return u


def parse_shortcode(value: str) -> str:
    """Accept a post URL or a bare shortcode, return the shortcode."""
    v = (value or "").strip()
    m = PERMALINK_RE.search(v)
    if m:
        return m.group(1)
    if SHORTCODE_RE.match(v):
        return v
    raise CompetitorLookupError(
        f"Could not read a post shortcode out of '{value}'.",
        hint="Pass a post URL like https://www.instagram.com/p/ABC123/ "
        "or the bare shortcode ABC123.",
    )


def _clamp(value: Optional[int], default: int, low: int, high: int) -> int:
    try:
        n = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


def _check_date(value: Optional[str], label: str) -> Optional[str]:
    if not value:
        return None
    if not DATE_RE.match(value.strip()):
        raise CompetitorLookupError(
            f"{label} must be YYYY-MM-DD, got '{value}'.",
        )
    return value.strip()


# --- Graph plumbing -------------------------------------------------------


def explain_error(exc: InstagramAPIError) -> Tuple[str, Optional[str]]:
    """Map a Graph error onto something the model can act on."""
    msg = getattr(exc, "message", None) or str(exc)
    code = getattr(exc, "error_code", None)
    low = msg.lower()

    if code == 190 or "access token" in low:
        return msg, (
            "The Instagram access token is invalid or expired. Rotate "
            "INSTAGRAM_*_ACCESS_TOKEN on the gateway EC2 .env."
        )
    if code in (4, 17, 32, 613) or "rate limit" in low or "too many calls" in low:
        return msg, (
            "Graph API rate limit hit. Lower max_pages / limit and retry later."
        )
    if code in (10, 200) or "permission" in low:
        return msg, (
            "The token is missing a permission. business_discovery needs "
            "instagram_basic + instagram_manage_insights + pages_read_engagement "
            "(plus ads_read/ads_management if the Page role came from Business "
            "Manager)."
        )
    if "invalid user id" in low or "does not exist" in low or "cannot be loaded" in low:
        return msg, (
            "Graph could not resolve that handle. business_discovery only sees "
            "PUBLIC Business or Creator accounts — personal, private and "
            "age-gated accounts return nothing. Check the spelling too."
        )
    return msg, None


async def _discover(
    client: InstagramClient, username: str, sub_fields: str
) -> Dict[str, Any]:
    """Run one business_discovery call and return the inner object."""
    account_id = client.settings.instagram_business_account_id
    if not account_id:
        raise CompetitorLookupError(
            "INSTAGRAM_BUSINESS_ACCOUNT_ID is not configured on this container."
        )

    fields = f"business_discovery.username({username}){{{sub_fields}}}"
    try:
        body = await client._make_request("GET", account_id, params={"fields": fields})
    except InstagramAPIError as exc:
        message, hint = explain_error(exc)
        raise CompetitorLookupError(message, hint) from exc

    discovered = body.get("business_discovery")
    if not discovered:
        raise CompetitorLookupError(
            f"Graph returned no business_discovery payload for '{username}'.",
            hint="Usually means the account is not a public Business/Creator "
            "account.",
        )
    return discovered


def _media_fields(include_media_urls: bool) -> str:
    fields = [
        "id",
        "caption",
        "media_type",
        "media_product_type",
        "permalink",
        "timestamp",
        "like_count",
        "comments_count",
    ]
    child = ["id", "media_type"]
    if _view_count_supported:
        fields.append("view_count")
    if include_media_urls:
        fields.append("media_url")
        child.append("media_url")
    return ",".join(fields) + ",children{" + ",".join(child) + "}"


async def _media_page(
    client: InstagramClient,
    username: str,
    *,
    page_size: int,
    after: Optional[str],
    include_media_urls: bool,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[int]]:
    """Fetch one page of a competitor's media. Returns (items, cursor, followers)."""
    global _view_count_supported

    def build(with_followers: bool = True) -> str:
        media = f"media.limit({page_size})"
        if after:
            media += f".after({after})"
        media += "{" + _media_fields(include_media_urls) + "}"
        head = "username,followers_count," if with_followers else ""
        return head + media

    try:
        discovered = await _discover(client, username, build())
    except CompetitorLookupError as exc:
        # view_count is documented on IG Media but is not guaranteed on this
        # edge. If it is what Graph objected to, drop it permanently and retry
        # once rather than failing the whole lookup.
        if _view_count_supported and "view_count" in str(exc.message):
            logger.warning(
                "business_discovery rejected view_count; disabling it",
                error=exc.message,
            )
            _view_count_supported = False
            discovered = await _discover(client, username, build())
        else:
            raise

    media = discovered.get("media") or {}
    items = media.get("data") or []
    cursor = ((media.get("paging") or {}).get("cursors") or {}).get("after")
    # Graph omits the cursor once the feed is exhausted; also guard on a short
    # page so we never loop on an account with fewer posts than page_size.
    if len(items) < page_size:
        cursor = None
    return items, cursor, discovered.get("followers_count")


# --- shaping --------------------------------------------------------------


def shape_post(
    item: Dict[str, Any],
    *,
    followers: Optional[int],
    caption_chars: int,
    include_media_urls: bool,
) -> Dict[str, Any]:
    """Trim one raw media object down to what a content agent actually reads."""
    caption = item.get("caption") or ""
    permalink = item.get("permalink") or ""
    likes = item.get("like_count") or 0
    comments = item.get("comments_count") or 0
    children = ((item.get("children") or {}).get("data")) or []

    m = PERMALINK_RE.search(permalink)
    out: Dict[str, Any] = {
        "shortcode": m.group(1) if m else None,
        "permalink": permalink,
        "type": item.get("media_type"),
        "posted_at": (item.get("timestamp") or "")[:10],
        "likes": likes,
        "comments": comments,
        "engagement": likes + comments,
        "caption": caption[:caption_chars],
    }
    # Only when it says something FEED doesn't (REELS, AD, STORY).
    fmt = item.get("media_product_type")
    if fmt and fmt != "FEED":
        out["format"] = fmt
    if len(caption) > caption_chars:
        out["caption_cut"] = len(caption)
    if item.get("view_count") is not None:
        out["views"] = item["view_count"]
    if children:
        out["slides"] = len(children)
    if followers:
        out["eng_rate_pct"] = round((likes + comments) * 100.0 / followers, 2)
    if include_media_urls:
        urls = [c["media_url"] for c in children if c.get("media_url")]
        if not urls and item.get("media_url"):
            urls = [item["media_url"]]
        if urls:
            # Signed + short-lived on purpose — see the module docstring. The
            # caveat is stated ONCE at the top level, not per post.
            out["media_urls"] = urls
    return out


def _in_window(
    item: Dict[str, Any], since: Optional[str], until: Optional[str]
) -> bool:
    day = (item.get("timestamp") or "")[:10]
    if not day:
        return False
    if since and day < since:
        return False
    if until and day > until:
        return False
    return True


# --- tool-level operations ------------------------------------------------


async def get_profile(client: InstagramClient, username: str) -> Dict[str, Any]:
    """Public profile snapshot for one competitor."""
    u = clean_username(username)
    discovered = await _discover(client, u, PROFILE_FIELDS)
    return {
        "username": discovered.get("username"),
        "name": discovered.get("name"),
        "biography": discovered.get("biography"),
        "website": discovered.get("website"),
        "followers": discovered.get("followers_count"),
        "following": discovered.get("follows_count"),
        "posts": discovered.get("media_count"),
        "profile_picture_url": discovered.get("profile_picture_url"),
    }


async def list_posts(
    client: InstagramClient,
    username: str,
    *,
    limit: int = POSTS_DEFAULT,
    since: Optional[str] = None,
    until: Optional[str] = None,
    sort: str = "recent",
    include_media_urls: bool = False,
    caption_chars: int = CAPTION_CHARS_DEFAULT,
    max_pages: int = 4,
) -> Dict[str, Any]:
    """Recent (or top-performing) posts from one competitor.

    ``sort="engagement"`` ranks by likes+comments across everything scanned —
    that is the best of the last ``max_pages`` pages, NOT the account's all-time
    best. Graph exposes no server-side sort on this edge.
    """
    u = clean_username(username)
    limit = _clamp(limit, POSTS_DEFAULT, 1, 50)
    caption_chars = _clamp(caption_chars, CAPTION_CHARS_DEFAULT, 0, CAPTION_CHARS_CAP)
    max_pages = _clamp(max_pages, 4, 1, MAX_PAGES_CAP)
    since = _check_date(since, "since")
    until = _check_date(until, "until")
    if sort not in ("recent", "engagement"):
        sort = "recent"

    collected: List[Dict[str, Any]] = []
    followers: Optional[int] = None
    cursor: Optional[str] = None
    pages = 0
    exhausted = False
    reached_since = False

    while pages < max_pages:
        items, cursor, page_followers = await _media_page(
            client,
            u,
            page_size=PAGE_SIZE,
            after=cursor,
            include_media_urls=include_media_urls,
        )
        pages += 1
        if followers is None:
            followers = page_followers

        for item in items:
            day = (item.get("timestamp") or "")[:10]
            if since and day and day < since:
                # Feed is newest-first, so everything past here is older.
                reached_since = True
                break
            if _in_window(item, since, until):
                collected.append(item)

        if reached_since:
            break
        if not cursor:
            exhausted = True
            break
        # In "recent" mode with no date window, one page past the requested
        # count is already enough.
        if sort == "recent" and not since and not until and len(collected) >= limit:
            break

    posts = [
        shape_post(
            item,
            followers=followers,
            caption_chars=caption_chars,
            include_media_urls=include_media_urls,
        )
        for item in collected
    ]
    if sort == "engagement":
        posts.sort(key=lambda p: p.get("engagement", 0), reverse=True)

    result: Dict[str, Any] = {
        "username": u,
        "followers": followers,
        "sort": sort,
        "posts": posts[:limit],
        "scanned": {
            "posts": len(collected),
            "pages": pages,
            "feed_exhausted": exhausted,
        },
    }
    if include_media_urls:
        result["media_urls_note"] = MEDIA_URL_NOTE
    if since or until:
        result["window"] = {"since": since, "until": until}
    if sort == "engagement" and not exhausted:
        result["scanned"]["note"] = (
            "Top posts among those scanned, not the account's all-time best. "
            "Raise max_pages to widen the scan."
        )
    return result


async def find_post(
    client: InstagramClient,
    username: str,
    post: str,
    *,
    include_media_urls: bool = True,
    caption_chars: int = CAPTION_CHARS_CAP,
    max_pages: int = 8,
) -> Dict[str, Any]:
    """Resolve one specific competitor post from its URL (or shortcode).

    Requires the handle: Graph has no shortcode lookup, so this pages the
    account's feed comparing permalinks. A post older than ``max_pages`` * 25
    will not be found — widen max_pages or use ``since`` on list_posts instead.
    """
    u = clean_username(username)
    shortcode = parse_shortcode(post)
    caption_chars = _clamp(caption_chars, CAPTION_CHARS_CAP, 0, CAPTION_CHARS_CAP)
    max_pages = _clamp(max_pages, 8, 1, MAX_PAGES_CAP)

    cursor: Optional[str] = None
    followers: Optional[int] = None
    pages = 0
    scanned = 0

    while pages < max_pages:
        items, cursor, page_followers = await _media_page(
            client,
            u,
            page_size=PAGE_SIZE,
            after=cursor,
            include_media_urls=include_media_urls,
        )
        pages += 1
        scanned += len(items)
        if followers is None:
            followers = page_followers

        for item in items:
            m = PERMALINK_RE.search(item.get("permalink") or "")
            if m and m.group(1) == shortcode:
                # Deliberately the most generous response in this module: one
                # post, full caption, every slide. That is the point of the
                # call — the caller already narrowed it to a single post.
                out = {
                    "username": u,
                    "followers": followers,
                    "found": True,
                    "post": shape_post(
                        item,
                        followers=followers,
                        caption_chars=caption_chars,
                        include_media_urls=include_media_urls,
                    ),
                    "scanned": {"posts": scanned, "pages": pages},
                }
                if include_media_urls:
                    out["media_urls_note"] = MEDIA_URL_NOTE
                return out
        if not cursor:
            break

    return {
        "username": u,
        "shortcode": shortcode,
        "found": False,
        "scanned": {"posts": scanned, "pages": pages},
        "hint": (
            f"Not among the {scanned} most recent posts of @{u}. Either the post "
            "belongs to a different account, or it is older than the scan — "
            "raise max_pages."
        ),
    }


async def compare(
    client: InstagramClient,
    usernames: List[str],
    *,
    limit_per_account: int = 3,
    caption_chars: int = 200,
) -> Dict[str, Any]:
    """Side-by-side snapshot of several competitors: profile + top recent posts.

    Deliberately small per account — this is the "who is worth a closer look"
    call, then drill in with list_posts / find_post.
    """
    if not usernames:
        raise CompetitorLookupError("Pass at least one username.")
    if len(usernames) > MAX_USERNAMES:
        raise CompetitorLookupError(
            f"At most {MAX_USERNAMES} accounts per call — the tool result would "
            "outgrow what claude.ai accepts. Split the list."
        )
    limit_per_account = _clamp(limit_per_account, 3, 1, 10)
    caption_chars = _clamp(caption_chars, 200, 0, 1000)

    accounts: List[Dict[str, Any]] = []
    for raw in usernames:
        try:
            u = clean_username(raw)
            posts = await list_posts(
                client,
                u,
                limit=limit_per_account,
                sort="engagement",
                include_media_urls=False,
                caption_chars=caption_chars,
                max_pages=1,
            )
            engagements = [p.get("engagement", 0) for p in posts["posts"]]
            accounts.append(
                {
                    "username": u,
                    "followers": posts.get("followers"),
                    "avg_engagement_top": (
                        round(sum(engagements) / len(engagements), 1)
                        if engagements
                        else None
                    ),
                    "top_posts": posts["posts"],
                }
            )
        except CompetitorLookupError as exc:
            accounts.append({"username": raw, "error": exc.message, "hint": exc.hint})

    return {"accounts": accounts, "compared": len(accounts)}

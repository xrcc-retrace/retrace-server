"""PDF page → PNG renderer (PyMuPDF).

Used by services.manual_ingest after Gemini returns a `source_page` (and
optional `source_bbox`) per step. Renders either the full page or just
the per-step figure region at a configurable DPI so the iOS PiP overlay
has something concrete to show during a manual-driven coaching session.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

DEFAULT_DPI = 144  # 2× of standard 72 dpi — sharp enough for a 160×120 PiP

# A bbox crop covering less than this fraction of the page is treated as
# a Gemini hallucination (or a degenerate clamp) and rejected. Callers
# should fall through to "no image for this step" rather than rendering
# something unhelpful.
_MIN_BBOX_AREA_FRACTION = 0.01


class BBoxRejected(ValueError):
    """Raised when a provided bbox produces an empty / out-of-bounds /
    sub-`_MIN_BBOX_AREA_FRACTION`-of-page crop region. The caller should
    treat this as 'no image for this step' rather than falling back to a
    full-page render — per product direction, a step without a real
    figure shows no image at all.
    """


def render_page_png(
    pdf_path: Path,
    page_index: int,
    out_path: Path,
    dpi: int = DEFAULT_DPI,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Render one PDF page (or a region of it) to PNG.

    `page_index` is **0-indexed**; callers that get 1-indexed page numbers
    from Gemini must subtract 1.

    `bbox` is `(x_pct, y_pct, w_pct, h_pct)` in **page-percent**
    coordinates (0..100; top-left origin). When omitted the full page
    renders. When provided but the resulting clip region is empty,
    out of bounds, or smaller than 1% of the page area, raises
    `BBoxRejected` so the caller can skip this step's image entirely
    (no full-page fallback).

    Raises `IndexError` for an out-of-range `page_index` and `BBoxRejected`
    for a degenerate `bbox`. Other exceptions (file I/O, PyMuPDF errors)
    propagate to the caller.
    """
    pdf_path = Path(pdf_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(
                f"page_index {page_index} out of range for {pdf_path.name} "
                f"(page_count={doc.page_count})"
            )
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        clip = _build_clip(page, bbox) if bbox is not None else None
        pix = page.get_pixmap(matrix=matrix, alpha=False, clip=clip)
        pix.save(str(out_path))
    logger.debug(
        "Rendered %s page %d (bbox=%s) → %s",
        pdf_path.name, page_index + 1, bbox, out_path,
    )


def _build_clip(page: "fitz.Page", bbox: tuple[float, float, float, float]) -> "fitz.Rect":
    """Convert a page-percent bbox into a clamped `fitz.Rect`. Raises
    `BBoxRejected` when the result is unusable."""
    x_pct, y_pct, w_pct, h_pct = bbox
    if w_pct <= 0 or h_pct <= 0:
        raise BBoxRejected(f"non-positive bbox dimensions: w={w_pct}, h={h_pct}")

    pr = page.rect
    candidate = fitz.Rect(
        pr.x0 + pr.width  * max(0.0, min(100.0, x_pct))         / 100.0,
        pr.y0 + pr.height * max(0.0, min(100.0, y_pct))         / 100.0,
        pr.x0 + pr.width  * max(0.0, min(100.0, x_pct + w_pct)) / 100.0,
        pr.y0 + pr.height * max(0.0, min(100.0, y_pct + h_pct)) / 100.0,
    )
    candidate = candidate & pr  # clamp to page bounds
    page_area = pr.width * pr.height
    cand_area = candidate.width * candidate.height
    if candidate.is_empty or cand_area < _MIN_BBOX_AREA_FRACTION * page_area:
        raise BBoxRejected(
            f"bbox {bbox} clamped to {candidate} is empty or "
            f"<{_MIN_BBOX_AREA_FRACTION:.0%} of page area"
        )
    return candidate


def page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""
    with fitz.open(Path(pdf_path)) as doc:
        return doc.page_count


# Filter band for "is this raster image plausibly a step illustration?"
# Below the lower bound: header logos, decorative bullets, page numbers.
# Above the upper bound: the page is essentially one big background image
# (PDF cover pages do this). Both rejected so the snap layer doesn't try
# to crop them.
_FIGURE_MIN_AREA_FRACTION = 0.010   # >= 1.0% of page area
_FIGURE_MAX_AREA_FRACTION = 0.50    # <= 50% of page area


def page_rect(pdf_path: Path, page_index: int) -> "fitz.Rect":
    """Return the bounding rect (PDF points) of a single page. Used by
    callers that need to convert between absolute coordinates and
    page-percent bboxes."""
    with fitz.open(Path(pdf_path)) as doc:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(
                f"page_index {page_index} out of range for {Path(pdf_path).name} "
                f"(page_count={doc.page_count})"
            )
        return fitz.Rect(doc.load_page(page_index).rect)


def list_image_regions(
    pdf_path: Path,
    page_index: int,
    *,
    min_area_fraction: float = _FIGURE_MIN_AREA_FRACTION,
    max_area_fraction: float = _FIGURE_MAX_AREA_FRACTION,
) -> list["fitz.Rect"]:
    """Return absolute bboxes (PDF points) of plausible per-step
    illustrations on `page_index`.

    Detection runs in two passes and merges the results:
      1. Embedded raster images via `page.get_images()` + `get_image_bbox()`.
         Catches photos and pre-rendered diagrams.
      2. Vector figure clusters via `page.get_drawings()` clustered by
         spatial proximity (within ~8pt of each other). Catches inline
         line-art illustrations that have no embedded raster.

    Both passes are filtered to the same `min_area_fraction` /
    `max_area_fraction` band so header logos, page numbers, decorative
    rules, and full-page backgrounds are rejected. Overlapping regions
    across the two passes are deduplicated (a vector outline drawn over
    a raster shouldn't be detected twice).

    The snap-to-detected step in `manual_ingest` picks one of these as
    the actual crop region per step, overriding Gemini's looser bbox so
    the crop captures only the illustration and not the surrounding
    caption text.
    """
    pdf_path = Path(pdf_path)
    out: list[fitz.Rect] = []
    with fitz.open(pdf_path) as doc:
        if page_index < 0 or page_index >= doc.page_count:
            return out
        page = doc.load_page(page_index)
        page_area = page.rect.width * page.rect.height

        # --- Pass 1: embedded raster images ---
        for img in page.get_images(full=True):
            try:
                bbox = page.get_image_bbox(img)
            except Exception:
                continue
            if bbox.is_empty:
                continue
            bbox = bbox & page.rect  # clamp
            area = bbox.width * bbox.height
            if area < min_area_fraction * page_area:
                continue
            if area > max_area_fraction * page_area:
                continue
            out.append(fitz.Rect(bbox))

        # --- Pass 2: vector figure clusters ---
        # Cluster individual drawing paths into figure regions when they
        # sit within `_VECTOR_CLUSTER_TOLERANCE` points of each other.
        # Then filter clusters to the same area band as raster images.
        # Skip clusters that overlap a raster bbox already detected so a
        # vector outline drawn over a photo doesn't get added twice.
        path_rects: list[fitz.Rect] = []
        for d in page.get_drawings():
            r = d.get("rect")
            if r is None or r.is_empty:
                continue
            # Drop pure horizontal/vertical lines (decorative dividers).
            if r.width < 2 or r.height < 2:
                continue
            path_rects.append(fitz.Rect(r))

        clusters = _cluster_rects(path_rects, _VECTOR_CLUSTER_TOLERANCE)
        for cluster in clusters:
            cluster = cluster & page.rect  # clamp
            area = cluster.width * cluster.height
            if area < min_area_fraction * page_area:
                continue
            if area > max_area_fraction * page_area:
                continue
            # Dedupe: skip clusters that significantly overlap a raster
            # region we already added.
            if any(_iou(cluster, existing) > 0.5 for existing in out):
                continue
            out.append(fitz.Rect(cluster))
    return out


# Two vector paths within this many points of each other (in any
# direction) are merged into one figure cluster. Tuned for typical
# manual layouts where line-art illustrations have many small strokes
# packed tightly together. Decorative dividers and page numbers won't
# survive the area filter even if they cluster.
_VECTOR_CLUSTER_TOLERANCE = 8.0


def _cluster_rects(rects: list["fitz.Rect"], tolerance: float) -> list["fitz.Rect"]:
    """Greedy union-find: any two rects within `tolerance` points of each
    other in both axes are grouped into a single bounding rect. Repeats
    until no more merges happen.
    """
    if not rects:
        return []
    merged = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        new_merged: list[fitz.Rect] = []
        consumed = [False] * len(merged)
        for i in range(len(merged)):
            if consumed[i]:
                continue
            acc = fitz.Rect(merged[i])
            consumed[i] = True
            for j in range(i + 1, len(merged)):
                if consumed[j]:
                    continue
                other = merged[j]
                expanded = fitz.Rect(
                    acc.x0 - tolerance, acc.y0 - tolerance,
                    acc.x1 + tolerance, acc.y1 + tolerance,
                )
                if expanded.intersects(other):
                    acc |= other
                    consumed[j] = True
                    changed = True
            new_merged.append(acc)
        merged = new_merged
    return merged


def _iou(a: "fitz.Rect", b: "fitz.Rect") -> float:
    """Intersection-over-union for two rects. Used to dedupe vector
    clusters that lie on top of a raster image we already detected."""
    inter = a & b
    if inter.is_empty:
        return 0.0
    inter_area = inter.width * inter.height
    union_area = (a.width * a.height) + (b.width * b.height) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def rect_to_page_percent(
    rect: "fitz.Rect", page_rect: "fitz.Rect"
) -> tuple[float, float, float, float]:
    """Convert an absolute `fitz.Rect` into the `(x_pct, y_pct, w_pct, h_pct)`
    tuple that `render_page_png` expects via its `bbox` parameter."""
    return (
        (rect.x0 - page_rect.x0) / page_rect.width  * 100.0,
        (rect.y0 - page_rect.y0) / page_rect.height * 100.0,
        (rect.x1 - rect.x0)      / page_rect.width  * 100.0,
        (rect.y1 - rect.y0)      / page_rect.height * 100.0,
    )


def percent_to_rect(
    bbox_pct: tuple[float, float, float, float], page_rect: "fitz.Rect"
) -> "fitz.Rect":
    """Inverse of `rect_to_page_percent`."""
    x_pct, y_pct, w_pct, h_pct = bbox_pct
    return fitz.Rect(
        page_rect.x0 + page_rect.width  * x_pct          / 100.0,
        page_rect.y0 + page_rect.height * y_pct          / 100.0,
        page_rect.x0 + page_rect.width  * (x_pct + w_pct) / 100.0,
        page_rect.y0 + page_rect.height * (y_pct + h_pct) / 100.0,
    )

#!/usr/bin/env python3
"""Reproducible GitHub-profile terminal banner generator.

Source of truth:
  - portrait-source.png
  - this file

Outputs:
  - dark.svg / light.svg (1180x610 animated SVG)
  - metrics.json

The portrait pipeline follows the supplied brief: 300x340 crop, autocontrast
cutoff 1, contrast 1.3, UnsharpMask(3, 140), serpentine Floyd-Steinberg, and
compact crisp SVG runs.  The dark treatment additionally builds a hard subject
mask using background rejection, morphological closing, hole filling, and
largest-component selection.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import statistics
import struct
import zipfile
from collections import deque
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


W, H = 1180, 610
PW, PH = 300, 340
PX, PY = 74, 164
BANDS = 94
TRAVELLERS = 900
INTRO_GROUPS = 60
INTRO_SECONDS = 3.2
LOOP_SECONDS = 14.2
SEED = 24072003
KEY_TIMES = "0;0.21127;0.30282;0.44366;0.53521;0.67606;0.76761;0.90845;1"

PROFILE = {
    "Subject": "RUBENS.RAFAEL",
    "Handle": "@FaelDev-ux",
    "Role": "Web Developer",
    "Origin": "Brasil",
    "Education": "Eng. de Software",
    "Status": "Building + Learning + Shipping",
    "ToolChain": "VS Code · Git · Vercel",
    "Core.Lang": "TypeScript · JavaScript · Python",
    "Core.Frontend": "Next.js · React · Tailwind",
    "Core.Backend": "FastAPI · Supabase",
    "Core.Database": "PostgreSQL · Supabase",
    "Core.Infra": "Vercel · Firebase",
    "Grid.Mail": "rubensnobrega2003@gmail.com",
    "Grid.Portfolio": "www.r2labss.dev",
    "Grid.Instagram": "@fael.rdgs",
    "Grid.GitHub": "FaelDev-ux",
}

THEMES = {
    "dark": {
        "bg": "#0A101F",
        "panel": "#0D1528",
        "panel2": "#101B32",
        "border": "#223251",
        "text": "#E6EDF8",
        "muted": "#7E8CA8",
        "faint": "#34435F",
        "portrait": "#A78BFA",
        "chrome": "#22D3EE",
        "accent": "#10B981",
        "red": "#FB7185",
        "shadow": "#030712",
    },
    "light": {
        "bg": "#F4F6FB",
        "panel": "#FFFFFF",
        "panel2": "#F8FAFC",
        "border": "#CFD8E8",
        "text": "#172033",
        "muted": "#5E6B84",
        "faint": "#D8DFEB",
        "portrait": "#7C3AED",
        "chrome": "#0891B2",
        "accent": "#10B981",
        "red": "#E11D48",
        "shadow": "#AAB5C8",
    },
}


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def crop_portrait(source: Image.Image) -> Image.Image:
    """Crop around the face/upper torso to the exact 300x340 map."""
    source = source.convert("RGB")
    return ImageOps.fit(
        source,
        (PW, PH),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.38),
    )


def prepared_density(crop: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(crop)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    return ImageOps.invert(gray)


def fill_holes(bits: list[list[bool]]) -> list[list[bool]]:
    h, w = len(bits), len(bits[0])
    seen = [[False] * w for _ in range(h)]
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if not bits[0][x]:
            seen[0][x] = True
            q.append((x, 0))
        if not bits[h - 1][x]:
            seen[h - 1][x] = True
            q.append((x, h - 1))
    for y in range(h):
        if not bits[y][0] and not seen[y][0]:
            seen[y][0] = True
            q.append((0, y))
        if not bits[y][w - 1] and not seen[y][w - 1]:
            seen[y][w - 1] = True
            q.append((w - 1, y))
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and not bits[ny][nx] and not seen[ny][nx]:
                seen[ny][nx] = True
                q.append((nx, ny))
    return [[bits[y][x] or not seen[y][x] for x in range(w)] for y in range(h)]


def largest_component(bits: list[list[bool]]) -> list[list[bool]]:
    h, w = len(bits), len(bits[0])
    seen = [[False] * w for _ in range(h)]
    best: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not bits[y][x] or seen[y][x]:
                continue
            q = [(x, y)]
            seen[y][x] = True
            comp: list[tuple[int, int]] = []
            for cx, cy in q:
                comp.append((cx, cy))
                for nx, ny in (
                    (cx - 1, cy),
                    (cx + 1, cy),
                    (cx, cy - 1),
                    (cx, cy + 1),
                ):
                    if 0 <= nx < w and 0 <= ny < h and bits[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        q.append((nx, ny))
            if len(comp) > len(best):
                best = comp
    out = [[False] * w for _ in range(h)]
    for x, y in best:
        out[y][x] = True
    return out


def dark_subject_mask(crop: Image.Image) -> list[list[bool]]:
    """Reject neutral mid-tone studio background, then close/fill/select subject."""
    px = crop.load()
    raw = Image.new("L", (PW, PH), 0)
    out = raw.load()
    for y in range(PH):
        for x in range(PW):
            r, g, b = px[x, y]
            hi, lo = max(r, g, b), min(r, g, b)
            sat = hi - lo
            lum = (54 * r + 183 * g + 19 * b) >> 8
            # Studio background is neutral and mostly in this middle range.
            foreground = sat >= 13 or lum <= 37 or lum >= 79
            out[x, y] = 255 if foreground else 0
    # Binary closing: dilation then erosion.
    closed = raw.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))
    bits = [[closed.getpixel((x, y)) >= 128 for x in range(PW)] for y in range(PH)]
    bits = fill_holes(bits)
    bits = largest_component(bits)
    return bits


def floyd_steinberg(
    density: Image.Image, mask: list[list[bool]] | None
) -> list[tuple[int, int]]:
    """1-bit serpentine Floyd-Steinberg error diffusion."""
    rows = [[float(v) for v in density.crop((0, y, PW, y + 1)).getdata()] for y in range(PH)]
    if mask is not None:
        for y in range(PH):
            for x in range(PW):
                if not mask[y][x]:
                    rows[y][x] = 0.0
    dots: list[tuple[int, int]] = []
    for y in range(PH):
        forward = (y % 2) == 0
        xs = range(PW) if forward else range(PW - 1, -1, -1)
        for x in xs:
            old = max(0.0, min(255.0, rows[y][x]))
            new = 255.0 if old >= 127.5 else 0.0
            if new:
                dots.append((x, y))
            err = old - new
            if forward:
                nbrs = ((x + 1, y, 7), (x - 1, y + 1, 3), (x, y + 1, 5), (x + 1, y + 1, 1))
            else:
                nbrs = ((x - 1, y, 7), (x + 1, y + 1, 3), (x, y + 1, 5), (x - 1, y + 1, 1))
            for nx, ny, weight in nbrs:
                if 0 <= nx < PW and 0 <= ny < PH:
                    rows[ny][nx] += err * weight / 16.0
    return dots


def band_map(dots: list[tuple[int, int]]) -> tuple[list[list[tuple[int, int]]], float]:
    rng = random.Random(SEED)
    # Linear interpolation damps the knots; 5.45 yields an observed σ ≈ 4 px.
    knots = [rng.gauss(0.0, 5.45) for _ in range(PW // 12 + 2)]
    bands: list[list[tuple[int, int]]] = [[] for _ in range(BANDS)]
    noise_samples: list[float] = []
    for x, y in dots:
        k = x // 12
        t = (x % 12) / 12.0
        smooth = knots[k] * (1.0 - t) + knots[k + 1] * t
        noise = smooth + 0.55 * math.sin((x * 0.21) + (y * 0.09))
        noise_samples.append(noise)
        b = int((y + noise) * BANDS / PH)
        b = max(0, min(BANDS - 1, b))
        bands[b].append((x, y))
    return bands, statistics.pstdev(noise_samples)


def compact_path(points: list[tuple[int, int]]) -> str:
    rows: dict[int, list[int]] = {}
    for x, y in points:
        rows.setdefault(y, []).append(x)
    chunks: list[str] = []
    for y in sorted(rows):
        xs = sorted(rows[y])
        start = prev = xs[0]
        for x in xs[1:] + [10_000]:
            if x != prev + 1:
                width = prev - start + 1
                chunks.append(f"M{start} {y}h{width}v1h-{width}z")
                start = x
            prev = x
    return "".join(chunks)


def schedule_metrics(
    order: list[int],
    band_cell_counts: list[list[int]],
    band_totals: list[int],
) -> tuple[float, float]:
    rank = [0] * BANDS
    for i, b in enumerate(order):
        rank[b] = i
    checkpoints = (0.18, 0.35, 0.5, 0.68, 0.84)
    spatial_errors: list[float] = []
    grand_total = sum(band_totals)
    cell_totals = [sum(band_cell_counts[b][c] for b in range(BANDS)) for c in range(36)]
    for f in checkpoints:
        cutoff = int(round(f * BANDS))
        visible = [rank[b] < cutoff for b in range(BANDS)]
        global_ratio = sum(band_totals[b] for b in range(BANDS) if visible[b]) / max(1, grand_total)
        for c in range(36):
            if cell_totals[c]:
                cell_visible = sum(
                    band_cell_counts[b][c] for b in range(BANDS) if visible[b]
                )
                spatial_errors.append(abs(cell_visible / cell_totals[c] - global_ratio))
    spatial = sum(spatial_errors) / max(1, len(spatial_errors))
    mean_b = (BANDS - 1) / 2
    mean_r = mean_b
    num = sum((b - mean_b) * (rank[b] - mean_r) for b in range(BANDS))
    den = math.sqrt(
        sum((b - mean_b) ** 2 for b in range(BANDS))
        * sum((rank[b] - mean_r) ** 2 for b in range(BANDS))
    )
    straight = abs(num / den) if den else 0.0
    return spatial, straight


def choose_schedule(band_points: list[list[tuple[int, int]]]) -> tuple[list[int], float, float]:
    best: tuple[float, list[int], float, float] | None = None
    band_cell_counts = [[0] * 36 for _ in range(BANDS)]
    band_totals = [len(points) for points in band_points]
    for b, points in enumerate(band_points):
        for x, y in points:
            gx = min(5, int(x * 6 / PW))
            gy = min(5, int(y * 6 / PH))
            band_cell_counts[b][gy * 6 + gx] += 1
    # Modular permutations are low-discrepancy along Y and avoid a wipe boundary.
    for step in range(1, BANDS):
        if math.gcd(step, BANDS) != 1:
            continue
        for offset in range(0, BANDS, 3):
            order = [((i * step) + offset) % BANDS for i in range(BANDS)]
            spatial, straight = schedule_metrics(order, band_cell_counts, band_totals)
            score = spatial + 0.6 * straight
            if best is None or score < best[0]:
                best = (score, order, spatial, straight)
    assert best is not None
    return best[1], best[2], best[3]


def evenly_sample(points: list[tuple[int, int]], n: int) -> list[tuple[float, float]]:
    if not points:
        return [(PW / 2, PH / 2)] * n
    ranked = sorted(points, key=lambda p: ((p[0] * 73856093) ^ (p[1] * 19349663) ^ SEED))
    if len(ranked) >= n:
        step = len(ranked) / n
        return [(float(ranked[int(i * step)][0]), float(ranked[int(i * step)][1])) for i in range(n)]
    return [(float(ranked[i % len(ranked)][0]), float(ranked[i % len(ranked)][1])) for i in range(n)]


def sample_polyline(vertices: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    segs = []
    total = 0.0
    for a, b in zip(vertices, vertices[1:]):
        length = math.dist(a, b)
        segs.append((a, b, length))
        total += length
    out = []
    for i in range(n):
        target = (i + 0.5) * total / n
        acc = 0.0
        for a, b, length in segs:
            if target <= acc + length:
                t = (target - acc) / length
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
                break
            acc += length
    return out


def react_points(n: int) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    center_n = 54
    ring_n = n - center_n
    for axis in (0.0, math.pi / 3, -math.pi / 3):
        count = ring_n // 3
        ca, sa = math.cos(axis), math.sin(axis)
        for i in range(count):
            t = 2 * math.pi * i / count
            ex, ey = 110 * math.cos(t), 42 * math.sin(t)
            out.append((150 + ex * ca - ey * sa, 170 + ex * sa + ey * ca))
    for i in range(n - len(out)):
        t = 2 * math.pi * i / max(1, n - len(out))
        out.append((150 + 18 * math.cos(t), 170 + 18 * math.sin(t)))
    return out[:n]


def code_points(n: int) -> list[tuple[float, float]]:
    parts = [
        [(82, 108), (34, 170), (82, 232)],
        [(179, 93), (121, 247)],
        [(218, 108), (266, 170), (218, 232)],
    ]
    lengths = [sum(math.dist(a, b) for a, b in zip(p, p[1:])) for p in parts]
    counts = [round(n * length / sum(lengths)) for length in lengths]
    counts[-1] += n - sum(counts)
    out: list[tuple[float, float]] = []
    for p, count in zip(parts, counts):
        out.extend(sample_polyline(p, count))
    return out[:n]


def halton(index: int, base: int) -> float:
    result, f = 0.0, 1.0
    while index:
        f /= base
        result += f * (index % base)
        index //= base
    return result


def vercel_points(n: int) -> list[tuple[float, float]]:
    a, b, c = (150.0, 78.0), (48.0, 246.0), (252.0, 246.0)
    out = []
    for i in range(1, n + 1):
        u, v = halton(i, 2), halton(i, 3)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        out.append((a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
                    a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1])))
    return out


def greedy_nearest(
    source: list[tuple[float, float]], target: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Deterministic shortest-available assignment (greedy Euclidean matching)."""
    remaining = list(target)
    matched = []
    for sx, sy in source:
        best_i = min(
            range(len(remaining)),
            key=lambda i: (remaining[i][0] - sx) ** 2 + (remaining[i][1] - sy) ** 2,
        )
        matched.append(remaining.pop(best_i))
    return matched


def dot_path(points: list[tuple[float, float]], size: float = 1.55) -> str:
    chunks = []
    s = f"{size:.2f}".rstrip("0").rstrip(".")
    neg = f"{-size:.2f}".rstrip("0").rstrip(".")
    for x, y in points:
        chunks.append(f"M{x:.1f} {y:.1f}h{s}v{s}h{neg}z")
    return "".join(chunks)


def npy_bytes(values: list[int] | list[float], shape: tuple[int, ...], descr: str) -> bytes:
    """Write a small NumPy v1 .npy payload without requiring NumPy at runtime."""
    header_dict = {"descr": descr, "fortran_order": False, "shape": shape}
    header = repr(header_dict)
    # Magic + version + uint16 header length consume 10 bytes; align to 16.
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    header_bytes = (header + (" " * padding) + "\n").encode("latin1")
    prefix = b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(header_bytes))
    if descr == "|u1":
        payload = bytes(int(v) & 0xFF for v in values)
    else:
        code = {"<i2": "h", "<f4": "f"}[descr]
        payload_buffer = bytearray()
        for start in range(0, len(values), 4096):
            chunk = values[start : start + 4096]
            payload_buffer.extend(struct.pack(f"<{len(chunk)}{code}", *chunk))
        payload = bytes(payload_buffer)
    return prefix + header_bytes + payload


def write_intermediate_npz(
    output: Path,
    mask: list[list[bool]],
    theme_intermediate: dict[str, dict[str, object]],
) -> None:
    entries: dict[str, bytes] = {}
    entries["subject_mask.npy"] = npy_bytes(
        [int(v) for row in mask for v in row], (PH, PW), "|u1"
    )
    for theme_name in ("dark", "light"):
        payload = theme_intermediate[theme_name]
        dots = payload["dots"]
        bands = payload["bands"]
        travellers = payload["travellers"]
        dot_grid = [0] * (PW * PH)
        for x, y in dots:
            dot_grid[y * PW + x] = 1
        band_grid = [-1] * (PW * PH)
        for b, points in enumerate(bands):
            for x, y in points:
                band_grid[y * PW + x] = b
        flat_travellers = [
            coord
            for state in travellers
            for point in state
            for coord in point
        ]
        entries[f"{theme_name}_dither.npy"] = npy_bytes(dot_grid, (PH, PW), "|u1")
        entries[f"{theme_name}_bands.npy"] = npy_bytes(
            band_grid, (PH, PW), "<i2"
        )
        entries[f"{theme_name}_travellers.npy"] = npy_bytes(
            flat_travellers, (5, TRAVELLERS, 2), "<f4"
        )
    metadata = {
        "canvas": [W, H],
        "portrait_grid": [PW, PH],
        "portrait_offset": [PX, PY],
        "bands": BANDS,
        "travellers": TRAVELLERS,
        "seed": SEED,
        "intro_seconds": INTRO_SECONDS,
        "loop_seconds": LOOP_SECONDS,
        "key_times": [float(x) for x in KEY_TIMES.split(";")],
    }
    entries["metadata.json"] = (
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    with zipfile.ZipFile(output, "w") as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, entries[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if name.endswith(".npy"):
                assert archive.read(name).startswith(b"\x93NUMPY")


def movement_metric(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(math.dist(p, q) for p, q in zip(a, b)) / len(a)


def row_svg(label: str, value: str, y: int, t: dict[str, str], x: int = 506, width: int = 622) -> str:
    font = 12.5
    char = 7.45
    label_text = label.upper()
    label_width = min(118.0, len(label_text) * char)
    value_width = min(330.0, len(value) * char)
    leader_start = x + label_width + 13
    value_x = x + width
    leader_end = value_x - value_width - 13
    dots = max(2, int((leader_end - leader_start) / 6.3))
    leader_width = max(8.0, leader_end - leader_start)
    return (
        f'<text x="{x}" y="{y}" class="label" textLength="{label_width:.1f}" '
        f'lengthAdjust="spacingAndGlyphs">{esc(label_text)}</text>'
        f'<text x="{leader_start:.1f}" y="{y}" class="leader" textLength="{leader_width:.1f}" '
        f'lengthAdjust="spacing">{"·" * dots}</text>'
        f'<text x="{value_x}" y="{y}" class="value" text-anchor="end" '
        f'textLength="{value_width:.1f}" lengthAdjust="spacingAndGlyphs">{esc(value)}</text>'
    )


def build_svg(
    theme_name: str,
    dots: list[tuple[int, int]],
    bands: list[list[tuple[int, int]]],
    order: list[int],
    travellers: tuple[
        list[tuple[float, float]],
        list[tuple[float, float]],
        list[tuple[float, float]],
        list[tuple[float, float]],
        list[tuple[float, float]],
    ],
) -> str:
    t = THEMES[theme_name]
    rank = [0] * BANDS
    for i, b in enumerate(order):
        rank[b] = i
    band_chunks = []
    for b, pts in enumerate(bands):
        group = min(INTRO_GROUPS - 1, int(rank[b] * INTRO_GROUPS / BANDS))
        begin = 0.10 + group * (2.78 / (INTRO_GROUPS - 1))
        dx = ((b * 17) % 7 - 3) * 0.22
        dy = ((b * 29) % 5 - 2) * 0.16
        dur = 5.8 + (b % 9) * 0.37
        path = compact_path(pts)
        band_chunks.append(
            f'<g opacity="0">'
            f'<animate attributeName="opacity" begin="{begin:.3f}s" dur=".22s" '
            f'values="0;1" fill="freeze"/>'
            f'<animateTransform attributeName="transform" type="translate" additive="sum" '
            f'begin="{INTRO_SECONDS}s" dur="{dur:.2f}s" values="0 0;{dx:.2f} {dy:.2f};0 0" '
            f'keyTimes="0;.47;1" repeatCount="indefinite"/>'
            f'<path d="{path}"/></g>'
        )
    p0, react, code, vercel, p1 = travellers
    d_values = ";".join(
        dot_path(state)
        for state in (p0, p0, react, react, code, code, vercel, vercel, p1)
    )
    info_rows = "".join(
        (
            row_svg("Role", PROFILE["Role"], 214, t),
            row_svg("Origin", PROFILE["Origin"], 238, t),
            row_svg("Education", PROFILE["Education"], 262, t),
            row_svg("Status", PROFILE["Status"], 286, t),
            row_svg("ToolChain", PROFILE["ToolChain"], 330, t),
            row_svg("Core.Lang", PROFILE["Core.Lang"], 354, t),
            row_svg("Core.Frontend", PROFILE["Core.Frontend"], 378, t),
            row_svg("Core.Backend", PROFILE["Core.Backend"], 402, t),
            row_svg("Core.Database", PROFILE["Core.Database"], 426, t),
            row_svg("Core.Infra", PROFILE["Core.Infra"], 450, t),
            row_svg("Grid.Mail", PROFILE["Grid.Mail"], 494, t),
            row_svg("Grid.Portfolio", PROFILE["Grid.Portfolio"], 518, t),
            row_svg("Grid.Instagram", PROFILE["Grid.Instagram"], 542, t),
            row_svg("Grid.GitHub", PROFILE["Grid.GitHub"], 566, t),
        )
    )
    dot_count = len(dots)
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
  <title id="title">Rubens Rafael — animated developer profile terminal</title>
  <desc id="desc">Dithered portrait morphing into React, code, and Vercel symbols beside Rubens Rafael's developer profile.</desc>
  <style>
    text {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
    .micro {{ font-size: 10px; font-weight: 700; letter-spacing: 1.8px; fill: {t["muted"]}; }}
    .section {{ font-size: 11px; font-weight: 800; letter-spacing: 2.6px; fill: {t["chrome"]}; }}
    .label {{ font-size: 12.5px; font-weight: 700; fill: {t["muted"]}; }}
    .leader {{ font-size: 12px; fill: {t["faint"]}; }}
    .value {{ font-size: 12.5px; font-weight: 650; fill: {t["text"]}; }}
  </style>
  <rect width="{W}" height="{H}" rx="22" fill="{t["bg"]}"/>
  <rect x="18" y="18" width="1144" height="574" rx="18" fill="{t["shadow"]}" opacity=".18"/>
  <rect x="18" y="16" width="1144" height="574" rx="18" fill="{t["panel"]}" stroke="{t["border"]}"/>
  <path d="M18 70H1162" stroke="{t["border"]}"/>
  <circle cx="46" cy="43" r="6" fill="{t["red"]}"/>
  <circle cx="67" cy="43" r="6" fill="#FBBF24"/>
  <circle cx="88" cy="43" r="6" fill="{t["accent"]}"/>
  <text x="118" y="48" font-size="13" font-weight="700" fill="{t["text"]}">profile.sh --live</text>
  <text x="1012" y="47" class="micro">SESSION 01</text>
  <g transform="translate(1092 32)">
    <rect width="50" height="22" rx="11" fill="{t["panel2"]}" stroke="{t["border"]}"/>
    <circle cx="12" cy="11" r="4" fill="{t["red"]}">
      <animate attributeName="r" values="3;5;3" keyTimes="0;.5;1" dur="1.4s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values=".55;1;.55" dur="1.4s" repeatCount="indefinite"/>
    </circle>
    <text x="21" y="15" font-size="9" font-weight="800" fill="{t["red"]}">LIVE</text>
  </g>

  <text x="54" y="112" class="section">VISUAL.MAP</text>
  <text x="506" y="112" class="section">SYSTEM.INFO</text>
  <path d="M54 124H433M506 124H1128" stroke="{t["border"]}"/>

  <rect x="54" y="144" width="340" height="380" rx="12" fill="{t["panel2"]}" stroke="{t["border"]}"/>
  <path d="M62 155h16M62 155v16M386 155h-16M386 155v16M62 513h16M62 513v-16M386 513h-16M386 513v-16"
        stroke="{t["chrome"]}" stroke-width="1.2" fill="none" opacity=".8"/>
  <g transform="translate({PX} {PY})" fill="{t["portrait"]}" shape-rendering="crispEdges">
    <g>
      <animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s"
        values="1;1;0;0;0;0;0;0;1" keyTimes="{KEY_TIMES}" repeatCount="indefinite"/>
      {''.join(band_chunks)}
    </g>
    <path d="{dot_path(p0)}" opacity=".16">
      <animate attributeName="d" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s"
        values="{d_values}" keyTimes="{KEY_TIMES}" calcMode="linear" repeatCount="indefinite"/>
      <animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s"
        values=".16;.16;.96;.96;.96;.96;.96;.96;.16" keyTimes="{KEY_TIMES}" repeatCount="indefinite"/>
    </path>
  </g>
  <rect x="68" y="533" width="312" height="25" rx="12.5" fill="{t["panel2"]}" stroke="{t["border"]}"/>
  <circle cx="82" cy="545.5" r="3" fill="{t["accent"]}"/>
  <text x="93" y="549" class="micro" style="letter-spacing:1px">FS/1BIT · {dot_count:05d} PTS · SIGMA≈4</text>

  <text x="506" y="151" class="micro">SUBJECT</text>
  <text x="506" y="183" font-size="29" font-weight="850" letter-spacing=".8" fill="{t["text"]}"
        textLength="311" lengthAdjust="spacingAndGlyphs">{PROFILE["Subject"]}</text>
  <g transform="translate(952 151)">
    <rect width="176" height="31" rx="15.5" fill="{t["panel2"]}" stroke="{t["chrome"]}" opacity=".98"/>
    <circle cx="15" cy="15.5" r="3.5" fill="{t["accent"]}"/>
    <text x="27" y="20" font-size="12" font-weight="750" fill="{t["chrome"]}"
          textLength="132" lengthAdjust="spacingAndGlyphs">{PROFILE["Handle"]}</text>
  </g>
  {info_rows}
  <text x="506" y="308" class="micro">CORE.STACK / RUNTIME</text>
  <text x="506" y="472" class="micro">GRID.CONTACT / ROUTES</text>
  <path d="M506 314H1128M506 478H1128" stroke="{t["border"]}"/>
</svg>
'''
    return svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("portrait-source.png"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.source)
    crop = crop_portrait(source)
    density = prepared_density(crop)
    mask = dark_subject_mask(crop)

    theme_data = {}
    all_metrics: dict[str, object] = {
        "canvas": [W, H],
        "portrait_grid": [PW, PH],
        "bands": BANDS,
        "traveller_dots": TRAVELLERS,
        "intro_groups": INTRO_GROUPS,
        "intro_seconds": INTRO_SECONDS,
        "loop_seconds": LOOP_SECONDS,
        "key_times": [float(x) for x in KEY_TIMES.split(";")],
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
    }
    theme_intermediate: dict[str, dict[str, object]] = {}

    for theme_name in ("dark", "light"):
        dots = floyd_steinberg(density, mask if theme_name == "dark" else None)
        bands, sigma = band_map(dots)
        order, spatial, straight = choose_schedule(bands)
        p0 = evenly_sample(dots, TRAVELLERS)
        react = greedy_nearest(p0, react_points(TRAVELLERS))
        code = greedy_nearest(react, code_points(TRAVELLERS))
        vercel = greedy_nearest(code, vercel_points(TRAVELLERS))
        p1 = greedy_nearest(vercel, p0)
        travellers = (p0, react, code, vercel, p1)
        theme_intermediate[theme_name] = {
            "dots": dots,
            "bands": bands,
            "travellers": travellers,
        }
        svg = build_svg(theme_name, dots, bands, order, travellers)
        out = args.output_dir / f"{theme_name}.svg"
        out.write_text(svg, encoding="utf-8", newline="\n")
        ET.parse(out)
        theme_data[theme_name] = {
            "dot_count": len(dots),
            "foreground_mask_pixels": sum(sum(row) for row in mask) if theme_name == "dark" else PW * PH,
            "noise_sigma": round(sigma, 4),
            "intro_spatial_evenness": round(spatial, 5),
            "straight_boundary_metric": round(straight, 5),
            "mean_travel_px": {
                "portrait_to_react": round(movement_metric(p0, react), 3),
                "react_to_code": round(movement_metric(react, code), 3),
                "code_to_vercel": round(movement_metric(code, vercel), 3),
                "vercel_to_portrait": round(movement_metric(vercel, p1), 3),
            },
            "file_bytes": out.stat().st_size,
            "xml_valid": True,
        }
    npz_path = args.output_dir / "portrait-data.npz"
    write_intermediate_npz(npz_path, mask, theme_intermediate)
    all_metrics["intermediate"] = {
        "file": npz_path.name,
        "file_bytes": npz_path.stat().st_size,
        "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "npz_valid": True,
        "members": [
            "subject_mask.npy",
            "dark_dither.npy",
            "dark_bands.npy",
            "dark_travellers.npy",
            "light_dither.npy",
            "light_bands.npy",
            "light_travellers.npy",
            "metadata.json",
        ],
    }
    all_metrics["themes"] = theme_data
    (args.output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

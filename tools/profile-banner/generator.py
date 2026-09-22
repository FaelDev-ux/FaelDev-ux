#!/usr/bin/env python3
"""Reproducible GitHub-profile terminal banner generator.

Source of truth:
  - assets/legacy-portrait-data.npz
  - assets/* technology logos
  - this file

Outputs:
  - dark.svg / light.svg (1180x610 animated SVG)
  - metrics.json
  - portrait-data.npz

The legacy portrait point map is preserved exactly. Its particles morph through
the original symbols and the supplied technology logos in one continuous loop.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import math
import random
import statistics
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw


W, H = 1180, 610
PW, PH = 300, 340
PX, PY = 74, 164
BANDS = 94
TRAVELLERS = 3600
PARTICLE_SIZE = 2.8
INTRO_GROUPS = 60
INTRO_SECONDS = 3.2
SEED = 24072003
HOLD_SECONDS = 1.4
TRANSITION_SECONDS = 0.8

TRAVELLER_STATES = (
    "portrait",
    "nextjs",
    "code",
    "vercel",
    "javascript",
    "typescript",
    "react",
    "tailwind",
    "python",
    "supabase",
    "portrait",
)
DISPLAY_STATES = TRAVELLER_STATES[:-1]
LOOP_SECONDS = len(DISPLAY_STATES) * (HOLD_SECONDS + TRANSITION_SECONDS)


def animation_key_times() -> tuple[float, ...]:
    elapsed = 0.0
    times = [0.0]
    for _ in DISPLAY_STATES:
        elapsed += HOLD_SECONDS
        times.append(elapsed / LOOP_SECONDS)
        elapsed += TRANSITION_SECONDS
        times.append(elapsed / LOOP_SECONDS)
    return tuple(round(value, 6) for value in times)


KEY_TIME_VALUES = animation_key_times()
KEY_TIMES = ";".join(f"{value:.6f}".rstrip("0").rstrip(".") for value in KEY_TIME_VALUES)

TECHNOLOGIES = (
    ("JavaScript", "javascript.png", "image/png"),
    ("TypeScript", "typescript.webp", "image/webp"),
    ("React", "react.png", "image/png"),
    ("Tailwind CSS", "tailwind.png", "image/png"),
    ("Python", "python.jpeg", "image/jpeg"),
    ("Supabase", "supabase.jpeg", "image/jpeg"),
)

TECH_LOOP = (
    ("javascript", "javascript.png"),
    ("typescript", "typescript.webp"),
    ("react", "react.png"),
    ("tailwind", "tailwind.png"),
    ("python", "python.jpeg"),
    ("supabase", "supabase.jpeg"),
)

TECH_COLORS = {
    "javascript": "#F7DF1E",
    "typescript": "#3178C6",
    "react": "#61DAFB",
    "tailwind": "#38BDF8",
    "python": "#3776AB",
    "supabase": "#3ECF8E",
}

PROFILE = {
    "Subject": "RUBENS.RAFAEL",
    "Handle": "@FaelDev-ux",
    "Role": "Web Developer",
    "Origin": "Brasil",
    "Education": "Eng. de Software",
    "Status": "Building + Learning + Shipping",
    "ToolChain": "VS Code · Git · Vercel",
    "Core.Lang": "JavaScript · Python",
    "Core.Frontend": "Next.js · Tailwind",
    "Core.Database": "Firebase · Supabase",
    "Grid.Mail": "rubensnobrega2003@gmail.com",
    "Grid.Portfolio": "www.r2labss.dev",
    "Grid.Instagram": "@r.noobrega",
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


def read_npy_u1(blob: bytes) -> tuple[tuple[int, ...], bytes]:
    """Read the uint8 NumPy payloads stored in the legacy portrait archive."""
    if not blob.startswith(b"\x93NUMPY"):
        raise ValueError("Invalid NPY payload")
    major = blob[6]
    if major == 1:
        header_len = struct.unpack("<H", blob[8:10])[0]
        header_start = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", blob[8:12])[0]
        header_start = 12
    else:
        raise ValueError(f"Unsupported NPY version: {major}")
    header_end = header_start + header_len
    header = ast.literal_eval(blob[header_start:header_end].decode("latin1").strip())
    if header["descr"] != "|u1" or header["fortran_order"]:
        raise ValueError("Expected a row-major uint8 NPY payload")
    return tuple(header["shape"]), blob[header_end:]


def load_legacy_grid(archive_path: Path, member: str) -> list[list[bool]]:
    with zipfile.ZipFile(archive_path) as archive:
        shape, payload = read_npy_u1(archive.read(member))
    if shape != (PH, PW) or len(payload) != PW * PH:
        raise ValueError(f"Unexpected legacy portrait grid: {member} {shape}")
    return [
        [bool(payload[y * PW + x]) for x in range(PW)]
        for y in range(PH)
    ]


def load_legacy_portrait_points(
    archive_path: Path, theme_name: str
) -> list[tuple[int, int]]:
    grid = load_legacy_grid(archive_path, f"{theme_name}_dither.npy")
    return [(x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if value]


def technology_logo_points(path: Path, n: int) -> list[tuple[float, float]]:
    """Convert a supplied color logo into a centered particle silhouette."""
    source = Image.open(path).convert("RGBA")
    mask = Image.new("L", source.size, 0)
    mask_pixels = mask.load()
    source_pixels = source.load()
    for y in range(source.height):
        for x in range(source.width):
            red, green, blue, alpha = source_pixels[x, y]
            saturation = max(red, green, blue) - min(red, green, blue)
            if alpha >= 32 and saturation >= 32 and max(red, green, blue) >= 56:
                mask_pixels[x, y] = 255

    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"No colored logo pixels found in {path}")
    logo = mask.crop(bbox)
    logo.thumbnail((230, 230), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (PW, PH), 0)
    canvas.paste(logo, ((PW - logo.width) // 2, (PH - logo.height) // 2))
    points = [
        (x, y)
        for y in range(PH)
        for x in range(PW)
        if canvas.getpixel((x, y)) >= 96
    ]
    return evenly_sample(points, n)


def python_logo_points(n: int) -> list[tuple[float, float]]:
    """Build the complete Python mark so a cropped source image cannot clip its tail."""
    mask = Image.new("L", (PW, PH), 0)
    draw = ImageDraw.Draw(mask)
    scale = 230 / 56
    offset_x = (PW - 64 * scale) / 2
    offset_y = (PH - 64 * scale) / 2

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        return (offset_x + point[0] * scale, offset_y + point[1] * scale)

    def cubic(
        points: list[tuple[float, float]],
        control_a: tuple[float, float],
        control_b: tuple[float, float],
        end: tuple[float, float],
        steps: int = 12,
    ) -> None:
        start_x, start_y = points[-1]
        for step in range(1, steps + 1):
            t = step / steps
            inv = 1 - t
            points.append(
                (
                    inv**3 * start_x
                    + 3 * inv**2 * t * control_a[0]
                    + 3 * inv * t**2 * control_b[0]
                    + t**3 * end[0],
                    inv**3 * start_y
                    + 3 * inv**2 * t * control_a[1]
                    + 3 * inv * t**2 * control_b[1]
                    + t**3 * end[1],
                )
            )

    upper = [(30.25, 4.0)]
    cubic(upper, (23.4628, 4.0), (18.0, 7.9025), (18.0, 12.75))
    upper.extend(((18.0, 15.0), (32.0, 15.0), (32.0, 18.0), (12.75, 18.0)))
    cubic(upper, (7.9025, 18.0), (4.0, 23.4628), (4.0, 30.25))
    upper.append((4.0, 33.75))
    cubic(upper, (4.0, 40.5372), (7.9025, 46.0), (12.75, 46.0))
    upper.extend(((18.0, 46.0), (18.0, 39.0)))
    cubic(upper, (18.0, 35.122), (21.122, 32.0), (25.0, 32.0))
    upper.append((39.0, 32.0))
    cubic(upper, (39.0, 32.0), (46.0, 32.0), (46.0, 25.0))
    upper.append((46.0, 12.75))
    cubic(upper, (46.0, 7.9025), (40.5372, 4.0), (33.75, 4.0))

    lower = [(46.0, 18.0), (46.0, 25.0)]
    cubic(lower, (46.0, 28.878), (42.878, 32.0), (39.0, 32.0))
    lower.append((25.0, 32.0))
    cubic(lower, (25.0, 32.0), (18.0, 32.0), (18.0, 39.0))
    lower.append((18.0, 51.25))
    cubic(lower, (18.0, 56.0968), (23.4628, 60.0), (30.25, 60.0))
    lower.append((33.75, 60.0))
    cubic(lower, (40.5372, 60.0), (46.0, 56.0968), (46.0, 51.25))
    lower.extend(((46.0, 49.0), (32.0, 49.0), (32.0, 46.0), (51.25, 46.0)))
    cubic(lower, (56.0968, 46.0), (60.0, 40.5372), (60.0, 33.75))
    lower.append((60.0, 30.25))
    cubic(lower, (60.0, 23.4628), (56.0968, 18.0), (51.25, 18.0))

    draw.polygon([transform(point) for point in upper], fill=255)
    draw.polygon([transform(point) for point in lower], fill=255)
    eye_radius = 1.65 * scale
    for eye in ((25.5, 8.5), (39.5, 55.5)):
        eye_x, eye_y = transform(eye)
        draw.ellipse(
            (
                eye_x - eye_radius,
                eye_y - eye_radius,
                eye_x + eye_radius,
                eye_y + eye_radius,
            ),
            fill=0,
        )

    points = [
        (x, y)
        for y in range(PH)
        for x in range(PW)
        if mask.getpixel((x, y)) >= 96
    ]
    return evenly_sample(points, n)


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


def thick_line_points(
    start: tuple[float, float],
    end: tuple[float, float],
    n: int,
    width: float,
    layers: int,
) -> list[tuple[float, float]]:
    """Sample a deterministic, optically solid line from parallel dot rows."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    counts = [n // layers] * layers
    for i in range(n % layers):
        counts[i] += 1
    out: list[tuple[float, float]] = []
    for layer, count in enumerate(counts):
        offset = width * (layer / max(1, layers - 1) - 0.5)
        shifted = [
            (start[0] + nx * offset, start[1] + ny * offset),
            (end[0] + nx * offset, end[1] + ny * offset),
        ]
        out.extend(sample_polyline(shifted, count))
    return out


def nextjs_points(n: int) -> list[tuple[float, float]]:
    """Recognizable Next.js mark: enclosing ring and its diagonal N monogram."""
    out: list[tuple[float, float]] = []
    ring_n = round(n * 300 / 900)
    left_n = round(n * 180 / 900)
    diagonal_n = round(n * 300 / 900)
    right_n = n - ring_n - left_n - diagonal_n
    for i in range(ring_n):
        angle = (2 * math.pi * i / ring_n) - (math.pi / 2)
        out.append((150 + 111 * math.cos(angle), 170 + 111 * math.sin(angle)))

    # The long diagonal exits toward the lower-right, matching the familiar mark.
    out.extend(thick_line_points((98, 109), (98, 231), left_n, 10, 6))
    out.extend(thick_line_points((98, 109), (248, 271), diagonal_n, 11, 6))
    out.extend(thick_line_points((205, 109), (205, 204), right_n, 10, 6))
    assert len(out) == n
    return out


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


def dot_path(points: list[tuple[float, float]]) -> str:
    """Create tiny round-capped segments that render as circular particles."""
    return "".join(f"M{x:.1f} {y:.1f}h.01" for x, y in points)


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
            flat_travellers, (len(travellers), TRAVELLERS, 2), "<f4"
        )
    metadata = {
        "canvas": [W, H],
        "portrait_grid": [PW, PH],
        "portrait_offset": [PX, PY],
        "bands": BANDS,
        "travellers": TRAVELLERS,
        "traveller_states": list(TRAVELLER_STATES),
        "seed": SEED,
        "intro_seconds": INTRO_SECONDS,
        "loop_seconds": LOOP_SECONDS,
        "key_times": list(KEY_TIME_VALUES),
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


def image_data_uri(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def technology_cards_svg(
    theme: dict[str, str], technologies: list[tuple[str, str]]
) -> str:
    cards: list[str] = []
    for index, (name, uri) in enumerate(technologies):
        column = index % 3
        row = index // 3
        x = 506 + column * 211
        y = 326 + row * 46
        cards.append(
            f'<g transform="translate({x} {y})">'
            f'<title>{esc(name)}</title>'
            f'<rect width="200" height="38" rx="9" fill="{theme["panel2"]}" '
            f'stroke="{theme["border"]}"/>'
            f'<rect x="6" y="5" width="28" height="28" rx="6" fill="#FFFFFF"/>'
            f'<image x="8" y="7" width="24" height="24" href="{uri}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
            f'<text x="44" y="24" font-size="11.5" font-weight="750" '
            f'fill="{theme["text"]}">{esc(name)}</text>'
            f'</g>'
        )
    return "".join(cards)


def build_svg(
    theme_name: str,
    dots: list[tuple[int, int]],
    bands: list[list[tuple[int, int]]],
    order: list[int],
    travellers: list[list[tuple[float, float]]],
    technologies: list[tuple[str, str]],
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
    animation_states = [
        state
        for state in travellers[:-1]
        for _ in range(2)
    ] + [travellers[-1]]
    d_values = ";".join(dot_path(state) for state in animation_states)
    band_opacity_values = ";".join(
        value
        for state_name in DISPLAY_STATES
        for value in (("1", "1") if state_name == "portrait" else ("0", "0"))
    ) + ";1"
    particle_opacity_values = ";".join(
        value
        for state_name in DISPLAY_STATES
        for value in ((".16", ".16") if state_name == "portrait" else (".96", ".96"))
    ) + ";.16"
    state_colors = {
        "portrait": t["portrait"],
        "nextjs": t["portrait"],
        "code": t["portrait"],
        "vercel": t["portrait"],
        **TECH_COLORS,
    }
    particle_color_values = ";".join(
        color
        for state_name in DISPLAY_STATES
        for color in (state_colors[state_name], state_colors[state_name])
    ) + f';{t["portrait"]}'
    info_rows = "".join(
        (
            row_svg("Role", PROFILE["Role"], 214, t),
            row_svg("Origin", PROFILE["Origin"], 238, t),
            row_svg("Education", PROFILE["Education"], 262, t),
            row_svg("Status", PROFILE["Status"], 286, t),
            row_svg("ToolChain", PROFILE["ToolChain"], 430, t),
            row_svg("Grid.Mail", PROFILE["Grid.Mail"], 486, t),
            row_svg("Grid.Portfolio", PROFILE["Grid.Portfolio"], 510, t),
            row_svg("Grid.Instagram", PROFILE["Grid.Instagram"], 534, t),
            row_svg("Grid.GitHub", PROFILE["Grid.GitHub"], 558, t),
        )
    )
    technology_cards = technology_cards_svg(t, technologies)
    dot_count = len(dots)
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
  <title id="title">Rubens Rafael — animated developer profile terminal</title>
  <desc id="desc">The original dithered portrait morphs through developer symbols and JavaScript, TypeScript, React, Tailwind CSS, Python, and Supabase logos beside Rubens Rafael's developer profile.</desc>
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
  <g transform="translate({PX} {PY})" fill="{t["portrait"]}">
    <g shape-rendering="crispEdges">
      <animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s"
        values="{band_opacity_values}" keyTimes="{KEY_TIMES}" repeatCount="indefinite"/>
      {''.join(band_chunks)}
    </g>
    <path d="{dot_path(travellers[0])}" opacity=".16" fill="none"
          stroke="{t["portrait"]}" stroke-width="{PARTICLE_SIZE}" stroke-linecap="round">
      <animate attributeName="d" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s"
        values="{d_values}" keyTimes="{KEY_TIMES}" calcMode="linear" repeatCount="indefinite"/>
      <animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s"
        values="{particle_opacity_values}" keyTimes="{KEY_TIMES}" repeatCount="indefinite"/>
      <animate attributeName="stroke" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s"
        values="{particle_color_values}" keyTimes="{KEY_TIMES}" repeatCount="indefinite"/>
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
  {technology_cards}
  <text x="506" y="308" class="micro">CORE.STACK / RUNTIME</text>
  <text x="506" y="454" class="micro">GRID.CONTACT / ROUTES</text>
  <path d="M506 314H1128M506 460H1128" stroke="{t["border"]}"/>
</svg>
'''
    return svg


def main() -> None:
    parser = argparse.ArgumentParser()
    default_assets = Path(__file__).with_name("assets")
    parser.add_argument(
        "--portrait-data",
        type=Path,
        default=default_assets / "legacy-portrait-data.npz",
    )
    parser.add_argument("--assets-dir", type=Path, default=default_assets)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mask = load_legacy_grid(args.portrait_data, "subject_mask.npy")
    technologies = [
        (name, image_data_uri(args.assets_dir / filename, mime_type))
        for name, filename, mime_type in TECHNOLOGIES
    ]
    logo_targets = {
        state_name: (
            python_logo_points(TRAVELLERS)
            if state_name == "python"
            else technology_logo_points(args.assets_dir / filename, TRAVELLERS)
        )
        for state_name, filename in TECH_LOOP
    }

    theme_data = {}
    all_metrics: dict[str, object] = {
        "canvas": [W, H],
        "portrait_grid": [PW, PH],
        "bands": BANDS,
        "traveller_dots": TRAVELLERS,
        "particle_size": PARTICLE_SIZE,
        "python_loop_source": "procedural-complete-mark",
        "intro_groups": INTRO_GROUPS,
        "intro_seconds": INTRO_SECONDS,
        "loop_seconds": LOOP_SECONDS,
        "key_times": list(KEY_TIME_VALUES),
        "portrait_source": args.portrait_data.name,
        "source_sha256": hashlib.sha256(args.portrait_data.read_bytes()).hexdigest(),
        "technology_assets_sha256": {
            filename: hashlib.sha256((args.assets_dir / filename).read_bytes()).hexdigest()
            for _, filename, _ in TECHNOLOGIES
        },
    }
    theme_intermediate: dict[str, dict[str, object]] = {}

    for theme_name in ("dark", "light"):
        dots = load_legacy_portrait_points(args.portrait_data, theme_name)
        bands, sigma = band_map(dots)
        order, spatial, straight = choose_schedule(bands)
        p0 = evenly_sample(dots, TRAVELLERS)
        targets = (
            ("nextjs", nextjs_points(TRAVELLERS)),
            ("code", code_points(TRAVELLERS)),
            ("vercel", vercel_points(TRAVELLERS)),
            *((state_name, logo_targets[state_name]) for state_name, _ in TECH_LOOP),
        )
        travellers = [p0]
        current = p0
        for _, target in targets:
            current = greedy_nearest(current, target)
            travellers.append(current)
        travellers.append(greedy_nearest(current, p0))
        assert len(travellers) == len(TRAVELLER_STATES)
        theme_intermediate[theme_name] = {
            "dots": dots,
            "bands": bands,
            "travellers": travellers,
        }
        svg = build_svg(theme_name, dots, bands, order, travellers, technologies)
        out = args.output_dir / f"{theme_name}.svg"
        out.write_text(svg, encoding="utf-8", newline="\n")
        ET.parse(out)
        theme_data[theme_name] = {
            "dot_count": len(dots),
            "foreground_mask_pixels": (
                sum(sum(row) for row in mask) if theme_name == "dark" else PW * PH
            ),
            "noise_sigma": round(sigma, 4),
            "intro_spatial_evenness": round(spatial, 5),
            "straight_boundary_metric": round(straight, 5),
            "mean_travel_px": {
                f"{source_name}_to_{target_name}": round(
                    movement_metric(source_points, target_points), 3
                )
                for source_name, target_name, source_points, target_points in zip(
                    TRAVELLER_STATES,
                    TRAVELLER_STATES[1:],
                    travellers,
                    travellers[1:],
                )
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

"""Small dependency-free PNG renderer for Telegram PnL status charts."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Iterable, List


RGB = tuple[int, int, int]


def render_pnl_chart(
    output_path: Path,
    realized_pnl: Iterable[float],
    open_pnl: Iterable[float] = (),
    width: int = 900,
    height: int = 500,
) -> Path:
    """Render a compact realized/open PnL chart and return the PNG path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = _Canvas(width, height, (11, 18, 32))
    canvas.rect(0, 0, width, height, (11, 18, 32))
    canvas.rect(36, 42, width - 72, height - 94, (18, 28, 45))

    left, top, right, bottom = 76, 76, width - 60, height - 94
    values = [0.0]
    total = 0.0
    for pnl in realized_pnl:
        total += float(pnl)
        values.append(total)
    open_values = [float(pnl) for pnl in open_pnl]
    scale_values = values + open_values + [0.0]
    low, high = min(scale_values), max(scale_values)
    if low == high:
        low -= 1.0
        high += 1.0
    padding = max((high - low) * 0.15, 0.01)
    low -= padding
    high += padding

    zero_y = _scale(0.0, low, high, bottom, top)
    canvas.line(left, zero_y, right, zero_y, (73, 88, 112), 2)
    canvas.line(left, top, left, bottom, (45, 58, 80), 2)
    canvas.line(left, bottom, right, bottom, (45, 58, 80), 2)

    if len(values) == 1:
        values.append(values[0])
    points: List[tuple[int, int]] = []
    for index, value in enumerate(values):
        x = left + round((right - left) * index / max(1, len(values) - 1))
        y = _scale(value, low, high, bottom, top)
        points.append((x, y))
    for start, end in zip(points, points[1:]):
        color = (68, 211, 146) if end[1] <= zero_y else (248, 113, 113)
        canvas.line(start[0], start[1], end[0], end[1], color, 4)
    for x, y in points[-12:]:
        canvas.rect(x - 4, y - 4, 8, 8, (226, 232, 240))

    bar_left = left
    bar_width = 24
    for index, pnl in enumerate(open_values[:20]):
        x = bar_left + index * (bar_width + 10)
        if x + bar_width > right:
            break
        y = _scale(pnl, low, high, bottom, top)
        color = (34, 197, 94) if pnl >= 0 else (239, 68, 68)
        canvas.rect(x, min(y, zero_y), bar_width, max(3, abs(zero_y - y)), color)

    canvas.write_png(output_path)
    return output_path


class _Canvas:
    def __init__(self, width: int, height: int, background: RGB) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(background * width * height)

    def rect(self, x: int, y: int, width: int, height: int, color: RGB) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + width)
        y1 = min(self.height, y + height)
        for yy in range(y0, y1):
            row = yy * self.width * 3
            for xx in range(x0, x1):
                offset = row + xx * 3
                self.pixels[offset : offset + 3] = bytes(color)

    def line(
        self, x0: int, y0: int, x1: int, y1: int, color: RGB, thickness: int = 1
    ) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            radius = max(0, thickness // 2)
            self.rect(x0 - radius, y0 - radius, thickness, thickness, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def write_png(self, output_path: Path) -> None:
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * stride
            raw.extend(self.pixels[start : start + stride])
        with output_path.open("wb") as file:
            file.write(b"\x89PNG\r\n\x1a\n")
            _write_chunk(file, b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            _write_chunk(file, b"IDAT", zlib.compress(bytes(raw), level=9))
            _write_chunk(file, b"IEND", b"")


def _write_chunk(file: object, kind: bytes, data: bytes) -> None:
    file.write(struct.pack(">I", len(data)))
    file.write(kind)
    file.write(data)
    file.write(struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _scale(value: float, low: float, high: float, bottom: int, top: int) -> int:
    return round(bottom - ((value - low) / (high - low)) * (bottom - top))

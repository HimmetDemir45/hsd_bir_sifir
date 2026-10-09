"""Video dosyası için zamana bağlı bölge (demo): <video>.zones.json = [[başlangıç_sn, bölge_id], ...].

Gerçek kurulumda her kameranın tek bölgesi var (config cameras[].zone_id). Sunum demosu tek video dosyası olduğu
için parçaları farklı kameralar/bölgeler gibi gösterir: olaylar kat planında farklı yerlere düşer.
"""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path


class ZoneTimeline:
    def __init__(self, entries: list[tuple[float, str]]) -> None:
        entries = sorted((float(t), str(z)) for t, z in entries)
        self.starts = [t for t, _ in entries]
        self.zones = [z for _, z in entries]

    @classmethod
    def for_source(cls, source: str, known_zones: dict) -> "ZoneTimeline | None":
        """Video yanında .zones.json varsa yükler; config'de olmayan bölge varsa hata (kat planına düşmez)."""
        if not isinstance(source, str):
            return None
        path = Path(source).with_suffix(".zones.json")
        if not path.is_file():
            return None
        tl = cls([tuple(e) for e in json.loads(path.read_text(encoding="utf-8"))])
        unknown = set(tl.zones) - set(known_zones)
        if unknown:
            raise ValueError(f"{path.name}: config.yaml zones'da olmayan bölge(ler): {sorted(unknown)}")
        return tl

    def zone_at(self, pos_s: float, default: str) -> str:
        """Video saniyesindeki bölge (ilk girişten önce: default)."""
        i = bisect_right(self.starts, pos_s) - 1
        return self.zones[i] if i >= 0 else default

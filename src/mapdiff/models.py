"""Versioned data model: what a floorplan *means* (snapshot) and how it changed (changeset).

Coordinates are normalised to the page: (0, 0) is top-left, (1, 1) bottom-right.
Presentation (legend symbols, colours, marker codes) is stored for traceability
but is never used as identity when comparing versions.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from mapdiff.taxonomy import Category


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Point(BaseModel):
    x: float
    y: float

    def dist(self, other: Point) -> float:
        return math.dist((self.x, self.y), (other.x, other.y))


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def around(cls, p: Point, half: float) -> BBox:
        """Square box of half-size ``half`` centred on ``p``, shifted to stay inside the page."""
        half = min(half, 0.5)
        x0, y0 = min(max(0.0, p.x - half), 1 - 2 * half), min(max(0.0, p.y - half), 1 - 2 * half)
        return cls(x0=x0, y0=y0, x1=x0 + 2 * half, y1=y0 + 2 * half)

    def expanded(self, fraction: float) -> BBox:
        """Grown by ``fraction`` of its size on every side, clipped to the page."""
        dx, dy = fraction * self.width, fraction * self.height
        return BBox(x0=max(0.0, self.x0 - dx), y0=max(0.0, self.y0 - dy),
                    x1=min(1.0, self.x1 + dx), y1=min(1.0, self.y1 + dy))

    def contains(self, p: Point, margin: float = 0.0) -> bool:
        return (self.x0 - margin <= p.x <= self.x1 + margin) and (self.y0 - margin <= p.y <= self.y1 + margin)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> Point:
        return Point(x=(self.x0 + self.x1) / 2, y=(self.y0 + self.y1) / 2)


FULL_PAGE = BBox(x0=0, y0=0, x1=1, y1=1)

PanelKind = Literal["map", "inset_map", "legend", "table", "title", "decoration", "other"]


class Panel(BaseModel):
    """A region of the page: a floor map, an inset (often another level), a legend, ..."""

    id: str
    page: int = 0
    kind: PanelKind
    title: str | None = None
    level: str | None = Field(None, description="Floor/level shown in this panel, e.g. 'departures', 'satellite lower'")
    bbox: BBox

    @property
    def is_map(self) -> bool:
        return self.kind in ("map", "inset_map")


class LegendEntry(BaseModel):
    """One line of this map's own visual language. Presentation only."""

    page: int = 0
    kind: Literal["marker_code", "color", "icon", "line", "pattern", "other"]
    symbol: str = Field(description="What it looks like / the code, e.g. '8', 'purple fill', 'icon: man+woman'")
    meaning: str = Field(description="What it means, e.g. 'Culto Café', 'Food & Drinks', 'Toilets'")
    category: Category | None = None


class Verification(BaseModel):
    """Independent assessment of one entity by the mapping-confidence agent."""

    score: float
    verdict: Literal["correct", "incorrect", "unsure"]
    method: str = Field(description="'vlm', 'text_layer', or 'vlm+text_layer'")
    issue: str = ""
    suggested_fix: str = ""


class Entity(BaseModel):
    """Something that is somewhere on the map."""

    id: str
    lineage_id: str | None = Field(None, description="Stable id carried across versions once matched")
    page: int = 0
    panel: str | None = None
    level: str | None = None
    category: Category
    name: str | None = Field(None, description="Specific identity: brand, airline, gate/desk number. None for generic facilities")
    label: str | None = Field(None, description="Code/marker as drawn on the map (presentation)")
    anchor: Point
    bbox: BBox | None = None
    evidence: str = ""
    confidence: float = Field(1.0, description="Verified score once assessed; the extractor's own estimate before")
    extractor_confidence: float | None = None
    verification: Verification | None = None


class Relation(BaseModel):
    """A non-spatial fact, typically from a table: 'Emirates' --checks_in_at--> 'desk 14'."""

    page: int = 0
    subject: str
    predicate: str
    object: str
    evidence: str = ""
    confidence: float = 1.0


class QualityRound(BaseModel):
    round: int  # 0 = after understanding, n = after n-th correction
    mapping_confidence: float
    entities: int
    low_confidence: int
    corrected: int = 0


class MappingQuality(BaseModel):
    threshold: float
    mapping_confidence: float
    passed: bool
    corrections_used: int
    max_corrections: int
    rounds: list[QualityRound] = []


class AgentUsage(BaseModel):
    """Model usage of one agent during a run."""

    agent: str
    models: list[str] = []
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0  # includes thinking tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None  # None if a model's price is unknown
    api_seconds: float = 0.0  # summed request time
    wall_seconds: float = 0.0  # elapsed time with parallel requests counted once


class Lesson(BaseModel):
    """A mistake made on this map before, kept so that later versions do not repeat it."""

    kind: Literal["misread", "false_change"]
    page: int = 0
    at: Point
    wrong: str  # what was read, or the change that was claimed
    right: str  # what it really was
    why: str = ""
    learned_from: str = ""  # source file of the version where it was found


class Snapshot(BaseModel):
    schema_version: int = 1
    map_id: str
    version: int = 0
    source_file: str
    source_sha256: str
    created_at: datetime = Field(default_factory=utcnow)
    title: str | None = None
    map_date: str | None = Field(None, description="Date printed on the map, if any")
    pages: int = 1
    panels: list[Panel] = []
    legend: list[LegendEntry] = []
    entities: list[Entity] = []
    relations: list[Relation] = []
    notes: list[str] = []
    quality: MappingQuality | None = None
    llm: dict[str, str] = {}  # agent -> "provider/model@effort"
    usage: list[AgentUsage] = []  # model usage while building this snapshot
    lessons: list[Lesson] = []  # mistakes made on this map so far (this version and earlier ones)


class ChangeType(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    REPLACED = "replaced"  # same spot & kind of place, different occupant (cafe X -> cafe Y)
    REPURPOSED = "repurposed"  # same spot, different kind of place (shop -> cafe)
    MOVED = "moved"
    RENAMED = "renamed"  # same occupant, name spelled differently / rebranded (adjudicated)
    RELATION_ADDED = "relation_added"
    RELATION_REMOVED = "relation_removed"
    RELATION_CHANGED = "relation_changed"  # e.g. airline moved to another desk


class Change(BaseModel):
    type: ChangeType
    summary: str
    old: Entity | None = None
    new: Entity | None = None
    old_relation: Relation | None = None
    new_relation: Relation | None = None
    confidence: float = 1.0
    needs_review: bool = False
    rejected: bool = Field(False, description="Change-confidence agent judged it extraction noise, not a real change")
    unconfirmed: bool = Field(False, description="Looks like a reading difference; kept only if the visual check confirms it")
    confidence_breakdown: dict[str, float] = {}
    verification_note: str = ""
    reason: str = ""

    @property
    def page(self) -> int:
        if self.new or self.old:
            return (self.new or self.old).page
        return (self.new_relation or self.old_relation).page


class Alignment(BaseModel):
    """Similarity transform mapping new-version coordinates into the old version's frame."""

    page: int
    method: Literal["identity", "similarity"]
    anchors: int
    scale: float = 1.0
    rotation_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    residual: float = 0.0


class ChangeSet(BaseModel):
    map_id: str
    from_version: int
    to_version: int
    created_at: datetime = Field(default_factory=utcnow)
    alignment: list[Alignment] = []
    changes: list[Change] = []
    unchanged: int = 0
    presentation_notes: list[str] = Field(
        default=[], description="Styling differences that were deliberately ignored"
    )
    usage: list[AgentUsage] = []  # model usage of change scoring

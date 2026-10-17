"""Request and response models for the mushroom classifier API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EXAMPLE_MUSHROOM = {
    "cap-shape": "x", "cap-surface": "s", "cap-color": "n", "bruises": "t", "odor": "p",
    "gill-attachment": "f", "gill-spacing": "c", "gill-size": "n", "gill-color": "k",
    "stalk-shape": "e", "stalk-root": "e", "stalk-surface-above-ring": "s",
    "stalk-surface-below-ring": "s", "stalk-color-above-ring": "w", "stalk-color-below-ring": "w",
    "veil-type": "p", "veil-color": "w", "ring-number": "o", "ring-type": "p",
    "spore-print-color": "k", "population": "s", "habitat": "u",
}


class Mushroom(BaseModel):
    """One mushroom using the categorical columns from mushrooms.csv."""

    model_config = {"extra": "forbid", "populate_by_name": True,
                    "json_schema_extra": {"examples": [EXAMPLE_MUSHROOM]}}

    cap_shape: str = Field(alias="cap-shape", min_length=1)
    cap_surface: str = Field(alias="cap-surface", min_length=1)
    cap_color: str = Field(alias="cap-color", min_length=1)
    bruises: str = Field(min_length=1)
    odor: str = Field(min_length=1)
    gill_attachment: str = Field(alias="gill-attachment", min_length=1)
    gill_spacing: str = Field(alias="gill-spacing", min_length=1)
    gill_size: str = Field(alias="gill-size", min_length=1)
    gill_color: str = Field(alias="gill-color", min_length=1)
    stalk_shape: str = Field(alias="stalk-shape", min_length=1)
    stalk_root: str = Field(alias="stalk-root", min_length=1)
    stalk_surface_above_ring: str = Field(alias="stalk-surface-above-ring", min_length=1)
    stalk_surface_below_ring: str = Field(alias="stalk-surface-below-ring", min_length=1)
    stalk_color_above_ring: str = Field(alias="stalk-color-above-ring", min_length=1)
    stalk_color_below_ring: str = Field(alias="stalk-color-below-ring", min_length=1)
    veil_type: str = Field(alias="veil-type", min_length=1)
    veil_color: str = Field(alias="veil-color", min_length=1)
    ring_number: str = Field(alias="ring-number", min_length=1)
    ring_type: str = Field(alias="ring-type", min_length=1)
    spore_print_color: str = Field(alias="spore-print-color", min_length=1)
    population: str = Field(min_length=1)
    habitat: str = Field(min_length=1)


class MushroomPrediction(BaseModel):
    label: Literal["e", "p"]
    class_name: Literal["edible", "poisonous"]
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    algorithm: str | None = None
    trained_at: str | None = None
    metrics: dict[str, float] = {}

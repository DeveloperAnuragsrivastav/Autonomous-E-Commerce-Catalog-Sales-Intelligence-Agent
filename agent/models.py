"""Typed records passed between pipeline steps."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Dimensions(BaseModel):
    """Assembled product size in inches, using the vendor's L x W x H convention (L = depth)."""

    length: float
    width: float
    height: float

    def label(self) -> str:
        return f'{_num(self.length)}"L x {_num(self.width)}"W x {_num(self.height)}"H'


class Carton(BaseModel):
    weight_lb: float
    length: float
    width: float
    height: float


class Product(BaseModel):
    """A vendor SKU after cleaning: facts only, nothing generated."""

    sku: str
    masked_sku: str
    vendor: str
    brand: str
    category: str
    product_type: str | None = None
    collection: str | None = None
    style: str | None = None
    colors: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    assembly_required: bool | None = None
    assembly_note: str | None = None
    weight_capacity: str | None = None
    product_weight_lb: float | None = None
    dimensions: Dimensions | None = None
    dimension_details: list[str] = Field(default_factory=list)
    included: str | None = None
    features: list[str] = Field(default_factory=list)
    image_url: str | None = None
    cartons: list[Carton] = Field(default_factory=list)

    def fact_sheet(self) -> dict:
        """The only information content generation is allowed to use."""
        return self.model_dump(exclude={"masked_sku", "image_url", "cartons", "vendor"})


def _num(x: float) -> str:
    return f"{x:g}"

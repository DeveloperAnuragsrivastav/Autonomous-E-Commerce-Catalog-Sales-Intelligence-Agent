"""Unit tests for catalog agent components (grounding, models, bullet parsing)."""
import pytest
from agent.grounding import numbers_in, fact_numbers, ungrounded
from agent.models import Product, Dimensions, Carton
from agent.vendor import DIMENSION_RE, CROSS_SELL_RE, CAPACITY_RE


def test_numbers_in():
    text = "Revenue grew to $1.4M with 25 SKUs sold and 3 channels."
    nums = numbers_in(text)
    values = [val for _, val in nums]
    assert 1.4 in values
    assert 25.0 in values
    assert 3.0 in values


def test_fact_grounding():
    facts = {
        "revenue": 1400000.0,
        "units": 2500,
        "growth": 14.2,
    }
    # Valid grounded text
    text_ok = "Total net revenue reached $1.4M with 2500 units sold."
    assert ungrounded(text_ok, facts) == []

    # Text containing hallucinated numbers
    text_bad = "Total profit was $999K with 888 returns."
    bad_nums = ungrounded(text_bad, facts)
    assert len(bad_nums) > 0


def test_regex_bullet_routing():
    dim_str = "41L x 37W x 41H"
    m_dim = DIMENSION_RE.search(dim_str)
    assert m_dim is not None
    assert m_dim.group(1) == "41"
    assert m_dim.group(2) == "37"
    assert m_dim.group(3) == "41"

    collection_str = "Check out our Jasi collection for matching pieces."
    m_coll = CROSS_SELL_RE.search(collection_str)
    assert m_coll is not None
    assert m_coll.group(1).lower() == "jasi"

    cap_str = "Supports 300lbs of distributed weight"
    m_cap = CAPACITY_RE.search(cap_str)
    assert m_cap is not None


def test_product_schema():
    p = Product(
        sku="BM349936",
        masked_sku="MSKU-349936",
        vendor="WareHouse",
        brand="Benjara",
        category="Furniture",
        product_type="Dresser Chest",
        collection="Jasi",
        colors=["Oak Brown"],
        materials=["Solid Wood"],
        style="Modern",
        assembly_required=False,
        features=["Sturdy build", "Spacious drawers"],
    )
    assert p.sku == "BM349936"
    assert p.collection == "Jasi"
    assert p.brand == "Benjara"
    assert p.assembly_required is False
    assert "collection" in p.fact_sheet()

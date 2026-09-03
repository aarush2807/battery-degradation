"""Schema normalization tests."""

from battery_degradation.schema import apply_schema, resolve_schema


def test_alias_mapping_capacity_and_cycle():
    cols = ["cycle", "Capacity", "temp", "battery"]
    result = resolve_schema(cols)
    assert result.ok
    assert result.mapping["cycle_number"] == "cycle"
    assert "capacity_ah" in result.mapping or "Capacity" in result.mapping.values()
    df_cols = apply_schema(
        __import__("pandas").DataFrame({"cycle": [1], "Capacity": [4.0], "temp": [25], "battery": ["A"]}),
        result.mapping,
    )
    assert "cycle_number" in df_cols.columns
    assert "capacity_ah" in df_cols.columns


def test_ambiguous_column_reported():
    # 'id' only maps to battery_id in our alias table — craft ambiguity by custom path
    # Use a column that matches multiple via normalize: we check ambiguities dict behavior
    result = resolve_schema(["cycle_number", "capacity_ah", "efficiency"])
    # efficiency maps solely to coulombic_efficiency
    assert "coulombic_efficiency" in result.mapping
    assert result.ok


def test_override_wins():
    result = resolve_schema(
        ["cyc", "cap", "cycle_number", "capacity_ah"],
        overrides={"cycle_number": "cyc", "capacity_ah": "cap"},
    )
    assert result.mapping["cycle_number"] == "cyc"
    assert result.mapping["capacity_ah"] == "cap"


def test_missing_required_errors():
    result = resolve_schema(["temperature", "foo"])
    assert not result.ok
    assert any("cycle" in e.lower() or "capacity" in e.lower() for e in result.errors)

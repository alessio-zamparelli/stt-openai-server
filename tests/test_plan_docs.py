"""PLAN-performance.md item 3H drift-guard tests.

Every latency knob the server reads, documents, or exposes as a preset must be
consistent across app.py, the README configuration table, the README preset
table, and docker-compose.yml — the four sources of truth for WHISPER_* knobs.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_KNOB = re.compile(r"WHISPER_[A-Z0-9_]+")


def _extract(text: str) -> set:
    return set(_KNOB.findall(text))


def _readme_section(marker: str) -> str:
    """Verbatim text of a README section, bounded by the next heading."""
    readme = (ROOT / "README.md").read_text()
    _head, _sep, tail = readme.partition(marker)
    assert _sep, f"README marker {marker!r} not found"
    out = []
    for line in tail.splitlines():
        if line.startswith(("# ", "## ", "### ")):
            break
        out.append(line)
    return "\n".join(out)


def _config_table_knobs() -> set:
    return _extract(_readme_section("## ⚙️ Configuration"))


def _preset_table_knobs() -> set:
    return _extract(_readme_section("### 🧪 Reproduce the measured baseline"))


def test_preset_knobs_in_config_table_and_compose():
    """Every knob in the baseline-preset table has a README config row and a
    token in docker-compose.yml (the 1:1 mapping Plan-3H requires)."""
    compose = _extract((ROOT / "docker-compose.yml").read_text())
    config = _config_table_knobs()
    presets = _preset_table_knobs()
    assert presets, "no WHISPER_* knobs found in the preset table"
    assert presets <= config, f"presets missing from config table: {presets - config}"
    assert presets <= compose, f"presets missing from compose: {presets - compose}"


def test_app_env_knobs_documented_in_readme_config():
    """No env knob the server reads may be undocumented."""
    app_knobs = _extract((ROOT / "app.py").read_text())
    config = _config_table_knobs()
    assert app_knobs <= config, f"undocumented knobs: {app_knobs - config}"


def test_readme_config_knobs_read_by_app():
    """No documented knob may be a phantom the server never reads."""
    config = _config_table_knobs()
    app_knobs = _extract((ROOT / "app.py").read_text())
    assert config <= app_knobs, f"phantom knobs: {config - app_knobs}"


def test_compose_knobs_documented_in_readme():
    """Every knob token in docker-compose.yml (active or commented) is in the
    README configuration table."""
    compose = _extract((ROOT / "docker-compose.yml").read_text())
    config = _config_table_knobs()
    assert compose <= config, f"compose knobs missing from README: {compose - config}"


def test_settings_field_surface_covers_latency_knobs():
    """The Settings model exposes every health-reported latency knob (runtime
    counterpart of the docs drift-guard)."""
    import app as app_module  # noqa: E402  (unit-safe: no model load at import)

    needed = {
        "beam_size", "best_of", "temperature_schedule", "cpu_threads",
        "batch_size", "vad_filter", "hf_offline", "max_concurrent",
    }
    missing = needed - set(app_module.Settings.model_fields)
    assert not missing, f"Settings missing health-reported fields: {missing}"

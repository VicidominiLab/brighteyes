"""Check published examples and source for private filesystem locations."""

import json
import re
from pathlib import Path


def test_private_measurement_metadata_omits_acquisition_names() -> None:
    """Publish a neutral measurement ID without private acquisition details."""
    record = json.loads(Path("tests/data/deconv3/manifest.json").read_text("utf-8"))[
        "raw_input"
    ]
    assert re.fullmatch(r"measurement-\d{3}", record["measurement_id"])
    assert (
        not {"filename", "source_file", "source_path", "acquisition_date"}
        & record.keys()
    )


def test_published_files_omit_private_filesystem_locations() -> None:
    """Keep machine paths and internal server locations out of published files."""
    pattern = re.compile(
        r"(?<!\w)[A-Za-z]:[\\/]|\\\\[\w.-]+\\|/(?:home|Users|mnt|media|Volumes)/"
        r"|Data [A-Za-z]+ server|\.[A-Za-z0-9-]+\.local",
        re.IGNORECASE,
    )
    root = Path(__file__).resolve().parents[2]
    for directory in ("src", "examples", "scripts"):
        for path in sorted((root / directory).rglob("*.py")):
            assert not pattern.search(path.read_text("utf-8")), path.relative_to(root)
    for path in sorted((root / "examples").rglob("*.ipynb")):
        notebook = json.loads(path.read_text("utf-8"))
        pending = [notebook]
        while pending:
            value = pending.pop()
            match value:
                case dict():
                    pending.extend(
                        child
                        for key, child in value.items()
                        if key not in {"image/png", "image/jpeg"}
                    )
                case list():
                    pending.extend(value)
                case str():
                    assert not pattern.search(value), path.relative_to(root)

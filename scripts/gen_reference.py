"""Generate pipeline reference pages and navigation during the docs build."""

from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
PIPELINE = SOURCE / "boview" / "pipeline"

nav = mkdocs_gen_files.Nav()

for path in sorted(PIPELINE.rglob("*.py")):
    # Package initializer files don't need separate reference pages here.
    if path.name in {"__init__.py", "__main__.py"}:
        continue

    # Replaces <stage>/<functions>.py -> <stage>/<functions>.md
    relative = path.relative_to(PIPELINE)
    page_path = relative.with_suffix(".md")
    output_path = Path("pipeline/reference") / page_path

    # Looks like boview.pipeline.<stage>.<functions>
    module = ".".join(path.relative_to(SOURCE).with_suffix("").parts)

    # Mirror the source folders in the reference navigation
    nav[relative.with_suffix("").parts] = page_path.as_posix()

    with mkdocs_gen_files.open(output_path.as_posix(), "w") as page:
        page.write(f"# {module}\n\n::: {module}\n")

with mkdocs_gen_files.open("pipeline/reference/SUMMARY.md", "w") as summary:
    summary.writelines(nav.build_literate_nav())

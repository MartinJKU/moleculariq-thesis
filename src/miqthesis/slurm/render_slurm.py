from __future__ import annotations

import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--account", default="<PROJECT_ACCOUNT>")
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--time", default="24:00:00")
    args = parser.parse_args()
    template_path = Path(args.template)
    environment = Environment(
        loader=FileSystemLoader(template_path.parent),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    rendered = environment.get_template(template_path.name).render(**vars(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()


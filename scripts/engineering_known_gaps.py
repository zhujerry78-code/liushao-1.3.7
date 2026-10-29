"""Explicitly diagnose protected-core gaps without changing model or experiment files.

Exit 2 means at least one gap was observed, not a passing acceptance run. All inputs
are tiny engineering fixtures; their model outputs are not research measurements.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from liushao.model import fit_bundle, infer
from liushao.reviews import current_reviews, save_review


def diagnose():
    features = pd.DataFrame({"a": [0., .1, .2, .3, 5., 6., 7., 8.], "b": [1.] * 8})
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    bundle = fit_bundle(
        features, labels, features, labels, "engineering-fixture-not-an-experiment",
        n_estimators=8,
    )
    observations = []
    for field, incompatible in [
        ("sklearn", "0.0.0-incompatible-engineering-fixture"),
        ("algorithm_version", "unknown-incompatible-engineering-fixture"),
    ]:
        changed = copy.deepcopy(bundle)
        changed["config"][field] = incompatible
        observation = {
            "id": f"unchecked_{field}",
            "entrypoint": "liushao.model.infer",
            "scope": "version metadata validation; protected core unchanged",
            "declared_incompatible_value": incompatible,
        }
        try:
            predictions, _, rejected = infer(features, changed, "fixture", "fixture.csv")
            observation.update({
                "observed": "inference accepted incompatible version metadata",
                "predicted_rows": len(predictions), "rejected_rows": len(rejected),
                "gap_observed": True,
            })
        except Exception as error:
            observation.update({
                "observed": "inference rejected incompatible version metadata",
                "error_type": type(error).__name__, "error": str(error), "gap_observed": False,
            })
        observations.append(observation)

    observations.append({
        "id": "schema_version_contract_absent",
        "entrypoint": "liushao.model.fit_bundle config",
        "observed": "schema_version absent from current bundle config",
        "gap_observed": "schema_version" not in bundle["config"],
        "coordination_required": "Agree the schema/version compatibility contract before editing core.",
    })
    with tempfile.TemporaryDirectory(prefix="liushao-engineering-gap-") as temporary:
        database = Path(temporary) / "reviews.sqlite"
        for index, status in enumerate(["确认异常", "确认正常"]):
            record_id = f"fixture:{index}"
            try:
                save_review(database, record_id, "fixture-version", status, "")
                saved = current_reviews(database, "fixture-version").get(record_id)
                observations.append({
                    "id": f"confirmation_without_reason_{index}",
                    "entrypoint": "liushao.reviews.save_review", "status": status,
                    "observed": "confirmation with empty reasoning persisted",
                    "gap_observed": saved is not None and saved["note"] == "",
                })
            except ValueError as error:
                observations.append({
                    "id": f"confirmation_without_reason_{index}",
                    "entrypoint": "liushao.reviews.save_review", "status": status,
                    "observed": "empty confirmation reasoning rejected", "error": str(error),
                    "gap_observed": False,
                })
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Explicit gap observations; not passing tests or model-performance results.",
        "fixture": "8 synthetic rows, 2 features; separate temporary SQLite database",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "sklearn": sklearn.__version__},
        "diagnostic_mutates_core_files": False,
        "observations": observations,
        "observed_gap_count": sum(item["gap_observed"] for item in observations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("test-results/engineering/known-gaps.json"))
    args = parser.parse_args()
    result = diagnose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "observed_gap_count": result["observed_gap_count"]}, ensure_ascii=False))
    return 2 if result["observed_gap_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

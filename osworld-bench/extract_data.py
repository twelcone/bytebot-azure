#!/usr/bin/env python3
"""Extract OSWorld task data into a consolidated JSON dataset.

Reads test_all.json for the canonical task list, loads each task JSON from
examples/{domain}/{task_id}.json, and outputs a single consolidated dataset file.

Usage:
    python extract_data.py
    python extract_data.py --domain chrome --output chrome_tasks.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OSWorld benchmark tasks into a consolidated JSON dataset."
    )
    parser.add_argument(
        "--examples-dir",
        default="./OSWorld/evaluation_examples/examples",
        help="Path to the OSWorld examples directory (default: ./OSWorld/evaluation_examples/examples)",
    )
    parser.add_argument(
        "--test-all",
        default="./OSWorld/evaluation_examples/test_all.json",
        help="Path to test_all.json (default: ./OSWorld/evaluation_examples/test_all.json)",
    )
    parser.add_argument(
        "--output",
        default="./osworld_dataset.json",
        help="Output JSON file path (default: ./osworld_dataset.json)",
    )
    parser.add_argument(
        "--domain",
        default="all",
        help="Filter to a single domain (e.g. 'chrome'), or 'all' for everything (default: all)",
    )
    return parser.parse_args()


def load_test_all(path: str) -> dict[str, list[str]]:
    """Load test_all.json which maps domain names to lists of task UUIDs."""
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(
        "Loaded test_all.json: %d domains, %d total tasks",
        len(data),
        sum(len(v) for v in data.values()),
    )
    return data


def load_task(examples_dir: str, domain: str, task_id: str) -> dict | None:
    """Load a single task JSON and normalize it with an explicit domain field."""
    path = os.path.join(examples_dir, domain, f"{task_id}.json")
    if not os.path.exists(path):
        logger.warning("Task file not found: %s", path)
        return None

    with open(path, "r") as f:
        task = json.load(f)

    # Normalize: ensure task_id field and add domain
    return {
        "task_id": task.get("id", task_id),
        "domain": domain,
        "instruction": task.get("instruction", ""),
        "snapshot": task.get("snapshot", ""),
        "related_apps": task.get("related_apps", []),
        "source": task.get("source", ""),
        "config": task.get("config", []),
        "evaluator": task.get("evaluator", {}),
        "trajectory": task.get("trajectory", ""),
        "proxy": task.get("proxy", False),
        "fixed_ip": task.get("fixed_ip", False),
        "possibility_of_env_change": task.get("possibility_of_env_change", "low"),
    }


def extract_dataset(
    examples_dir: str,
    test_all: dict[str, list[str]],
    domain_filter: str,
) -> list[dict]:
    """Iterate test_all, load each task, return flat list of task dicts."""
    tasks = []
    domains = (
        {domain_filter: test_all[domain_filter]}
        if domain_filter != "all" and domain_filter in test_all
        else test_all
    )

    if domain_filter != "all" and domain_filter not in test_all:
        logger.error(
            "Domain '%s' not found in test_all.json. Available: %s",
            domain_filter,
            ", ".join(test_all.keys()),
        )
        sys.exit(1)

    for domain, task_ids in sorted(domains.items()):
        loaded = 0
        for task_id in task_ids:
            task = load_task(examples_dir, domain, task_id)
            if task:
                tasks.append(task)
                loaded += 1
        logger.info("Domain '%s': loaded %d/%d tasks", domain, loaded, len(task_ids))

    return tasks


def main():
    args = parse_args()

    test_all = load_test_all(args.test_all)
    tasks = extract_dataset(args.examples_dir, test_all, args.domain)

    # Build domain counts
    domain_counts: dict[str, int] = {}
    for task in tasks:
        domain_counts[task["domain"]] = domain_counts.get(task["domain"], 0) + 1

    dataset = {
        "metadata": {
            "total_tasks": len(tasks),
            "domains": domain_counts,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        },
        "tasks": tasks,
    }

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(
        "Dataset written to %s (%d tasks across %d domains)",
        args.output,
        len(tasks),
        len(domain_counts),
    )


if __name__ == "__main__":
    main()

"""Local Day 9 API demonstration (no web server needed).

Usage (from repository root):

    python scripts/demo_day9_api.py            # health + Top-10 for one fixed test request
    python scripts/demo_day9_api.py --timing   # also measure start-up and latency

Uses Flask's test client against the real application factory. The
sample request is fixed (the first test-split request in link order), so
the printed recommendations are deterministic; timings naturally vary
between machines and runs.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timing", action="store_true", help="measure start-up and recommendation latency")
    args = parser.parse_args()

    t0 = time.perf_counter()
    app = create_app("production")
    startup = time.perf_counter() - t0
    client = app.test_client()
    service = app.extensions.get("hirewise_service")
    if service is None:
        raise SystemExit(f"recommender unavailable: {client.get('/api/health').get_json()}")

    print("GET /api/health ->", client.get("/api/health").get_json())
    test_links = sorted(service.data.requests.loc[service.data.requests["split"] == "test", "link"])
    link = test_links[0]
    body = client.get("/api/recommendations", query_string={"link": link, "k": 10}).get_json()
    request = body["request"]
    print(f"\nRequest: {request['title']!r} | category {request['category']} | skills {', '.join(request['required_skills'])}")
    print(f"Model: {body['model']['name']} weights {body['model']['weights']} | {body['n_returned']} of {body['n_candidates']} eligible\n")
    for rec in body["recommendations"]:
        w, c = rec["worker"], rec["contributions"]
        print(f"{rec['rank']:>2}. {w['worker_id']} ({w['primary_category']}, {w['country']}) score {rec['score']:.3f} "
              f"[CBF {c['cbf']:.3f} | CF {c['cf']:.3f} | KG {c['kg']:.3f}]")
        print(f"    {rec['explanation']}")

    if args.timing:
        sample = test_links[::100]
        first = []
        for l in sample:
            t = time.perf_counter()
            client.get("/api/recommendations", query_string={"link": l})
            first.append(time.perf_counter() - t)
        repeat = []
        for l in sample:
            t = time.perf_counter()
            client.get("/api/recommendations", query_string={"link": l})
            repeat.append(time.perf_counter() - t)
        print(f"\nstart-up (create_app incl. model loading): {startup:.2f} s (service build {service.startup_seconds:.2f} s)")
        print(f"first call per request   (n={len(first)}): median {statistics.median(first) * 1000:.0f} ms, mean {statistics.mean(first) * 1000:.0f} ms")
        print(f"repeat call per request  (n={len(repeat)}): median {statistics.median(repeat) * 1000:.0f} ms, mean {statistics.mean(repeat) * 1000:.0f} ms")
        print("models/data are built once in create_app and reused across calls; component scores are cached per request")


if __name__ == "__main__":
    main()

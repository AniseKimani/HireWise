"""Create a demo marketplace account, optionally linked to an IMPORTED
research profile (Day 12).

Public registration never links accounts to imported profiles; this
explicit script is the only way, so a presenter can demonstrate e.g. a
worker from the recommender's catalogue accepting a booking.

Usage (from repository root):

    python scripts/create_demo_account.py --role client --email client@demo.local --name "Demo Client"
    python scripts/create_demo_account.py --role worker --email worker@demo.local --name "Demo Worker" --worker-id W001674
    python scripts/create_demo_account.py --show-demo-recommendation   # top workers for the documented demo request
    python scripts/create_demo_account.py --demo-operator --email operator@example.com --name "HireWise Demo"

--demo-operator (Milestone 3) creates the DEMO OPERATOR: a verified worker
account with no linked catalogue profile. With HIREWISE_DEMO_MODE=1 and
HIREWISE_DEMO_OPERATOR_EMAIL set to its address, it accepts or declines
bookings for catalogue workers that have no linked account. Use an address
outside HIREWISE_NOTIFICATION_EMAIL_SKIP_DOMAINS (default demo.local) if the
operator should receive notification emails.

The password is read from HIREWISE_DEMO_PASSWORD if set, otherwise
prompted (not echoed). It is never stored in plain text, printed, or
written to any tracked file. Add --test-db to target POSTGRES_TEST_DB.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.app.marketplace.security import hash_password
from src.app.marketplace.validation import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH, validate_email
from src.db.database import MAIN_DATABASE_ENV, TEST_DATABASE_ENV, make_engine
from src.db.marketplace import DuplicateEmail, PostgresMarketplaceRepository, ProfileAlreadyLinked

# The documented presentation request (docs/day12_marketplace_report.md, "Demo flow").
DEMO_REQUEST = {"category": "Logo Design", "skills": ("Adobe Illustrator", "Logo Design"), "pay_type": "fixed", "budget": 150.0}


def show_demo_recommendation() -> int:
    from src.app.marketplace.runtime import RuntimeRecommender, RuntimeRequest
    from src.app.services import RecommendationService

    service = RecommendationService.build()
    runtime = RuntimeRecommender.build(service)
    tier = runtime.price_tiers.tier(DEMO_REQUEST["category"], "fixed", DEMO_REQUEST["budget"], None, None)
    result = runtime.recommend(RuntimeRequest(0, 0, DEMO_REQUEST["category"], DEMO_REQUEST["skills"], tier, None, None), 3)
    print(f"Demo request: {DEMO_REQUEST} (price tier {tier:.3f}) for a NEW client without history")
    for rec in result["recommendations"]:
        print(f"  #{rec['rank']} {rec['worker']['worker_id']}  score {rec['score']:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["client", "worker"])
    parser.add_argument("--email")
    parser.add_argument("--name", help="display name")
    parser.add_argument("--country")
    parser.add_argument("--client-id", help="link a client account to this imported client_id")
    parser.add_argument("--worker-id", help="link a worker account to this imported worker_id")
    parser.add_argument("--test-db", action="store_true")
    parser.add_argument("--show-demo-recommendation", action="store_true")
    parser.add_argument("--demo-operator", action="store_true",
                        help="create the demo operator: a worker account with no linked catalogue profile (Milestone 3)")
    args = parser.parse_args()
    if args.show_demo_recommendation:
        return show_demo_recommendation()
    if args.demo_operator:
        if args.role not in (None, "worker") or args.client_id or args.worker_id:
            parser.error("--demo-operator creates a worker account and cannot be linked to an imported profile")
        args.role = "worker"
    if not (args.role and args.email and args.name):
        parser.error("--role, --email and --name are required (--demo-operator implies --role worker)")
    if (args.client_id and args.role != "client") or (args.worker_id and args.role != "worker"):
        parser.error("--client-id is for client accounts and --worker-id for worker accounts")
    if args.role == "worker" and not args.worker_id and not args.demo_operator:
        parser.error("worker demo accounts must be linked to an imported profile with --worker-id "
                     "(or use --demo-operator for the demo operator account)")

    errors: dict = {}
    email = validate_email(args.email, errors)
    if errors:
        print(errors["email"])
        return 1
    password = os.environ.get("HIREWISE_DEMO_PASSWORD") or getpass.getpass("Password for the demo account: ")
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        print(f"Password must be {MIN_PASSWORD_LENGTH}-{MAX_PASSWORD_LENGTH} characters.")
        return 1

    repo = PostgresMarketplaceRepository(make_engine(database_env=TEST_DATABASE_ENV if args.test_db else MAIN_DATABASE_ENV))
    if args.client_id and not repo.imported_client_exists(args.client_id):
        print(f"Unknown imported client_id {args.client_id}.")
        return 1
    if args.worker_id and not repo.imported_worker_exists(args.worker_id):
        print(f"Unknown imported worker_id {args.worker_id}.")
        return 1
    try:
        user_id = repo.create_user(email, hash_password(password), args.role, args.name, args.country,
                                   client_id=args.client_id, worker_id=args.worker_id)
    except DuplicateEmail:
        print("An account with this email already exists.")
        return 2
    except ProfileAlreadyLinked:
        print("That imported profile is already linked to another account.")
        return 2
    link = args.client_id or args.worker_id
    print(f"Created {args.role} account #{user_id} for {email}" + (f", linked to imported profile {link}" if link else ""))
    if args.demo_operator:
        print("This is the demo operator account. To use it, set HIREWISE_DEMO_MODE=1 and "
              "HIREWISE_DEMO_OPERATOR_EMAIL to this address, then restart the app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

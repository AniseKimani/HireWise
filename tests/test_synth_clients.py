import unittest
from dataclasses import replace

import pandas as pd

from src.data.config import load_config
from src.data.generator_config import load_generator_config
from src.data.synth_clients import finalize_client_observables, generate_clients

GEN_CFG = load_generator_config("config/generator_v1.yaml")


def make_requests(n_us: int, n_gb: int) -> pd.DataFrame:
    rows = []
    for i in range(n_us):
        rows.append({"link": f"us-{i}", "category": "Writing" if i % 2 == 0 else "Design", "country_clean": "United States"})
    for i in range(n_gb):
        rows.append({"link": f"gb-{i}", "category": "Design", "country_clean": "United Kingdom"})
    return pd.DataFrame(rows)


class TestGenerateClients(unittest.TestCase):
    def test_every_request_assigned_exactly_once(self):
        cfg = replace(load_config(), synthetic_clients=20)
        requests = make_requests(40, 20)
        clients_df, private_df, assignment_df = generate_clients(requests, cfg, GEN_CFG)
        self.assertEqual(len(assignment_df), len(requests))
        self.assertEqual(set(assignment_df["link"]), set(requests["link"]))
        self.assertEqual(assignment_df["link"].duplicated().sum(), 0)

    def test_client_count_matches_config(self):
        cfg = replace(load_config(), synthetic_clients=20)
        requests = make_requests(40, 20)
        clients_df, _, _ = generate_clients(requests, cfg, GEN_CFG)
        self.assertEqual(len(clients_df), 20)

    def test_every_client_country_has_at_least_one_request_available(self):
        cfg = replace(load_config(), synthetic_clients=20)
        requests = make_requests(40, 20)
        clients_df, _, assignment_df = generate_clients(requests, cfg, GEN_CFG)
        assigned_countries = requests.set_index("link").loc[assignment_df["link"], "country_clean"].values
        client_countries = clients_df.set_index("client_id").loc[assignment_df["client_id"], "country"].values
        self.assertTrue((assigned_countries == client_countries).all())

    def test_deterministic(self):
        cfg = replace(load_config(), synthetic_clients=20)
        requests = make_requests(40, 20)
        c1, p1, a1 = generate_clients(requests, cfg, GEN_CFG)
        c2, p2, a2 = generate_clients(requests, cfg, GEN_CFG)
        self.assertTrue(c1.equals(c2))
        self.assertTrue(a1.equals(a2))

    def test_private_taste_not_in_observable_frame(self):
        cfg = replace(load_config(), synthetic_clients=20)
        requests = make_requests(40, 20)
        clients_df, private_df, _ = generate_clients(requests, cfg, GEN_CFG)
        taste_cols = [c for c in private_df.columns if c.startswith("taste_")]
        self.assertEqual(len(taste_cols), GEN_CFG.client_taste_dim)
        self.assertFalse(any(c.startswith("taste_") for c in clients_df.columns))

    def test_finalize_client_observables_adds_pay_and_price_tier(self):
        cfg = replace(load_config(), synthetic_clients=20)
        requests = make_requests(40, 20)
        requests_full = requests.assign(pay_type="fixed", price_tier=0.5)
        clients_df, _, assignment_df = generate_clients(requests, cfg, GEN_CFG)
        finalized = finalize_client_observables(clients_df, assignment_df, requests_full)
        self.assertIn("preferred_pay_type", finalized.columns)
        self.assertIn("median_price_tier", finalized.columns)


if __name__ == "__main__":
    unittest.main()

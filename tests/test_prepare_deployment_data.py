"""scripts/prepare_deployment_data.py: the runtime-data package for deployment.

Every packaging test works on a throw-away fake repository in a temporary directory (small dummy files, matching
pipeline manifests, a dummy KG fingerprint, and decoy private/raw files that must never be copied). The real data
is only READ, by one opt-out check that the real sources pass verification; nothing real is copied or changed.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prepare_deployment_data", REPO_ROOT / "scripts" / "prepare_deployment_data.py")
pdd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdd)

EXPECTED_FILES = {
    "data/processed/synthetic/workers.csv", "data/processed/synthetic/requests.csv",
    "data/processed/synthetic/interactions.csv", "data/processed/sampled_requests.csv",
    "data/processed/prepared_jobs.csv", "data/processed/kg/nodes_category.csv", "data/processed/kg/nodes_skill.csv",
    "data/processed/kg/nodes_worker.csv", "data/processed/kg/rel_has_skill.csv", "data/processed/kg/rel_hired.csv",
    "data/processed/kg/rel_interested_in.csv", "data/processed/kg/rel_related_to.csv",
    "data/processed/kg/rel_reviewed.csv", "data/processed/kg/rel_specializes_in.csv",
}
DECOYS = ("data/processed/synthetic_private/truth.csv", "data/processed/synthetic_private/workers_private.csv",
          "data/raw/upwork-jobs.csv", "data/processed/synthetic/clients.csv", "data/processed/kg/nodes_client.csv",
          "data/processed/eligible_jobs.csv", ".env")
FINGERPRINT = "f" * 64


def fake_fingerprint(kg_dir):
    """Stands in for the project's KG fingerprint: derived from the KG files' bytes, so tampering changes it."""
    digest = hashlib.sha256()
    for rel in sorted(f for f in EXPECTED_FILES if f.startswith("data/processed/kg/")):  # only what the KG view reads
        name = rel.rsplit("/", 1)[1]
        digest.update(name.encode() + (Path(kg_dir) / name).read_bytes())
    return digest.hexdigest()


class FakeRepo:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "repo"
        manifests = {m: [] for m in pdd.SOURCE_MANIFESTS}
        for rel in sorted(EXPECTED_FILES):
            data = f"id,value\n{rel},1\n".encode()
            self.write(rel, data)
            owner = ("data/processed/kg/kg_prepare_manifest.json" if "/kg/" in rel else
                     "data/processed/synthetic/manifest.json" if "/synthetic/" in rel else "data/processed/manifest.json")
            manifests[owner].append({"path": rel, "rows": 1, "sha256": hashlib.sha256(data).hexdigest()})
        for rel in DECOYS:
            self.write(rel, b"PRIVATE-OR-RAW-DO-NOT-COPY\n")
        manifests["data/processed/synthetic/manifest.json"].append(
            {"path": "data/processed/synthetic_private/truth.csv", "sha256": "0" * 64})
        for rel, files in manifests.items():
            self.write(rel, json.dumps({"files": files}).encode())
        self.write(pdd.GRAPH_SNAPSHOT, json.dumps({"fingerprint": fake_fingerprint(self.root / pdd.KG_DIR)}).encode())
        self.dest = self.root / pdd.PACKAGE_DIR

    def write(self, rel, data: bytes):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def build(self, **kwargs):
        return pdd.build_package(self.root, self.dest, fingerprint_fn=fake_fingerprint, **kwargs)

    def verify(self):
        return pdd.verify_package(self.root, self.dest, fingerprint_fn=fake_fingerprint)

    def package_files(self):
        return {p.relative_to(self.dest).as_posix() for p in self.dest.rglob("*") if p.is_file()}

    def cleanup(self):
        self._tmp.cleanup()


def symlinks_supported() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.symlink(Path(tmp) / "missing", Path(tmp) / "link")
            return True
        except (OSError, NotImplementedError):
            return False


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.repo = FakeRepo()
        self.addCleanup(self.repo.cleanup)

    def test_allow_list_is_exactly_the_14_runtime_files(self):
        self.assertEqual(len(pdd.REQUIRED_FILES), 14)
        self.assertEqual(set(pdd.REQUIRED_FILES), EXPECTED_FILES)
        pdd.check_allow_list()
        for bad in (("data/processed/../../.env",), ("data/raw/upwork-jobs.csv",), ("data/processed/synthetic_private/truth.csv",),
                    ("/etc/passwd",), ("results/day8/day8_manifest.json",), ("data/processed/a.csv", "data/processed/a.csv")):
            with self.assertRaises(pdd.PackagingError, msg=bad):
                pdd.check_allow_list(bad)

    def test_package_holds_exactly_the_allow_listed_files_byte_for_byte(self):
        info = self.repo.build()
        self.assertEqual(set(info), EXPECTED_FILES)
        self.assertEqual(self.repo.package_files(), EXPECTED_FILES | {pdd.PACKAGE_MANIFEST})
        for rel in EXPECTED_FILES:
            self.assertEqual((self.repo.dest / rel).read_bytes(), (self.repo.root / rel).read_bytes())
        manifest = json.loads((self.repo.dest / pdd.PACKAGE_MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual({e["path"] for e in manifest["files"]}, EXPECTED_FILES)
        self.assertFalse(list(self.repo.dest.parent.glob(".*staging*")))  # no staging directory left behind

    def test_private_raw_and_other_data_are_never_copied(self):
        self.repo.build()
        everything = "\n".join(sorted(self.repo.package_files()))
        for marker in ("synthetic_private", "data/raw", ".env", "clients.csv", "nodes_client", "eligible_jobs"):
            self.assertNotIn(marker, everything)
        for path in self.repo.dest.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"PRIVATE-OR-RAW-DO-NOT-COPY", path.read_bytes())

    def test_missing_source_file_is_refused_and_nothing_is_created(self):
        (self.repo.root / "data/processed/kg/rel_hired.csv").unlink()
        with self.assertRaises(pdd.PackagingError) as ctx:
            self.repo.build()
        self.assertIn("missing: data/processed/kg/rel_hired.csv", str(ctx.exception))
        self.assertFalse(self.repo.dest.exists())

    def test_source_checksum_mismatch_is_refused(self):
        self.repo.write("data/processed/prepared_jobs.csv", b"changed\n")
        with self.assertRaises(pdd.PackagingError) as ctx:
            self.repo.build()
        self.assertIn("checksum mismatch for data/processed/prepared_jobs.csv", str(ctx.exception))
        self.assertFalse(self.repo.dest.exists())

    def test_existing_package_needs_force(self):
        self.repo.build()
        marker = self.repo.dest / pdd.PACKAGE_MANIFEST
        before = marker.read_bytes()
        with self.assertRaises(pdd.PackagingError) as ctx:
            self.repo.build()
        self.assertIn("--force", str(ctx.exception))
        self.assertEqual(marker.read_bytes(), before)  # untouched
        (self.repo.dest / "stray.txt").write_text("left over")
        self.repo.build(force=True)
        self.assertEqual(self.repo.package_files(), EXPECTED_FILES | {pdd.PACKAGE_MANIFEST})

    def test_failed_rebuild_keeps_the_previous_package(self):
        self.repo.build()
        self.repo.write("data/processed/sampled_requests.csv", b"corrupted\n")
        with self.assertRaises(pdd.PackagingError):
            self.repo.build(force=True)
        self.assertEqual(self.repo.verify().keys(), set(EXPECTED_FILES))  # the earlier package is intact

    def test_verify_only_mode_checks_without_writing(self):
        self.repo.build()
        snapshot = {p: p.stat().st_mtime_ns for p in self.repo.root.rglob("*")}
        self.assertEqual(set(self.repo.verify()), EXPECTED_FILES)
        self.assertEqual({p: p.stat().st_mtime_ns for p in self.repo.root.rglob("*")}, snapshot)
        # tampered, extra, missing
        (self.repo.dest / "data/processed/synthetic/workers.csv").write_bytes(b"tampered\n")
        with self.assertRaisesRegex(pdd.PackagingError, "checksum mismatch for packaged"):
            self.repo.verify()
        self.repo.build(force=True)
        (self.repo.dest / "data/processed/extra.csv").write_text("x")
        with self.assertRaisesRegex(pdd.PackagingError, "unexpected: data/processed/extra.csv"):
            self.repo.verify()
        self.repo.build(force=True)
        (self.repo.dest / "data/processed/kg/rel_reviewed.csv").unlink()
        with self.assertRaisesRegex(pdd.PackagingError, "missing: data/processed/kg/rel_reviewed.csv"):
            self.repo.verify()
        shutil.rmtree(self.repo.dest)
        with self.assertRaisesRegex(pdd.PackagingError, "no package"):
            self.repo.verify()

    def test_private_data_inside_a_package_is_detected(self):
        self.repo.build()
        leak = self.repo.dest / "data/processed/synthetic_private/truth.csv"
        leak.parent.mkdir(parents=True)
        leak.write_text("secret")
        with self.assertRaisesRegex(pdd.PackagingError, "private or raw data"):
            self.repo.verify()

    def test_kg_fingerprint_mismatch_is_refused(self):
        self.repo.write(pdd.GRAPH_SNAPSHOT, json.dumps({"fingerprint": FINGERPRINT}).encode())
        with self.assertRaisesRegex(pdd.PackagingError, "KG fingerprint .* does not match the frozen Day 7 snapshot"):
            self.repo.build()
        self.assertFalse(self.repo.dest.exists())
        # a package whose KG files no longer reproduce the snapshot fails verification
        self.repo.write(pdd.GRAPH_SNAPSHOT, json.dumps({"fingerprint": fake_fingerprint(self.repo.root / pdd.KG_DIR)}).encode())
        self.repo.build()
        with mock.patch.object(pdd, "recorded_fingerprint", return_value=FINGERPRINT):
            with self.assertRaisesRegex(pdd.PackagingError, "KG fingerprint"):
                self.repo.verify()

    def test_destination_must_stay_inside_deployment(self):
        for bad in (self.repo.root / "data" / "pkg", self.repo.root.parent / "elsewhere", self.repo.root / "deployment",
                    self.repo.root / "deployment" / ".." / "data"):
            with self.assertRaises(pdd.PackagingError, msg=str(bad)):
                pdd.build_package(self.repo.root, bad, fingerprint_fn=fake_fingerprint)

    @unittest.skipUnless(symlinks_supported(), "creating symlinks is not permitted here")
    def test_symlinks_are_refused(self):
        target = self.repo.root / "data/processed/synthetic_private/truth.csv"
        source = self.repo.root / "data/processed/synthetic/workers.csv"
        source.unlink()
        os.symlink(target, source)
        with self.assertRaisesRegex(pdd.PackagingError, "refusing symlink"):
            self.repo.build()
        source.unlink()
        self.repo.write("data/processed/synthetic/workers.csv", b"id,value\ndata/processed/synthetic/workers.csv,1\n")
        self.repo.build()
        planted = self.repo.dest / "data/processed/link.csv"
        os.symlink(target, planted)
        with self.assertRaisesRegex(pdd.PackagingError, "symlink"):
            self.repo.verify()

    @unittest.skipUnless(os.name == "nt", "directory junctions are a Windows feature")
    def test_junctions_are_refused(self):
        import _winapi

        outside = Path(self.repo._tmp.name) / "outside-kg"
        shutil.copytree(self.repo.root / pdd.KG_DIR, outside)
        shutil.rmtree(self.repo.root / pdd.KG_DIR)
        _winapi.CreateJunction(str(outside), str(self.repo.root / pdd.KG_DIR))
        with self.assertRaisesRegex(pdd.PackagingError, "refusing symlink or junction"):
            self.repo.build()
        self.assertFalse(self.repo.dest.exists())
        os.rmdir(self.repo.root / pdd.KG_DIR)  # removes the junction only
        shutil.copytree(outside, self.repo.root / pdd.KG_DIR)
        self.repo.build()
        planted = self.repo.dest / "data" / "processed" / "linked"
        _winapi.CreateJunction(str(outside), str(planted))
        with self.assertRaisesRegex(pdd.PackagingError, "symlink or junction"):
            self.repo.verify()
        os.rmdir(planted)

    def test_path_escaping_the_repository_is_refused(self):
        with self.assertRaisesRegex(pdd.PackagingError, "escapes|plain repository-relative"):
            pdd.safe_path(self.repo.root, "../outside.csv")
        with mock.patch.object(pdd, "REQUIRED_FILES", pdd.REQUIRED_FILES[:-1] + ("data/processed/../../outside.csv",)):
            with self.assertRaises(pdd.PackagingError):
                self.repo.build()
        self.assertFalse(self.repo.dest.exists())

    def test_command_line_exit_codes_and_report(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(pdd.main(["--verify"], self.repo.root, self.repo.dest, fake_fingerprint), 1)
            self.assertEqual(pdd.main([], self.repo.root, self.repo.dest, fake_fingerprint), 0)
            self.assertEqual(pdd.main([], self.repo.root, self.repo.dest, fake_fingerprint), 1)  # exists, no --force
            self.assertEqual(pdd.main(["--verify"], self.repo.root, self.repo.dest, fake_fingerprint), 0)
            self.assertEqual(pdd.main(["--force"], self.repo.root, self.repo.dest, fake_fingerprint), 0)
        self.assertIn("14 files", out.getvalue())
        for rel in EXPECTED_FILES:
            self.assertIn(rel, out.getvalue())
        self.assertIn("FAILED: no package", err.getvalue())
        self.assertIn("--force", err.getvalue())


@unittest.skipUnless(all((REPO_ROOT / f).is_file() for f in pdd.REQUIRED_FILES + pdd.SOURCE_MANIFESTS),
                     "runtime data not present")
class RealSourcesTests(unittest.TestCase):
    """Read-only: the real sources match their pipeline checksums and the frozen KG fingerprint. Nothing is copied."""

    def test_real_sources_verify(self):
        def snapshot():  # source verification must not create, change or remove anything under deployment/
            root = REPO_ROOT / "deployment"
            return {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*")} if root.exists() else None

        before = snapshot()
        info = pdd.verify_sources(REPO_ROOT)
        self.assertEqual(set(info), EXPECTED_FILES)
        self.assertEqual(snapshot(), before)


if __name__ == "__main__":
    unittest.main()

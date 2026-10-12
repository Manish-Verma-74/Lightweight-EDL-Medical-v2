"""Two persistent checkpoint roles per run, with safe Colab-local -> Drive persistence.

The resumable checkpoint embeds the best weights as well as the current training state. This makes the
latest checkpoint authoritative if a Drive write is interrupted between the two named checkpoint files.
"""
import os
import shutil
import time

import torch

from edl.utils.hashing import sha256_file
from edl.utils.io import read_json, write_json_atomic

BEST = "best_checkpoint.pth"
LATEST = "latest_resumable.pth"
INDEX = "persist_index.json"


class CheckpointError(RuntimeError):
    pass


class PersistenceError(RuntimeError):
    pass


def atomic_torch_save(obj, path: str) -> None:
    """Save through a temp file, check that it can be loaded, then replace atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    try:
        torch.load(tmp, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp): os.remove(tmp)
        raise CheckpointError(f"checkpoint failed validation after writing: {e}") from e
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def check_identity(saved: dict, expected: dict, allow_commit_change: bool = False) -> None:
    """Reject run/config/split/source mismatches. A config-only commit may resume if source is identical."""
    keys = ["run_id", "protocol_version", "config_hash", "split_sha256", "phase", "debug_subset"]
    for k in keys:
        if saved.get(k) != expected.get(k):
            raise CheckpointError(f"identity mismatch on '{k}': checkpoint has {saved.get(k)!r}, "
                                  f"current run has {expected.get(k)!r}")
    old_fp, new_fp = saved.get("source_fingerprint"), expected.get("source_fingerprint")
    if old_fp != new_fp and (old_fp is not None or new_fp is not None):
        raise CheckpointError("identity mismatch on 'source_fingerprint': training-critical source code changed")
    if saved.get("code_commit") != expected.get("code_commit"):
        # A cap-only config commit is allowed because config_hash deliberately excludes only the cap and the
        # matching source fingerprint proves no training-critical Python source changed.
        if old_fp is not None and old_fp == new_fp:
            return
        if allow_commit_change:
            return
        raise CheckpointError(f"identity mismatch on 'code_commit': checkpoint has {saved.get('code_commit')!r}, "
                              f"current run has {expected.get('code_commit')!r}")


def sync_file(local_path: str, drive_dir: str, retries: int = 3, backoff: float = 2.0, fail_hook=None) -> str:
    """Copy a completed local file to Drive through temp + hash verification + atomic rename."""
    os.makedirs(drive_dir, exist_ok=True)
    name = os.path.basename(local_path)
    dst, tmp = os.path.join(drive_dir, name), os.path.join(drive_dir, ".tmp_" + name)
    digest = sha256_file(local_path)
    last = None
    for attempt in range(1, retries + 1):
        try:
            if fail_hook is not None: fail_hook(attempt)
            shutil.copyfile(local_path, tmp)
            if sha256_file(tmp) != digest:
                raise PersistenceError("hash mismatch after copying to Drive")
            os.replace(tmp, dst)
            return digest
        except Exception as e:  # noqa: BLE001
            last = e
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
            if attempt < retries: time.sleep(backoff * attempt)
    raise PersistenceError(f"could not persist {name} after {retries} attempts: {last}")


class RunCheckpointer:
    def __init__(self, local_dir: str, drive_dir: str, identity: dict, sync_every: int = 1,
                 retries: int = 3, backoff: float = 2.0, fail_hook=None):
        self.local_dir, self.drive_dir = local_dir, drive_dir
        self.identity = identity
        self.sync_every = max(1, int(sync_every))
        self.retries, self.backoff, self.fail_hook = retries, backoff, fail_hook
        self._best_dirty = False
        self._best_epoch = 0
        os.makedirs(local_dir, exist_ok=True)
        os.makedirs(drive_dir, exist_ok=True)

    def local(self, name): return os.path.join(self.local_dir, name)
    def drive(self, name): return os.path.join(self.drive_dir, name)

    def _persist(self, name: str, epoch: int) -> None:
        """Persist one role transactionally with a short-lived recovery copy for the index-update window.

        Drive cannot atomically replace a checkpoint and persist_index.json together. Before replacing an
        already-indexed role, preserve the old verified bytes under a transaction-temp name. If the process
        dies after replacing the checkpoint but before updating the index, the next reader can restore the
        old hash-matching copy. The temporary backup is removed as soon as the new file+index pair is valid.
        """
        dst = self.drive(name)
        idx_path = self.drive(INDEX)
        idx = read_json(idx_path) if os.path.exists(idx_path) else {}
        old_entry = idx.get(name)
        backup = self.drive(".txn_backup_" + name)
        backup_tmp = backup + ".tmp"
        if os.path.exists(dst) and old_entry:
            try:
                old_valid = sha256_file(dst) == old_entry.get("sha256")
                old_ck = load_checkpoint(dst) if old_valid else None
                old_valid = old_valid and int(old_ck.get("epoch", -1)) == int(old_entry.get("epoch", -2))
            except Exception:
                old_valid = False
            if old_valid:
                shutil.copyfile(dst, backup_tmp)
                if sha256_file(backup_tmp) != old_entry["sha256"]:
                    try: os.remove(backup_tmp)
                    except OSError: pass
                    raise PersistenceError(f"failed to create verified transaction backup for {name}")
                os.replace(backup_tmp, backup)
        digest = sync_file(self.local(name), self.drive_dir, self.retries, self.backoff, self.fail_hook)
        idx[name] = {"sha256": digest, "epoch": int(epoch), "size": os.path.getsize(self.local(name))}
        write_json_atomic(idx, idx_path)
        # The file and its index entry now agree. The backup is no longer needed.
        for tmp in (backup, backup_tmp):
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass

    def save_best(self, model_state: dict, epoch: int, val_macro_f1: float, extra: dict = None) -> None:
        atomic_torch_save({"kind": "best", "identity": self.identity, "epoch": int(epoch),
                           "val_macro_f1": float(val_macro_f1), "model": model_state, "extra": extra or {}},
                          self.local(BEST))
        self._best_dirty = True
        self._best_epoch = int(epoch)

    def save_latest(self, state: dict, epoch: int, force: bool = False) -> bool:
        """Save the resume state; sync best first, then latest, when the cadence is due."""
        state = dict(state, kind="latest", identity=self.identity, epoch=int(epoch))
        atomic_torch_save(state, self.local(LATEST))
        if force or epoch % self.sync_every == 0:
            # If interrupted after BEST but before LATEST, the previous LATEST embeds its best weights and can
            # reconstruct the matching BEST. Never advertise a latest state before its best checkpoint exists.
            if self._best_dirty and os.path.exists(self.local(BEST)):
                self._persist(BEST, self._best_epoch)
                self._best_dirty = False
            self._persist(LATEST, epoch)
            return True
        return False

    def note_best_epoch(self, epoch: int) -> None:
        self._best_epoch = int(epoch)

    def finalize(self, epoch: int) -> None:
        if os.path.exists(self.local(BEST)) and self._best_dirty:
            self._persist(BEST, getattr(self, "_best_epoch", epoch))
            self._best_dirty = False
        if os.path.exists(self.local(LATEST)):
            self._persist(LATEST, epoch)
        if os.path.exists(self.local(BEST)):
            # Refresh the index/file pair even if the best was synced earlier; its exact identity is preserved.
            entry = self._valid_drive_copy(BEST)
            if entry is None or entry.get("epoch") != getattr(self, "_best_epoch", epoch):
                self._persist(BEST, getattr(self, "_best_epoch", epoch))

    def _valid_drive_copy(self, name: str):
        path, idx_path = self.drive(name), self.drive(INDEX)
        if not os.path.exists(idx_path): return None
        try:
            entry = read_json(idx_path).get(name)
            if not entry: return None
            if not os.path.exists(path) or sha256_file(path) != entry["sha256"]:
                # Recover the prior indexed version if a disconnect happened between the checkpoint
                # replace and the atomic persist_index.json update.
                backup = self.drive(".txn_backup_" + name)
                if os.path.exists(backup) and sha256_file(backup) == entry["sha256"]:
                    backup_ck = load_checkpoint(backup)
                    if int(backup_ck.get("epoch", -1)) == int(entry.get("epoch", -2)):
                        os.replace(backup, path)
                if not os.path.exists(path) or sha256_file(path) != entry["sha256"]: return None
            ck = load_checkpoint(path)
            expected_kind = "best" if name == BEST else "latest"
            if ck.get("kind") != expected_kind: return None
            if int(ck.get("epoch", -1)) != int(entry.get("epoch", -2)): return None
            # A valid current entry means any older transaction backup is obsolete.
            backup = self.drive(".txn_backup_" + name)
            if os.path.exists(backup):
                try: os.remove(backup)
                except OSError: pass
            backup_tmp = backup + ".tmp"
            if os.path.exists(backup_tmp):
                try: os.remove(backup_tmp)
                except OSError: pass
            return entry
        except Exception:  # noqa: BLE001
            return None

    def verify_drive_checkpoint(self, name: str) -> dict:
        """Verify a persistent checkpoint against persist_index.json and this run's identity."""
        if name not in (BEST, LATEST):
            raise CheckpointError(f"unsupported persistent checkpoint role: {name}")
        entry = self._valid_drive_copy(name)
        if entry is None:
            raise CheckpointError(f"persistent {name} is missing or fails persist_index.json/hash validation")
        ck = load_checkpoint(self.drive(name))
        check_identity(ck.get("identity", {}), self.identity, allow_commit_change=True)
        if int(ck.get("epoch", -1)) != int(entry.get("epoch", -2)):
            raise CheckpointError(f"persistent {name} epoch does not match persist_index.json")
        return ck

    def _restore_best_from_latest(self, ck: dict) -> None:
        best_epoch = int(ck.get("best_epoch", 0))
        if best_epoch < 1:
            raise CheckpointError("latest checkpoint has no valid best_epoch")
        if ck.get("best_model_state") is not None:
            best_blob = {"kind": "best", "identity": ck["identity"], "epoch": best_epoch,
                         "val_macro_f1": float(ck["best_f1"]), "model": ck["best_model_state"],
                         "extra": ck.get("best_extra", {})}
            atomic_torch_save(best_blob, self.local(BEST))
        else:
            # Compatibility path for unit fixtures/early development checkpoints: only trust an exact epoch match.
            candidates = []
            if os.path.exists(self.local(BEST)):
                try: candidates.append((self.local(BEST), load_checkpoint(self.local(BEST))))
                except Exception: pass
            if self._valid_drive_copy(BEST) is not None:
                candidates.append((self.drive(BEST), load_checkpoint(self.drive(BEST))))
            match = next(((p, b) for p, b in candidates if int(b.get("epoch", -1)) == best_epoch), None)
            if match is None:
                raise CheckpointError("latest checkpoint cannot reconstruct its best checkpoint (epoch mismatch)")
            if match[0] != self.local(BEST): shutil.copyfile(match[0], self.local(BEST))
        self._best_epoch = best_epoch
        drive_entry = self._valid_drive_copy(BEST)
        local_digest = sha256_file(self.local(BEST)) if os.path.exists(self.local(BEST)) else None
        self._best_dirty = not (drive_entry is not None and drive_entry.get("epoch") == best_epoch
                                and drive_entry.get("sha256") == local_digest)

    def load_resume(self, allow_commit_change: bool = False):
        """Load the newest valid latest checkpoint and reconstruct its matching best checkpoint locally."""
        candidates = []
        if os.path.exists(self.local(LATEST)):
            try:
                ck = load_checkpoint(self.local(LATEST))
                check_identity(ck["identity"], self.identity, allow_commit_change)
                candidates.append((int(ck["epoch"]), "local", ck))
            except CheckpointError: raise
            except Exception: pass
        entry = self._valid_drive_copy(LATEST)
        if entry is not None:
            ck = load_checkpoint(self.drive(LATEST))
            check_identity(ck["identity"], self.identity, allow_commit_change)
            candidates.append((int(ck["epoch"]), "drive", ck))
        if not candidates: return None
        _, source, ck = max(candidates, key=lambda c: c[0])
        if source == "drive": shutil.copyfile(self.drive(LATEST), self.local(LATEST))
        self._restore_best_from_latest(ck)
        ck["_source"] = source
        return ck

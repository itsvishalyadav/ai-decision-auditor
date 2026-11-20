"""Tamper-evident audit trail (accountability).

Every decision the AI makes is written to an append-only log. Each record
contains the SHA-256 hash of the previous record, forming a hash chain (the
same idea as a blockchain, without the network). If anybody edits or deletes a
past record, every hash after it stops matching and ``verify`` reports exactly
where the chain was broken.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def _hash(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class AuditLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def records(self) -> list[dict]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def append(self, model_id: str, inputs: dict, decision: int, probability: float,
               explanation: str, reviewer_flag: bool = False) -> dict:
        recs = self.records()
        prev = recs[-1]["hash"] if recs else GENESIS
        record = {
            "index": len(recs),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_id": model_id,
            "inputs": {k: (v.item() if hasattr(v, "item") else v) for k, v in inputs.items()},
            "decision": "APPROVED" if decision == 1 else "REJECTED",
            "probability": round(float(probability), 4),
            "explanation": explanation,
            "needs_human_review": bool(reviewer_flag),
            "prev_hash": prev,
        }
        record["hash"] = _hash(record)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record

    def verify(self) -> dict:
        prev = GENESIS
        for i, r in enumerate(self.records()):
            if r.get("prev_hash") != prev:
                return {"valid": False, "broken_at": i, "reason": "link to previous record is broken"}
            if _hash(r) != r.get("hash"):
                return {"valid": False, "broken_at": i, "reason": "record content was modified"}
            prev = r["hash"]
        return {"valid": True, "broken_at": None, "reason": "all records intact"}

    def tamper(self, index: int = 0) -> None:
        """DEMO ONLY: silently change a past decision to show detection works."""
        recs = self.records()
        if not recs:
            return
        index = min(index, len(recs) - 1)
        r = recs[index]
        r["decision"] = "APPROVED" if r["decision"] == "REJECTED" else "REJECTED"
        self.path.write_text("".join(json.dumps(x, default=str) + "\n" for x in recs), encoding="utf-8")

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")

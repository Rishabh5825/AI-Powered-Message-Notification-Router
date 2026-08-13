"""
Self-Evaluation — Compare predictions against sample_messages.csv ground truth.

Checks action accuracy, message_type accuracy, and flags mismatches.

Usage:
    cd code
    python evaluation/main.py
"""

import csv
import os
import sys

# Ensure code/ is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET_DIR, OUTPUT_PATH


def evaluate():
    """Compare output.csv predictions against sample_messages.csv ground truth."""
    sample_path = DATASET_DIR / "sample_messages.csv"
    output_path = OUTPUT_PATH

    if not output_path.exists():
        print("ERROR: output.csv not found. Run main.py first.")
        return

    # Load ground truth samples
    with open(sample_path, encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    # Load predictions keyed by message_id
    with open(output_path, encoding="utf-8") as f:
        preds = {row["message_id"]: row for row in csv.DictReader(f)}

    # Evaluate
    action_correct = 0
    type_correct = 0
    total = 0
    mismatches = []

    for sample in samples:
        msg_id = sample["message_id"]
        if msg_id not in preds:
            print(f"  ⚠ {msg_id} not found in predictions")
            continue

        pred = preds[msg_id]
        total += 1

        action_match = pred["action"] == sample["action"]
        type_match = pred["message_type"] == sample["message_type"]

        if action_match:
            action_correct += 1
        if type_match:
            type_correct += 1

        if not action_match or not type_match:
            mismatches.append({
                "message_id": msg_id,
                "pred_action": pred["action"],
                "expected_action": sample["action"],
                "action_ok": "✓" if action_match else "✗",
                "pred_type": pred["message_type"],
                "expected_type": sample["message_type"],
                "type_ok": "✓" if type_match else "✗",
                "pred_reason": pred.get("reason", "")[:80],
            })

    # Print results
    print("=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"\n  Action Accuracy:       {action_correct}/{total} "
          f"({action_correct/max(total,1)*100:.1f}%)")
    print(f"  Message Type Accuracy: {type_correct}/{total} "
          f"({type_correct/max(total,1)*100:.1f}%)")

    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        print("-" * 60)
        for m in mismatches:
            print(f"  {m['message_id']}:")
            print(f"    Action: {m['action_ok']} pred={m['pred_action']}, "
                  f"expected={m['expected_action']}")
            print(f"    Type:   {m['type_ok']} pred={m['pred_type']}, "
                  f"expected={m['expected_type']}")
            print(f"    Reason: {m['pred_reason']}")
            print()

    # Target thresholds
    action_pct = action_correct / max(total, 1) * 100
    type_pct = type_correct / max(total, 1) * 100

    print("=" * 60)
    if action_pct >= 90 and type_pct >= 80:
        print("  [PASS] — Ready to submit!")
    elif action_pct >= 80:
        print("  [WARN] — Close — action accuracy OK, improve message_type.")
    else:
        print("  [FAIL] — Needs work — review mismatches above.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()

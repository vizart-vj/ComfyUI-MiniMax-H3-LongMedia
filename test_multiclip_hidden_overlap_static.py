from pathlib import Path
text=(Path(__file__).resolve().parents[1]/"nodes.py").read_text(encoding="utf-8")
assert "local_lengths[segment_index] = int(local_lengths[segment_index]) + int(hidden_overlap)" not in text
assert "multiclip_native_overlap = 5" in text
print("MULTICLIP_HIDDEN_OVERLAP_RETIRED_STATIC: PASS")

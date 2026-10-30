"""Derive the Strata configs used in the benchmark from the one Strata's setup writes.

  python strata_variants.py strata-unsloth-ud-q4_k_xl.json OUT_DIR

setup-default.json      as setup wrote it (KV streaming on: --kv-resident 32768)
one-user.json           KV cache kept on the card (--kv-resident removed, as Strata's docs/TROUBLESHOOTING.md describes)
one-user-no-draft.json  the same without the MTP draft (--mtp removed; the native pack still needs --spec)
"""
import json
import os
import sys


def without(args, flags):
    out, skip = [], 0
    for x in args:
        if skip:
            skip -= 1
            continue
        if x in flags:
            skip = 1
            continue
        out.append(x)
    return out


src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
base = json.load(open(src))
json.dump(base, open(os.path.join(out, "setup-default.json"), "w"), indent=1)
one = dict(base, args=without(base["args"], {"--kv-resident"}))
json.dump(one, open(os.path.join(out, "one-user.json"), "w"), indent=1)
json.dump(dict(one, args=without(one["args"], {"--mtp"}), mtp=False),
          open(os.path.join(out, "one-user-no-draft.json"), "w"), indent=1)
print("one-user args:", " ".join(one["args"]))

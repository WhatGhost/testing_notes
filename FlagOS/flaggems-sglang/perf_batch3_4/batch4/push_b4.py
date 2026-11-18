"""Recreate the local batch-4 commit on the fork on top of upstream 5c2ed0f via the
GitHub git-data API (git push over TLS fails here). New branch only, no force."""

import base64
import json
import subprocess
import sys

FORK = "WhatGhost/FlagGems-sglang"
UPSTREAM = "flagos-ai/FlagGems-sglang"
BASE = "5c2ed0f15b6d644f8e4bcbeae7236a22b6959eaa"
BRANCH = "perf/amd-gfx1100-softmax-pool-dsv3-rotary"
REPO = open("/tmp/fgs-b4a.path").read().strip()
MSG = "perf(amd): speed up gfx1100 fused_softmax_pool, per_group_transpose, ltx2_split_rotary, dsv3_fused_a_gemm, embedding_lora_a and apply_token_bitmask"
AUTHOR = {"name": "whatghost", "email": "yuyang.gao@amd.com"}


def gh(path, method="GET", payload=None, check=True):
    cmd = ["gh", "api", path, "-X", method]
    if payload is not None:
        cmd += ["--input", "-"]
    p = subprocess.run(cmd, input=json.dumps(payload) if payload is not None else None,
                       capture_output=True, text=True)
    if check and p.returncode != 0:
        sys.exit(f"gh api {method} {path} failed: {p.stdout[-500:]} {p.stderr[-500:]}")
    return p.returncode, (json.loads(p.stdout) if p.stdout.strip().startswith(("{", "[")) else p.stdout)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


files = git("diff", "--name-only", "HEAD~1", "HEAD").split()
print("files:", files)

# 1. the branch must not exist yet
rc, _ = gh(f"repos/{FORK}/git/ref/heads/{BRANCH}", check=False)
if rc == 0:
    sys.exit("branch already exists on the fork; stopping")

# 2. upstream base: tree, and the changed files must match the local base blobs
_, base_commit = gh(f"repos/{FORK}/git/commits/{BASE}")
base_tree = base_commit["tree"]["sha"]
for f in files:
    in_base = subprocess.run(["git", "-C", REPO, "cat-file", "-e", f"HEAD~1:{f}"]).returncode == 0
    rc, info = gh(f"repos/{UPSTREAM}/contents/{f}?ref={BASE}", check=False)
    if not in_base:
        if rc == 0:
            sys.exit(f"{f} is new locally but exists upstream")
        print("new file:", f)
        continue
    local_base = git("rev-parse", f"HEAD~1:{f}")
    if rc != 0 or info["sha"] != local_base:
        sys.exit(f"base mismatch for {f}")
print("base blobs match upstream", BASE[:7], "tree", base_tree[:7])

# 3. blobs, checked against the local commit
entries = []
for f in files:
    content = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{f}"], capture_output=True, check=True).stdout
    _, blob = gh(f"repos/{FORK}/git/blobs", "POST",
                 {"content": base64.b64encode(content).decode(), "encoding": "base64"})
    local = git("rev-parse", f"HEAD:{f}")
    if blob["sha"] != local:
        sys.exit(f"blob mismatch for {f}: {blob['sha']} vs {local}")
    entries.append({"path": f, "mode": "100644", "type": "blob", "sha": blob["sha"]})
print("blobs ok")

# 4. tree and commit
_, tree = gh(f"repos/{FORK}/git/trees", "POST", {"base_tree": base_tree, "tree": entries})
date = git("log", "-1", "--format=%aI", "HEAD")
who = dict(AUTHOR, date=date)
_, commit = gh(f"repos/{FORK}/git/commits", "POST",
               {"message": MSG, "tree": tree["sha"], "parents": [BASE], "author": who, "committer": who})
print("commit", commit["sha"], "tree", tree["sha"][:7])

# 5. new branch (no force)
_, ref = gh(f"repos/{FORK}/git/refs", "POST", {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]})
print("ref", ref["ref"], "->", ref["object"]["sha"])

# 6. verify: compare with upstream base touches exactly the changed files
_, cmp = gh(f"repos/{UPSTREAM}/compare/{BASE}...WhatGhost:{BRANCH}")
print("compare:", cmp["status"], "ahead", cmp["ahead_by"], "files", sorted(x["filename"] for x in cmp["files"]),
      "+%d -%d" % (sum(x["additions"] for x in cmp["files"]), sum(x["deletions"] for x in cmp["files"])))

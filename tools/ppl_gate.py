#!/usr/bin/env python3
"""Perplexity and KL divergence through the decode path, against a llama.cpp base.

The base is `llama-perplexity --kl-divergence-base`'s file (wikitext-2, 512-token chunks, second half scored). The
engine runs as `--serve` with the model plus --ppl-base and --ppl-out; per chunk it reads the first half as a prompt
and teacher-forces the rest through verify windows of T tokens (see the engine's ppl_gate.hpp). Flags this tool does
not know are passed to the engine as they are; it sets only what the measurement needs (--spec 5,
no adaptive swaps, the first two visible GPUs when there are two: --gpu 0,1), and an engine flag given here replaces
that value.

  python3 tools/ppl_gate.py --model FILE.basalt --base base.kld --out DIR [--engine-bin BIN] [--chunks 20] [--t 3]
                            [--gpu 0 ...]
  python3 tools/ppl_gate.py --report DIR/rows.tsv          # summary of a finished run
  python3 tools/ppl_gate.py --diff A/rows.tsv B/rows.tsv   # row by row: identical rows, largest NLL/KLD change
"""
from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

ENGINE_BIN = Path(__file__).resolve().parents[1] / "bin" / "basalt"


def read_base(path: str):
    with open(path, "rb") as f:
        header = f.read(20)
        if len(header) != 20 or header[:8] != b"_logits_":
            raise ValueError(f"{path}: not a llama.cpp KL-divergence base")
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", header[8:])
        if n_ctx < 4 or n_vocab <= 0 or n_chunk <= 0:
            raise ValueError(f"{path}: invalid base dimensions")
        token_bytes = 4 * n_ctx * n_chunk
        row_bytes = (2 * ((n_vocab + 1) // 2) + 4) * 2
        expected = 20 + token_bytes + n_chunk * (n_ctx - 1 - n_ctx // 2) * row_bytes
        f.seek(0, 2)
        if f.tell() != expected:
            raise ValueError(f"{path}: base size does not match its dimensions")
        f.seek(20)
        toks = struct.unpack(f"<{n_ctx * n_chunk}i", f.read(token_bytes))
        if any(t < 0 or t >= n_vocab for t in toks):
            raise ValueError(f"{path}: token outside the base vocabulary")
    return n_ctx, n_vocab, n_chunk, [toks[k * n_ctx:(k + 1) * n_ctx] for k in range(n_chunk)]


def load_rows(path: str) -> dict:
    """Read the engine's ten-column table, rejecting corrupt or duplicate evidence."""
    rows = {}
    with open(path) as source:
        for number, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split("\t")
            try:
                if len(fields) != 10:
                    raise ValueError("expected ten columns")
                ints = {i: int(fields[i]) for i in (0, 1, 2, 6, 7, 8)}
                values = [float(fields[i]) for i in (3, 4, 5, 9)]
                if any(v < 0 for v in ints.values()) or ints[6] not in (0, 1):
                    raise ValueError("invalid row indices or top-1 flag")
                if ints[6] != int(ints[7] == ints[8]):
                    raise ValueError("top-1 flag disagrees with token IDs")
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("non-finite metric")
                key = ints[0], ints[1]
                if key in rows:
                    raise ValueError("duplicate chunk/position")
                rows[key] = fields
            except ValueError as e:
                raise ValueError(f"{path}:{number}: {e}") from e
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def summary(rows_path: str, extra: dict | None = None) -> str:
    nll, nllb, kld, same, dlp = [], [], [], 0, []
    chunks = set()
    for f in load_rows(rows_path).values():
        chunks.add(int(f[0]))
        nll.append(float(f[3])); nllb.append(float(f[4])); kld.append(float(f[5]))
        same += int(f[6]); dlp.append(float(f[9]))
    n = len(nll)
    ks = sorted(kld)
    pct = lambda q: ks[min(n - 1, int(round(q * (n - 1))))]
    mn = sum(nll) / n
    se = math.sqrt(max(0.0, sum(v * v for v in nll) / n - mn * mn) / max(1, n - 1))
    out = [f"chunks {len(chunks)}  rows {n}",
           f"PPL {math.exp(mn):.4f} +/- {math.exp(mn) * se:.4f}   base PPL {math.exp(sum(nllb) / n):.4f}   "
           f"ln(PPL/base) {mn - sum(nllb) / n:.5f}",
           f"KLD mean {sum(kld) / n:.5f}  median {pct(0.5):.5f}  p99 {pct(0.99):.4f}  max {ks[-1]:.4f}",
           f"same top-1 {100.0 * same / n:.2f}%   mean |dlogp| of the base top-1 {sum(dlp) / n:.5f}"]
    if extra:
        out.append("  ".join(f"{k} {v}" for k, v in extra.items()))
    return "\n".join(out)


def diff(pa: str, pb: str) -> str:
    a, b = load_rows(pa), load_rows(pb)
    if a.keys() != b.keys():
        raise ValueError(f"row sets differ: {len(a.keys() - b.keys())} only in {pa}, "
                         f"{len(b.keys() - a.keys())} only in {pb}")
    keys = sorted(a)
    for key in keys:
        if int(a[key][2]) != int(b[key][2]):
            raise ValueError(f"target token differs at chunk {key[0]} pos {key[1]}")
    same = sum(a[k][3:] == b[k][3:] for k in keys)
    top = sum(a[k][7] == b[k][7] for k in keys)
    dn = max(abs(float(a[k][3]) - float(b[k][3])) for k in keys)
    dk = max(abs(float(a[k][5]) - float(b[k][5])) for k in keys)
    first = next((k for k in keys if a[k][3:] != b[k][3:]), None)
    return (f"rows {len(keys)}: identical {same} ({100.0 * same / len(keys):.2f}%), same top-1 {top}, "
            f"max |dNLL| {dn:.6f}, max |dKLD| {dk:.6f}, first difference "
            f"{'none' if first is None else 'chunk %s pos %s' % first}")


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def set_args(args: list[str], extra: list[str]) -> list[str]:
    """`args` with each flag of `extra` replacing the same flag (and its value) in `args`, or added."""
    out = list(args)
    i = 0
    while i < len(extra):
        flag = extra[i]
        has_value = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        value = [extra[i + 1]] if has_value else []
        if flag in out:
            j = out.index(flag)
            old_has_value = j + 1 < len(out) and not out[j + 1].startswith("--")
            out[j:j + 1 + old_has_value] = [flag, *value]
        else:
            out += [flag, *value]
        i += 1 + has_value
    return out


def run(a) -> int:
    n_ctx, n_vocab, n_chunk, chunks = read_base(a.base)
    if a.chunks < 0 or a.chunks > n_chunk:
        raise ValueError(f"--chunks must be between 0 and {n_chunk}")
    if not 1 <= a.t <= 8 or a.progress < 0:
        raise ValueError("--t must be 1..8 and --progress must be nonnegative")
    nc = a.chunks or n_chunk
    first = n_ctx // 2
    scored = n_ctx - 1 - first
    args = ["--model", str(Path(a.model).resolve()), "--spec", "5",
            "--adapt-swaps", "0", "--peer-adapt-swaps", "0", "--max-context", "4096"]
    if gpu_count() > 1:
        args += ["--gpu", "0,1"]
    args = set_args(args, a.engine_args + ["--ppl-base", str(Path(a.base).resolve())])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "rows.tsv"
    rows.unlink(missing_ok=True)
    (out / "report.txt").unlink(missing_ok=True)
    args = set_args(args, ["--ppl-out", str(rows)])
    log = open(out / "engine.log", "w")
    log.write(f"ppl_gate: engine={a.engine_bin} T={a.t} chunks={a.chunks} args={' '.join(args)}\n")
    log.flush()
    t0 = time.time()
    eng = subprocess.Popen([a.engine_bin, "--serve", *args], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    try:
        for line in eng.stdout:
            if line.startswith("READY"):
                break
        else:
            print("the engine exited before READY; see", out / "engine.log")
            return 1
        t_load = time.time() - t0
        hits = look = off = 0
        pp_ms = dec_ms = 0.0
        t1 = time.time()
        for k in range(nc):
            ids = ",".join(str(t) for t in chunks[k][:first + 1])
            eng.stdin.write(f"GEN {n_ctx - 1 - first} ppl={k} ppl_t={a.t} {ids}\n")
            eng.stdin.flush()
            for line in eng.stdout:
                if line.startswith("ERR"):
                    print(f"chunk {k}: {line.strip()}")
                    return 1
                if line.startswith("DONE"):
                    f = line.split()
                    try:
                        if len(f) < 9 or int(f[1]) != scored or int(f[2]) != first + 1 or f[5] not in ("stop", "length"):
                            raise ValueError("incomplete DONE")
                        times = float(f[3]), float(f[4])
                        if any(not math.isfinite(v) or v < 0 for v in times):
                            raise ValueError("invalid DONE timing")
                    except (ValueError, IndexError) as e:
                        print(f"chunk {k}: invalid completion: {e}")
                        return 1
                    pp_ms += times[0]; dec_ms += times[1]
                    if f[8] != "0":
                        print(f"chunk {k}: {f[8]} prompt tokens reused from an earlier request")
                    if len(f) > 10:
                        hits += int(f[9]); look += int(f[10])
                    if len(f) > 15:
                        off += int(f[15])
                    break
            else:
                print(f"chunk {k}: the engine exited; see", out / "engine.log")
                return 1
            if a.progress and (k + 1) % a.progress == 0:
                print(f"  {k + 1}/{nc} chunks, {time.time() - t1:.0f} s", flush=True)
        t_run = time.time() - t1
        eng.stdin.write("QUIT\n")
        eng.stdin.flush()
    finally:
        try:
            eng.wait(timeout=60)
        except subprocess.TimeoutExpired:
            eng.kill()
            eng.wait(timeout=60)
        finally:
            eng.stdin.close()
            eng.stdout.close()
            log.close()
    if eng.returncode != 0:
        print(f"the engine exited with status {eng.returncode}; see", out / "engine.log")
        return 1
    try:
        measured = load_rows(str(rows))
        expected = {(k, pos): chunks[k][pos + 1] for k in range(nc) for pos in range(first, n_ctx - 1)}
        if measured.keys() != expected.keys():
            raise ValueError(f"expected {len(expected)} scored rows for {nc} chunks; got {len(measured)}")
        for key, target in expected.items():
            fields = measured[key]
            if int(fields[2]) != target or any(int(fields[i]) >= n_vocab for i in (7, 8)):
                raise ValueError(f"invalid target or top token at chunk {key[0]} pos {key[1]}")
    except (OSError, ValueError) as e:
        print(f"invalid quality output: {e}")
        return 1
    extra = {"T": a.t, "load s": f"{t_load:.0f}", "run s": f"{t_run:.1f}",
             "prompt ms/chunk": f"{pp_ms / nc:.0f}", "decode ms/chunk": f"{dec_ms / nc:.0f}",
             "decode routed experts": f"{look + off}: VRAM hits {100.0 * hits / max(1, look + off):.1f}%, "
                                      f"peer GPU/PCIe {100.0 * off / max(1, look + off):.1f}%, "
                                      f"CPU {100.0 * (look - hits) / max(1, look + off):.1f}%"}
    rep = summary(str(rows), extra)
    (out / "report.txt").write_text(f"engine {a.engine_bin}\nargs {' '.join(args)}\n{rep}\n")
    print(rep)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="the .basalt file")
    ap.add_argument("--engine-bin", default=str(ENGINE_BIN) if ENGINE_BIN.is_file() else None,
                    help="the engine binary (default: bin/basalt in the repo)")
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--chunks", type=int, default=0, help="the first N chunks (0: all)")
    ap.add_argument("--t", type=int, default=3, help="tokens per verify window")
    ap.add_argument("--progress", type=int, default=10, help="a progress line every N chunks (0: none)")
    ap.add_argument("--report", help="print the summary of an existing rows.tsv and exit")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="compare two rows.tsv files row by row and exit")
    a, extra = ap.parse_known_args()
    a.engine_args = extra
    if a.diff:
        print(diff(*a.diff))
        return 0
    if a.report:
        print(summary(a.report))
        return 0
    if not (a.model and a.base and a.out):
        ap.error("--model, --base and --out are required")
    if not a.engine_bin:
        ap.error("no engine binary: pass --engine-bin")
    return run(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, OverflowError, struct.error) as e:
        print(f"ppl_gate: {e}", file=sys.stderr)
        sys.exit(1)

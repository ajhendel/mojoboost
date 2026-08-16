#!/usr/bin/env python3
"""Refuse an artifact that contains instructions the target baseline forbids.

    python3 packaging/isa_baseline.py python/dist/mojotrees-*.whl
    python3 packaging/isa_baseline.py python/mojotrees/_mojotrees.so
    python3 packaging/isa_baseline.py <file> --profile linux-aarch64 --verbose

Exit status 0 when every object is within its baseline, 1 when any is not, 2
when the check could not be performed. A check that cannot run is a failure,
not a skip: an inspection that quietly skipped is worse than no inspection,
because it prints a passing line.

WHY THIS FILE EXISTS
--------------------
`mojo build` defaults `--target-cpu` and `--target-features` to the HOST.
`packaging/build_target.sh` now pins a baseline instead, and that flag is the
actual fix. This file is the alarm that tells you the flag came off, or that a
new build path was added that does not go through those scripts.

It is needed because nothing else in the repository can see the failure.
`packaging/macos/inspect_wheel.py` reads Mach-O `cputype` and `minos`;
`packaging/matrix/validate_artifact.py` reads the same. On arm64 the Mach-O
`cpusubtype` field stays `ARM64_ALL` no matter what `-mcpu` the compiler was
given, and there is no load command that records the feature set. A wheel built
on an M4 with `+bf16 +i8mm +sme2` is byte-for-byte indistinguishable, at the
header level, from one built for apple-m1. It passes every existing check and
SIGILLs on an M1. The only place the difference is visible is the instruction
stream, so this reads the instruction stream.

WHAT IT IS AND IS NOT
---------------------
This is a DENY-LIST over disassembled mnemonics and operands. It is a smoke
alarm, not a proof.

    It WILL catch      the families a native build on this project's actual
                       hardware introduces: bf16, i8mm, SME/SME2, SVE/SVE2 on
                       ARM; AVX, AVX2, BMI, AVX-512, AMX on x86.
    It will NOT catch  a post-baseline instruction in a family nobody added to
                       the table, an instruction hidden in a data section that
                       the disassembler renders as data, or anything in a
                       library this project did not compile and cannot recompile
                       (those are reported, and their findings are advisory,
                       because the remedy is a different one).

Treat a clean run as "the alarm did not go off", never as "the artifact is
certified portable". The certification is the build flag plus a run on the
oldest supported machine (packaging/matrix/smoke/).

WHY DISASSEMBLY AND NOT A HEADER FIELD
--------------------------------------
Because there is no header field. See above. `otool -tv` on macOS and
`objdump -d` (or `llvm-objdump -d`) on Linux are the tools; both are read-only
and neither loads or executes the object.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# The rule table.
#
# Each rule is (feature, mnemonic regex or None, operand regex or None, note).
# A rule fires when the mnemonic regex matches the mnemonic (suffixes already
# stripped) or the operand regex matches the operand text. A profile lists the
# features its baseline ALLOWS; every other rule for that architecture is live.
#
# Mnemonic matching is done on a normalized mnemonic: lowercased, with any
# Apple-style `.4s` / `.16b` arrangement suffix removed, and with a leading
# `{` or trailing `,` never present because the mnemonic is the first
# whitespace token of the instruction.
#
# Operand matching is done on operand text with symbol references (`<foo>`),
# trailing comments (`; ...`, `// ...`, `# ...`) and hex literals removed, so a
# symbol named `_bfdot_kernel` or an address that happens to spell a register
# name cannot fire a rule.
# ---------------------------------------------------------------------------

Rule = tuple  # (feature, mnemonic_re, operand_re, note)

AARCH64_RULES: list[Rule] = [
    (
        "bf16",
        r"^(bfdot|bfmmla|bfmlalb|bfmlalt|bfmlslb|bfmlslt|bfcvt|bfcvtn|bfcvtn2"
        r"|bfcvtnt)$",
        None,
        "FEAT_BF16, ARMv8.6. Present from apple-m2; absent on apple-m1.",
    ),
    (
        "i8mm",
        r"^(smmla|ummla|usmmla|usdot|sudot)$",
        None,
        "FEAT_I8MM, ARMv8.6. Present from apple-m2; absent on apple-m1.",
    ),
    (
        "sme",
        r"^(smstart|smstop|rdsvl|addsvl|addspl|fmopa|fmops|smopa|smops|umopa"
        r"|umops|sumopa|usmopa|bfmopa|bfmops|addha|addva|psel|luti2|luti4)$",
        r"\bza\d*(\.[bhsdq])?\s*[\[,]|\bza\b",
        "FEAT_SME/SME2. Present from apple-m4; absent on m1, m2 and m3.",
    ),
    (
        "sve",
        None,
        r"\bz\d{1,2}\.[bhsdq]\b|\bp\d{1,2}/[mz]\b",
        "FEAT_SVE/SVE2. No Apple silicon has it; Neoverse V1/V2/N2 do.",
    ),
    (
        "dotprod",
        r"^[su]dot$",
        None,
        "FEAT_DotProd, ARMv8.2 extension. Present on all Apple silicon and on "
        "Neoverse N1 and later; absent on Cortex-A72 and plain ARMv8.0-A.",
    ),
    (
        "fp16fml",
        r"^(fmlal|fmlal2|fmlsl|fmlsl2)$",
        None,
        "FEAT_FHM, ARMv8.2 extension.",
    ),
    (
        "fullfp16",
        r"^f[a-z0-9]+$",
        r"\.[48]h\b|\bh\d{1,2}\b",
        "FEAT_FP16 half-precision arithmetic, ARMv8.2 extension. Note that "
        "fcvt to and from h registers is ARMv8.0 and is excluded below.",
    ),
    (
        "lse",
        r"^(cas|casp|swp|ld(add|clr|eor|set|smax|smin|umax|umin)"
        r"|st(add|clr|eor|set|smax|smin|umax|umin))[ab]{0,2}l?[bh]?$",
        None,
        "FEAT_LSE large-system atomics, ARMv8.1. The alternative is an "
        "ldxr/stxr retry loop, which is correct but slower under contention.",
    ),
    (
        "rcpc",
        r"^ldapr[bh]?$",
        None,
        "FEAT_LRCPC, ARMv8.3.",
    ),
    (
        "complxnum",
        r"^(fcmla|fcadd)$",
        None,
        "FEAT_FCMA, ARMv8.3.",
    ),
    (
        "jsconv",
        r"^fjcvtzs$",
        None,
        "FEAT_JSCVT, ARMv8.3.",
    ),
    (
        "rdm",
        r"^(sqrdmlah|sqrdmlsh)$",
        None,
        "FEAT_RDM, ARMv8.1.",
    ),
    (
        "crc",
        r"^crc32[a-z]*$",
        None,
        "FEAT_CRC32, ARMv8.1.",
    ),
    (
        "crypto",
        r"^(aese|aesd|aesmc|aesimc|pmull2?|sha1[a-z]*|sha256[a-z]*"
        r"|sha512[a-z]*|eor3|bcax|xar|rax1)$",
        None,
        "FEAT_AES/SHA/SHA3. Optional even where the architecture level is met.",
    ),
    (
        "pauth",
        r"^(pac[a-z]{2,4}|aut[a-z]{2,4}|xpac[a-z]*|retaa|retab|braa|brab"
        r"|blraa|blrab|braaz|brabz|blraaz|blrabz)$",
        None,
        "FEAT_PAuth, ARMv8.3. Emitted only under -mbranch-protection, but a "
        "toolchain default can turn that on.",
    ),
]

# `fcvt` between single/double/half is ARMv8.0-A and its operands contain an
# h register, so the fullfp16 operand rule would fire on it. Same for the
# ld1r/ldr/str forms that address an h register as plain 16-bit memory. These
# mnemonics are exempted from the fullfp16 rule only.
FULLFP16_EXEMPT = re.compile(
    r"^(fcvt|fcvtn|fcvtn2|fcvtl|fcvtl2|fcvtxn|fcvtxn2|ldr|str|ldur|stur"
    r"|ldp|stp|ld1|st1|ld1r|dup|mov|umov|smov|ins|fmov|tbl|tbx|ext|rev16"
    r"|rev32|rev64|zip1|zip2|uzp1|uzp2|trn1|trn2)$"
)

# AT&T-syntax disassemblers append an operand-size suffix to many mnemonics:
# GNU objdump prints `pdepl`, `lzcntl`, `movbew`. Intel syntax does not. Every
# x86 mnemonic pattern below therefore ends in `[bwlq]?`, which is why it is
# written out rather than assumed. This was found by running the check against a
# real objdump of an AVX2/BMI object, not by reading about it.
X86_RULES: list[Rule] = [
    # AVX-512 and AMX first, so a zmm/k-register hit is reported as the more
    # specific finding rather than as bare AVX.
    (
        "avx512",
        r"^v.*$",
        r"\bzmm\d+\b|%zmm\d+|\bk[0-7]\b(?!\w)|%k[0-7]\b",
        "AVX-512, x86-64-v4. Absent on every AMD before Zen 4 and on Intel "
        "E-cores; a GitHub-hosted EPYC runner does not have it.",
    ),
    (
        "amx",
        r"^(ldtilecfg|sttilecfg|tilerelease|tilezero|tileloadd|tileloaddt1"
        r"|tilestored|tdp[a-z0-9]+)[bwlq]?$",
        None,
        "AMX. Sapphire Rapids and later Xeon only.",
    ),
    (
        "avx",
        r"^v(?!err|erw|mcall|mlaunch|mresume|mxoff|mxon|mread|mwrite|mptrld"
        r"|mptrst|mclear|mfunc|mrun)[a-z0-9]+$",
        None,
        "Any VEX-encoded instruction. AVX and AVX2 are x86-64-v3; x86-64-v2 "
        "stops at SSE4.2. The exclusions in this pattern are the handful of "
        "v-prefixed instructions that are not AVX (VMX and VERR/VERW).",
    ),
    (
        "bmi",
        r"^(andn|bextr|blsi|blsmsk|blsr|bzhi|mulx|pdep|pext|rorx|sarx|shlx"
        r"|shrx|tzcnt)[bwlq]?$",
        None,
        "BMI1/BMI2, x86-64-v3.",
    ),
    (
        "lzcnt",
        r"^lzcnt[bwlq]?$",
        None,
        "LZCNT, x86-64-v3. Decodes as BSR on a CPU without it, which is the "
        "worst kind of portability bug: wrong answers, no fault.",
    ),
    (
        "movbe",
        r"^movbe[bwlq]?$",
        None,
        "MOVBE, x86-64-v3.",
    ),
    (
        "xsave",
        r"^(xsave[a-z0-9]*|xrstor[a-z0-9]*|xgetbv|xsetbv)[bwlq]?$",
        None,
        "XSAVE, x86-64-v3.",
    ),
    (
        "adx",
        r"^(adcx|adox)[bwlq]?$",
        None,
        "ADX, Broadwell and later.",
    ),
    (
        "aes-ni",
        r"^(aes[a-z]+|pclmul[a-z]*|sha1[a-z]+|sha256[a-z]+)[bwlq]?$",
        None,
        "AES-NI, PCLMULQDQ, SHA-NI. None is in x86-64-v2.",
    ),
    (
        "rdrand",
        r"^(rdrand|rdseed)[bwlq]?$",
        None,
        "RDRAND/RDSEED, Ivy Bridge and later.",
    ),
]


class Profile:
    def __init__(self, name, arch, target, allowed, note):
        self.name = name
        self.arch = arch          # "aarch64" or "x86_64"
        self.target = target      # the flag packaging/build_target.sh passes
        self.allowed = set(allowed)
        self.note = note

    @property
    def rules(self):
        table = AARCH64_RULES if self.arch == "aarch64" else X86_RULES
        return [r for r in table if r[0] not in self.allowed]


# The allowed sets are read from `mojo build --print-effective-target
# --target-cpu <cpu>`, not from memory, and the commands are in the notes so
# anyone can re-derive them. They must be kept in step with
# packaging/build_target.sh; check_profiles() below verifies that the profiles
# and that file still name the same targets.
PROFILES = {
    "macos-arm64": Profile(
        "macos-arm64", "aarch64", "--target-cpu apple-m1",
        allowed={"dotprod", "fp16fml", "fullfp16", "lse", "rcpc", "complxnum",
                 "jsconv", "rdm", "crc", "crypto", "pauth"},
        note="apple-m1. From `--print-effective-target --target-cpu apple-m1`: "
             "+aes,+altnzcv,+ccdp,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,"
             "+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,"
             "+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs. No bf16, no "
             "i8mm, no sme, no sve.",
    ),
    "linux-aarch64": Profile(
        "linux-aarch64", "aarch64", "--target-cpu generic --target-features +lse",
        allowed={"lse"},
        note="ARMv8.0-A plus LSE atomics, i.e. an ARMv8.1-A floor. From "
             "`--print-effective-target --target-triple aarch64-unknown-linux-gnu "
             "--target-cpu generic --target-features +lse`: +lse,+ete,"
             "+fp-armv8,+neon.",
    ),
    "linux-x86_64": Profile(
        "linux-x86_64", "x86_64", "--target-cpu x86-64-v2",
        allowed=set(),
        note="x86-64-v2. From `--print-effective-target --target-triple "
             "x86_64-unknown-linux-gnu --target-cpu x86-64-v2`: +cmov,+crc32,"
             "+cx16,+cx8,+fxsr,+mmx,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,"
             "+sse4.2,+ssse3,+x87. Nothing in the x86 table above is in it, "
             "which is why `allowed` is empty.",
    ),
}

# Objects this project does not compile and cannot recompile. Their findings
# are reported and do not fail the run, because the remedy for one of these is
# to change what the wheel bundles or to raise the declared floor, not to pass
# a different flag to `mojo build`. They are NOT ignored: a post-baseline
# instruction in a vendored runtime is still a crash on a user's machine, and
# the operator has to see it to decide.
VENDORED = re.compile(
    r"(^|/)(\.dylibs|\.libs)/|"
    r"lib(KGENCompilerRTShared|AsyncRTMojoBindings|MSupportGlobals"
    r"|AsyncRTRuntimeGlobals)\."
)

MH_MAGIC_64 = 0xFEEDFACF
MH_CIGAM_64 = 0xCFFAEDFE
ELF_MAGIC = b"\x7fELF"


# ---------------------------------------------------------------------------
# Disassembly
# ---------------------------------------------------------------------------


def object_format(blob: bytes) -> str | None:
    if blob[:4] == ELF_MAGIC:
        return "elf"
    if len(blob) >= 4:
        (magic,) = struct.unpack_from("<I", blob, 0)
        if magic in (MH_MAGIC_64, MH_CIGAM_64):
            return "macho"
    return None


def object_arch(blob: bytes, fmt: str) -> str | None:
    """Architecture, so the right rule table is used even without a profile."""
    if fmt == "elf":
        if len(blob) < 20:
            return None
        (e_machine,) = struct.unpack_from("<H", blob, 18)
        return {62: "x86_64", 183: "aarch64"}.get(e_machine)
    (cputype,) = struct.unpack_from("<i", blob, 4)
    return {0x0100000C: "aarch64", 0x01000007: "x86_64"}.get(cputype & 0xFFFFFFFF)


def disassemble(path: Path, fmt: str) -> tuple[list[str], str]:
    """Text disassembly of `path`, and the command that produced it.

    Raises RuntimeError when no disassembler is available. That propagates to
    exit status 2 rather than to a pass: this check has exactly one mechanism
    and it either ran or it did not.
    """
    candidates = []
    if fmt == "macho":
        if shutil.which("otool"):
            candidates.append(["otool", "-tv", str(path)])
        if shutil.which("llvm-objdump"):
            candidates.append(["llvm-objdump", "-d", "--macho", str(path)])
    else:
        for tool in ("objdump", "llvm-objdump", "aarch64-linux-gnu-objdump"):
            if shutil.which(tool):
                candidates.append([tool, "-d", str(path)])
    if not candidates:
        want = "otool or llvm-objdump" if fmt == "macho" else "objdump or llvm-objdump"
        raise RuntimeError(
            f"no disassembler for a {fmt} object: {want} is required and "
            f"neither is on PATH. On macOS otool ships with the Xcode command "
            f"line tools; on Linux objdump is in binutils and is present in "
            f"every manylinux image. This is reported as a failure rather than "
            f"a skip on purpose."
        )
    last = None
    for cmd in candidates:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.splitlines(), " ".join(cmd)
        last = (cmd, proc.returncode, (proc.stderr or "").strip()[:400])
    raise RuntimeError(f"every disassembler failed; last was {last}")


# A disassembly line looks like one of:
#   otool:          0000000100003f40\tsub\tsp, sp, #0x40
#   llvm-objdump:   100003f40: d10103ff     \tsub\tsp, sp, #0x40
#   GNU objdump:           0: c4 e2 75 a8 d0    \tvfmadd213ps\t%ymm0, %ymm1, %ymm2
#   objdump aarch64:   4ae24:\t33 90 86 6e\tudot\tv19.4s, v1.16b, v6.16b
# In every one the address comes first, then zero or more fields of encoded
# bytes, then the mnemonic.
#
# The address is one to sixteen hex digits, NOT four to sixteen. An objdump of
# a relocatable object starts at 0 and prints a single digit. That mistake cost
# this file a silent pass on an artifact full of AVX-512, which is why
# scan_object below now refuses to report on a disassembly it could not parse
# rather than reporting zero findings.
_ADDR = re.compile(r"^\s*([0-9a-fA-F]{1,16}):?(?=[\s\t])")
# A field of encoded bytes: space-separated byte pairs, or one 8/16-digit word.
# Deliberately not "any even run of hex", because `fadd` is four hex digits and
# is a mnemonic.
_HEXBYTES = re.compile(
    r"^(?:[0-9a-fA-F]{2})(?:\s+[0-9a-fA-F]{2})*$|^[0-9a-fA-F]{8}$|^[0-9a-fA-F]{16}$"
)
# `//` and `;` always start a comment. `#` only does when a space follows it:
# on AArch64 `#0x40` and `#-1` are immediates, and truncating the operands
# there would be wrong even though no rule currently keys on an immediate.
_COMMENT = re.compile(r"(?:;|//).*$|#\s.*$")
_SYMREF = re.compile(r"<[^>]*>")
_HEXLIT = re.compile(r"\b0x[0-9a-fA-F]+\b|#-?\d+")
_SUFFIX = re.compile(r"\.(2s|4s|2d|1d|4h|8h|8b|16b|b|h|s|d|q)$")


def parse_instruction(line: str) -> tuple[str, str] | None:
    """(normalized mnemonic, cleaned operand text), or None for a non-instruction."""
    m = _ADDR.match(line)
    if not m:
        return None
    rest = line[m.end():].lstrip(":")
    fields = [f.strip() for f in rest.split("\t")]
    fields = [f for f in fields if f]
    # Drop leading byte-dump fields, but never the last field: on a line with
    # one field left, that field is the instruction whatever it looks like.
    while len(fields) > 1 and _HEXBYTES.match(fields[0]):
        fields.pop(0)
    if not fields:
        return None
    bits = fields[0].split(None, 1)
    mnemonic = bits[0].lower()
    if not mnemonic[:1].isalpha():
        return None
    operands = bits[1] if len(bits) > 1 else ""
    if len(fields) > 1:
        operands = (operands + " " + " ".join(fields[1:])).strip()
    operands = _SYMREF.sub(" ", operands)
    operands = _COMMENT.sub("", operands)
    operands = _HEXLIT.sub(" ", operands).lower()
    return _SUFFIX.sub("", mnemonic), operands


def scan_object(lines: list[str], profile: Profile) -> dict:
    """Feature -> list of (mnemonic, operands, first line seen), plus a count."""
    compiled = []
    for feature, mre, ore, note in profile.rules:
        compiled.append((
            feature,
            re.compile(mre) if mre else None,
            re.compile(ore) if ore else None,
            note,
        ))
    hits: dict[str, dict] = {}
    total = 0
    # Every line that looks like it holds an instruction, whether or not the
    # parser managed to read one out of it. The two counts must agree; see
    # the guard in main(). A disassembler whose output format this parser does
    # not understand produces zero findings, which is indistinguishable from a
    # clean artifact unless something is counting.
    candidates = 0
    unparsed = 0
    unparsed_examples: list[str] = []
    for line in lines:
        # A symbol line (`0000000000000000 <f>:`) is address-prefixed and is
        # not an instruction, so it is not a candidate and its failure to parse
        # is not evidence of anything.
        is_candidate = bool(_ADDR.match(line)) and "<" not in line
        if is_candidate:
            candidates += 1
        parsed = parse_instruction(line)
        if parsed is None:
            if is_candidate:
                unparsed += 1
                if len(unparsed_examples) < 3:
                    unparsed_examples.append(line.rstrip())
            continue
        mnemonic, operands = parsed
        total += 1
        for feature, mre, ore, note in compiled:
            fired = False
            if mre is not None and mre.match(mnemonic):
                if feature == "fullfp16":
                    # This one needs both halves and an exemption list; a bare
                    # `f...` mnemonic is not evidence of anything.
                    fired = (
                        ore is not None
                        and ore.search(operands) is not None
                        and not FULLFP16_EXEMPT.match(mnemonic)
                    )
                elif ore is not None:
                    fired = ore.search(operands) is not None
                else:
                    fired = True
            elif ore is not None and mre is None:
                fired = ore.search(operands) is not None
            if fired:
                slot = hits.setdefault(feature, {"note": note, "count": 0,
                                                 "examples": []})
                slot["count"] += 1
                if len(slot["examples"]) < 5:
                    slot["examples"].append(f"{mnemonic}\t{operands}".strip())
                break
    # Did the parser actually read this disassembly, or did it read nothing and
    # therefore find nothing? A handful of unparsed lines is normal: a literal
    # pool inside .text disassembles to `.word 0x...`, which is data and has no
    # mnemonic. A systematic failure is not normal and must not be reported as
    # a clean artifact. The threshold is one percent, floored at eight lines so
    # that a tiny object is not judged on a ratio over a handful of lines.
    tolerated = max(8, candidates // 100)
    readable = total > 0 and unparsed <= tolerated
    return {
        "hits": hits,
        "instructions": total,
        "candidates": candidates,
        "unparsed": unparsed,
        "unparsed_examples": unparsed_examples,
        "readable": readable,
        "tolerated": tolerated,
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def collect(target: Path, tmp: Path) -> list[tuple[str, Path]]:
    """[(display name, path on disk)] for every object to check."""
    if target.suffix == ".whl":
        out = []
        with zipfile.ZipFile(target) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/"):
                    continue
                if not re.search(r"\.(so|dylib)(\.\d+)*$", name):
                    continue
                dest = tmp / name.replace("/", "__")
                dest.write_bytes(zf.read(name))
                out.append((name, dest))
        return out
    if target.is_dir():
        return [
            (str(p.relative_to(target)), p)
            for p in sorted(target.rglob("*"))
            if p.is_file() and re.search(r"\.(so|dylib)(\.\d+)*$", p.name)
        ]
    return [(target.name, target)]


def scan_blob(name: str, blob: bytes, profile_name: str | None = None) -> dict:
    """Scan one object already in memory. The entry point other checkers use.

    packaging/macos/inspect_wheel.py already holds every Mach-O member of the
    wheel as bytes, so re-opening the zip to get them again would be a second
    source of truth about what is in the artifact. It calls this instead.

    Returns a dict with `ok` (False means refuse), `error` (set when the check
    could not run, which is also not ok), `vendored`, `profile`, `hits` and
    `instructions`. A caller that treats a missing key as a pass has made the
    same mistake this whole file exists to prevent, so every key is always
    present.
    """
    out = {"name": name, "ok": False, "error": None, "vendored": bool(VENDORED.search(name)),
           "profile": None, "hits": {}, "instructions": 0}
    fmt = object_format(blob)
    if fmt is None:
        out["error"] = "not an ELF or Mach-O object"
        return out
    arch = object_arch(blob, fmt)
    if arch is None:
        out["error"] = f"unsupported architecture in the {fmt} header"
        return out
    profile = PROFILES[profile_name or default_profile(arch)]
    out["profile"] = profile
    if profile.arch != arch:
        out["error"] = f"object is {arch}, profile {profile.name} is {profile.arch}"
        return out
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "object.bin"
        path.write_bytes(blob)
        try:
            lines, _tool = disassemble(path, fmt)
        except RuntimeError as exc:
            out["error"] = str(exc)
            return out
        result = scan_object(lines, profile)
    out["instructions"] = result["instructions"]
    out["hits"] = result["hits"]
    if not result["readable"]:
        out["error"] = (
            f"the disassembly could not be read: {result['candidates']} "
            f"instruction-shaped lines, {result['instructions']} parsed, "
            f"{result['unparsed']} unparsed against a tolerance of "
            f"{result['tolerated']}"
            + (f"; first unparsed: {result['unparsed_examples'][0]!r}"
               if result["unparsed_examples"] else "")
        )
        return out
    # A vendored object's findings are advisory: this project does not compile
    # it and no flag here changes it. They are still returned, and the caller
    # is expected to print them.
    out["ok"] = out["vendored"] or not result["hits"]
    return out


def default_profile(arch: str) -> str:
    system = platform.system()
    if arch == "aarch64":
        return "macos-arm64" if system == "Darwin" else "linux-aarch64"
    return "linux-x86_64"


def check_profiles(root: Path) -> list[str]:
    """Every profile's target string must appear in packaging/build_target.sh.

    The two files are one decision written twice, and the failure mode of that
    is silent: a raised floor in build_target.sh with a stale profile here
    passes every artifact it should now reject. This is the cheapest way to
    make the drift loud, and it needs no build.
    """
    path = root / "packaging" / "build_target.sh"
    try:
        text = path.read_text()
    except OSError as exc:
        return [f"cannot read {path}: {exc}"]
    return [
        f"profile {p.name} targets {p.target!r}, which does not appear in "
        f"packaging/build_target.sh. One of the two was changed without the "
        f"other."
        for p in PROFILES.values() if p.target not in text
    ]


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Refuse an artifact containing post-baseline instructions.",
    )
    ap.add_argument("target", type=Path,
                    help="a .whl, a directory, or a single .so/.dylib")
    ap.add_argument("--profile", choices=sorted(PROFILES),
                    help="baseline to check against; inferred from the object's "
                         "architecture and this host when omitted")
    ap.add_argument("--verbose", action="store_true",
                    help="print the per-object instruction count and the tool used")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    drift = check_profiles(root)
    for line in drift:
        print(f"FAIL profiles  {line}")
    if drift:
        return 2

    if not args.target.exists():
        print(f"no such file: {args.target}", file=sys.stderr)
        return 2

    failed = False
    advisory = 0
    with tempfile.TemporaryDirectory() as td:
        objects = collect(args.target, Path(td))
        if not objects:
            print(f"FAIL  no .so or .dylib found in {args.target}. This check "
                  f"cannot run, which is a failure, not a skip.")
            return 2
        print(f"ISA baseline check: {args.target}")
        for name, path in objects:
            blob = path.read_bytes()
            fmt = object_format(blob)
            if fmt is None:
                print(f"FAIL  {name}: not an ELF or Mach-O object")
                failed = True
                continue
            arch = object_arch(blob, fmt)
            if arch is None:
                print(f"FAIL  {name}: unsupported architecture in the {fmt} header")
                failed = True
                continue
            profile = PROFILES[args.profile or default_profile(arch)]
            if profile.arch != arch:
                print(f"FAIL  {name}: object is {arch}, profile "
                      f"{profile.name} is {profile.arch}")
                failed = True
                continue
            try:
                lines, tool = disassemble(path, fmt)
            except RuntimeError as exc:
                print(f"FAIL  {name}: {exc}")
                return 2
            result = scan_object(lines, profile)
            vendored = bool(VENDORED.search(name))
            label = "vendored" if vendored else "ours"
            if args.verbose:
                print(f"      {name}: {result['instructions']} instructions, "
                      f"{label}, via `{tool}`")

            # The guard that makes a clean result mean something. An empty or
            # partly-read disassembly reports no findings, which reads exactly
            # like a portable artifact. It is a failure of this check, not of
            # the artifact, and it must be spelled differently from a pass.
            if not result["readable"]:
                print(f"FAIL  {name}: this check could not read the "
                      f"disassembly. `{tool}` produced {len(lines)} lines, "
                      f"{result['candidates']} of which look like "
                      f"instructions; {result['instructions']} parsed and "
                      f"{result['unparsed']} did not, against a tolerance of "
                      f"{result['tolerated']}.")
                for line in result["unparsed_examples"]:
                    print(f"          unparsed: {line}")
                print("          The output format changed or the tool is not "
                      "one this parser knows.")
                print("          Reported as a failure, never as a pass: a "
                      "parser that reads nothing")
                print("          finds nothing, and that is what a clean "
                      "artifact also looks like.")
                return 2
            if not result["hits"]:
                print(f"ok    {name}  within {profile.name} "
                      f"({profile.target}){'  [vendored]' if vendored else ''}")
                continue
            verdict = "note" if vendored else "FAIL"
            for feature, slot in sorted(result["hits"].items()):
                print(f"{verdict}  {name}  {feature}: {slot['count']} "
                      f"instruction(s) outside {profile.name} "
                      f"({profile.target})")
                print(f"          {slot['note']}")
                for ex in slot["examples"]:
                    print(f"          {ex}")
            if vendored:
                advisory += 1
                print("          This object is not compiled by this project, so "
                      "no build flag here changes it.")
                print("          It is reported and does not fail the run. If it "
                      "is above the floor the")
                print("          wheel claims, the wheel's real floor is this "
                      "object's, not ours.")
            else:
                failed = True
                print("          Built by this project. The fix is "
                      "packaging/build_target.sh: this")
                print("          object was compiled for a CPU above the "
                      "baseline, which means either")
                print("          MOJOTREES_BUILD_TARGET=native was set or a "
                      "build path was added that")
                print("          does not source that file.")

    print()
    if failed:
        print("REFUSED. Do not publish this artifact. It contains instructions "
              "that do not")
        print("exist on the oldest hardware its platform tag promises, and the "
              "failure on that")
        print("hardware is SIGILL inside the extension with no diagnostic the "
              "user can act on.")
        return 1
    if advisory:
        print(f"Passed, with {advisory} advisory finding(s) in vendored objects. "
              f"Read them.")
    else:
        print("Passed. No instruction outside the baseline was found.")
    print("This is a deny-list over a disassembly, not a proof of portability. "
          "It says the")
    print("alarm did not go off. What certifies the artifact is the build flag "
          "plus a run on")
    print("the oldest supported machine (packaging/matrix/smoke/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

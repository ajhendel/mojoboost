"""The package's bit-exactness contract: checked-in golden bit patterns.

WHAT THIS FILE IS
-----------------
Every other exactness test in this tree compares one change against its own
baseline inside one build: the new route against the old route, the parallel
path against the serial one, the device against the host. That is the right
test for a change in flight, and it is not a contract. It proves each change
exact against what it replaced; it proves nothing about what the composition
of five such changes produces, and a merge is exactly where two individually
exact edits compose into a moved multiply.

This file is the contract. It holds literal IEEE-754 bit patterns for six
fits, checked in, and it compares against them with integer equality. It
does not compare against anything computed in the same process, so nothing
in the package can move without this file saying so. Two lanes in the round
that produced these values hit fused multiply-add contraction, once on the
CPU (hoisting `learning_rate * value` out of a loop stopped a fusion and
moved 98 of 600 rows by one ulp, which changed the next tree) and once on
the GPU (moving a leaf id from a launch argument to a per-thread value
started one and moved every score by one ulp). Neither is visible to a
tolerance. Both are visible here.

WHAT IT COVERS
--------------
Six fixtures, chosen because they take different routes through the
trainer, not because they are different datasets:

  l2        plain L2 regression, dense, every feature, every row.
  logit     binary logistic: a different gradient, a different base score,
            and a non-identity link, so raw and response scales differ.
  multi     three-class softmax: the per-round, per-class tree loop and the
            row-major raw score buffer.
  bagged    L2 with `bagging_fraction=0.6, bagging_freq=1`. A bagged round
            ends by traversing the tree for every row rather than by leaf
            membership, so this is the score-update path the unbagged
            fixtures do not reach.
  ffrac     L2 with `feature_fraction=0.5`: the per-tree feature selection,
            which changes which features a split search may even see.
  misscat   L2 over a matrix with a reserved missing bin on one numerical
            column and an integer-coded categorical column. Covers NaN
            routing by node default and category-set splits.

For each fixture the recorded state is every tree's `feature`,
`threshold_bin`, `left`, and `right` arrays, every tree's `value` array as
bits, and every row's final raw score and prediction as bits. The routing
metadata a tree also carries (`missing_bin`, `default_left`, `cat_offset`,
`cat_bitset`) is not listed separately: it is covered through the per-row
scores, which are what routing decides.

THE INPUT
---------
Generated in the test, never loaded from disk, so the fixture is
reproducible on any machine with no data file. Every feature value is
`support._uniform(UInt64(k) + seed)`, that is, the top 53 bits of one round
of SplitMix64 over the column-major index `k = f * n_rows + r` offset by a
per-fixture seed, scaled into [0, 1). Targets and labels are closed-form
functions of those features, written out below. Nothing draws on wall clock,
address, thread count, or environment.

Bit-determinism across worker counts is a separate promise, made by
`parallel.mojo` and tested in `tests/test_cpu_parallel.mojo`; this file
relies on it and sets no worker count, so a CI runner with any core count
must produce these same bits.

HOW THE GOLDEN VALUES WERE GENERATED
------------------------------------
From the PRE-MERGE tree at commit `e812f7c`, the parent of the five-lane
merge into `perf-round-2`, built as its own package and run in generate
mode:

    mojo precompile -I <pre>/src <pre>/src/mojotrees \\
        -o <pre>/build/mojotrees.mojopkg
    mojo run -I <pre>/build -I <pre>/tests tests/test_golden_bits.mojo \\
        --generate

Generate mode prints the whole block between the GOLDEN BEGIN and GOLDEN END
markers below, paste-ready. Toolchain: Mojo 1.0.0 (ed45d567). Platform of
record: arm64-apple-darwin.

CHANGING THESE VALUES
---------------------
Bit-determinism is promised per toolchain version. These literals are the
behavior of the package, not a record of a passing run, so editing them is a
deliberate reviewed act that needs a stated reason in the commit message.
There are exactly two admissible reasons:

  1. A toolchain bump, where the new compiler contracts or splits a
     floating-point operation differently. Say which version, and update the
     version recorded above in the same commit.
  2. A numerics change that was chosen, argued for, and reviewed on its own
     merits, with the ulp movement stated.

"The test failed and I regenerated it" is not one of them. A failure here
means the package moved; the finding is the failure, and the fix belongs in
the source. Regenerating to make it pass destroys the only evidence that
anything changed.

A lane preparing a performance change should run this file first on its
branch point and last before merge, and a merge that composes two lanes
should run it on the merge commit even when both lanes ran it green on their
own, because that composition is what neither lane tested.

This file is accelerator-free by design: a CPU-only runner has to be able to
enforce the contract. GPU exactness lives in the `gpu` test set.
"""

# run_tests: cpu-safe -- opens no device; see tools/run_tests.sh gpu_by_content.

from std.math import isnan
from std.memory import bitcast
from std.sys import argv
from std.testing import TestSuite
from std.utils.numerics import nan

from mojotrees.bagging import BaggingParams
from mojotrees.binning import BinnedMatrix, bin_equal_width, fit_bins
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    MulticlassBooster,
    train,
    train_multiclass,
)
from mojotrees.categorical import CategoricalParams
from mojotrees.tree import Tree, TreeParams

from support import _uniform

comptime NAN = nan[DType.float64]()


# --- GOLDEN BEGIN ---
# Generated by `mojo run ... tests/test_golden_bits.mojo --generate`.
# Do not hand-edit; see this file's docstring before changing.

comptime _L2_INTS = """
8 15 0 13 1 2 1 16 5 6 1 19
3 4 0 27 7 8 0 23 13 14 0 7
9 10 0 9 11 12 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 15 0 17 1 2 1 13 3 4 1
15 5 6 0 5 9 10 0 5 7 8 0
23 13 14 0 26 11 12 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 15 0 13 1 2 1 17 5 6
1 19 3 4 0 26 7 8 -1 -1 -1 -1
1 4 11 12 0 9 13 14 1 8 9 10
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 15 0 17 1 2 1 21 3
4 1 10 5 6 0 7 7 8 0 8 13
14 -1 -1 -1 -1 0 23 9 10 -1 -1 -1
-1 1 8 11 12 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 15 0 19 1 2 1 10
3 4 1 18 5 6 0 8 9 10 0 3
7 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 1 25 11 12 -1 -1 -1 -1 2 23
13 14 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 15 0 12 1 2 1
21 9 10 0 28 3 4 1 26 5 6 -1
-1 -1 -1 2 23 7 8 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 0 2 11 12 -1
-1 -1 -1 -1 -1 -1 -1 7 24 13 14 -1
-1 -1 -1 -1 -1 -1 -1 15 0 20 1 2
1 3 3 4 1 6 5 6 -1 -1 -1 -1
2 9 7 8 -1 -1 -1 -1 2 13 13 14
-1 -1 -1 -1 3 13 9 10 1 27 11 12
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 15 0 24 1
2 1 21 3 4 1 9 5 6 0 4 7
8 3 22 11 12 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 2 22 9 10 -1 -1 -1
-1 -1 -1 -1 -1 1 27 13 14 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1
"""

comptime _L2_VALS = """
3e24ece540f4898d bfeea415a4f9d4d2 3fe55f158f17651c 3ff03fbeb8433333
bfa6a81242733333 bfd901e766690690 bff6d6e8ef777777 3fe801a60408d3dd
3ffa3fd12de9bd38 bfe6d16e60ba2e8c 3f998431cc71c71c bff9c0ca9b4b4b4b
bfe95cac3eaaaaab bfd1b9e5c52147ae 3fd49df14f400000 bf500033065e3fae
bfe24a110f526051 3fe23a25238511be bfa76efa49e50d79 bfebfe44a2700000
3fed7785cf1d89d9 3fc8b165d5e147ae bff3beae4d555555 bfe494b1cebb512c
bfdda65c52aaaaab 3fc21c3c5ee38e39 3fb0a1b0d9e66666 3fe48017b2e8ba2f
3fe4fce274d097b4 3ff291c174ec4ec5 bf2a6765ebf0b767 bfdfa161822f21ed
3fd60a2da434b4b5 3fe0efd80f7ccccd bfa16f2a11e66666 bfc9137b2596cccd
bfe8225458100000 3fd7ee48c8666666 3feacd55aa762762 3fe2ce5374555555
3fc9d81fd3e00000 3fc3ee3e2e666666 bfd3648b06f5ef7c bfeb8ceffa0f83e1
bfd97497fbcaaaab 3f221f25c569b621 bfd33c66f2cfd772 3fd340e8fe6cfd77
bfc02fb5a1965966 bfe25f0c5d41a41a 3fe24b9933f1c71c 3fc302b125000000
bfd42429e8c6318c 3fabc27a9e8ba2e9 bf9849acd5294a53 3fd2bbc420f1c71c
3fd0ae3bc7555555 bfbc91882f286bca bfe74ee16e79e79e bfd7e54de8f286bd
bf41b28ba59277b9 bfc71078fc600000 3fd0caa7d87063e7 3fb4882e447ce0c8
bfd3edcd77a80000 3fd962b53072f054 3faf4c7d6cb4b4b5 bfe2c2e17e969697
bfcde37214a40000 bfc337eb8f6db6db 3fc8a43c39db6db7 bfc1fccc08f9e79e
bfd9291c148590b2 3fb6b5dea0666666 3fd9b6e200e38e39 bf4c2099077b8ad3
bfca13bab8fa740e 3fbe73afcfc99326 3fab90a5a4cdcdce 3fd7ab43009d89d9
3fb8d420da0e8ba3 bfc9f857d6666666 3fa273868e507507 3fd4406b70d79436
bfbae8b4dd9bd37a bfd647fdabdf7777 bfd2d195fd555555 bfa31d81d8cccccd
bfbd2b667999999a 3fc1f11e9dba2e8c bf48694a0701460d bfb90dbc17ab5f35
3fc279edee86522c 3fc34d394abca1af bfc246c75756fd84 3fd95f9586492492
3fb71f0e85e5a5a6 bfcf3664f6fa0be8 bfb1b751cac10c97 bfca456013492492
3fa6f86249e1e1e2 bfbfec3f54000000 bfdb75b2f6db6db7 3f81c0edb22be2be
3fc5faa4e08d2d2d bf4250d9ed36c424 bfa935e731dd89d9 3fc50fc15634de9c
3f84179b2911be19 bfc3d27c8036db6e 3fd43beebe800000 3fb4bb489fd6b5ad
bfc1abde24000000 3fa64b1bc3d1bc25 bf5d2123bb08d3dd 3fc24cac793b13b1
bfc9236c8e2e8ba3 bf6378fe13b13b14 bfc08591d3bbbbbc bfd4590d63333333
"""

comptime _L2_RAW = """
4003717798d768a3 3fe5042b4e491ce5 3fd4ae5f8eaa9626 bfec23c7e797b7ac
3ff105e0d9232863 3fb2ccc21ee04fe1 3ff9730aca00d938 3fe0517de732ed82
3ff75db271d141d1 3fe83b86d961d983 bff06ff552a2b744 bfb0907458c2b7f3
3ffb3c2075535fae 3ff821c4db3c9b45 3feaff5b1b6532a5 3fc3f9347bd7ff64
bfbd114d820b4599 3fc84877f835a9a5 3f956fe58bf3600a 3fe4b77c92836faf
3fe3183d969336ce bf97dac62acb5113 3ff5b4b5ba5b02f4 3ff54604ecd44055
3ffd328862cb7176 3ff530b65dc562b1 3ff47b1317de05ec 3ff2d5e96dfeedbf
3ff214707023b27b 3ffb560420f0dc40 3fe9cc3880303b70 3ffecc7287f94c56
4001f22e0f6cfb9a 3fc896283af94179 3fdabfdc7d73191f 3fe5694e6f254413
40041c5bcbdbba5f 3fef85c255959d18 3ff54604ecd44055 3ff1b0d28c5e97a6
bfe354e5847b6eb6 bfef18bdc5f7455e 3ff9730aca00d938 3ffb560420f0dc40
4001f22e0f6cfb9a 40032e19aaee6fc4 3ffc4eba2dfe823c 3fefe3b5904b37b5
bfed4899e6f7455e bff06ff552a2b744 3fe8935f55ab01e2 3fdb47f7b4a6e8d8
4003924614a2083f 400092b405dbbcb3 3fe6482e3951d19c 3fe9da891c05a4f3
3ff2d5e96dfeedbf 3fb094dcb68bd8dd 3ff693ebb4181bcb 3fddc7cc21e4e2bc
3fe83a0dec1e65ab 3ff11230ae2864dc bfe95904c5314902 3ff693ebb4181bcb
3ffb853005ee1fab bfd23c99f8693ba1 bfeb6de74411544b 4001251960dea6fb
3f956fe58bf3600a 3fdb47f7b4a6e8d8 bfb3eabd870ed80e 3ffbbf5a13266007
3fe85b6217f527e5 3ff67d0e8dc233eb bfe4295ff47c882c 3fe31576c32b7fa2
40041c5bcbdbba5f bfd34c91e1ee85fe 3ff3363c6fe49e70 3fefe3b5904b37b5
3fe83a0dec1e65ab 3fddc7cc21e4e2bc bfec53345ba8483d bfec1139a9b57fa8
40013efa389f71b6 3fc6128ac63dbf1e 3ff0e879b8f766af 4002565a79209415
3fb9bb6cb1884668 3ff67d0e8dc233eb 40041c5bcbdbba5f 4000b2b9381afdc0
3ffe1561a18388d9 3ff54604ecd44055 4002565a79209415 40041c5bcbdbba5f
3fecd19d10a88432 bfdcb8a4dbd78aaa 40029d9bcbafa645 bfe95904c5314902
3fb330db6e81f6bb 3ff08b39fe31ff83 bfe8e112bafa6db6 3fce709f220bd4e1
bfe1c4ca5f525d01 3ffc1e5cf4580d37 bfd8bde062934644 bfb117cf64c0a3e5
3fff48e3c9dce430 3fe702adaedc9ff5 3ff8a554c53c47c5 400116e5a1ce963b
3fde59dde62964d7 3ff768a23d3f9766 3fe5fbf8447f9d0d bfe2593c157c882c
40003df875ca81d9 3ff3b867439a5ffb 3ff3202c44abd66e bfdd950072c5a487
3fd1b96f85408da5 bfed3e0b2311544b 3f956fe58bf3600a 3ffd328862cb7176
3ff08b39fe31ff83 3ff1b0d28c5e97a6 3ff768a23d3f9766 3fdb88618fd46f4f
3ff22de658fe8146 3ff2d5e96dfeedbf 3fefe3b5904b37b5 3ff4aefc2b3adbaf
3fe4b77c92836faf bfe7bc40bb9ae004 40041c5bcbdbba5f 3fc84877f835a9a5
bfe08f5c1a2c7195 3fd207a6a7f1268a 3ff22de658fe8146 bfe08f5c1a2c7195
40041c5bcbdbba5f 3ff3ca2f8f9bd4cd 3ffd328862cb7176 3fe83b86d961d983
3f5c00f8633d3a7a bfb0907458c2b7f3 3fb094dcb68bd8dd bfe49ce207839226
bfbe1d8abc3c1f60 bfe8a85dd9c2572a 4002565a79209415 3fff3cf0ef8cb348
400048c1067fec77 bfe163d68a2d8b0b 3fe3cb5223c3e357 4003924614a2083f
3ffe1561a18388d9 3ff26e501d71fea1 3fb094dcb68bd8dd 3ff185d655b7c909
3ffbb92b7ab2fa81 3ffe1561a18388d9 bfd04a2ea6acb510 3fefccb5c8dbdedf
3ff207401d5c8a6c 3ff54604ecd44055 3fefccb5c8dbdedf 3fe9cabf92ecc798
3ff04d94dd85ec6a 3ff54604ecd44055 3feb2f7a8f1a2a20 3ffbbf5a13266007
3fdbc1f5ed9e6883 bfed4899e6f7455e 3f5c00f8633d3a7a 3fe2e758b3836faf
bfbc2969197da8cd 3ff693ebb4181bcb 3ff937bdc4ec0e9e bfe8e112bafa6db6
3ffa4451a13d406a 3fbda838f2d4b8ef 3ffa1f0907821cc3 bfc52846f574d3d9
3fce709f220bd4e1 3fd3069963bdb6c3 bfe8e112bafa6db6 3fbe37670da1d119
3ffa8a444e1987e1 3ff5b4b5ba5b02f4 bfe94bb03f668287 400048c1067fec77
40041c5bcbdbba5f bfe1ba2354dc99f9 3ff22de658fe8146 3fe2a3c504d646f2
3fd1b96f85408da5 3ff024cb48496b1f 3ffa8a444e1987e1 bfed4899e6f7455e
"""

comptime _L2_PRED = """
4003717798d768a3 3fe5042b4e491ce5 3fd4ae5f8eaa9626 bfec23c7e797b7ac
3ff105e0d9232863 3fb2ccc21ee04fe1 3ff9730aca00d938 3fe0517de732ed82
3ff75db271d141d1 3fe83b86d961d983 bff06ff552a2b744 bfb0907458c2b7f3
3ffb3c2075535fae 3ff821c4db3c9b45 3feaff5b1b6532a5 3fc3f9347bd7ff64
bfbd114d820b4599 3fc84877f835a9a5 3f956fe58bf3600a 3fe4b77c92836faf
3fe3183d969336ce bf97dac62acb5113 3ff5b4b5ba5b02f4 3ff54604ecd44055
3ffd328862cb7176 3ff530b65dc562b1 3ff47b1317de05ec 3ff2d5e96dfeedbf
3ff214707023b27b 3ffb560420f0dc40 3fe9cc3880303b70 3ffecc7287f94c56
4001f22e0f6cfb9a 3fc896283af94179 3fdabfdc7d73191f 3fe5694e6f254413
40041c5bcbdbba5f 3fef85c255959d18 3ff54604ecd44055 3ff1b0d28c5e97a6
bfe354e5847b6eb6 bfef18bdc5f7455e 3ff9730aca00d938 3ffb560420f0dc40
4001f22e0f6cfb9a 40032e19aaee6fc4 3ffc4eba2dfe823c 3fefe3b5904b37b5
bfed4899e6f7455e bff06ff552a2b744 3fe8935f55ab01e2 3fdb47f7b4a6e8d8
4003924614a2083f 400092b405dbbcb3 3fe6482e3951d19c 3fe9da891c05a4f3
3ff2d5e96dfeedbf 3fb094dcb68bd8dd 3ff693ebb4181bcb 3fddc7cc21e4e2bc
3fe83a0dec1e65ab 3ff11230ae2864dc bfe95904c5314902 3ff693ebb4181bcb
3ffb853005ee1fab bfd23c99f8693ba1 bfeb6de74411544b 4001251960dea6fb
3f956fe58bf3600a 3fdb47f7b4a6e8d8 bfb3eabd870ed80e 3ffbbf5a13266007
3fe85b6217f527e5 3ff67d0e8dc233eb bfe4295ff47c882c 3fe31576c32b7fa2
40041c5bcbdbba5f bfd34c91e1ee85fe 3ff3363c6fe49e70 3fefe3b5904b37b5
3fe83a0dec1e65ab 3fddc7cc21e4e2bc bfec53345ba8483d bfec1139a9b57fa8
40013efa389f71b6 3fc6128ac63dbf1e 3ff0e879b8f766af 4002565a79209415
3fb9bb6cb1884668 3ff67d0e8dc233eb 40041c5bcbdbba5f 4000b2b9381afdc0
3ffe1561a18388d9 3ff54604ecd44055 4002565a79209415 40041c5bcbdbba5f
3fecd19d10a88432 bfdcb8a4dbd78aaa 40029d9bcbafa645 bfe95904c5314902
3fb330db6e81f6bb 3ff08b39fe31ff83 bfe8e112bafa6db6 3fce709f220bd4e1
bfe1c4ca5f525d01 3ffc1e5cf4580d37 bfd8bde062934644 bfb117cf64c0a3e5
3fff48e3c9dce430 3fe702adaedc9ff5 3ff8a554c53c47c5 400116e5a1ce963b
3fde59dde62964d7 3ff768a23d3f9766 3fe5fbf8447f9d0d bfe2593c157c882c
40003df875ca81d9 3ff3b867439a5ffb 3ff3202c44abd66e bfdd950072c5a487
3fd1b96f85408da5 bfed3e0b2311544b 3f956fe58bf3600a 3ffd328862cb7176
3ff08b39fe31ff83 3ff1b0d28c5e97a6 3ff768a23d3f9766 3fdb88618fd46f4f
3ff22de658fe8146 3ff2d5e96dfeedbf 3fefe3b5904b37b5 3ff4aefc2b3adbaf
3fe4b77c92836faf bfe7bc40bb9ae004 40041c5bcbdbba5f 3fc84877f835a9a5
bfe08f5c1a2c7195 3fd207a6a7f1268a 3ff22de658fe8146 bfe08f5c1a2c7195
40041c5bcbdbba5f 3ff3ca2f8f9bd4cd 3ffd328862cb7176 3fe83b86d961d983
3f5c00f8633d3a7a bfb0907458c2b7f3 3fb094dcb68bd8dd bfe49ce207839226
bfbe1d8abc3c1f60 bfe8a85dd9c2572a 4002565a79209415 3fff3cf0ef8cb348
400048c1067fec77 bfe163d68a2d8b0b 3fe3cb5223c3e357 4003924614a2083f
3ffe1561a18388d9 3ff26e501d71fea1 3fb094dcb68bd8dd 3ff185d655b7c909
3ffbb92b7ab2fa81 3ffe1561a18388d9 bfd04a2ea6acb510 3fefccb5c8dbdedf
3ff207401d5c8a6c 3ff54604ecd44055 3fefccb5c8dbdedf 3fe9cabf92ecc798
3ff04d94dd85ec6a 3ff54604ecd44055 3feb2f7a8f1a2a20 3ffbbf5a13266007
3fdbc1f5ed9e6883 bfed4899e6f7455e 3f5c00f8633d3a7a 3fe2e758b3836faf
bfbc2969197da8cd 3ff693ebb4181bcb 3ff937bdc4ec0e9e bfe8e112bafa6db6
3ffa4451a13d406a 3fbda838f2d4b8ef 3ffa1f0907821cc3 bfc52846f574d3d9
3fce709f220bd4e1 3fd3069963bdb6c3 bfe8e112bafa6db6 3fbe37670da1d119
3ffa8a444e1987e1 3ff5b4b5ba5b02f4 bfe94bb03f668287 400048c1067fec77
40041c5bcbdbba5f bfe1ba2354dc99f9 3ff22de658fe8146 3fe2a3c504d646f2
3fd1b96f85408da5 3ff024cb48496b1f 3ffa8a444e1987e1 bfed4899e6f7455e
"""

comptime _LOGIT_INTS = """
8 11 0 18 1 2 1 4 7 8 1 24
3 4 -1 -1 -1 -1 0 27 5 6 -1 -1
-1 -1 -1 -1 -1 -1 0 9 9 10 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 15 0
18 1 2 1 4 7 8 1 24 3 4 -1
-1 -1 -1 0 27 5 6 -1 -1 -1 -1 -1
-1 -1 -1 2 12 9 10 0 17 13 14 -1
-1 -1 -1 6 15 11 12 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 15
0 18 1 2 1 4 7 8 1 24 3 4
7 6 13 14 0 27 5 6 -1 -1 -1 -1
-1 -1 -1 -1 2 12 9 10 -1 -1 -1 -1
-1 -1 -1 -1 6 15 11 12 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
15 0 16 1 2 3 25 11 12 1 14 3
4 -1 -1 -1 -1 0 25 5 6 3 10 7
8 -1 -1 -1 -1 -1 -1 -1 -1 1 22 9
10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 2 19 13 14 -1 -1 -1 -1 -1 -1 -1
-1 15 0 16 1 2 3 28 13 14 1 14
3 4 -1 -1 -1 -1 0 25 5 6 3 10
9 10 2 8 7 8 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 1 22 11 12 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 15 0 16 1 2 3 21 5 6 1
24 3 4 0 20 11 12 2 8 9 10 -1
-1 -1 -1 2 20 7 8 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1
7 13 14 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 15 0 16 1 2 6 24 7 8
1 14 3 4 -1 -1 -1 -1 0 25 5 6
7 12 11 12 2 8 13 14 -1 -1 -1 -1
2 20 9 10 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 15 0 18 1 2 1 7 5
6 1 24 3 4 7 6 13 14 4 23 7
8 2 12 9 10 -1 -1 -1 -1 4 10 11
12 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1
"""

comptime _LOGIT_VALS = """
3e325ccf37bf863a bff56aaab84377ea 3ffa7912cac68e52 4001b0143a5928cf
bfcb8ab76253e225 bff01c405e28858b 3fef35ffdf4b2c1a bfd1838e4dfa3286
bff7faffdc51a484 bfe9f2fcea09339b 3fdd925b7f3992f3 bf952e8fb24693c2
bff1a7c6aaf48316 3ff26af892dee219 3ff84f9897ee000e bfc52170f4b91b89
bfead568bb3c478f 3fe84119c8bd1126 bfcc56d3a1164cdc bff40664a960d7cb
bfedaa151f960c40 3fd648131a2e4f62 bfd767fdc7eb101e 3fec46f375d9e36c
bff4a80ed41cbb0f bfdb9b2e49c61d1e bf9e72b168fdabb9 bfee8defed8a0ca8
3fed4ac58d2562c6 3ff388c81ba7cf2f bfc0ded367e8bb0a bfe6c681f5b347fa
3fe35d857eb1e22c bfc869407395b861 bff1a028155a53fe bfe9fea199a92d23
3fd26e5494c72f15 bfd414fa6ff2bc5e 3fe72d8fe4d0758d 3fe46ed9dfadb188
3ff5fbf63909b5c3 bf99f92c916f2de4 bfee7d962f535373 3fe5f469d88b7728
3ff45cc618bb463a 3fbd6aabad796e6a bfdf8928a135b25b 3fe70c267a9d8f11
bff2450dd23316da 3fc91c52b8827189 3feb58cbe05ccc13 bfe6881b3c99fceb
bff19986ccd93013 bfa83f966c4ce273 bfe6979f51663bdf 3fe8e7c244c5bf92
bf9a2e9ad936abb6 bfeb2c2f55b9d0dc 3fe24d4844d7fc6c 3ff1f9356b69bbf8
3fb42b6c45220984 bfdbccbae215e67d 3fe203ec1fd80b07 bfc89af4829fab03
3ff04750498359e9 bfef06eede5167d6 3fc56d087222b8cd 3fe87152a47c55d2
bfe45ab41254a405 bfeed3c617110597 3fc5f57780c92296 bf933f4585b16ce6
bfe7e0338f6cbdd0 3fdf3f78a853432a 3fe8bfc179e9baac bfd01bf7058b0fa2
bfeffa78b4ba2cf2 bfc1086d548d7483 bfe969fa7f56f672 3feb70e52b32aa69
bfe53f3c5038ae2f 3fd5d13fe6bf031d 3fc832ccef4006f1 3fef169cd46bbb1f
3fee974638e95f14 bfe61242e156ac81 bf92908078ea88f8 bfe586a99877ff07
3fdabd513e71f250 3fedc4b80ac3bdab 3f9ebfd9e702f887 bfda9164893517f6
3fdbfce609309861 bfec42aebef43bbf 3fb23fe5b8c07fb4 bfe2b216777fe123
3fe8e1b3d9159ff5 bfe8aee2c2918288 3fc535dadbc58cc9 bfbb4e65c4305ddc
3fe9664874f416c6 bf87f8db90327e5c bfe08feea6b01280 3fdaee180721c432
3fe6a6f2a46ef913 bfc50d0dc37cfdaa 3fbc44228953b2bb bfe9d5f48a03c907
bfddcde7d0767644 3fde7f355e820fb0 bfe05511cae1eb12 3fdca672f1dfae60
3fd1d23200adf153 bfe844039ff5c997 3fc5f7d7187e1170 3fed9fff9b80ae49
"""

comptime _LOGIT_RAW = """
c005035103c851f6 c00771e924cc267c c00828608cbb7d88 400377fba10ee54e
c00770bb875f9055 4005605acbe48055 bfc82c8da9525018 3fffc97fcc265ba0
c00828608cbb7d88 c00828608cbb7d88 4005605acbe48055 3ff47a2fe1081eb2
c00828608cbb7d88 4005605acbe48055 bf99b0227da935ff c0052549e61c6b1e
bff0f7cc7435dc84 c00828608cbb7d88 4005605acbe48055 4001cd1f10e8c8d8
4001c3922ba56857 c00828608cbb7d88 400377fba10ee54e 3fd102450b418ddb
4005605acbe48055 c00770bb875f9055 c00828608cbb7d88 c0052549e61c6b1e
c00770bb875f9055 3ff172785cc2cf36 4005605acbe48055 4005605acbe48055
400556cde6a11fd4 4001cd1f10e8c8d8 4005605acbe48055 3ff5a8ae90c6a510
c00828608cbb7d88 c00770bb875f9055 bfe1e5d2e87e9fa7 4005605acbe48055
400556cde6a11fd4 c003feff50c2e40f 4005605acbe48055 4005605acbe48055
c0042c0c5aedf374 bffec43beb61d772 c0042c0c5aedf374 bff7e68220290d8e
c00532da6fd87506 4001cd1f10e8c8d8 c00828608cbb7d88 bffd02846cbf4e45
c00828608cbb7d88 c006f2b2102b52a5 3ff7b3d33c1a3fae c007aa5715873fd8
3ffb6069c2a1c194 3fffc97fcc265ba0 c00770bb875f9055 c00828608cbb7d88
3ffeda4ab211aeaa c00828608cbb7d88 4001cd1f10e8c8d8 c00771e924cc267c
bfe93007a4e15f78 c00828608cbb7d88 bfc1663d68c4f49e 4001cd1f10e8c8d8
c007aa5715873fd8 c00828608cbb7d88 c0012bcb3373d3f2 bfff14f0095b196b
bffe218fc8b7fd8a 400556cde6a11fd4 bffabcac4508f356 c00828608cbb7d88
400377fba10ee54e c0006f0b324c6466 4005605acbe48055 c0038242fad12687
c003b3864f2353c8 c007aa5715873fd8 400377fba10ee54e c00828608cbb7d88
bffec43beb61d772 c00771e924cc267c c00828608cbb7d88 3fd7323452455d77
c00771e924cc267c 4001acdf4225dbc3 c00828608cbb7d88 3ff386cfa06502d1
400377fba10ee54e 4001c3922ba56857 4005605acbe48055 c00828608cbb7d88
c00828608cbb7d88 bff7e68220290d8e c005035103c851f6 bffec43beb61d772
c003475a4b66f6dc 4001cd1f10e8c8d8 400556cde6a11fd4 bffbe40ba55456a0
c00828608cbb7d88 bf99b0227da935ff c00770bb875f9055 c00828608cbb7d88
c00828608cbb7d88 c007aa5715873fd8 400377fba10ee54e 3ff386cfa06502d1
bf99b0227da935ff c0038242fad12687 c00828608cbb7d88 400377fba10ee54e
c0052549e61c6b1e c00828608cbb7d88 c006f2b2102b52a5 c005035103c851f6
c00770bb875f9055 4005605acbe48055 c00828608cbb7d88 c003b3864f2353c8
3ff5a8ae90c6a510 3fc9a9a08ecb96e6 bfc1663d68c4f49e 4005605acbe48055
c00439e8002d13ba 3fd61a5cd47983e2 c003feff50c2e40f 3ffccf2606be140d
c006b84869b4bd20 4005605acbe48055 c003b3864f2353c8 c003b3864f2353c8
c00828608cbb7d88 c00828608cbb7d88 4001acdf4225dbc3 4005605acbe48055
4001cd1f10e8c8d8 4005605acbe48055 c007aa5715873fd8 c0034928b44722ce
c000dfc4fc021f3a bfeb5a5199e3e997 3ff386cfa06502d1 400556cde6a11fd4
3ff7b3d33c1a3fae 4005605acbe48055 c007430aece55664 c001a197e5122931
4005605acbe48055 c00828608cbb7d88 c003feff50c2e40f c00771e924cc267c
c006f2b2102b52a5 c000dfc4fc021f3a c00828608cbb7d88 4005605acbe48055
c00828608cbb7d88 3fedad29e440cb50 c00828608cbb7d88 c006b84869b4bd20
c00828608cbb7d88 bfff8f73b592fa80 4001acdf4225dbc3 c00828608cbb7d88
4005605acbe48055 c003feff50c2e40f c0012bcb3373d3f2 3ffb6069c2a1c194
bffe49384bb26873 c00828608cbb7d88 c00532da6fd87506 3fe71f247db95463
c00439e8002d13ba c00601d101c56614 3fb546471f0aa08b c00828608cbb7d88
c00828608cbb7d88 4005605acbe48055 4005605acbe48055 c00828608cbb7d88
c00328544b3e3d77 4005605acbe48055 4005605acbe48055 c000eaa5eea4aab7
4001acdf4225dbc3 c00828608cbb7d88 c00770bb875f9055 4005605acbe48055
c0042c0c5aedf374 bff7e68220290d8e c00828608cbb7d88 c00828608cbb7d88
c006f2b2102b52a5 4005605acbe48055 3ff9d16e771dc8bc 4001c3922ba56857
"""

comptime _LOGIT_PRED = """
3fb1440f714356ad 3fa9f02b7de6eb53 3fa7d47f5cf33184 3fed6b567e798d52
3fa9f3cc1b767535 3fedee71c4de5c32 3fdcfcb8d275189c 3fec23fcb1884476
3fa7d47f5cf33184 3fa7d47f5cf33184 3fedee71c4de5c32 3fe90997b72afe26
3fa7d47f5cf33184 3fedee71c4de5c32 3fdf9940d7285a43 3fb1002c9230a731
3fd076332f505927 3fa7d47f5cf33184 3fedee71c4de5c32 3fece12a433be5c8
3fecddcbfe4630f9 3fa7d47f5cf33184 3fed6b567e798d52 3fe21d1a44493864
3fedee71c4de5c32 3fa9f3cc1b767535 3fa7d47f5cf33184 3fb1002c9230a731
3fa9f3cc1b767535 3fe7f36ea8019778 3fedee71c4de5c32 3fedee71c4de5c32
3fedec21406965ad 3fece12a433be5c8 3fedee71c4de5c32 3fe96e6f9cb0fa3c
3fa7d47f5cf33184 3fa9f3cc1b767535 3fd746ff6ec1d74e 3fedee71c4de5c32
3fedec21406965ad 3fb36db10434636f 3fedee71c4de5c32 3fedee71c4de5c32
3fb309863f31c1ea 3fc0532695f02438 3fb309863f31c1ea 3fc77832175e107f
3fb0e556f2497ec3 3fece12a433be5c8 3fa7d47f5cf33184 3fc1f41bf712affe
3fa7d47f5cf33184 3fab82d99de1778e 3fea12b69e4eda57 3fa9449d302944a2
3feb1a62c00fb128 3fec23fcb1884476 3fa9f3cc1b767535 3fa7d47f5cf33184
3febf01c7cf59671 3fa7d47f5cf33184 3fece12a433be5c8 3fa9f02b7de6eb53
3fd404bbb79e5b3f 3fa7d47f5cf33184 3fddd41365799b66 3fece12a433be5c8
3fa9449d302944a2 3fa7d47f5cf33184 3fbacba1265286b5 3fc00bd60efec41f
3fc0e61cadcac770 3fedec21406965ad 3fc4429a310a2d44 3fa7d47f5cf33184
3fed6b567e798d52 3fbd16a22d823e7b 3fedee71c4de5c32 3fb48cf6126bf29f
3fb419bd984fcd56 3fa9449d302944a2 3fed6b567e798d52 3fa7d47f5cf33184
3fc0532695f02438 3fa9f02b7de6eb53 3fa7d47f5cf33184 3fe2de415bda9ae4
3fa9f02b7de6eb53 3fecd5bd937c4b43 3fa7d47f5cf33184 3fe8b556f77a87b4
3fed6b567e798d52 3fecddcbfe4630f9 3fedee71c4de5c32 3fa7d47f5cf33184
3fa7d47f5cf33184 3fc77832175e107f 3fb1440f714356ad 3fc0532695f02438
3fb519d4cdd9907f 3fece12a433be5c8 3fedec21406965ad 3fc30f7c4589bff0
3fa7d47f5cf33184 3fdf9940d7285a43 3fa9f3cc1b767535 3fa7d47f5cf33184
3fa7d47f5cf33184 3fa9449d302944a2 3fed6b567e798d52 3fe8b556f77a87b4
3fdf9940d7285a43 3fb48cf6126bf29f 3fa7d47f5cf33184 3fed6b567e798d52
3fb1002c9230a731 3fa7d47f5cf33184 3fab82d99de1778e 3fb1440f714356ad
3fa9f3cc1b767535 3fedee71c4de5c32 3fa7d47f5cf33184 3fb419bd984fcd56
3fe96e6f9cb0fa3c 3fe1993b59341c58 3fddd41365799b66 3fedee71c4de5c32
3fb2eb165ac749a5 3fe2bc591be3e1bf 3fb36db10434636f 3feb76871968e4a8
3fac435e6cb26166 3fedee71c4de5c32 3fb419bd984fcd56 3fb419bd984fcd56
3fa7d47f5cf33184 3fa7d47f5cf33184 3fecd5bd937c4b43 3fedee71c4de5c32
3fece12a433be5c8 3fedee71c4de5c32 3fa9449d302944a2 3fb5157616d8ac1f
3fbbb2fbad7bb14d 3fd31982f639e241 3fe8b556f77a87b4 3fedec21406965ad
3fea12b69e4eda57 3fedee71c4de5c32 3faa81ec4102461c 3fb972506d45bfc0
3fedee71c4de5c32 3fa7d47f5cf33184 3fb36db10434636f 3fa9f02b7de6eb53
3fab82d99de1778e 3fbbb2fbad7bb14d 3fa7d47f5cf33184 3fedee71c4de5c32
3fa7d47f5cf33184 3fe6edeffad18744 3fa7d47f5cf33184 3fac435e6cb26166
3fa7d47f5cf33184 3fbf43238d49842a 3fecd5bd937c4b43 3fa7d47f5cf33184
3fedee71c4de5c32 3fb36db10434636f 3fbacba1265286b5 3feb1a62c00fb128
3fc0c1e293c9a488 3fa7d47f5cf33184 3fb0e556f2497ec3 3fe58a98f838f519
3fb2eb165ac749a5 3faebd1d50a16216 3fe0aa192a1799c0 3fa7d47f5cf33184
3fa7d47f5cf33184 3fedee71c4de5c32 3fedee71c4de5c32 3fa7d47f5cf33184
3fb56564127f6207 3fedee71c4de5c32 3fedee71c4de5c32 3fbb91764bf0b669
3fecd5bd937c4b43 3fa7d47f5cf33184 3fa9f3cc1b767535 3fedee71c4de5c32
3fb309863f31c1ea 3fc77832175e107f 3fa7d47f5cf33184 3fa7d47f5cf33184
3fab82d99de1778e 3fedee71c4de5c32 3feaaf6e35a5737c 3fecddcbfe4630f9
"""

comptime _MULTI_INTS = """
12 11 0 6 1 2 1 16 3 4 1 4
5 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 2 8 7 8 1 13 9 10 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 11 0
5 1 2 1 16 5 6 0 27 3 4 1
4 7 8 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 0 19 9 10 -1
-1 -1 -1 -1 -1 -1 -1 7 0 19 1 2
-1 -1 -1 -1 1 16 3 4 1 12 5 6
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
11 0 6 1 2 1 16 5 6 1 4 3
4 -1 -1 -1 -1 2 8 7 8 -1 -1 -1
-1 -1 -1 -1 -1 0 13 9 10 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 11 0 28
1 2 0 6 3 4 -1 -1 -1 -1 1 16
5 6 1 4 7 8 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 1 23 9 10 -1 -1
-1 -1 -1 -1 -1 -1 9 0 19 1 2 -1
-1 -1 -1 1 12 3 4 -1 -1 -1 -1 2
10 5 6 2 5 7 8 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 11 0 15 1 2
1 13 3 4 -1 -1 -1 -1 2 13 5 6
0 4 9 10 -1 -1 -1 -1 0 5 7 8
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 11 0 27 1 2 0 6 3
4 -1 -1 -1 -1 1 16 5 6 1 2 7
8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 0 19 9 10 -1 -1 -1 -1 -1 -1 -1
-1 9 0 19 1 2 -1 -1 -1 -1 1 17
3 4 4 21 5 6 2 9 7 8 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 11 0 15 1 2 2 13 3 4 -1
-1 -1 -1 1 24 9 10 1 9 5 6 0
5 7 8 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 11
0 28 1 2 0 6 3 4 -1 -1 -1 -1
2 24 5 6 5 11 7 8 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 2 6 9 10
-1 -1 -1 -1 -1 -1 -1 -1 9 0 19 1
2 -1 -1 -1 -1 1 17 3 4 4 21 5
6 2 9 7 8 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1
"""

comptime _MULTI_VALS = """
3e4781ff812478f8 3ff9144c2e000f85 bfe04da0444a1a0e 40017682bac62605
bfc1ca460253d267 3ff05bb4970a88c0 bfe663f370c027bd bfd2a462909c5856
bfeb416a8d9c0fd3 3fe87e5423c4337a bfe8a0529b344971 be3b36f94be8f7f6
bfea9ca35ddb2f86 3fcd7adf8f173dfa 3fdeca2f16ddf1ba bff0e4a65c393fba
bff401b97b1d5ecd 3fd895e4dd55dac1 bfdb4a3a924df253 3fe33d2559b9d880
3feb22c908ff0a18 3f9cdf9fc123ae8a be2d1fbecc4ac96a bfe92aba841c0ac6
3ffb1124fe604dbf bfcb324e6f48f724 4004d66c475f568a bfe2bf06b8dc7c4e
3fdfeb92354d8e45 bf825109b81a44e4 3fed697e528d6252 bfdaecf03879fc71
3fe53b9c3b0bc87a bfe3665077827f92 3ff315b5bcd3846f bfb7e894b40cd65a
bfc80c630ba29884 bfe8e3dc39be8336 3fe811cd68ffc4ae bfe6cda592e32eaf
3f96f36a013c89e4 3fc173936cb04185 bfea97a9696614b1 bfe0b111ad90295c
3fd839bd9c1b3fd0 bfec650ee15fc841 3fd9072dd3670b53 bfd2ce210e0c79ff
3fde6e000a768f1e 3fe60091343ba0e4 3f985c95f927abfe bf9919c168b68276
bfe78e3acf98b976 3fecdce42ba078f9 bfe215426efcd8f8 3ff367cea3bd4fd8
3fdd0154d59d2caf 3ff78feb5008cd34 3feb789cdcb6009d bfa7d827af68f9f6
bf8c3fa787bc4655 3fd81718a0541622 bfe7fc765edb9ee5 3fe80844e0870caa
bfd6e655f42a5d9b 3ff34e77d2df69fe 3fc9b86bea2dda3f 3fe6ccb28922958d
bfe49160ebbecb9e 3fdba7431403733c bfe5fd66e56e01c3 3f97d79ef9399dab
3fc00fb15e901090 bfe3969513efa9db bfd93cd57eaf4707 3fd41b0728e49928
bfe66694bff7ca85 3fd309984b7749c3 bfd03e3c938de9a9 3fd8f7daf3a8c1dd
3fe20114bb86d484 3faf5d5a70c2623e bf94f68e32dfb9f5 bfe65e4c5404c5d3
3fe56e13657e7872 bfca8a4b18e76679 3ff0649c35a9f00b bfe422db79a73b29
3fe19e67650f9ebd 3fdaf5781696d968 3ff2e49d2f3112eb bf91fbb9759df22c
3fd49ef5246509e8 bfe6d4f2bf03a699 3fe73da9dfd06363 bfc3fde696b27fab
3fd51f528ad173e2 bfe74d286d6bb740 3fe79888d0025ad6 bfe19e80593a1828
3fec20f9733fa6bb bfdb1b38e4e86c54 3f99828d0386218e 3fba7331b8672e69
bfe20e23a7d0a3b0 bfd38962208848d2 3fcec46761dda076 bfe0eb692b4ff8f6
3fe02bfe7728ee3c 3fdffbf9ecd69e78 3fa9156593c9d5e9 bfe125eecddced97
3fd0b4b8c908d6f4 bf93cc9f3ec325f7 bfe54f4d28a2bc23 3fe0873b14677606
bfcd0f5692a51fdb 3fea4e4bf798a058 bfe331c52ae7d7e9 3fda8e164338dd44
3fd3cb78c6b6cbb9 3fef1249dd4599f0
"""

comptime _MULTI_RAW = """
c002681b2d3d4c32 bff7fafd30336852 3fc85c0943f00f7e bffc29919cef9795
bfbc7295553ca2ed c0045b84a1459e9f bffcdeaa33ecdb9f 3fc060a92b70196b
c0045b84a1459e9f 3fb31984ac9f77cf bff96fb61f64d178 c0045b84a1459e9f
c0020d8ee1beaa2d bff7fafd30336852 bfe623825b06b64b bfd2b13649b96783
bfbc7295553ca2ed c0045b84a1459e9f bffcdeaa33ecdb9f 3fc98b69da852aba
c0045b84a1459e9f 3fb31984ac9f77cf bff96fb61f64d178 c0045b84a1459e9f
bffb623d40022104 3f3c68dd91669e66 c0045b84a1459e9f c002681b2d3d4c32
bfd12d7ee38b8380 bfd0fd30d4a677de bffde0bd47fc5c5f bfd1c24a13f95c79
c0045b84a1459e9f c002681b2d3d4c32 bfddec199969bf34 3fc85c0943f00f7e
c002681b2d3d4c32 bff7fafd30336852 3fc85c0943f00f7e c002681b2d3d4c32
bff7fafd30336852 3fc85c0943f00f7e bffcdeaa33ecdb9f 3fc060a92b70196b
c0045b84a1459e9f c002681b2d3d4c32 bff7fafd30336852 3fc85c0943f00f7e
c002681b2d3d4c32 3fc98b69da852aba c0045b84a1459e9f bfceebc18b0359a6
bfe71b77422ecedd c0045b84a1459e9f bff0ca76ef2b7ef8 bfdaaa313d5f074d
c0045b84a1459e9f c002681b2d3d4c32 bfd12d7ee38b8380 bff74ab3f990cd57
c0024acad79ce1a9 3fc060a92b70196b c0045b84a1459e9f c002681b2d3d4c32
bfddec199969bf34 bfe623825b06b64b c0024acad79ce1a9 3fc98b69da852aba
c0045b84a1459e9f bff881626a7866aa bfbb3d65d5f38bf0 c0045b84a1459e9f
bff3c7839f0c885d 3fc98b69da852aba c0045b84a1459e9f 3fd0fb9637dd9759
bfeeba5a3f051a3b c0045b84a1459e9f c002681b2d3d4c32 bfc9303d1801f5b2
c00323b1da935462 c0020d8ee1beaa2d bfb239188098bc00 c0045b84a1459e9f
c002681b2d3d4c32 bfefd50fbb8be6ac 3fc85c0943f00f7e bffb623d40022104
3f3c68dd91669e66 c0045b84a1459e9f bfd75d092ce6f9b3 bff479002d6c4864
c0045b84a1459e9f bff881626a7866aa bfbb3d65d5f38bf0 c0045b84a1459e9f
bfceebc18b0359a6 bfe18fb1b92989fd c0045b84a1459e9f c002681b2d3d4c32
bfd12d7ee38b8380 c00323b1da935462 c0020d8ee1beaa2d bff7fafd30336852
bfdae68b8172afa7 3fd0fb9637dd9759 bff96fb61f64d178 c0045b84a1459e9f
bffcbceb26d80e6a bfc9303d1801f5b2 bffbbeada1ddca4d c002681b2d3d4c32
bfddec199969bf34 3fc85c0943f00f7e 3fd0fb9637dd9759 bff96fb61f64d178
c0045b84a1459e9f c0020d8ee1beaa2d 3f3c68dd91669e66 c0045b84a1459e9f
c002681b2d3d4c32 bff7fafd30336852 3fc85c0943f00f7e bffcbceb26d80e6a
bfc9303d1801f5b2 c00323b1da935462 3fb31984ac9f77cf bff96fb61f64d178
c0045b84a1459e9f c001c264302c04de 3fc060a92b70196b c0045b84a1459e9f
c0022f7d98162400 3fc060a92b70196b c0045b84a1459e9f 3fb31984ac9f77cf
bff479002d6c4864 c0045b84a1459e9f c0020d8ee1beaa2d 3f3c68dd91669e66
c0045b84a1459e9f bff39abcf2abec95 bfe18fb1b92989fd c0045b84a1459e9f
c002681b2d3d4c32 bfddec199969bf34 3fc85c0943f00f7e c0024acad79ce1a9
bfb239188098bc00 c0045b84a1459e9f bff881626a7866aa bfdaaa313d5f074d
c0045b84a1459e9f bfceebc18b0359a6 bfe71b77422ecedd c0045b84a1459e9f
c002681b2d3d4c32 3fc060a92b70196b c0045b84a1459e9f bffcbceb26d80e6a
bfe03d3c67495c78 c00323b1da935462 c0020d8ee1beaa2d bfd3db400b2d6470
c0045b84a1459e9f c002681b2d3d4c32 bff7fafd30336852 bfe983fbcc8fddc0
bfd2b13649b96783 3fc060a92b70196b c0045b84a1459e9f 3fd0fb9637dd9759
bff96fb61f64d178 c0045b84a1459e9f c002681b2d3d4c32 3fc98b69da852aba
c0045b84a1459e9f c0024acad79ce1a9 3fc060a92b70196b c0045b84a1459e9f
3fb31984ac9f77cf bff96fb61f64d178 c0045b84a1459e9f bff881626a7866aa
bfbb3d65d5f38bf0 c0045b84a1459e9f 3fb31984ac9f77cf bff96fb61f64d178
c0045b84a1459e9f bff0ca76ef2b7ef8 bfdaaa313d5f074d c0045b84a1459e9f
3fa0e33ad3610907 bff96fb61f64d178 c0045b84a1459e9f c0020d8ee1beaa2d
bfe69c89c2387a52 bfdae68b8172afa7 3fd0fb9637dd9759 bff96fb61f64d178
c0045b84a1459e9f c0020d8ee1beaa2d bfd956b941df368d bfe623825b06b64b
c002681b2d3d4c32 bfddec199969bf34 3fc85c0943f00f7e c0024acad79ce1a9
3fc98b69da852aba c0045b84a1459e9f c0024acad79ce1a9 bfb239188098bc00
c0045b84a1459e9f c0024acad79ce1a9 3fc98b69da852aba c0045b84a1459e9f
c002681b2d3d4c32 bff7fafd30336852 3fc85c0943f00f7e c0020d8ee1beaa2d
3fc98b69da852aba c0045b84a1459e9f 3fb31984ac9f77cf bff96fb61f64d178
c0045b84a1459e9f 3fd0fb9637dd9759 bff96fb61f64d178 c0045b84a1459e9f
c0022f7d98162400 3fc060a92b70196b c0045b84a1459e9f c0022f7d98162400
3fc98b69da852aba c0045b84a1459e9f c0024acad79ce1a9 3fc060a92b70196b
c0045b84a1459e9f bff39abcf2abec95 bfe966276df41331 c0045b84a1459e9f
c002681b2d3d4c32 bfd12d7ee38b8380 bff74ab3f990cd57 3fb31984ac9f77cf
bff479002d6c4864 c0045b84a1459e9f c0024acad79ce1a9 3fc98b69da852aba
c0045b84a1459e9f c002681b2d3d4c32 bfddec199969bf34 3fc85c0943f00f7e
3fb31984ac9f77cf bff96fb61f64d178 c0045b84a1459e9f c0020d8ee1beaa2d
bfe69c89c2387a52 bfdae68b8172afa7 c002681b2d3d4c32 bff7fafd30336852
c00323b1da935462 c0020d8ee1beaa2d 3f3c68dd91669e66 c0045b84a1459e9f
bff521a6c3f3f315 bfbc7295553ca2ed c0045b84a1459e9f bffde0bd47fc5c5f
bfd1c24a13f95c79 c0045b84a1459e9f c0022f7d98162400 3fc060a92b70196b
c0045b84a1459e9f c0020d8ee1beaa2d 3fc060a92b70196b c0045b84a1459e9f
bfd2b13649b96783 bfbc7295553ca2ed c0045b84a1459e9f bfceebc18b0359a6
bff0865231bbd6f5 c0045b84a1459e9f c0020d8ee1beaa2d bfc9303d1801f5b2
bffa3d630a11be02 c002681b2d3d4c32 bfd12d7ee38b8380 c00323b1da935462
bfceebc18b0359a6 bfe71b77422ecedd c0045b84a1459e9f c002681b2d3d4c32
3f3c68dd91669e66 c0045b84a1459e9f bfceebc18b0359a6 bff0865231bbd6f5
c0045b84a1459e9f c002681b2d3d4c32 bff7fafd30336852 3fc85c0943f00f7e
c002681b2d3d4c32 bfc9303d1801f5b2 c00323b1da935462 c001c264302c04de
3fc060a92b70196b c0045b84a1459e9f 3fa0e33ad3610907 bff96fb61f64d178
c0045b84a1459e9f bffcdeaa33ecdb9f 3fc060a92b70196b c0045b84a1459e9f
c001c264302c04de 3fc060a92b70196b c0045b84a1459e9f c001c264302c04de
3fc98b69da852aba c0045b84a1459e9f c002681b2d3d4c32 bfd12d7ee38b8380
bffbbeada1ddca4d bffb623d40022104 3f3c68dd91669e66 c0045b84a1459e9f
c002681b2d3d4c32 3fc98b69da852aba c0045b84a1459e9f c002681b2d3d4c32
bfd12d7ee38b8380 c00323b1da935462 c002681b2d3d4c32 bfe72b1234d7847e
c00323b1da935462 c001c264302c04de 3fc98b69da852aba c0045b84a1459e9f
c002681b2d3d4c32 bfd956b941df368d 3fc85c0943f00f7e c002681b2d3d4c32
bff7fafd30336852 bfc531a5953df9cb bff4e62aecce7b3c bfe71b77422ecedd
c0045b84a1459e9f c0020d8ee1beaa2d bfbc7295553ca2ed c0045b84a1459e9f
3fb31984ac9f77cf bff96fb61f64d178 c0045b84a1459e9f c002681b2d3d4c32
bfddec199969bf34 3fc85c0943f00f7e c001c264302c04de 3fc060a92b70196b
c0045b84a1459e9f c002681b2d3d4c32 bff7fafd30336852 3fc85c0943f00f7e
3fd0fb9637dd9759 bfeeba5a3f051a3b c0045b84a1459e9f c002681b2d3d4c32
bfd12d7ee38b8380 bff479b823983d98 c001c264302c04de 3fc98b69da852aba
c0045b84a1459e9f 3fd0fb9637dd9759 bff96fb61f64d178 c0045b84a1459e9f
3fb31984ac9f77cf bff96fb61f64d178 c0045b84a1459e9f bfd2b13649b96783
bfbc7295553ca2ed c0045b84a1459e9f c0022f7d98162400 3fc060a92b70196b
c0045b84a1459e9f c002681b2d3d4c32 3fc98b69da852aba c0045b84a1459e9f
c0020d8ee1beaa2d bfd12d7ee38b8380 bfe623825b06b64b bfd2b13649b96783
3fc98b69da852aba c0045b84a1459e9f bff629d1ccaf74ad bfdaaa313d5f074d
c0045b84a1459e9f c0024acad79ce1a9 3f3c68dd91669e66 c0045b84a1459e9f
3fb31984ac9f77cf bff96fb61f64d178 c0045b84a1459e9f c002681b2d3d4c32
bfd12d7ee38b8380 3fc85c0943f00f7e c0024acad79ce1a9 3fc060a92b70196b
c0045b84a1459e9f bffc6a6848bda29c bfdaaa313d5f074d c0045b84a1459e9f
bfceebc18b0359a6 bff0865231bbd6f5 c0045b84a1459e9f c002681b2d3d4c32
bfd956b941df368d 3fc85c0943f00f7e c002681b2d3d4c32 bff7fafd30336852
bfe983fbcc8fddc0 3fd0fb9637dd9759 bff96fb61f64d178 c0045b84a1459e9f
c0024acad79ce1a9 bfb239188098bc00 c0045b84a1459e9f c0024acad79ce1a9
bfb239188098bc00 c0045b84a1459e9f bffcdeaa33ecdb9f 3fc060a92b70196b
c0045b84a1459e9f 3fb31984ac9f77cf bff96fb61f64d178 c0045b84a1459e9f
bff4e62aecce7b3c bfe71b77422ecedd c0045b84a1459e9f bff3c7839f0c885d
3fc98b69da852aba c0045b84a1459e9f bffb623d40022104 3f3c68dd91669e66
c0045b84a1459e9f bfe76870d6d3fcc5 3fc98b69da852aba c0045b84a1459e9f
c002681b2d3d4c32 bfefd50fbb8be6ac 3fc85c0943f00f7e c0020d8ee1beaa2d
bff7fafd30336852 bfdae68b8172afa7
"""

comptime _MULTI_PRED = """
3fb0b9f314e21abf 3fc2a6a2c289f917 3fe93f18ecc13e63 3fc3396c38f353eb
3fe900320ad82872 3fb18b9737581491 3fbe8a465cf17c66 3fea5c97593a805b
3fad21fdb2750187 3fe95a7364cd19c1 3fc332c51308dd34 3fad8db5670aef2e
3fc02bd1f8f25575 3fd1408160240fe7 3fe354cad1b162af 3fdbc8bf514cd3c8
3fe0a5c4abeb07c2 3fa75dbab6e8e59a 3fbcc7a6b9ff06fe 3feaafc9073d1ff3
3fab7422182ff2be 3fe95a7364cd19c1 3fc332c51308dd34 3fad8db5670aef2e
3fc25a70a98d7af7 3fe96ad6f7285537 3fafe8cde744c0b3 3faf6f42d77234cf
3fddfdb61da1b925 3fde146187700042 3fc3f71668880fb9 3fe8792626bc6238
3fb448a1f90cceda 3faa7ceb3d692d3c 3fd4b56375034cb7 3fe3fd7f91a7c6d1
3fb0b9f314e21abf 3fc2a6a2c289f917 3fe93f18ecc13e63 3fb0b9f314e21abf
3fc2a6a2c289f917 3fe93f18ecc13e63 3fbe8a465cf17c66 3fea5c97593a805b
3fad21fdb2750187 3fb0b9f314e21abf 3fc2a6a2c289f917 3fe93f18ecc13e63
3fb252de759538ce 3febea2b593904c8 3facb78d814541d0 3fe29f395a177768
3fd7089d2fc0760c 3fadc780e084d91a 3fd4992aa7ecd2a1 3fe364517e4ba409
3fb278c96def9537 3fb75b1d6748321d 3fe6488a1802ec8b 3fcb3048ec5034c2
3fb3c23bafef40fa 3feb9f4e77539d08 3fae86a12ae7ad82 3fb4e4e1a78f89c6
3fe055cec38b4cc6 3fda1b2a0f058402 3fb29196c12a80ce 3febe2cd8f8cf5ca
3facaff984dba1bb 3fc72e652b39cc6b 3fe819b7d53440a1 3fb0d576ffea6217
3fc7630ed9b9e421 3fe892c34d1ad718 3fa9478fc76afdf3 3fe7a331297c2a93
3fcbc20ba357596c 3fa6c4bedadff125 3fb95128ce5edc1c 3fe9f29e7e742dd3
3fb719e33dffb54e 3fb80cf1a89250ac 3feabd63ea2847c6 3fb207ef062b711c
3fae89653c6c186a 3fcc2ec7a60b0d53 3fe70bb7c2b67b24 3fc25a70a98d7af7
3fe96ad6f7285537 3fafe8cde744c0b3 3fe523879b1b75a2 3fd0f10952add50c
3fb31f9ddc6cfec4 3fc72e652b39cc6b 3fe819b7d53440a1 3fb0d576ffea6217
3fe16f3f62f37cfd 3fd9a5546fb4cf32 3fabe1665321b6a1 3fbad1ef6db8323e
3fe996a288c71765 3fb878fc4c0f1293 3fbb36c42581ee78 3fcd0863f786a015
3fe5570e7d6e1a2b 3fea4d531f82918c 3fc07540266c3d03 3fa955cd6e25f346
3fc23ff00c10bf35 3fe6953c95de2d07 3fc36b1d9c768cae 3faa7ceb3d692d3c
3fd4b56375034cb7 3fe3fd7f91a7c6d1 3fea4d531f82918c 3fc07540266c3d03
3fa955cd6e25f346 3fb6a548393465bc 3feb0c0da9720fb0 3fb0fa4a7b3b1cc8
3fb0b9f314e21abf 3fc2a6a2c289f917 3fe93f18ecc13e63 3fc3b0c4784167b2
3fe85da412e61efa 3fb5b156784c38c2 3fe95a7364cd19c1 3fc332c51308dd34
3fad8db5670aef2e 3fb50201b7d0bc52 3feb79ead31180fc 3fae5d4f5f46778e
3fb400cfb9b7219a 3feb97fd568cde5a 3fae7e8b23c3d72a 3fe80aac6b601506
3fc8d3badef5aa8f 3fac064dce280570 3fb6a548393465bc 3feb0c0da9720fb0
3fb0fa4a7b3b1cc8 3fd3c9c822affbf6 3fe37615975e91bb 3fb52832ba4b8250
3faa7ceb3d692d3c 3fd4b56375034cb7 3fe3fd7f91a7c6d1 3fb7682973cbc1b8
3fead063085c8cdd 3fb214be494fd75b 3fcd0229339ac57f 3fe61d5d52b209a0
3fb510c3033a2807 3fe29f395a177768 3fd7089d2fc0760c 3fadc780e084d91a
3fb37fd2e5959a5b 3feba712448771cf 3fae8f35ec5dae4a 3fc8b76ed8e38ab8
3fe66ac97a00e724 3fbb3ad67e31b166 3fbd3f5c59d761b8 3fe99a68d82ab085
3fb5ed5ce4d31a21 3fc0907ef4c2bbad 3fd27869972f7598 3fe29fab77379649
3fd85c832defb853 3fe289f12d8efbfa 3fa47cd3b7927db4 3fea4d531f82918c
3fc07540266c3d03 3fa955cd6e25f346 3fb252de759538ce 3febea2b593904c8
3facb78d814541d0 3fb3c23bafef40fa 3feb9f4e77539d08 3fae86a12ae7ad82
3fe95a7364cd19c1 3fc332c51308dd34 3fad8db5670aef2e 3fc72e652b39cc6b
3fe819b7d53440a1 3fb0d576ffea6217 3fe95a7364cd19c1 3fc332c51308dd34
3fad8db5670aef2e 3fd4992aa7ecd2a1 3fe364517e4ba409 3fb278c96def9537
3fe921a72a5bc01a 3fc3d6d332a4d8db 3fae8a408fb09af2 3fb55c425b75d256
3fd928f5936af4bc 3fe0bffceadbcb57 3fea4d531f82918c 3fc07540266c3d03
3fa955cd6e25f346 3fb4f772ee97688c 3fe0d8e955406146 3fd9105099d96351
3faa7ceb3d692d3c 3fd4b56375034cb7 3fe3fd7f91a7c6d1 3fb29196c12a80ce
3febe2cd8f8cf5ca 3facaff984dba1bb 3fb7682973cbc1b8 3fead063085c8cdd
3fb214be494fd75b 3fb29196c12a80ce 3febe2cd8f8cf5ca 3facaff984dba1bb
3fb0b9f314e21abf 3fc2a6a2c289f917 3fe93f18ecc13e63 3fb3171b2fe44b24
3febd31f1ec7ecd8 3fac9fd7b3b89c1e 3fe95a7364cd19c1 3fc332c51308dd34
3fad8db5670aef2e 3fea4d531f82918c 3fc07540266c3d03 3fa955cd6e25f346
3fb400cfb9b7219a 3feb97fd568cde5a 3fae7e8b23c3d72a 3fb2ccb2ba535b31
3febdbdc554bffc0 3faca8d536994d89 3fb3c23bafef40fa 3feb9f4e77539d08
3fae86a12ae7ad82 3fd6ccf20f25eeae 3fe18d706296601e 3fb860b4aeb5445a
3fb75b1d6748321d 3fe6488a1802ec8b 3fcb3048ec5034c2 3fe80aac6b601506
3fc8d3badef5aa8f 3fac064dce280570 3fb29196c12a80ce 3febe2cd8f8cf5ca
3facaff984dba1bb 3faa7ceb3d692d3c 3fd4b56375034cb7 3fe3fd7f91a7c6d1
3fe95a7364cd19c1 3fc332c51308dd34 3fad8db5670aef2e 3fb55c425b75d256
3fd928f5936af4bc 3fe0bffceadbcb57 3fcee603ff2d4e82 3fe13a1363dbf982
3fcc31ae7162cb76 3fb6a548393465bc 3feb0c0da9720fb0 3fb0fa4a7b3b1cc8
3fcb8cadc9c607e7 3fe7165a30aae22b 3fb033d2e71cded4 3fc3f71668880fb9
3fe8792626bc6238 3fb448a1f90cceda 3fb400cfb9b7219a 3feb97fd568cde5a
3fae7e8b23c3d72a 3fb44f9342ff3c54 3feb8ec7bc43ac4f 3fae745db5c6c257
3fdbc8bf514cd3c8 3fe0a5c4abeb07c2 3fa75dbab6e8e59a 3fe49a25ddc00c9f
3fd2ad6f7406e7b1 3fb0791341e3fc43 3fb7ee5f23b8a17b 3fe7776d8a62aece
3fc62b1a4498f40d 3fbad1ef6db8323e 3fe996a288c71765 3fb878fc4c0f1293
3fe29f395a177768 3fd7089d2fc0760c 3fadc780e084d91a 3fb5bfda54166ab0
3feb26a54029ebff 3fb10afbaa9a355a 3fe49a25ddc00c9f 3fd2ad6f7406e7b1
3fb0791341e3fc43 3fb0b9f314e21abf 3fc2a6a2c289f917 3fe93f18ecc13e63
3fb95128ce5edc1c 3fe9f29e7e742dd3 3fb719e33dffb54e 3fb50201b7d0bc52
3feb79ead31180fc 3fae5d4f5f46778e 3fe921a72a5bc01a 3fc3d6d332a4d8db
3fae8a408fb09af2 3fbe8a465cf17c66 3fea5c97593a805b 3fad21fdb2750187
3fb50201b7d0bc52 3feb79ead31180fc 3fae5d4f5f46778e 3fb3bfb69ae9a231
3febbf51a43d648c 3fac8b78865672d4 3fb8a0720f0ac6e1 3fe77eee1dba2fa0
3fc5b40e8191de12 3fc25a70a98d7af7 3fe96ad6f7285537 3fafe8cde744c0b3
3fb252de759538ce 3febea2b593904c8 3facb78d814541d0 3fbad1ef6db8323e
3fe996a288c71765 3fb878fc4c0f1293 3fc2f50d04e9e6a0 3fe6efaabc5e6af6
3fc14c48099c6d84 3fb3bfb69ae9a231 3febbf51a43d648c 3fac8b78865672d4
3fa9ddd745e9ce00 3fd5b961f938b651 3fe385718f0507f7 3fb5e67b03800ff5
3fc86b91a29da922 3fe7284c36e893b9 3fd4c1fe8f9a00d8 3fe29cf34c2d63a5
3fb8106b602cdf74 3fb8dd311c436fe1 3fea8fd7f478c71c 3fb2a40f3ff6573b
3fe95a7364cd19c1 3fc332c51308dd34 3fad8db5670aef2e 3faa7ceb3d692d3c
3fd4b56375034cb7 3fe3fd7f91a7c6d1 3fb50201b7d0bc52 3feb79ead31180fc
3fae5d4f5f46778e 3fb0b9f314e21abf 3fc2a6a2c289f917 3fe93f18ecc13e63
3fe7a331297c2a93 3fcbc20ba357596c 3fa6c4bedadff125 3fb67047d98234a9
3fe5687d48999b10 3fcf25e6f0d8796c 3fb3bfb69ae9a231 3febbf51a43d648c
3fac8b78865672d4 3fea4d531f82918c 3fc07540266c3d03 3fa955cd6e25f346
3fe95a7364cd19c1 3fc332c51308dd34 3fad8db5670aef2e 3fdbc8bf514cd3c8
3fe0a5c4abeb07c2 3fa75dbab6e8e59a 3fb400cfb9b7219a 3feb97fd568cde5a
3fae7e8b23c3d72a 3fb252de759538ce 3febea2b593904c8 3facb78d814541d0
3fb390cddb171dd6 3fe1dc1b35f750f6 3fd763961d4b969e 3fd75b53b18c5f1a
3fe3180d8e5fd0f3 3fa3a4898d9ff7fb 3fd036209280743e 3fe55a1795bbe57d
3fb456c1081f0325 3fb60936203b4c07 3feb1e249106334d 3fb105a557931990
3fe95a7364cd19c1 3fc332c51308dd34 3fad8db5670aef2e 3fa8b9a22884e59c
3fd796f61a72092d 3fe2a8ead03ead0f 3fb3c23bafef40fa 3feb9f4e77539d08
3fae86a12ae7ad82 3fc7e4960d6ef792 3fe741ec66378311 3fb62770b365f861
3fe49a25ddc00c9f 3fd2ad6f7406e7b1 3fb0791341e3fc43 3fa9ddd745e9ce00
3fd5b961f938b651 3fe385718f0507f7 3fc0907ef4c2bbad 3fd27869972f7598
3fe29fab77379649 3fea4d531f82918c 3fc07540266c3d03 3fa955cd6e25f346
3fb7682973cbc1b8 3fead063085c8cdd 3fb214be494fd75b 3fb7682973cbc1b8
3fead063085c8cdd 3fb214be494fd75b 3fbe8a465cf17c66 3fea5c97593a805b
3fad21fdb2750187 3fe95a7364cd19c1 3fc332c51308dd34 3fad8db5670aef2e
3fd4c1fe8f9a00d8 3fe29cf34c2d63a5 3fb8106b602cdf74 3fc7630ed9b9e421
3fe892c34d1ad718 3fa9478fc76afdf3 3fc25a70a98d7af7 3fe96ad6f7285537
3fafe8cde744c0b3 3fd14bb19ed6d2dd 3fe5f10038f764ec 3fa6926f79d31a3e
3fae89653c6c186a 3fcc2ec7a60b0d53 3fe70bb7c2b67b24 3fbb36c42581ee78
3fcd0863f786a015 3fe5570e7d6e1a2b
"""

comptime _BAG_INTS = """
8 15 0 13 1 2 1 16 5 6 1 19
3 4 0 26 7 8 -1 -1 -1 -1 0 7
9 10 0 9 13 14 2 23 11 12 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 15 0 17 1 2 1 11 5 6 1
15 3 4 0 23 11 12 0 23 9 10 0
9 13 14 0 5 7 8 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 15 0 8 1 2 1 21 5 6
1 19 3 4 1 8 7 8 0 19 11 12
-1 -1 -1 -1 -1 -1 -1 -1 0 18 13 14
0 19 9 10 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 15 0 13 1 2 1 12 5
6 0 28 3 4 1 16 7 8 -1 -1 -1
-1 0 5 11 12 0 3 9 10 -1 -1 -1
-1 1 26 13 14 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 15 0 17 1 2 1 14
5 6 0 27 3 4 1 13 7 8 -1 -1
-1 -1 1 4 9 10 2 28 11 12 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 1 24 13 14 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 15 0 6 1 2 1
15 11 12 1 10 3 4 0 23 5 6 2
22 7 8 -1 -1 -1 -1 -1 -1 -1 -1 0
22 9 10 -1 -1 -1 -1 -1 -1 -1 -1 1
24 13 14 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 15 0 12 1 2
2 23 5 6 1 9 3 4 3 7 9 10
1 26 7 8 1 28 13 14 -1 -1 -1 -1
3 18 11 12 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 15 1 20 1
2 0 2 5 6 0 19 3 4 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 3 18 7
8 1 9 9 10 -1 -1 -1 -1 2 11 13
14 0 21 11 12 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1
"""

comptime _BAG_VALS = """
3f83d5f33c333333 bfec350240bd70a4 3fe46351798c2b45 3ff00f8948e4c416
bfb3e4199e1eb852 bfd9eeaaee124925 bff6c475621642c8 3fe6afa2a2200000
3ff87e00a4000000 bfe63896e0888889 bfb0fe0110924925 3fe10c643ed55555
3ff1997f05555555 bff949a44aaaaaab bfe6d2ff85555555 3f75d410a42c8591
bfe25828e16e5847 3fe2aeb872dee584 3fee0a13eac00000 3fc21f4dbb097b42
3fb1f11798800000 bfe993fcb42fa0bf bff30388fbbbbbbc bfe241b406c234f7
bfc0347db50f0f0f 3fe161abe745d174 3fe56cc94ad2d2d3 3ff2a848fb000000
bfcaa6041b6db6db 3fd080f14699999a bfac8528e4f4ac2d bfe701739fa83a84
3fce92252a474a88 3fdac81116a2e8ba bfc53c132feb851f bfda6434378e38e4
bfef899c0c71c71c 3fe3387d85a12f68 3fce012f9058469f bf9ccd35b6276276
3fdaf7fee6a5a5a6 bfd5f4b47baaaaab 3fb73cf59a2e8ba3 3fdbd590e9000000
3fe8b0b9d1555555 3f9373690986d6f6 bfd5c71cc073ecae 3fd0a053d8951034
3fc2b3fc56b00000 3fe657d2a8000000 bf8e6b88af286bca bfe03b19fb333333
3fd3ded8becccccd bfa9dbc1eeeeeeef bfea37abd5555555 bfd89a09b12f684c
bfd0d75366000000 3fc371cb01555555 3fafd9293999999a bfd00dba0745d174
3f7ba16540000000 bfcfc40c6838e38e 3fd1d859aa4e1a09 3fbefef8c8000000
3fe29292dc666666 bf9a53f805249249 bfda83e7b0471c72 3fd3178b9f000000
3f66fad666666666 3fccc8407c71c71c bfc18f3538e66666 bfde97f3adcccccd
bfb4fedf19249249 bfd3e5ce8d924925 bfe2cd188c3c3c3c 3f313d2920d20d21
bfd51298df5b6db7 3fba4fafe22d82d8 3fd3d6f95f000000 bf87401d8ea4e1a1
3fc5f96a7cf3cf3d 3fe0d6d892aaaaab bfb0567b72b21643 3fc4b779e1249249
bfc34f01867ae148 3fa371d4268ba2e9 bfc715cc4c4ec4ec bfdb7fb087e00000
3fbd99378c800000 bfc28c8682492492 3f5e2e2773b06ada bfc55f4fdbcf3cf4
3fb8fe491f300376 3fd1cd0fc537a6f5 3f904b093dd63b14 bfcc790c85f15f16
3fb8ba6710000000 3faf21a385936536 bfc6e0302dcccccd 3fbade5a40000000
3fd87252ef6db6db bf6e7383526c9249 3fc5bfc275000000 bfc6087b2d555555
bfdbf685eaaaaaab 3f806de912dc11f7 3fb121f837460dd6 bfb93654749f3832
bfc7b2e7a12aaaab 3f971388c971c71c bfc2bd0bd4000000 3fb765c4bc17e16f
3fa49a3348f11dc4 3fc67e623647ae14 3fbebc377de79e7a bfa19b9625e42c86
bfbc5a79c5249249 3fb372243539999a 3fa1dad7e3555555 3fcaea5fe6666666
"""

comptime _BAG_RAW = """
40038503bc2a1568 3fe2a54db3cc43ee 3fda7e2351c9b3de bfea0b27a0254138
3ff338deb8a8b7a7 bfa58e26ca1dedf8 3ffb794ae790d359 3fd6b38965e4e6a1
3ff7ce996cd24ffe 3fe89a5e4f1e579d bff0e9172dfd7dc2 3fc9ba2a342109eb
3ffb5bfbdf81ecf7 3ff841d42e4c5095 3ff25387145e0cfc 3fba8a2d50710640
bfc002f805de64b0 3fd0015d00876d55 3fc244f615909b7b 3fe3763f6c50fe64
3fe20804112c0b90 bfb219a0765546d9 3ff6811bda15a9d1 3ff44615c3e25ee2
3ffba54526467f41 3ff4255f84ec4fc7 3ff69ffd34a350a2 3ff4fc896a4f91d2
3ff2aa7e55c88f92 3ffacdbb93c441d1 3fe817123bcfa61f 400081fb31474d16
4001758ad4bab71f 3fc93ae36c9d6848 3fd6beba93968a29 3fe2e520897fe916
4003727f83475869 3fe9e44938e0ab28 3ff65d44549fa140 3ff089fd83dd11e4
bfe284600ce9301c bff0e9172dfd7dc2 3ffa0e2c756695e8 3ffc22234e1faf87
40021d85714a8049 4003727f83475869 3ffb855d8702326f 3fecb97dfdf5a151
bfec9f94ba8ba79e bfef47704f6161eb 3feb178eb06201fd 3fe0611f578c3bb9
4003071cb8aead58 40001bfd595b75ab 3fe66825da2c6c1b 3ff2d0653c373d42
3ff43242e0421281 3fb82372a41caa76 3ff6dbf005746b95 3fe03f8e1d7735cd
3fec50a8de899797 3feb709bcc9ae9b1 bfe297ce87588403 3ff6dbf005746b95
3ffb6a54c2d01d6f bfd319a70f813950 bfeb1aa49d287e5d 40012f4a923efb2a
3fc244f615909b7b 3fdc5d37d9b5b948 bfa58e26ca1dedf8 3ffc399ed461cd60
3fe84eb4058627cd 3ff55007fb83be4c bfddbd8a58ef3259 3fe2caa3b6cd1a75
4003de444c887721 bfca2df03844cb16 3ff4062222448eb0 3fecb97dfdf5a151
3fe9d05fe5e5b4d9 3fda7e2351c9b3de bfecb30334fafb85 bfef47704f6161eb
4001e440deb2fac9 3fba2c04ec57b02e 3ff15f59558ca22d 40021d85714a8049
bfa1e0fc914bc40c 3ff55007fb83be4c 4004733e9b359a87 40009ecdc9c95665
3ffe5e429e734c0e 3ff732a0264f3189 400239ecce986307 4004887829b62dfc
3fed0b93d73447e0 bfd8a80e3fbbff26 400271d1c0cfea52 bfe20942dc475f71
3fc93ae36c9d6848 3fecd62fa6bc18df bfe6bfb07628746b 3fd47aa5c11fdb0d
bfe27f01444021b5 3ffba54526467f41 bfe23b4ed83a898d bfc253a35f776e25
40007288941c00ee 3fe817123bcfa61f 3ff9300a92a441d1 4000bb3527173923
3fe1864197bc4c7e 3ff65a9f7202ae31 3fea7726bccde17a bfd8a80e3fbbff26
4002d7b45b326754 3ff19efcc556f89a 3ff1ad97f5ffe12f bfdef67906546137
3fd015ffe5fbee2a bfeda562a9c217f7 3fc244f615909b7b 3ffacdbb93c441d1
3fe9e44938e0ab28 3ff15f59558ca22d 3ff890046e3a6af3 3fdc0fc926cd16fb
3fefd847a4ecaa16 3ff43242e0421281 3fecb97dfdf5a151 3ff51f0ecea7837a
3fe367a9659a103d bfe6bfb07628746b 4004733e9b359a87 3fd0015d00876d55
bfdfae4b5ed4ced1 3fbfe29e91e65118 3fefd847a4ecaa16 bfe284600ce9301c
4004733e9b359a87 3ff49a0c2ced65c5 3ffc22234e1faf87 3fe6cf7ec8ee02c4
3f81410a3e2689f4 bfc002f805de64b0 3f8116ffdccf1772 bfe27f01444021b5
bfc3cb0a73954db1 bfeb1aa49d287e5d 400191f2320899dd 3ffe5e429e734c0e
4000133a8a876ff9 bfddbd8a58ef3259 3fe1ee7a7f1a6fd1 4003b15095dc6433
3ffe14f957aeb9c4 3ff4e792462fd874 3faa2028bfb9cb04 3ff1e462028f1e0d
3ffaacf62e612915 3fffb2aa58ceb9c4 bfd7b2b995b83eb2 3ff089fd83dd11e4
3ff0c2cc3e78d760 3ff44615c3e25ee2 3ff089fd83dd11e4 3fe9c1c9df2ec6b2
3ff2917878103494 3ff44615c3e25ee2 3fed2fca02adba97 3ffae9a99b423b0a
3fd6b38965e4e6a1 bfec9f94ba8ba79e 3f81410a3e2689f4 3fe3763f6c50fe64
bfa106f20216643b 3ff83057bfcfd94b 3ffacdbb93c441d1 bfe6bfb07628746b
3ffa1c5547f4f7f8 3fcea39718a61c81 3ffb966bad7dbc9f bfb4788dd2cefd3e
3fcea39718a61c81 3fd749dc9e530e40 bfe71eedbf254138 3fba8a2d50710640
3ffa84724cffaf87 3ff7d11113353c27 bfe7d8a43573e222 4000bb3527173923
4003de444c887721 bfdc56630fa9305b 3ff368f84e5b7a40 3fd489411db3ee87
3fd015ffe5fbee2a 3fe9e44938e0ab28 3ffa079425267f41 bfec9f94ba8ba79e
"""

comptime _BAG_PRED = """
40038503bc2a1568 3fe2a54db3cc43ee 3fda7e2351c9b3de bfea0b27a0254138
3ff338deb8a8b7a7 bfa58e26ca1dedf8 3ffb794ae790d359 3fd6b38965e4e6a1
3ff7ce996cd24ffe 3fe89a5e4f1e579d bff0e9172dfd7dc2 3fc9ba2a342109eb
3ffb5bfbdf81ecf7 3ff841d42e4c5095 3ff25387145e0cfc 3fba8a2d50710640
bfc002f805de64b0 3fd0015d00876d55 3fc244f615909b7b 3fe3763f6c50fe64
3fe20804112c0b90 bfb219a0765546d9 3ff6811bda15a9d1 3ff44615c3e25ee2
3ffba54526467f41 3ff4255f84ec4fc7 3ff69ffd34a350a2 3ff4fc896a4f91d2
3ff2aa7e55c88f92 3ffacdbb93c441d1 3fe817123bcfa61f 400081fb31474d16
4001758ad4bab71f 3fc93ae36c9d6848 3fd6beba93968a29 3fe2e520897fe916
4003727f83475869 3fe9e44938e0ab28 3ff65d44549fa140 3ff089fd83dd11e4
bfe284600ce9301c bff0e9172dfd7dc2 3ffa0e2c756695e8 3ffc22234e1faf87
40021d85714a8049 4003727f83475869 3ffb855d8702326f 3fecb97dfdf5a151
bfec9f94ba8ba79e bfef47704f6161eb 3feb178eb06201fd 3fe0611f578c3bb9
4003071cb8aead58 40001bfd595b75ab 3fe66825da2c6c1b 3ff2d0653c373d42
3ff43242e0421281 3fb82372a41caa76 3ff6dbf005746b95 3fe03f8e1d7735cd
3fec50a8de899797 3feb709bcc9ae9b1 bfe297ce87588403 3ff6dbf005746b95
3ffb6a54c2d01d6f bfd319a70f813950 bfeb1aa49d287e5d 40012f4a923efb2a
3fc244f615909b7b 3fdc5d37d9b5b948 bfa58e26ca1dedf8 3ffc399ed461cd60
3fe84eb4058627cd 3ff55007fb83be4c bfddbd8a58ef3259 3fe2caa3b6cd1a75
4003de444c887721 bfca2df03844cb16 3ff4062222448eb0 3fecb97dfdf5a151
3fe9d05fe5e5b4d9 3fda7e2351c9b3de bfecb30334fafb85 bfef47704f6161eb
4001e440deb2fac9 3fba2c04ec57b02e 3ff15f59558ca22d 40021d85714a8049
bfa1e0fc914bc40c 3ff55007fb83be4c 4004733e9b359a87 40009ecdc9c95665
3ffe5e429e734c0e 3ff732a0264f3189 400239ecce986307 4004887829b62dfc
3fed0b93d73447e0 bfd8a80e3fbbff26 400271d1c0cfea52 bfe20942dc475f71
3fc93ae36c9d6848 3fecd62fa6bc18df bfe6bfb07628746b 3fd47aa5c11fdb0d
bfe27f01444021b5 3ffba54526467f41 bfe23b4ed83a898d bfc253a35f776e25
40007288941c00ee 3fe817123bcfa61f 3ff9300a92a441d1 4000bb3527173923
3fe1864197bc4c7e 3ff65a9f7202ae31 3fea7726bccde17a bfd8a80e3fbbff26
4002d7b45b326754 3ff19efcc556f89a 3ff1ad97f5ffe12f bfdef67906546137
3fd015ffe5fbee2a bfeda562a9c217f7 3fc244f615909b7b 3ffacdbb93c441d1
3fe9e44938e0ab28 3ff15f59558ca22d 3ff890046e3a6af3 3fdc0fc926cd16fb
3fefd847a4ecaa16 3ff43242e0421281 3fecb97dfdf5a151 3ff51f0ecea7837a
3fe367a9659a103d bfe6bfb07628746b 4004733e9b359a87 3fd0015d00876d55
bfdfae4b5ed4ced1 3fbfe29e91e65118 3fefd847a4ecaa16 bfe284600ce9301c
4004733e9b359a87 3ff49a0c2ced65c5 3ffc22234e1faf87 3fe6cf7ec8ee02c4
3f81410a3e2689f4 bfc002f805de64b0 3f8116ffdccf1772 bfe27f01444021b5
bfc3cb0a73954db1 bfeb1aa49d287e5d 400191f2320899dd 3ffe5e429e734c0e
4000133a8a876ff9 bfddbd8a58ef3259 3fe1ee7a7f1a6fd1 4003b15095dc6433
3ffe14f957aeb9c4 3ff4e792462fd874 3faa2028bfb9cb04 3ff1e462028f1e0d
3ffaacf62e612915 3fffb2aa58ceb9c4 bfd7b2b995b83eb2 3ff089fd83dd11e4
3ff0c2cc3e78d760 3ff44615c3e25ee2 3ff089fd83dd11e4 3fe9c1c9df2ec6b2
3ff2917878103494 3ff44615c3e25ee2 3fed2fca02adba97 3ffae9a99b423b0a
3fd6b38965e4e6a1 bfec9f94ba8ba79e 3f81410a3e2689f4 3fe3763f6c50fe64
bfa106f20216643b 3ff83057bfcfd94b 3ffacdbb93c441d1 bfe6bfb07628746b
3ffa1c5547f4f7f8 3fcea39718a61c81 3ffb966bad7dbc9f bfb4788dd2cefd3e
3fcea39718a61c81 3fd749dc9e530e40 bfe71eedbf254138 3fba8a2d50710640
3ffa84724cffaf87 3ff7d11113353c27 bfe7d8a43573e222 4000bb3527173923
4003de444c887721 bfdc56630fa9305b 3ff368f84e5b7a40 3fd489411db3ee87
3fd015ffe5fbee2a 3fe9e44938e0ab28 3ffa079425267f41 bfec9f94ba8ba79e
"""

comptime _FFRAC_INTS = """
8 15 1 24 1 2 1 4 3 4 1 30
5 6 -1 -1 -1 -1 3 16 9 10 2 3
7 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 7 13 11 12 3 22 13 14 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 15 0 13 1 2 0 6 5 6 0
21 3 4 9 27 7 8 0 26 9 10 -1
-1 -1 -1 6 4 11 12 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 9 18 13 14 -1 -1 -1 -1 -1
-1 -1 -1 15 0 13 1 2 1 13 3 4
1 18 5 6 -1 -1 -1 -1 0 8 9 10
0 24 11 12 0 18 7 8 -1 -1 -1 -1
3 10 13 14 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 15 1 12 1 2 1 4 7
8 1 25 3 4 2 12 5 6 1 29 9
10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 10 24 11 12 -1 -1 -1
-1 10 5 13 14 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 15 0 13 1 2 0 4
5 6 0 24 3 4 7 12 9 10 -1 -1
-1 -1 -1 -1 -1 -1 2 12 7 8 -1 -1
-1 -1 -1 -1 -1 -1 0 16 11 12 -1 -1
-1 -1 -1 -1 -1 -1 7 7 13 14 -1 -1
-1 -1 -1 -1 -1 -1 15 1 15 1 2 1
1 7 8 1 27 3 4 3 26 5 6 -1
-1 -1 -1 1 19 9 10 -1 -1 -1 -1 -1
-1 -1 -1 9 19 11 12 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 5 21 13 14 -1
-1 -1 -1 -1 -1 -1 -1 15 1 12 1 2
-1 -1 -1 -1 1 25 3 4 7 21 5 6
6 17 9 10 7 8 7 8 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 9 19 13 14
7 19 11 12 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 15 0 17 1
2 0 4 3 4 0 26 5 6 -1 -1 -1
-1 0 12 7 8 10 11 13 14 -1 -1 -1
-1 8 18 9 10 -1 -1 -1 -1 -1 -1 -1
-1 7 6 11 12 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1
"""

comptime _FFRAC_VALS = """
bdecd0e028c19790 3fd1ac4cb6347ae1 bfe97d6ea4c4ec4f 3fee5c08a4000000
3fbeebce886a2577 bfe43c7009c71c72 bff9edac7c000000 bff75baf5e38e38e
bfda7efe1cf914c2 bfbe098903a92492 3fd42ab6c7da5a5a 3fceb90da8f1c71c
bfdbd7b949777777 3fe51aea3e93b13b 3fb97a2302fa0be8 bf683ab19649dee3
bfe5f111a2a00000 3fe71746bdbc14e6 3fdaa5f8f8eaaaab 3fefd49547c3c3c4
bfedb6f2bee8ba2f bfd9e7b87128f5c3 3fdde5bc351dc477 bf913fd0aaaaaaab
3fe895859f2aaaab 3ff273ceaf6db6db bfb0daecd2aaaaab bfdc38bf5a444444
bfe180640a611a7c bfcdfc95b52d2d2d bf535762afc2dd9d bfdf0bd09384ec4f
3fe0655f81e343eb bfa85faf90000000 bfe8fb705b6b5ad7 3fe98b4073db6db7
3fc0677c20d65359 bfd4cce6baaaaaab 3fd2d25dec100000 bfee07a64499999a
bfde3c1b6d90b216 3fe44074da298376 3ff107548b99999a bf74a21451745d17
3fdb89ccc5d1745d bf3500867e687014 3fd4ce49f49d999a bfcb5a9907bb0432
bfaeaba0fd909f96 bfdddb2f81b7a6f5 bfd011c4a2ce4925 3fa8c22c3428f5c3
3fe056bda7436db7 3fcc485cf9c13522 bfd7d95c7157c57c bfe671c993555555
bfde7e56970aaaab bfc12540e3aaaaab bfc7766821c71c72 bfe3921a4c880000
bf5e9bdf028c1979 bfd523b1d929d89e 3fd6303e59249249 3fcb73e2f0000000
3fe1e25ebe7c8a61 bfe0bb44fcd55555 bfcb12f24745d174 bfd634cf8a800000
bfb4e56787507507 3fb2089e7313b13b 3fd3d53399306eb4 bfc2a6730a000000
3fc43108a3ca1af3 3fd27bf168aaaaab bfadfd7ed4000000 bf55618b8a3065e4
3fc9ba3ea6765d97 bfc6424fb3e3d107 bfb931c01b032916 bfd83d30bf34f72c
bfc2eb1d29387878 3fc3017d55800000 3fdbc0711c888889 3fc3bf89bc5df985
bf8e8e13d1745d17 bfca7a3896df51b4 3fb6e8b452e00000 3fcf9254cae00000
3fc495ff76e147ae 3fdefa4a6c000000 3f24532234380a30 3fc548fa9a4ccccd
bfbbd9757cdf79b4 bfa212f002249249 bfcd5e027dbd37a7 3f8ef6dc68c92492
bfc4bda96fa2e8ba bfbd00b598b0c30c 3fb6f06a1f38e38e bfc279f62c492492
bfd5ef83850d7943 bfdc0d0f4edb6db7 bfb00a9db5555555 bfa70787c1c71c72
bfd2ce0c168ba2e9 bf533b711f4898d6 bfc678c3c5b3def8 3fd1ab268a0d20d2
bfd5be654c690690 bfb95c2f68129aca 3fc850ca5cdcdcdd 3fdb133310924925
bfc242aa12e39ce7 3f8aa9bec7ae147b bfc8324d13bacf91 bfb2389e8464ec4f
bfc7c3d45fe9999a 3f3752b569696969 3fd2c6cb2e492492 3fc2cc79e1435e51
"""

comptime _FFRAC_RAW = """
3ff10604c2226a62 bfdbf42444229c3f 3fb8cae8cd944cd8 3ffd5ed9c7d72a4e
bfeaeff86c184782 bfcf4081d1153449 3feefeaf73fa40fa bfe5baf630ebe770
3fddb2e5eee3c757 3fc995d5d839e650 bfd4a76e64c4fa15 bfc4afbff1cfff9f
3ffb9d062f34d6dc bfd56010c1bd394c 3ff3d443cad34f0b 3fd227bff8add96a
bfe457e8e8798905 3feb7098494ae773 3fecc5a4bed90c64 3ffa85b216eba6cc
3fea129a673997a2 3ffb39fd634a37c4 3ff36c59dd55de27 3fee7d8e4fddc445
3ff832b75b0eeab5 bfed069f065db526 bfd15f730b0ce2b7 3ff9390b2a0b065f
3fe3536063ddfd6a 3fd227bff8add96a 3fe65b975bdc929e 3ffa53c95fff31c0
bfece9f87c5469ef 3fef29359cff0f7f 3ff224f58ed3e37b 3ffaf88ddc6d522c
3fdadd8ac49eb9d6 3fd0c91a604341dd 3fd15a4a580b6436 bfe8dc0f5d74f384
3ffcf0afbbbdf5e9 3ffb5b5a5b307a63 bfe18045ae3e0aea 3ffe462168940c10
3fc9d8f2c1f19f03 bff0b5de2e6fef81 3ffb5b5a5b307a63 3fd41eda253b1fc7
3ffb19a4fca11674 3fe8832e5156bd5b 3fd0d41c39179630 3fc59ae6f7e2842a
3fd0c91a604341dd 3ff5a52f981477c8 3ff8a0e316c64066 3fd37bb3662e5ed5
3fc11bc4c4782339 bfc7e1b4695930b8 3fff117816901cd3 3ff265d5a67b650e
3fe6a11dcb70784b 3fc995d5d839e650 3fd42674afc16e56 bfb3bb9017e10e68
3ff088f70c916646 bfc09d512c86349e 3fc9faacc59c2c16 bfecd5e623ca4568
3ff7868cf5c98838 3ffcb8fbce894b8d 3ffa904d2f9aed20 3fb685b5c3ccf10d
3fecac1bd46233f3 3fff117816901cd3 3ff83a97a0cfb107 3fd70e65226057e4
3ff29bfa8cd6d37b 3fd5568c55a3f72d 3ff9dc428494c451 bfdfa38af8009e07
bfd836f69aa6fcfd 3fd6d53f6ac217fe 3ff967efb307b14d 3ff339407154120d
3fbf911709a9d2b5 3ff36c59dd55de27 4001bfcd5b0b08ac 3fee7d8e4fddc445
3fddb4b3300cb3b5 3ffd0df8a9e96ce8 3fe7724af016514d 3ff9f6277b6b996b
3fe49cd3e942c5a4 3fe16d5d96864061 3ff0c09c367bcb14 bff1e99e01fdab3d
3fe3d7c1f6727b0d 3fe31b97015271c0 bfb4bfd92b050e2f 3ffbbfd0689d9d5e
3fbacd9e5b9d43f2 3ff1461f4bfa9b0d 3fbc894b2f8dd12a bfdbd7fb5c44c946
3fe5450edbd53fcd 3ff40ab8e03e1101 3ff2658eab9f3c78 3fee45255698d6a3
bfde09d69203d687 3fd0c91a604341dd 3ffc5110da3affab 3ff27ebc5283460f
bff1e99e01fdab3d 3fea2ff426879808 3fd0d8a613de7a72 3fc7ce964cf8caa5
3ff0370d2f4a524a 3fee7d8e4fddc445 3ff215814c7f1162 3ff2bb60fe7123ed
3fe3cf78e8f5fd68 3fe6a11dcb70784b 3fe73e6d7ddd3e86 bfcf5ef57cfc407d
3fe91e1124fd688e 3fc6c65b6e54c9c8 4001bfcd5b0b08ac 3ff166c234de0ce3
3fe7670c074701e0 bfaf518165cb84d6 3fe4b3afe3df466d bff138175dc47c07
3ffd0df8a9e96ce8 3fedf7c09334a251 3fd9c0fe284b74dc bfd15f730b0ce2b7
3ff2a5a682756831 3fba1680a7ff747f 4001bfcd5b0b08ac 3fef643d2b34f3a6
3ff2d0d2579233c5 3fe31695239b65f4 bfe76e8e6449c34e 3ffd0f05b64c34ec
bfc722facf521f09 3fe57e55667c10e2 3fe04b3de73ffbd6 3fc15e190d40aaed
3fd68d46656ba0d4 3fecb464e1deb198 3ff0e32ed19e64ae 3ff45a96410a2225
bfbf53df2ce7d6b3 bfcdb191a582a923 3fe71d93fce8c5ca 3ff8c8b8e09016ac
3ff5117f2c315d50 3fd0c91a604341dd 3fde72e676777fcd 3ff5d6d871cdf423
3fd1dde45a1bdb0f 3fc6b7689f3730cf 3fa18647df22da30 3fe2774739600448
3ff48575f6d5c6d6 3ffc73db13a3c62d 3fed775fad385db3 bfe9a86c831d526e
3fee54975779a395 3ff967efb307b14d 3ff5c597c717aeaa 3fbf911709a9d2b5
3ffaf88ddc6d522c bfd15f730b0ce2b7 3ffae7f695b59303 3fe7fba868caab40
3ff113363f368b2e bfd71f42e43f81df 3fe614a67fcfc39d bfd595a20ce350e9
3ff951af28a414bb 3ff6db2819636144 3ff2a5a682756831 3ff3d78e32b69a89
3fee9731c59e7b01 bfd50ed9beeae47f 3fdd5b779ba89611 3ffd4b82dae809b8
bfd9f05fc5223030 3fb5bec4c657f478 bfcd1b04895dda17 3ffd65a2100412cb
3ff58a71f2ae39de 3fdd8547a09819dd bff004578a36c04b 3fead58f6c211a2b
3ff68a6b3ce75da5 bfee40cdee54d658 bfd83292c00c5359 bfd5a398a428d07e
"""

comptime _FFRAC_PRED = """
3ff10604c2226a62 bfdbf42444229c3f 3fb8cae8cd944cd8 3ffd5ed9c7d72a4e
bfeaeff86c184782 bfcf4081d1153449 3feefeaf73fa40fa bfe5baf630ebe770
3fddb2e5eee3c757 3fc995d5d839e650 bfd4a76e64c4fa15 bfc4afbff1cfff9f
3ffb9d062f34d6dc bfd56010c1bd394c 3ff3d443cad34f0b 3fd227bff8add96a
bfe457e8e8798905 3feb7098494ae773 3fecc5a4bed90c64 3ffa85b216eba6cc
3fea129a673997a2 3ffb39fd634a37c4 3ff36c59dd55de27 3fee7d8e4fddc445
3ff832b75b0eeab5 bfed069f065db526 bfd15f730b0ce2b7 3ff9390b2a0b065f
3fe3536063ddfd6a 3fd227bff8add96a 3fe65b975bdc929e 3ffa53c95fff31c0
bfece9f87c5469ef 3fef29359cff0f7f 3ff224f58ed3e37b 3ffaf88ddc6d522c
3fdadd8ac49eb9d6 3fd0c91a604341dd 3fd15a4a580b6436 bfe8dc0f5d74f384
3ffcf0afbbbdf5e9 3ffb5b5a5b307a63 bfe18045ae3e0aea 3ffe462168940c10
3fc9d8f2c1f19f03 bff0b5de2e6fef81 3ffb5b5a5b307a63 3fd41eda253b1fc7
3ffb19a4fca11674 3fe8832e5156bd5b 3fd0d41c39179630 3fc59ae6f7e2842a
3fd0c91a604341dd 3ff5a52f981477c8 3ff8a0e316c64066 3fd37bb3662e5ed5
3fc11bc4c4782339 bfc7e1b4695930b8 3fff117816901cd3 3ff265d5a67b650e
3fe6a11dcb70784b 3fc995d5d839e650 3fd42674afc16e56 bfb3bb9017e10e68
3ff088f70c916646 bfc09d512c86349e 3fc9faacc59c2c16 bfecd5e623ca4568
3ff7868cf5c98838 3ffcb8fbce894b8d 3ffa904d2f9aed20 3fb685b5c3ccf10d
3fecac1bd46233f3 3fff117816901cd3 3ff83a97a0cfb107 3fd70e65226057e4
3ff29bfa8cd6d37b 3fd5568c55a3f72d 3ff9dc428494c451 bfdfa38af8009e07
bfd836f69aa6fcfd 3fd6d53f6ac217fe 3ff967efb307b14d 3ff339407154120d
3fbf911709a9d2b5 3ff36c59dd55de27 4001bfcd5b0b08ac 3fee7d8e4fddc445
3fddb4b3300cb3b5 3ffd0df8a9e96ce8 3fe7724af016514d 3ff9f6277b6b996b
3fe49cd3e942c5a4 3fe16d5d96864061 3ff0c09c367bcb14 bff1e99e01fdab3d
3fe3d7c1f6727b0d 3fe31b97015271c0 bfb4bfd92b050e2f 3ffbbfd0689d9d5e
3fbacd9e5b9d43f2 3ff1461f4bfa9b0d 3fbc894b2f8dd12a bfdbd7fb5c44c946
3fe5450edbd53fcd 3ff40ab8e03e1101 3ff2658eab9f3c78 3fee45255698d6a3
bfde09d69203d687 3fd0c91a604341dd 3ffc5110da3affab 3ff27ebc5283460f
bff1e99e01fdab3d 3fea2ff426879808 3fd0d8a613de7a72 3fc7ce964cf8caa5
3ff0370d2f4a524a 3fee7d8e4fddc445 3ff215814c7f1162 3ff2bb60fe7123ed
3fe3cf78e8f5fd68 3fe6a11dcb70784b 3fe73e6d7ddd3e86 bfcf5ef57cfc407d
3fe91e1124fd688e 3fc6c65b6e54c9c8 4001bfcd5b0b08ac 3ff166c234de0ce3
3fe7670c074701e0 bfaf518165cb84d6 3fe4b3afe3df466d bff138175dc47c07
3ffd0df8a9e96ce8 3fedf7c09334a251 3fd9c0fe284b74dc bfd15f730b0ce2b7
3ff2a5a682756831 3fba1680a7ff747f 4001bfcd5b0b08ac 3fef643d2b34f3a6
3ff2d0d2579233c5 3fe31695239b65f4 bfe76e8e6449c34e 3ffd0f05b64c34ec
bfc722facf521f09 3fe57e55667c10e2 3fe04b3de73ffbd6 3fc15e190d40aaed
3fd68d46656ba0d4 3fecb464e1deb198 3ff0e32ed19e64ae 3ff45a96410a2225
bfbf53df2ce7d6b3 bfcdb191a582a923 3fe71d93fce8c5ca 3ff8c8b8e09016ac
3ff5117f2c315d50 3fd0c91a604341dd 3fde72e676777fcd 3ff5d6d871cdf423
3fd1dde45a1bdb0f 3fc6b7689f3730cf 3fa18647df22da30 3fe2774739600448
3ff48575f6d5c6d6 3ffc73db13a3c62d 3fed775fad385db3 bfe9a86c831d526e
3fee54975779a395 3ff967efb307b14d 3ff5c597c717aeaa 3fbf911709a9d2b5
3ffaf88ddc6d522c bfd15f730b0ce2b7 3ffae7f695b59303 3fe7fba868caab40
3ff113363f368b2e bfd71f42e43f81df 3fe614a67fcfc39d bfd595a20ce350e9
3ff951af28a414bb 3ff6db2819636144 3ff2a5a682756831 3ff3d78e32b69a89
3fee9731c59e7b01 bfd50ed9beeae47f 3fdd5b779ba89611 3ffd4b82dae809b8
bfd9f05fc5223030 3fb5bec4c657f478 bfcd1b04895dda17 3ffd65a2100412cb
3ff58a71f2ae39de 3fdd8547a09819dd bff004578a36c04b 3fead58f6c211a2b
3ff68a6b3ce75da5 bfee40cdee54d658 bfd83292c00c5359 bfd5a398a428d07e
"""

comptime _MISSCAT_INTS = """
8 15 2 -1 1 2 0 5 5 6 0 22
3 4 4 23 7 8 2 -1 13 14 -1 -1
-1 -1 4 1 11 12 3 25 9 10 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 15 2 -1 1 2 0 24 7 8 2
-1 3 4 0 8 9 10 0 17 5 6 1
14 13 14 -1 -1 -1 -1 0 11 11 12 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 15 2 -1 1 2 0 29 7 8
2 -1 3 4 0 11 5 6 0 17 9 10
-1 -1 -1 -1 -1 -1 -1 -1 1 17 11 12
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 8 13 14 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 15 2 -1 1 2 0 25 7
8 2 -1 3 4 0 14 11 12 0 18 5
6 1 16 13 14 1 5 9 10 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 15 2 -1 1 2 0 8
9 10 2 -1 3 4 0 22 13 14 0 18
5 6 0 5 11 12 1 12 7 8 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 15 2 -1 1 2 0
23 5 6 2 -1 3 4 0 5 9 10 -1
-1 -1 -1 2 -1 7 8 1 18 13 14 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1
16 11 12 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 15 2 -1 1 2
1 18 5 6 2 -1 3 4 1 16 7 8
1 24 9 10 0 20 11 12 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 0 10 13 14
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 15 2 -1 1
2 1 14 5 6 0 9 3 4 1 16 7
8 1 21 9 10 0 18 11 12 0 23 13
14 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1
"""

comptime _MISSCAT_VALS = """
be073e68701460cc 3ff49bd3fcddc83d bff49bd3fd3a4c0a bff9917fe2f79436
bfd54641c5ec4ec5 3fd21c82af38e38e 3ff7cf2bb2d86186 bff4f4a005d6f182
c001584476400000 bff7d4f97e71745d bfd8d0e630cccccd 3fdc79a351555555
3ff8c687b7c26e2d 3fd3d36e9a4ec4ec bfecf5b06d249249 bf6ce80a67cd0e03
bff66e0d77dc0000 3fe4a3dbe94b2164 3ff49ddedba1642d 3f579c9c457c57c5
bfd6a91c0271c71c 3fd77e1433be2be3 bff936f42ecade30 bfe085cd612aaaab
3fe54410450ccccd 3ff832a60ccccccd bffdab836313b13b bff42d54336db6db
bfe25d1774e38e39 bfc0487d0af286bd bf750da4f065e3fb bfeff25573120000
3fdd270ae52519f9 3fed2a8e1a21642d bf3bb2e3a83a83a8 3fe14a41ed649249
3ff231d28f79e79e bff1413b85056c79 bfa6b523e0000000 bfd03c7f33aaaaab
3fd0a567901d41d4 bff44612203a83a8 bfe8ad239ebd70a4 bffae0bff2492492
bfee4c8b322e8ba3 bf797c968e32f1fd bfe6e103a0000000 3fd4a41af3508591
3fee6ef50be5be5c 3fb2fc79806051ec bfc40c7b6bce7627 3fd453311ca92492
bfe9969b86d61bed bfc6c2326999999a bfc7115404000000 3fd9e8cd7d3f3cf4
3fe6342717333333 3ff2921b60000000 bfd1ebc70fc80000 3fa3e32e1aadb6db
bf6928d9226357e1 bfeced4cace739ce 3fc483a175fd017f 3fe5c288cf7cb7cb
3f7b47931905c641 bfc4ba2869e4b17e 3fcc4512c0a4e1a1 bf633648f999999a
3fd4ec8db70ccccd bff353173ba2e8ba bfe674f7530c30c3 bfd3eecf9c649249
bfb2432f33555555 3fe20aaa4f5ad6b6 3ff0131f8e38e38e bf667e65b2f1fd74
3fd09d6cfb720f35 bfd0f6f48b288df1 bfb7bcc417d4bb7e bfe4d7184b5ad6b6
3fc434faea2a150b 3fe0376ae91ee584 3fd9e13a4b400000 bfa141cfd030c30c
bfd6ab5e45ddddde bf96d1fc3a1af287 bfc33a6fec444444 3fbd96ed2f000000
3fd78033c3ae8ba3 3fea789c60000000 bf52a74379277b8b 3fd1632878e6076c
bfc242e0305cdf47 bfdd946490421084 bfa719671345979d 3fc5f5b6f26e6e6e
3fe0d5e64d0d7943 bfe5ca97aec4ec4f bfd27117e1435e51 bfb7c40f1f4ea68e
3fc2a44d3579e79e 3fb18b8d7cc00000 3fd4fb10a999999a bfcc700c1f5ed098
bf9ea96d82d79436 bf50cd56d6ece541 3fc8db0b9a3b5cc1 bfba2fd40399fc26
bfd04b1c1b0c7ce1 bfa16f612144d134 3fa89a90358469ee 3fd2bd6279ed44af
bfd979fef5266666 bfbcd160f11745d1 bfb49bab3dc2b448 3fbc5df8c3bd37a7
bfb51cc245800000 3fc8ce308c924925 3fcb75f83e555555 3fddb39a28000000
"""

comptime _MISSCAT_RAW = """
400a1f187448db64 400eb57f27cd8516 bfef3e76c0046bc7 4007f67921f5c744
401073140c3612a8 3ffa5690b2f40f60 bfd2fcb75bb8c31e 400a1f187448db64
4010200b36610b58 3fca07762391eaeb 4002dcc86e220a38 400f09d6b5ee5e71
400f09d6b5ee5e71 3fef42f9c0e7c3d4 400d755f79d9a9f1 40065f1a8578a033
3fda0b33fc40d03b 400ea8c03611dcdb 4000e1d734b9ac42 400d411f49a2a0a6
3ff494c483268278 3feb133a8de75707 40065b8748d13659 4003ca68864b3b45
bfc05c6a6d69c06f 40113d4faa406ce9 400211a37e0a3242 3fe142aa5aa44223
bfadd399fd490466 40091b4a9896f5da 400007b4a4273042 400a1f187448db64
400a1f187448db64 3fe51577c8062da8 400fcea1806987c8 400d755f79d9a9f1
3fb4c54450338e2a 4010200b36610b58 40003a9eae8bb4c1 400b0ead1abc085d
3ff494c483268278 3ffab250b5f41d82 bfc882995ee5a31d bfef3e76c0046bc7
3fb72cb71af0a9b0 3fca07762391eaeb 4000e1d734b9ac42 3feea7ca1b917e55
400f09d6b5ee5e71 400198527b58790f 3ff765485f9a86c6 bfc05c6a6d69c06f
40034857f9c8d9c3 bfba7b8c26dbb806 bfd535ae212cbafe 3fcbea6fd42caaab
3fe269431172f9f0 bfef3e76c0046bc7 bfba7b8c26dbb806 3fb9a2578303b30f
40086c7b963170dd 3fcbea6fd42caaab 3fda0b33fc40d03b 400c27fcf1069df4
bfef3e76c0046bc7 3ff381272a066102 40030018bd280812 400a9d8989e06f6e
40113d4faa406ce9 3fe842cea51ae123 bfe8bb6704d8297d 3ffbe8a61514cf17
400f480eb5a905d1 3fe89b23f2eaf077 400c5f8fae9fbd87 3fe61aac87d42bad
400fcea1806987c8 3fed1bbf0ea780ea 4001aeb9fa4d9bb6 400007b4a4273042
40021a9aae9ccc9a bfc4966b7a0dd900 400c04fe050c681d 4005f943c33d3098
400d755f79d9a9f1 400c27fcf1069df4 4000e1d734b9ac42 400b7181aa67d127
400687c4e4a4a2b2 4006706d28686565 400ce035e7b8f5b9 3ff1b5fc5a74015a
3ff9281e750ce82d 3ffba16dc7ab9105 3fc9b60ff3b958e8 3fda0b33fc40d03b
bfdec41137c7a5cf 3ff48a09c1fcc436 4008acf468949411 40035252c48c0a08
3ff381272a066102 3fb4c54450338e2a 4004afcc74f4d272 3fe61aac87d42bad
3fc882568e5acb25 3ff1c9ad9608fc12 3fc882568e5acb25 400007b4a4273042
3fef42f9c0e7c3d4 3fc882568e5acb25 4003be3378db3aec 3ff51934097a2bd1
bf846c2bc72aa413 4000e1d734b9ac42 40071595cc176d00 3ff51934097a2bd1
4003755ce8abff6a 3ff6579a9898e3df 40065b8748d13659 4006706d28686565
3fca07762391eaeb 4005a61d5f453232 3fb4c54450338e2a 400007b4a4273042
40082eb824b36fb2 3ff3e835c44ae914 40071595cc176d00 400eafd386f3c538
400eb57f27cd8516 4005a61d5f453232 400d411f49a2a0a6 400dff03e12eb849
400b7181aa67d127 3ff88e20719a031e bfc4966b7a0dd900 3fdfee0e0736c8d8
3fb374e9693d5eaa 3ff5451db6ce8184 400fcea1806987c8 3ffdd9818c4234ad
3ffec5b6b06bf368 3ff494c483268278 4004afcc74f4d272 4003755ce8abff6a
400f09d6b5ee5e71 4003be3378db3aec bfc0547d20510633 bf846c2bc72aa413
3ff710287d72cc25 bfc05c6a6d69c06f 3fe142aa5aa44223 400f480eb5a905d1
400130387351018e 4010200b36610b58 400b466d5603bad5 400c04fe050c681d
bfe8baba4ac87e65 bfe8baba4ac87e65 400be5bbd59ae3cb 3ffec5b6b06bf368
bfef3e76c0046bc7 40023978e8765c24 3fe61aac87d42bad 40102b7688ebc586
400ccf4dce2f9b50 400007b4a4273042 bfb3ee8afb4c31f6 400ce035e7b8f5b9
40057a1c3e1805a5 bfc4966b7a0dd900 4005a61d5f453232 4006706d28686565
4001a5c2c9bb015e 3ff4eb500f868f10 4009689d2daa0e97 401073140c3612a8
bfcb55b491625fcc 3fdf2de025307c97 3ffa24af884cae95 3ff9281e750ce82d
4007f67921f5c744 400fcea1806987c8 3fe269431172f9f0 3ffa24af884cae95
4009689d2daa0e97 3ffbe8a61514cf17 3ffec5b6b06bf368 4001392fa3e39be6
3fda0b33fc40d03b 40028f570c6882cf 400f480eb5a905d1 bfdec41137c7a5cf
400be5bbd59ae3cb 3fea71e45a3074e1 4006706d28686565 3ff4eb500f868f10
40023978e8765c24 4003be3378db3aec bfe8baba4ac87e65 3ffdd9818c4234ad
"""

comptime _MISSCAT_PRED = """
400a1f187448db64 400eb57f27cd8516 bfef3e76c0046bc7 4007f67921f5c744
401073140c3612a8 3ffa5690b2f40f60 bfd2fcb75bb8c31e 400a1f187448db64
4010200b36610b58 3fca07762391eaeb 4002dcc86e220a38 400f09d6b5ee5e71
400f09d6b5ee5e71 3fef42f9c0e7c3d4 400d755f79d9a9f1 40065f1a8578a033
3fda0b33fc40d03b 400ea8c03611dcdb 4000e1d734b9ac42 400d411f49a2a0a6
3ff494c483268278 3feb133a8de75707 40065b8748d13659 4003ca68864b3b45
bfc05c6a6d69c06f 40113d4faa406ce9 400211a37e0a3242 3fe142aa5aa44223
bfadd399fd490466 40091b4a9896f5da 400007b4a4273042 400a1f187448db64
400a1f187448db64 3fe51577c8062da8 400fcea1806987c8 400d755f79d9a9f1
3fb4c54450338e2a 4010200b36610b58 40003a9eae8bb4c1 400b0ead1abc085d
3ff494c483268278 3ffab250b5f41d82 bfc882995ee5a31d bfef3e76c0046bc7
3fb72cb71af0a9b0 3fca07762391eaeb 4000e1d734b9ac42 3feea7ca1b917e55
400f09d6b5ee5e71 400198527b58790f 3ff765485f9a86c6 bfc05c6a6d69c06f
40034857f9c8d9c3 bfba7b8c26dbb806 bfd535ae212cbafe 3fcbea6fd42caaab
3fe269431172f9f0 bfef3e76c0046bc7 bfba7b8c26dbb806 3fb9a2578303b30f
40086c7b963170dd 3fcbea6fd42caaab 3fda0b33fc40d03b 400c27fcf1069df4
bfef3e76c0046bc7 3ff381272a066102 40030018bd280812 400a9d8989e06f6e
40113d4faa406ce9 3fe842cea51ae123 bfe8bb6704d8297d 3ffbe8a61514cf17
400f480eb5a905d1 3fe89b23f2eaf077 400c5f8fae9fbd87 3fe61aac87d42bad
400fcea1806987c8 3fed1bbf0ea780ea 4001aeb9fa4d9bb6 400007b4a4273042
40021a9aae9ccc9a bfc4966b7a0dd900 400c04fe050c681d 4005f943c33d3098
400d755f79d9a9f1 400c27fcf1069df4 4000e1d734b9ac42 400b7181aa67d127
400687c4e4a4a2b2 4006706d28686565 400ce035e7b8f5b9 3ff1b5fc5a74015a
3ff9281e750ce82d 3ffba16dc7ab9105 3fc9b60ff3b958e8 3fda0b33fc40d03b
bfdec41137c7a5cf 3ff48a09c1fcc436 4008acf468949411 40035252c48c0a08
3ff381272a066102 3fb4c54450338e2a 4004afcc74f4d272 3fe61aac87d42bad
3fc882568e5acb25 3ff1c9ad9608fc12 3fc882568e5acb25 400007b4a4273042
3fef42f9c0e7c3d4 3fc882568e5acb25 4003be3378db3aec 3ff51934097a2bd1
bf846c2bc72aa413 4000e1d734b9ac42 40071595cc176d00 3ff51934097a2bd1
4003755ce8abff6a 3ff6579a9898e3df 40065b8748d13659 4006706d28686565
3fca07762391eaeb 4005a61d5f453232 3fb4c54450338e2a 400007b4a4273042
40082eb824b36fb2 3ff3e835c44ae914 40071595cc176d00 400eafd386f3c538
400eb57f27cd8516 4005a61d5f453232 400d411f49a2a0a6 400dff03e12eb849
400b7181aa67d127 3ff88e20719a031e bfc4966b7a0dd900 3fdfee0e0736c8d8
3fb374e9693d5eaa 3ff5451db6ce8184 400fcea1806987c8 3ffdd9818c4234ad
3ffec5b6b06bf368 3ff494c483268278 4004afcc74f4d272 4003755ce8abff6a
400f09d6b5ee5e71 4003be3378db3aec bfc0547d20510633 bf846c2bc72aa413
3ff710287d72cc25 bfc05c6a6d69c06f 3fe142aa5aa44223 400f480eb5a905d1
400130387351018e 4010200b36610b58 400b466d5603bad5 400c04fe050c681d
bfe8baba4ac87e65 bfe8baba4ac87e65 400be5bbd59ae3cb 3ffec5b6b06bf368
bfef3e76c0046bc7 40023978e8765c24 3fe61aac87d42bad 40102b7688ebc586
400ccf4dce2f9b50 400007b4a4273042 bfb3ee8afb4c31f6 400ce035e7b8f5b9
40057a1c3e1805a5 bfc4966b7a0dd900 4005a61d5f453232 4006706d28686565
4001a5c2c9bb015e 3ff4eb500f868f10 4009689d2daa0e97 401073140c3612a8
bfcb55b491625fcc 3fdf2de025307c97 3ffa24af884cae95 3ff9281e750ce82d
4007f67921f5c744 400fcea1806987c8 3fe269431172f9f0 3ffa24af884cae95
4009689d2daa0e97 3ffbe8a61514cf17 3ffec5b6b06bf368 4001392fa3e39be6
3fda0b33fc40d03b 40028f570c6882cf 400f480eb5a905d1 bfdec41137c7a5cf
400be5bbd59ae3cb 3fea71e45a3074e1 4006706d28686565 3ff4eb500f868f10
40023978e8765c24 4003be3378db3aec bfe8baba4ac87e65 3ffdd9818c4234ad
"""

comptime _LARGE_INTS = """
8 61 0 118 1 2 1 127 5 6 1 119
3 4 0 186 9 10 0 185 7 8 0 55
11 12 0 59 13 14 1 179 17 18 1 193
15 16 1 57 27 28 1 56 29 30 1 71
21 22 1 59 19 20 1 186 23 24 1 184
25 26 0 223 31 32 0 221 37 38 0 151
41 42 0 155 33 34 0 89 45 46 0 88
47 48 0 36 55 56 -1 -1 -1 -1 0 37
57 58 0 22 51 52 0 87 59 60 0 91
53 54 0 150 39 40 0 148 35 36 0 228
43 44 0 219 49 50 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 61 0 129 1 2 1
105 3 4 1 150 5 6 0 64 11 12 0
72 7 8 0 197 9 10 0 200 13 14 1
167 15 16 1 204 21 22 1 83 17 18 1
80 19 20 1 37 25 26 1 47 27 28 1
216 23 24 1 209 33 34 0 41 35 36 0
44 29 30 0 169 31 32 0 166 39 40 3
148 57 58 -1 -1 -1 -1 1 153 37 38 -1
-1 -1 -1 3 136 43 44 0 164 51 52 -1
-1 -1 -1 0 23 41 42 2 174 49 50 2
172 47 48 1 226 53 54 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 2 169 55 56 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 2 146 45 46 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 2
130 59 60 -1 -1 -1 -1 -1 -1 -1 -1 61
0 111 1 2 1 146 5 6 1 141 3 4
0 176 7 8 0 175 9 10 0 46 11 12
0 57 13 14 1 67 19 20 1 38 15 16
1 224 27 28 0 239 23 24 1 83 21 22
1 82 17 18 2 161 33 34 2 161 29 30
-1 -1 -1 -1 2 154 25 26 3 125 43 44
-1 -1 -1 -1 3 163 51 52 2 151 41 42
-1 -1 -1 -1 -1 -1 -1 -1 1 217 31 32
-1 -1 -1 -1 0 215 39 40 3 96 35 36
3 136 55 56 -1 -1 -1 -1 1 217 53 54
3 119 47 48 3 110 57 58 -1 -1 -1 -1
-1 -1 -1 -1 3 128 37 38 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
3 115 49 50 -1 -1 -1 -1 2 139 45 46
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 2 121 59 60 -1 -1 -1 -1
-1 -1 -1 -1 61 0 141 1 2 1 101 3
4 1 104 5 6 0 81 9 10 0 79 7
8 0 209 13 14 0 214 11 12 1 218 15
16 1 195 19 20 2 160 23 24 2 185 41
42 1 186 17 18 1 173 27 28 3 136 29
30 0 239 49 50 3 113 21 22 -1 -1 -1
-1 2 191 39 40 2 199 45 46 2 158 37
38 2 208 53 54 0 29 47 48 2 126 25
26 0 28 35 36 3 153 31 32 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 3 119 51
52 -1 -1 -1 -1 2 105 33 34 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 3 157 43 44 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 3 158 55 56 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 3 92 59
60 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 2 165 57
58 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 61 0 106
1 2 1 155 5 6 1 160 3 4 0 203
7 8 0 189 9 10 1 29 11 12 0 19
17 18 1 30 13 14 1 30 25 26 3 75
31 32 3 90 39 40 0 13 47 48 3 144
15 16 2 141 55 56 3 90 19 20 0 23
37 38 2 181 21 22 -1 -1 -1 -1 3 183
27 28 0 161 51 52 2 141 23 24 0 21
49 50 -1 -1 -1 -1 -1 -1 -1 -1 3 181
53 54 -1 -1 -1 -1 3 157 29 30 1 233
43 44 2 113 45 46 -1 -1 -1 -1 2 169
35 36 -1 -1 -1 -1 2 136 33 34 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 3 56 59 60 -1 -1
-1 -1 2 137 41 42 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 3 94 57 58 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 61 0 158 1 2 1 92 3 4 1
92 5 6 0 75 9 10 0 41 7 8 2
205 21 22 3 153 13 14 1 202 23 24 2
189 11 12 3 69 25 26 1 18 31 32 1
208 17 18 3 161 19 20 1 229 27 28 2
113 15 16 -1 -1 -1 -1 2 213 45 46 3
178 37 38 -1 -1 -1 -1 3 60 47 48 -1
-1 -1 -1 0 226 33 34 3 69 43 44 2
173 41 42 2 203 57 58 -1 -1 -1 -1 2
107 29 30 0 232 35 36 -1 -1 -1 -1 -1
-1 -1 -1 3 200 59 60 -1 -1 -1 -1 2
86 49 50 1 20 53 54 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 0 97 39 40 2
71 55 56 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 3
72 51 52 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 61 0 99 1 2
1 162 5 6 1 132 3 4 0 225 9 10
0 228 7 8 3 87 11 12 3 70 17 18
1 238 21 22 3 197 35 36 3 77 15 16
2 131 29 30 0 16 33 34 2 138 13 14
1 49 31 32 1 17 39 40 -1 -1 -1 -1
2 200 19 20 -1 -1 -1 -1 2 102 27 28
1 23 37 38 3 204 49 50 3 70 23 24
-1 -1 -1 -1 -1 -1 -1 -1 2 98 25 26
-1 -1 -1 -1 3 206 41 42 0 29 57 58
3 202 47 48 -1 -1 -1 -1 3 61 51 52
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 47 53 54 -1 -1 -1 -1 2 184 59 60
-1 -1 -1 -1 0 160 43 44 -1 -1 -1 -1
0 12 45 46 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
3 212 55 56 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
61 0 145 1 2 1 136 3 4 1 71 5
6 0 33 7 8 2 217 11 12 2 187 19
20 2 207 9 10 2 204 35 36 3 191 17
18 1 204 15 16 3 191 29 30 0 27 13
14 3 195 31 32 1 211 57 58 1 234 21
22 0 243 23 24 -1 -1 -1 -1 1 38 27
28 2 85 25 26 1 11 39 40 3 44 55
56 3 115 33 34 -1 -1 -1 -1 3 57 37
38 -1 -1 -1 -1 -1 -1 -1 -1 2 230 51
52 -1 -1 -1 -1 2 232 43 44 3 26 45
46 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 2 57 47 48 0 7 41
42 3 46 59 60 -1 -1 -1 -1 2 76 49
50 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 1 11 53 54 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1
"""

comptime _LARGE_VALS = """
3df773bd5c376071 bfe9a3f50e21ca17 3fe66f973d42fe48 3ff39cb9908e1001
3fcd085cd3d7dfb9 bfd3a4a9310fd0bf bff4a9c8957a5c22 bfc770245b80dce0
3fe42b9c08e435c4 3fea1d78ccefa630 3ffa06a5fc6f67e7 bfe620a9c9561d86
3fa323536169fcdf bffa18151d7e00ba bfee618e2335d839 3fec68ff1b99b638
3fd4ef9fb68e70e7 3fbfad7d55c61ff5 bfdaa07eca4c27ed 3fd45744091c7dc1
bfca4d5ee7f57b24 bfdd69cfe1f5c997 bfef669b66ad4ad5 bff5d69f80247681
bffdf3596572f966 bfe5a02bdab8f343 bff2a9c4e07b3f43 3ff0c5beac5b7a33
3fe2b5b4cedd1451 3ffdbd8ed494aae6 3ff691301558c03e 3fe62ee8cb6a052c
3ff1d2cd3f86ac49 bfe2b1792575407d bfcaa8c79955b57c 3fd6db396b08b192
3fe8cb56aba3d0a9 3fc15a627d764c5d 3fe12de6dd3fa7ea 3feac59813860717
3ff3dd37b8930931 bfb62e6ed4570e04 3fd45b9278377777 3ffb22212fdc61f3
4000f9463e9e6374 3fc2118705210c97 3fe09ff411869ee6 bfd7f9319290be47
bf9878ae1abe88b6 3ff3834c7116c90c 3ff94dea91af286c c000f846b1e79e7a
bffbd20f5f3e2808 bff5179d6ae99b89 bfef7f833ce15649 bfe213794f5f068a
bfcc363e9f23e4d0 bff7df623aa0fbc3 bff2564dd7df7df8 bfead8ed368f286c
bfe098f406cb4666 3ec1be90015d7f08 bfe0929e0d053711 3fe13b60765fcc0d
bfb89e04fdf4eead bfea0ff5ef0f65a1 3feace984813a5ce 3fbb6cc6dee550ce
bff0bd76159b054f bfe077489797c3eb 3fe312793d7ea84f 3ff1f82cc31e3b2d
bfd800f8a6ad3a12 3fc6ba8018a7b961 bfbffce77235c6dd 3fda694e79d82465
bfe9d44328f5aff2 bff387bf524e660d 3fe8ebf23a439f65 3fd70c603082dfb0
3ff4f7e98cdcdcdd 3fecb9fd754cdedb bfd7aca9305c4743 bfe9c5979ff66666
3f041b13a05d4d8f bfd632ca60f286bd bfc369b4e2940000 bfdfee45e4efae97
3fd61d78df5b2693 3fa24dc13318323d bff58707788c6f2d bff083a042068942
3fe4bcbf7862259f 3feee2d1f5ebfda5 3fe191666ef5d35d 3fce5ee36389b72d
bfee1eadc2b8c632 bfe3e0fbe0114f34 bfcc6a92b09508b9 bfe01a2a959ea29d
3fcbd9fa54aaf301 3fe10c34dc4bb2ed bfe5b86203590bac bfd7ecdbf791e0df
bfbf2b5812914043 3fc34899aaf6cc8a bf98373df9d6217f 3fd6d9c294d82d83
bfac687a65bf5443 3fce398ec3825c05 3fce368dc63c8773 3fe1673d42e45419
bfe04cc43731745d bfc852b7433ebc35 bff3fb33c4196370 bff8e07db587ab0f
3fdbef6069fa1643 3fe790afe319d5ba 3ff352e470f5209e 3ff757af58de5ab2
3ff405b90a5d1746 3ffa8c5b741d41d4 3eff2927daea435d bfdb21ef058a6c04
3fd5402e62109a6a 3fe1d8570cf4814a 3fa635cd2d9c6487 bfcb47509ffc77e5
bfe6b138a0701d42 3fd51a9b138654e2 3fe7d8473f355fe0 bfc944713cfced5f
3fcee0af31f9ca56 bfda531fe02aec57 bfb105bd0937c1c3 bfeb9341acc71619
bfe18d5104d87411 3fef106eda981f4b 3fe4f6a1be3733c1 3fb260af02dcaf74
bfce4c13fe748ace 3fde80ecbc4c5e34 3fc818083f37917d bfd1e6783e4d1537
bfe286f41cd6de88 3fc699c83d89f1dd 3fe0275fe8f6c56b 3fe197ad047711dc
3fea2853b60fd5c6 bfbf69ad926949c8 bfd994e85385e20d bfe4cfb81947b3b7
bfd78ded01a65e25 3fd0cf3473f7564f 3f5056beba4e9607 bfeebef8d2ad73fc
bfe5dd95eb147488 3fe2a3a64cb03927 3fee7a9a30590b21 bfec4c323b94b94c
bfde463658c8c8c9 3fda53c2cad82d83 3fe5f18fd68995fe 3fb3e977687e5284
3fd635efab328dfc bfac77747597c091 3fc790c9c9e50800 3f99f704d7033623
3fd7d8a4894c4ec5 bfe26d59f2b277f4 bfca05f38806e34d 3fc0fcb5ef471c72
3fe0b7ad7e0fac68 3fd8727816f46e65 3fe4a1b07c555555 bfe1bdf3f60772c2
bfeac884a8d9df52 bfcc8b0a03cc1e91 bf3105e47dd1745d 3fbff9f79b837cdc
3fd721e3bac92fc8 3fc99bb0b5b8b043 3fe0427ed58c82e3 3f05df0efeb9e218
bfce7084c0c97e73 3fd320f475a4ec22 bf903ed8adc3a2c5 bfd89807674e2d87
3fe0990bbf9a4a3d 3fc24c9f50c934eb bfe052c62e2b9130 bfcc7a36a1c2fa0c
bfc38e6bcf431507 3fc5de499181a9fc 3fa9ea14f649d220 3fd3e31191dbe6fb
3fda3ecc97b29f1c 3fe5d54efdb03342 bfdc15535cc1ad7a bfe7b06363d2379c
3fc3a3ea63d96d90 bfb23e90e5cfd37a bfc0c980ca9e8bfb bfd76cd15bef7678
bfe1e1924708e4af bfd55e7e80863f08 bfceb1da223d1c9f bf5077d318f6db6e
bfde6e29b7fdbd51 bfc9541bd2fa290c 3fdc61ebcb7ef9db 3fc9b0783a68ec91
3fd3e067af58bad7 3fe0b71e5c82b8e9 bfc096fbdbb95956 3fc962fcccf2d174
3fd58a19f6129d11 3fe4abe139a6c709 bfd8c9ecaa0245f2 bfc350927edc035b
bfcbaf1eaaff2f35 3f7b4559ff1e5f75 3fb7c71960351962 3fd5ec860fa1f07c
3fbb787cb579adf4 3fd519160539d1bf bfbe6b8dc3255010 3fcbafaf88a5294a
bfbfa4c8d2ddc477 3fc0b3aa40d52308 bfe6454b0a152152 bfde119f70a46c3a
3fe377c4c105bf73 3fea4f8c851a5d8e 3fb3481a7634f52f 3fd3dc3b69b058c5
bfdacfe8e8778946 bfc2985406a81e91 3fc90d751fa084d8 3fe0426c2f4c9cf7
3fc9547bf113698e 3fe08f5a7d2b0b54 bfc039fb464590b2 3fd1981a51e5b476
3f11771a72d3e85d bfcc8893477d14f2 3fc4a487e04125b5 3fd0c4612a94abde
bf880cf61b37437f bfbe6a2799213e5b bfd8b3cf3ab9812d 3fc76cf4f5714928
3fda5ff890aff7e6 bfbc852f61e835ac 3fbe2d088e3ae7e4 3fb477a7b95567b1
bfc54d7347eccc4e 3fd80a15b2168549 3fc1864caa55b97e bfcf3d827d51400b
bfaf5cad9a17c7b4 bfe342c0c9661862 bfd5d555c6f2444a 3f949f3a85e72cfe
3fc9b2109573fdf6 bfc1a198a4945454 3fc1ecc9b49bee68 3fbba6d6115d1a0b
3fd3a4c1645e218d 3fe37b6e948f4a44 3fd75798845a0499 bfd977ec42bdecc7
bfc8e29b88ced794 3fd24eb676cfef8d 3fdf2521e1571d65 bfcee9e2aee41d01
bfaa291eb666d6b6 bfc43dd7741e65a4 3fb37b57d3c5c79d 3fd808f6c086a63c
3fe5d912fdc47712 bfd9009e66d07662 bfc935fdae48f7f5 3f54df16b7e8e6fa
3fc75f55975bc297 3fb3ef132a74cbb7 3fd30ff166e91a47 bfd6c2558e36aced
bfe14883c2c1ad8b bfd590da1f23d040 bfb6e73e5e84ba2f bfc5a5d72ec2e8ba
3fbf790164bd3a75 bfd425fff2b9435e bfb7fba39b94a371 bfa7e63773c192e3
3fbdd61f9f977ab3 3fcd0fa400a56b01 3fd987acb108aa24 3fd31674ce471c72
3fdecaf68fc3d43d 3fd39c46d7ac7692 3fe31498c315f15f bfd174545da05ba2
bfc25af2f2fc236c 3f128e1c98e4298e bfbb2249136293ff 3fc67d53c948d51e
3f966b422734b3a3 bfc6d039e64e5256 3fd39cd5c48eb884 3fb93cfc946d710f
bfd410c2860c7c52 bfc0aaa47e8142eb bfb1e857d3c8b2b5 3fbb39e1b2aff746
bfc5f340fac83bdb bf855735f9e7fb7b 3fa3fe184a98fb1a 3fc778a9367cdcdd
3faf5ed2156f2b9d 3fd25f7809287ac0 bfc0cdc4aa704d07 bfd1b2a594996533
bfb807242f3c71c7 3fc07b19cb39999a 3fd14118fc1e6e32 3fdde879fc559b93
bfd087570e468ed8 bfdc13c3c90e12c4 bfca6d223682ed45 bf98315fda991fbc
3fb16cc9d4e8db05 bfc0774c25804255 bfc0453a874ef46c 3fa8db4fb12a5e27
3fce9f376a2677d4 3fb2b0a0a7646a4f 3fccef03f1801103 3fd7db1f4da9d260
3fa10d651c387744 3fc798174f91e706 bfc536b12f1fcb96 bfa82c5d2e3686ee
bfcce64e99ee9923 bfbb8d8c479af532 bfd3a047e5257bf4 bfc3326c764bb0a6
3fd16f9b3acccccd 3fe16a32851dfb63 3fce117d1ef71249 3fda45ffb83f0f0f
bfc937c3d90a12f7 bf9f04fc1285745d bf78d223e28ddb0d 3fbd4715d8f3e453
bf7eda058ad4873f 3fc4dd389afeab50 3fd6023b17d8cdc8 3fc8b8464e299f06
bfc4199e3545782a 3f8a5dfb553ec089 bfdefedd136f96f9 bfcdb70187bf193d
3f37210bc1b56fd8 3fc60f7cf420cccd 3f11bbe55b53b9f2 bfbf400c0ba13c9a
3fb42f691d5a28fc 3fc2c65e66aa1749 3f709f1d80359e7f bfb181170f1aa5c7
bfcbb7991391c13d bf99a4100bbf0541 3fc27477a98defb5 3fbe5cad3d2dca09
3fd13203eaf70113 bfc37fbb0b309ec1 bf9adda43807ef9e bfb799768c63d66f
3faa7f42dae2d39d 3fa42dc31fac9855 3fc3bf8eae79a231 bfd53fcee18db45a
bfc61f9c5a2bf541 3fbf143febc23f17 3fd19bbf19552b99 bf728aa3f367b6e3
bfc449a5cfac1d69 bfb4a98273e27ca8 3f9b76df24a90025 bfaa92632d6ada93
3fb39359c0c08f96 bfd04fd72a3bf593 bfbe6bd1c370f617 3fc89cf730488b82
3fd6497bc9d26e07 3f62c40374727c06 bfc0e34126ca66ab bfd271b09433c679
bfbf74bf1628987c 3fb98578d59a0c53 3fd263d5101d0cb6 3fccbf9bd6fba1a3
3fb9831a2d8e73e6 3fcbe19cafa74c5a 3f9cd0dd378380a3 3fa4b9694c2dddde
3fc5896ea3ba1fe5 3faa1c637f237143 3fc250fd4e985625 bfbeb3038f99e0d6
3fab057f1a0aaaab bfc413321f5926bb bf93b2161413b13b 3fcc875f31da6f4e
3fd956e5e02cd8c2 3fc921a57259999a 3fd95233f6f93cdc bfa24c4a4cd45d17
bfc3be9c2da688a5 3f964caefed592b2 3fc43555a197a4b0 bfd7db548d4a0cb2
bfcb9d46ae31ae94 3fcae3b643ba6863 3fdefa982f96f970 3f13ec0cffc02d0d
bfafeaa26e6500c8 3fb56a5c00ca3463 bf894ad72e2c1d2e bfbebfe615ee83e5
3fc5f609769b0d32 3fa901e1913f76fc bfb995d7541fdbc1 3f9081232245b48b
3f9bed341c3a76c9 3fc275a190081e35 bfc1eeb7b1848b0e bf6c1ebae3459e6d
bfcf66d3f2c70933 bfbe2ef81cbddfe7 3fac04397c3534f6 bfa59db49bc78788
bf82921ad17ca9dc 3fb8558959bf8c5d 3fc1d983c3621b25 3fd129aaae04f345
bfb8d96bea08ae77 bfcb21c42f601a6b 3fa4c4d931dc4e48 3fc7362bfe180000
bf85efb79a8bb083 3fc2a80d99581f82 3facd73caa7eb6ac bfa1d7c18fc4cc42
3fba548cf786e5f1 3fd0a0f024da9b53 bfa95afe782f805d 3fc1ac5d78bafdc6
bfc1e65513c83232 bfb022174aa56a95 bfc036a4eb552050 3f75cc396fbd37a7
bf98d740bd97d05f 3fae39a9b035c695 3fce9cf6541bacf9 3fbe7876c1a58e59
bfcc3831ca930c31 bfb9163392cf3fc5 bfa78ce34e6c854e 3fb32ba2e0e1d41d
bfa2b77a57cc26e3 3fc07775c9d2e99c bfc1474fd2362762 bfa3c41b8f51ba17
3f8e7f0e7c459fb8 3fb5f1bbdb6ecea7 3fbf42af998364d9 3fd328c506000000
3fae74ca3a36db6e bfbd8f8df7d422e3 3fc0a52293cb4b4b 3fd3246fbcb2c41f
bfca54ae8c051034 bfd4ac1ee84a2068 bfc42206f1f2df2e 3fa8f21e7f150a85
"""

comptime _LARGE_RAW = """
3ff4d0c9b40fccae 3ff1d3f0f9cd3c1b 3fe0840810436778 3fe5130f377e7d63
3ff04d8a433336be 3ff015e85418d5de 3fe0ac16eccfb6c2 3fc94a087e17e2b8
3fc82f35e18064cb bff0234f44bae28c 4000624daf22be56 3ff1defbc338de10
bfb43ee803ec5090 bfe12ddf089a4850 3febabad97cb4dfb bfcb36db202e0672
3fdb67bc863c2e07 3ff8bde67777f514 3ff5c6c1133688bd 3ff488d1f2da9408
3fd60cbebb87f0fa 3ff1cd48a2861364 3feee4b36ee8a458 3fee130197ad52ff
3ff73384517680fb 3ffd4d6ea1232825 3fff9f710314fd2f bfdb224f06b20602
3fce025a63f7acbe bfd8308f6ff0ed5f 3fecce89050dc26a 3ff194477cb1aad9
3feb523a75295378 bf983d3d81fec8f6 bfe6523423e2e8fe bf918b1cd8cd05ec
3fef6c7322225744 3feed6dd51d0f181 3ffac00a7d7e4b8b 3ff21e2a45f7e6d4
3ff4b2ff743c2271 3ff91c2b684db4ca 4005b7bb31535a1c bff53504882818b9
3ff7dd270653d8f0 3fc7ec9128c96636 3ff01d6eecd1f064 4001d8b3d2d3e210
bfe024c8a6821372 bfd949bb5febd9c4 3ff6befdecdaddc0 3ff27ef5eccf11ed
3fd1f486ad381053 3ff0a493269f873e 40037cb4fee3e08e 4001ed5343dad88a
3fe56309a395a977 bfdb1f90d1b59eca 3fdf4ef4286e913e bfde768d69375dde
3fe4b2e294aadad6 bfde79e4793476e4 3fea1393a3d5e840 3fe09dd58abd0b22
3f853ddbac29a108 3ff0cb6bc68315b2 40029ad9d7ae6c8f 3ff1005d1fb71d69
3ff80edcc0960039 4003bd0fc8c5c220 3fd39629c85364be 3ffddb03111acbca
40002ff96da1e7bd 3ffe618b012e4819 bff21e59fd931c94 3ff5aa8e1a6dd614
3fece6a7acff1419 bfe07ea453919f84 3ff1ed8cf740cfbc 3fe88ba4709d42b1
3fff41d6186b6fbe 3ff553e1fcc0dc13 bff5b30482d548f9 4004c2a1ec77711c
bfcdc317e3ade285 3fe55da541d6ce35 4007cefd04be2567 bfdd4d00eab293d0
3ff4e241b34d8e4f 3ff6dbf169339b9a bfd959af95781982 bfec88c1a7b40d46
400165e5ba3db3b5 3fe2f2f3d91bea8d 3fffa761e8ae709a bfe8a816ac63edd5
4005c3dc03ca7234 3feb91c58eab6358 3fde1a7c39672320 3f6e17ee71565c39
3fe8a655ea142193 3ffa05e2f01f988b 3fea6953d228c2d6 3ff8fc61a815c6cb
3fffcaba5728b6f1 3ff454e16d2489b0 3ffa3ba4c34e24f9 3feae201d1e7e8e7
4004c2a1ec77711c 4002da43dce3815a 40029ef64ecb2c44 3f879bec4f2c9ebe
40028c0d28eb0333 bfc12db1526a4a14 bfd48e706e6c02dc 4002299f4099df80
3fd117001c13a715 3fe449fe7635c3b9 3ffb3e700b8cdc23 bfe4571a70c9f97e
3feb59965b7b04f3 3ff3469dfdbec0b0 40033079a4b1aa9e 3fea9d6a3b261821
3ff788914252fe39 3fd382d51bcd9613 bf9561f576bdda4e 3ffdad73c732b509
bfaa4bcdb9ba81cf 3ffd276bdd214742 40034b190c3f6e42 40016c395b3d1318
3fefe4a6a57121b2 3ff4e955c11035ea 3ffa4f76285b70f6 4000600cc45f801a
40034b190c3f6e42 bfecaf03b816f5d8 3ff3c39182ae40a2 bfc22ecf7770ab55
bfc9f4e498a6fe5a 3fbc76cff7e8ff22 4001d8b3d2d3e210 3ff910ba32c1bd20
bfa0249cdeb19631 3ff010ebf3e57d9b 3ffe0faa302512e8 bf90412561a5bdd0
3fe68d19a26af196 3ff07232967f090d 3fe64667b863a228 3feb1f6eba616712
bfe0101880a25a5b 3fe178a3f1038501 bfe537569b336969 bfdf5a092c732c23
3fdddc1a39589614 bfdfb58fe94ec87e 3fe8d4ce23f0dac5 bfb871093bd4581d
bfd27766acc4e94d 3ffa3a41a636a8fc 40010a32583a1ca6 40031f934238aeac
3fbd86dc1366b2a6 3ffc7397df6d2c11 3ff7bd1880ef88ca 3ff9ce747de75f47
3ffe4ecc6b572fcc 40037ace7d6a1468 3fd9c4ab483f585b 3fff9b54b97c0de2
40006b84556cfeb3 3ff5ed05b801f9aa 3ffc567b8efb00ca 4006ebc42d3dfbb4
4005010831278566 3ff81367c9f6e32c bfea84e8d5b55c2c 3fb336e10d86eec9
3fe80fa8f3bf95b7 bfcb539429953d45 3fef35c1e61819a4 bfe62539ea69e3c5
3fee2aeeb9d9f020 400041ccb8d3d737 4002611733ace0e5 3fd364fc742288cd
bfc4dd560eef9b5e 3ff164e7b8f159dd 4000742daa6f16a4 400824f2dec21ab3
3ff7d7b804d01482 3ff0535aaedc0d99 3fd9747cc897ad9e 40008ced0e7f0955
bff3aaf3507590dc 3fdcb4c6e65254c3 3ff6a71bdbf4de52 bfd72f705a9861ff
bfea2b7b0d6a6cb2 bfeb147f70d58396 3fbc76cff7e8ff22 3ff425ccc8f26d18
3ff56f3654046477 3fd0bf2b58b7ab22 bfc6fdc6fd6cdbca 3ffc43def5abafb3
3ffbe344b61df6f5 3feffc9899b54f00 3ffcbaf27843df5f 3febe7b5b9d5b45d
3fe5b1172c751431 3fecb23319161d23 3feb1a29092fb1bd 3fe456e35712506e
3feb12f33cf95b17 3fe7ad2314dbffc5 4006fb3d5cf284c7 3fcb88203490af72
3fe9540236906e7e 3fbe3e305141d0d1 3ffbdb4fa156d6d6 3ffcf9bfbbd595f6
bfc4b50c4ed7c341 3ffa288854de79d2 400089bdd1a84c80 3fe72b1e3149ccd5
bfe3793e6627b62d 400278bbd910cad5 bfd975ed39a847f2 3ffda3b7c5c68ca1
3fdbf851e6d35312 3fefc2162a3617ee 3ffb58bb1b654e05 3fd0af717f1bf177
bfcb3502268390f1 bff18b352ffcc57f bfe03dd4d04e5340 3fe7620e37aadef1
3fcbae6fcbc1840e 3ffd0632adfc1fcc 3ff2cb66351dd1aa 3ff853af63f001f6
bfe03d22adc74ef6 3ff36772b58c552d bff3425e7a4abe19 3fb8de2c928c2731
bfe12ddf089a4850 3fdfae55b2b69a43 3fdd2eaa9f0dd389 bfcdc6a7b58eaf23
3ffb96de7e6a2a95 3ff1009d0dda7091 3ffbc7155efb8182 3fcbfc1adbdf17d0
bfcb3502268390f1 3fe7c4afcf97fc2c bfa59b2b13e262eb 400064abaa5c6409
bfc19583e5d86770 3ff29b91e75ee398 3fe6b4d062ae0879 3ffd6cebc7e3a284
3fda7ff9139bd628 3fed15f1fd5188c9 40004353702e1105 bfbbded6e7d75b7a
bff18b352ffcc57f 400001ced020096d 3fed6ec4bfdf7b8a 3fe5c7a35af1bf98
3fe3101d4b8c8243 3fd7703538ab2e7c 40053738a14a5201 bfe6a9dfd0ec783e
3feb9fb3d75aab66 3ffaad6f9ea1a7a9 3ff9ff54ce1d3c52 4000e9c4a42b9543
3fe5a3d0213c68d3 3ff94d195e115f72 3feec41b61ea296f 3ff3a452832185cb
bff580f415f3f58c 3ff8bc07e0c8a71f 3fe4f3cd67761d80 bfd4c460c0a2c8e8
4000ead3f96f059d bfbe3824033a2faa 3ff2b5d16d876a95 bfe7018cc8cc7ef9
3ffece1663244d90 3ff217c7bfc993f2 bfee300bbde52875 bfe80044bbd3af05
4003cf0fa66eaf05 bff14715f43bd3b4 3fcd4368bd1443ad 3fe5184c78990f02
3ffc87f07b6a7220 bfe84f3cfbd94528 3fe98fa39e3a1b53 3fec74f62f990bbf
3fe2e5375987e721 3ff15ff0c9da2259 3ff2cef496f87fea bfe1b69c235763b3
3ffe83229fb588e5 3fe2aeb77129be5a 3ff7c0e2543fc73e 4004452f09016d61
3ff1cf883d9992fd bfe16aa5953472f5 3fe3388d176d898a 3fe63e0c6e4a957a
bfe9704cb6cf055e 3fffc554c7d71ee5 3ffca67de8f4e3ed 3fe677a7a6c92453
3fd0b0bcf4ebc8c1 400496fe22c7dfa1 bfcf136a730819ec bfc624f78fff0acf
3fe4180f7520437d 3fefc19ca51c9ef7 bfe2208c528b4852 3fd6339795cb23b1
bfe6f414734de00b bfdee46da7d4b254 3fe85200168adcd3 3ff2cef496f87fea
3fe3970b5c24e1ce 3fea7db997acb7ed 3ff1e9ccbee7185e 3ff9e6c8640a9581
40058108ef73afea bfc48fd87507fc99 3ffcd3e71669f430 3f94ce70d6562d4d
3fff8be39fefe4fc 3fc2c9ce0694050d 3ff3a4ef5a911c4a 3fe6e8c9f28f76a0
4003213de43080d8 4001e51969fcf4f6 bfe5b4acea89978b 3fd1f4691f39cc3b
3fe690aa879da93d 40067823bbeda75c 3fdf3c0d627d9b8c bfc554906ce44848
3fd61d4e598ae9ad 3fe3d3fecb050a47 3fecb1c70d922960 bfd175a499fde504
3fe94ed6c8d1bbb8 3fa6bf2217d108a8 3fbb688190129b2d bff247bd4cc1da06
bfbcfec1a4a0663d 3fe5a5db6f435b03 4000928bb07d99bf 4001aa2160d9afb7
3fe2ebdb4ee6d2d3 3fea7dd77abcfc21 400205ec566a62c1 bfd4e824973d4bef
3ff73384517680fb bfe7155a02eecc82 3ff0cf4c6ce44525 3fd5d711bac29f0d
3ff102206215f098 bfd32057048b6119 3ff6459f1c8972f1 3fecdbf1476db1e0
40033079a4b1aa9e bfe11c4506619fa8 3ff20664d169386e 40015a767545de54
3ff03829038cb5da 4002f949f278fc02 3ffbcd99956ef8c9 4002815c1dfd6052
40061f3e352b3a5f 40058b688959d3bc 3ff9746943398640 3fae40d63d6af0b3
3ff1f51d63d20f8e 3ffb3a5353d8f755 bfdda18c2a1bc750 3fdd9c18ae499bc3
3ff4b5841ac04ad4 3fc341b92954e686 bfc443c7ce7cd69f 3ff4c7f0f324f8ce
3fec1287b1874b65 3fe91ec97d6f554a bfb45a745cecbcf8 3ff56019e27b9714
3fb74152f298b843 3ffbad132c9f81c2 3ff0e59cd1c21c53 3fef1411a7db0df2
4005e52a77ccc515 3ff737c173bf0bd6 4002f949f278fc02 bfc17e30a7d034b9
3fe2c2531b0ebd2d 3ff59cc79ec8af19 3fc3736783da2014 4000b42dc5401d0b
bfc20c1cf1a5d315 bfbd073aae097674 3fdb48b98bb7c35e 3ffb12641904587a
3fc2cd5b53fc5473 bff3d280a676831f 3fefd7e2430bb3bd 4000d0d3d91243b1
3fced67fc8692ae3 3ffb4e311d2ee095 3fda4e5a2f2f63c6 bfe853297820c2d8
3ffd11047f06c9d9 3fa7dbfee30226dc 3feab287d0adb6ba 3ff7a0ee24a8f186
3ffc6c6e6472c700 3fd63d0a5148b33e 3fe677a7a6c92453 3fe00bdc990c943f
3ff454e16d2489b0 3ff86056cf171db0 3ff7e9fa2015bc71 4005db7a20eb7171
3fea3be18017d85d 3ffd5b5677e34e3c bfd1e81fca328632 bfdb00bd2be3667c
3fdef2ec487226e1 3fd3455b14cbee12 40037ef0ed5410b3 3fbb2316167f3664
3ff43ff2fa9d10af 3fe03ff2b432ef77 3ff65c62ac58e68f 400631c20751a5e7
4006fb3d5cf284c7 3fd6f8349c2c36f6 3fb8b6e24c6f44a1 3fee284247e73263
3ff87f759d67d78a bfd30ca15bc39cee bfbd073aae097674 3fddee860d49309d
3fd3e8996b8a8f09 3fced012708f9890 3fe3afb46af34b3f 3ff8cb1802c9d362
bfe3bb85932c4bfd 3ffc2d7614dc9204 3ff47a35d0f98bed 40017547251688b0
3febb469bca6c741 3fc5962c2cb9e4e4 3fed6f50d48f49db bfdce2e3f3c43aaa
400235ecc4b25e1a 3ffb440cd58e47f4 bff580f415f3f58c bfe0720f4d6a5934
3fdeee5c699b2ecb 3fe8861192a3badd 3fdaff3a39b0cda0 3fdca95aa31f352f
3ff121ece812bbf0 4001a2f527dcb893 3ffeab9e9d6e29ab bfd16b7c667237cf
bfd977b06fbcd796 3fe39da142fd24ac 3fd6842dbd539002 3fe6a6b746700838
3fef0bee1f29ee31 bfaaa88167b0b7ed 3ff2787bfd579863 3fe8cc53b74f6fd6
3ff36bd75b29b981 bfe42d4590e7e270 3ff48f8f252cc517 3fe09dd58abd0b22
bfe0d4af4bc20a33 3ff10b2f63283347 bfa5dfce2e04193e 3fc49fc4746e2c58
3ff9033332916084 3ff06939041cd06b 4001ed5343dad88a 3fe4a0c4b3f6ecf4
3fdccae51ec51eca bfe4f228ec3efb6d bfd46fb0c74110bd 3ff675a37604dac3
bf9fd618e1412c5d 3ff42c554e2b0c24 3fedbcb82c120c15 bfc326144ef7bc4d
3fe425923dfd0568 3fecba0a85cb146e 3ff2d93ac7adc2ae bfe0101880a25a5b
3ffd25e8a4331610 3fe0eb35a387e9e9 bfe31e3a39c6e79a 3fea38661ca8ada8
bfd542c53cdb5514 3fe5bcf61048c23a 40019526d63462be 3fe42262a84e8183
3ff1e90580e6b583 3fed9176c0d64d48 3ffc83e3221cc32d 3ff3b3d88d122b3b
3fb5cb6d14ed4199 bff1396ffce05efb 3fd5baf2419d91b2 3fd0add720c05b51
3fb23a4f06524abb 3ff436fd8e8ea1c3 bff29a23b6e682b2 bff29d077106bb8a
bfe0a5049d50bdd3 3fe9511386519c11 3fac729cd6049050 3fe652d5a40ca09d
3fec673126b4fb5a 3fa6bf2217d108a8 3fffb90e482f9dae 3fd3c1c1737844ad
bff3425e7a4abe19 bff4a9b00c9cacac 3ff28f22db60e6c5 3ff01d6eecd1f064
3fec1e1fcfc5db08 3ffa445b5555a4c0 bfcceb3f99e63e8b 3fd76a1d10c0d8ae
3feb23cd2cbbb6a9 3fb260affa233702 3fb77085ae1df27d bfbccae45bf1f721
3ff0ab022d74fb71 bfeadb9553e5ec3c 3ff57c959eb5eebe 4003097ceb579361
400196fa5f520748 3ff8005dcd5725b1 bfc6c5b2eb1b2018 bfe7c1c4a420739e
3fc4ccf63bc6971e 3ff27a4b27707a63 3f9a6496c7f9dfb2 3ff0cf1034d6dafc
3fdbc4fc2d311713 3ff627317d808bd4 3fe5e99259d9aafd bfe08ce869a7b4ee
3fe09b413a740769 bff2d6a090bdd101 3ff86ba8dadef697 3ff587c4393c76a1
bff45dc07ed0cfd9 3fd3f5e03356ad06 3fea235286b9af97 3ffa06b24315f567
4002815c1dfd6052 bfb0df910a0f939c 3ff15df4cba0f223 3fb2dfc3fba4065e
400176fc1e5d0254 3fdae17cabd02f07 3fec6437e5af56be bfd19a6094cb5ea5
3ff0453116611488 bfc56161b632ef6d 3fd3826b51514267 3f5f1aa202d5768f
3fc3f9e56402c9c4 bfd0ecf0fa809c1a 3fccc90afdea0f5a 3feb7d819c413988
3fe5167f686cef3e bfd44eb373c1e787 bfda54bda3056d96 bf310b60afcf1dc3
3fe3eefe7a6cfb5f 3fedf412723e9443 bfdfb58fe94ec87e 3ff72e0a0829d846
4004a24fbda6eeba 3ff15075d84478b9 3fdfee34d0f0f4d7 3fd873455963a041
3fd562ab18edbbb9 3ff98da7f637aab4 40053738a14a5201 3fe4c5ad7f860525
3ffa811f11ea7c66 3fe47ab8aba02eb6 3feba55931acece4 3fdc2f9406252836
3ff55346f4908e9e 3ff303d6db2c8262 bfebf95bdbddda32 3ff55fbeb8547d2e
3ff69344e8fc6839 3fe495c4b991ee9d bfdc3a0c8b5a60fc 3fc53aca75ca8b60
bfe81b5214cca7b6 3ffb2f74dd767f0f 3ff7e0a97a1db0e6 3fdccbb0f068eedb
3fe4a0c4b3f6ecf4 40022f8d0436ee12 3fd827329dcc0328 3fe2c52936406dbc
3fd750f0be8314af 3fefe0dc6c76015b 3ff17282bf8842da 3ff9247065c038ff
3ff0ab152fb562d3 3ffe9ff7572ef2e1 bfb167b9b838084c 4002cab704009ebd
3fea7db997acb7ed 3ff1c87bc1983edf bfdc5d3e2fec0c1d 3ffec0fdb73856f3
bfc40e9dde9931af 3ffc1e4e0d0a3a09 3fd600732d4eb786 3ff8005dcd5725b1
3fec498bb7feb2a8 bfbe1e47e80286ac 3fe9e2a2eca2dcc7 bfd09300f437afe9
4004e320847f0e09 3fe87abfba20e2cd 3fe2cda9d804e09e 3fa6dedda75ec30e
4002b51a60156abb bfe0cd3f7cb34de4 3feec07d0db8656e bfa4388365bb147d
3ff0975e135fe112 3fd2ad71cd061395 3ff5e9c0ef74a8b4 bfef48f030f20e8b
3fffcaba5728b6f1 bfe1050a6017e694 3ff8709012e7d5f4 bfdd71da06bcc6a9
3fed9176c0d64d48 3fb66eb09ac2fd61 3fed48d8debd8a4f 3fff09129695ff29
bfe139d7bd37e0a7 3fdf74bc9ab0dba7 3fdf8e95429fda7f 3ffb4e311d2ee095
3fe1478b686bdea3 bfe1aa986025cc02 3f92b5438a738052 3fee583cbae46b11
3feb03be2974f51f 3fc886f512580560 3fe8eef2897404af 400218fc8d232315
bff4a9b00c9cacac 3ff2dc3ff29114a5 3fe02acadf1afcf0 3fd5baf2419d91b2
4006b32fcca34fb7 3ff41fd87e63dd7a 3fc794551dfaf6a2 3fed9176c0d64d48
3fd7545e57d86742 3fd3473809fc42e2 3fe3dd2551aa2a93 3ff24f970e0d36b6
bfccf9e2ade8b062 bfa9e0de99987dd2 3ff31caf8aff054f 3ffd7a587cd79a80
3fb8de2c928c2731 3fe918cb7d5fcf31 3fd6a08efcd34749 400365448f1485a7
bff5b30482d548f9 3fe2255f88eee5b2 bfdf8448df8d2851 3ff784d6fa1eefc9
3ff4c6f66083bf42 bfd72ec99e63a437 bfe098d59b5f75bd 4000f7d6366ef22c
3ff73bf560b3e3d6 bfcc1a02502f08c6 3fd873455963a041 bfc99eca19ddd549
bfd2e0b01f991145 3ffe39424d2f92d5 3fe4a8ade646cbac 3ff006d79db4776a
3fdae647f20c25b8 3ffaf14f1281cfa4 bfe07c0dbb34dbfe 3fea15fecd9fa9cc
40030aedc9cec4a3 3ff0412aeb1a79be 3ff264f088b2256a 3ff9fb8d7d071d1a
3fe4423d56707e42 3ff0937087cae9bd 40001847c0757363 3ff33fb44a45b9cd
3fe3f554feb811dc 3fe86fd99e24a653 3fe98460cbad3f9c bff29a23b6e682b2
40012f31380ab038 bfa258da92d328d6 3fffa761e8ae709a 3ffe08211b31ef94
3ffb440cd58e47f4 bfe42d4590e7e270 3fce0b0551a613da 3fdbbfecce0b559d
3ff07281c201a9d6 bfdb802a2eea4337 3fd86f208ab8ef11 3ffe9b72b778e299
3ff5addeeac3648d 3ff179ce3152f778 40015b8bd940b12e 4000c9e1b4656b3b
3fddbd1b62b444c7 40004092292b62e6 3fd8ae1f8d17b40a 3ff4e0e5d8b96f87
3ff2787bfd579863 3fdcd7bc079d20dc 3ff0a493269f873e 3ff60860b55a5ec6
3fdbc008043284f1 3fed1279062a4ba5 3ffc43def5abafb3 bfe852862ed71768
3fd43165ca8cf419 3ff79340621ce38c 3ff6d3c4113e4e5e 400003ca4edf349b
3fc5ad24a1ebc751 3ffaaf2d8a3275c7 3ff4bd5a1fdf5253 3fa6e659312df969
bfe84ccfc496563c 3fee39eacd1f15f1 3fd80da393857a4e 3fea65319d80e7a0
bfd379b49f640aa0 3ff108e7e66dd1d9 3fe70ed346414175 bfd4188fc5b56c82
3fc98fd0be6fe625 3fd051f3623bb03d 4001e57a20624036 bfe800413967712b
3fdb3fb4f626fb6a 4005010831278566 3fea915f55ac35ee 3ff91112df2ecc14
bff4a9b00c9cacac bfaeffbf3e8aa7d2 3ff14a95f4372be8 bfe30f7f08eff47b
bff18b352ffcc57f 3ff6bf73e2d6ccdd bfe7c85759bd4311 3ff294f06a853f2a
3fe42e3ce54aff00 3fffe39ce03ce5f0 3ffa13e5cff96be5 3fd87435ded69d7a
3fcb515aa51fafda 4003058c4cff0973 bfdfdf51a9e92141 3fb9af0a3393da43
bff1d74f9a6dc078 3fcd2458442708a1 bf6e366449f2a54c bfe2b3c52596f4fd
3fed8b0d2b578d7f 3ff93b160eb22c5a 3fd72a0312f887a6 3ffe274a1b4209d7
3fe1ae7782b2ae4c bff5b30482d548f9 bff29d386cdb84a4 3ff48f8f252cc517
bfc9e2e2641f189d 3ff3273241088154 bfc8faad085861de bfca7931eb51632c
bfa55cd143b30c88 3fdf205fe798019d 3ff4332031106e6a 400433aca9fdc7e8
3fd7275f5efebd1b bfe99ac49b5720ac bfe4f228ec3efb6d 3fc344e53fd8a741
bf9c61b2332d3cce 3ffe30f64fd6205f bf4cca04a6a4b1d0 3ffa9e3274e1b043
bfc5b8621641042d 3ffada703426f0b3 40007020c1d23522 3fdb48b98bb7c35e
3fed707d739872e6 3feccae3f1723644 3fefe500d31994c2 3ff9abde7e2ece67
3ff3ff66e7fef774 4003c8a2c0bbf43c bfdfb58fe94ec87e 3fea7db997acb7ed
3ff06939041cd06b 3fdebe8703ab4d27 3ffc43def5abafb3 3fe09c299fe4362c
bfb98b3c61bf4d3e 3ff9552dec3ade61 bfe99ac49b5720ac 3febc50745fd1f08
3fe46e4cd192a8bc 3feab03486ef8aa4 4005863a43155b98 40002828af253072
3ff161b9c8ece3eb 3ff2a5ebf51fe4f7 3fe0bc27643fdead 4001a99d63901957
3ff470bb3821f04d 40015c3dfaeca6c9 3fcce48209086f33 3ffb39d5056852c1
3fe3388d176d898a 3fd7545e57d86742 40031f934238aeac bfe15c68e4c6d905
3fe11e687c332dea 3ff6916d54ed9c62 3fe72a7726087105 3ff59ab6079859a9
40024dc680723bdf 3fd5a8cbc9884e19 3ff56019e27b9714 3ffe4ecc6b572fcc
3fe104eb8fb7488a bfaec59d0008b619 3fd3f75472c39931 4005ba9acfff1110
bfef4ff6901d8b95 bfbf995b5d3a3d91 40036dbd8d093b57 3ff92dc885afc18b
3ff68a09ae3bb994 3ff60860b55a5ec6 3ff9a24ac9919e9b 3ff5dac905a69805
3fe7fa00d94768d9 3fb23a4f06524abb 3fdf4ef4286e913e bfd746c90ae4b370
3ffd25e8a4331610 3ffd83a918df92a2 3ff0e89dcd21dc00 3fd87435ded69d7a
3ff627cc72109e22 3fdebee403408fe3 40058108ef73afea 40004071c6ba2e5d
3ffd276bdd214742 40020370b4f96ba4 3fe80fa8f3bf95b7 4002f949f278fc02
40045da878d56c81 bfce09cd5f0b04c2 4000a33441a2a751 40068c253c60f042
bfe9c2d817249bd2 bfe1627da133fd12 4005c3dc03ca7234 3fff5be688836a6c
bfb4f7ab9eb3bc0c 3fb5abd117d8e5fd 3ff18ce382d60b93 3fb2baa478ad28b7
3ff9348b851bd8c0 bfdc5373900b52b9 3fd1ef94363410e7 400238f31ddddb0d
3fec7d7449e91be2 40012f31380ab038 3fe2cda9d804e09e bff46d77f3c241bc
3fe0d63dac2b5d45 4002deaa8aeb385e 3fa6bf2217d108a8 bfdd0c255425875e
3fd93303f5ee304c 3ff1c87bc1983edf 3fef4ea94ace5f37 3ff7fa3ae2bc6feb
bfee02393a08ba8e bfc46bc68f23ac14 40009e5d42af42fb 3fc6a4dcd680de99
bf4cca04a6a4b1d0 bfe19fe97dcae326 bfb0b846c3f2b10d bfb45a745cecbcf8
3ff0eacd71d11c4e 3ffe14827605c2d1 3ffb4e7742e24802 3ff44b77be9c6a5a
3fe86f4ce8507f95 400121b23a8b188d 4002f949f278fc02 3ff90a82dae3e154
4001a99d63901957 3fcedc38b2a0a3cf bfd7d469bc1a4245 bfab9cc0fa0d16e3
40060e822a5eb127 3ff57f2df7ae0ced 3fc98fd0be6fe625 bfd8e09786e8c224
3fe5167f686cef3e 3fc4d0f2d9941b8e 3ff7593bd33f94ca 3ffe1577bc7d113e
bfae061ad4bae6c6 3ffb53a5f729c282 3ff846947265cc1a bf8b5d6f68c02bdb
400439db5bff7e96 3f7a05eb932fd6a4 3fec20b73caa86b9 4001efd8c816e1f8
3fc7667dc827a2ec 3ff102206215f098 40017639ee43fc06 bfe853297820c2d8
3f8b9f770238b16e 3fda2404b65ab6f7 3fbb1e7ca4e8dbf6 3fe8ba0179fcfff7
3ff849c5b2e44cd9 3ff4b2664d1370c0 3fe5130f377e7d63 3fea4c41eaec7f3c
4003c8a2c0bbf43c 3fe7b1836e8e95de 3ff4d9e9f797a5d2 bfebc6b291a06c98
bff43b07ed9a99e3 3ff5610e31653cac bf9c61b2332d3cce 3fea915f55ac35ee
bfd7d7f488440add 40020512a46a47ba bfe10473be01200d bfd0ac345e980b20
bfec644d8990a756 bff07b2548428ba3 40040bb5c31209e6 3ff63e371738ccf9
3ff8b40fdb84ef12 3fe8cc53b74f6fd6 3fc2438806c0cacb 3fcc6ee90d954ab0
bfb3411e4f0668d9 3ffb0775b1be7fb9 3fdf4ef4286e913e 3fc9613712642a20
3fdc93bd844c7d91 3fbda88ce2a8f313 3ff94881695d0aa7 bfee251a746520c2
3ffe2c33e22ff788 3febc0ad7d465cae bf9def03b9c10767 3fe8fdf23d61f882
4000eb9f1b23e4cb 3ffdb2d5e0ef3c5b bfbccae45bf1f721 3fc31c5d96ca6ca4
3fec707a4d756902 3fdb6cb1547aef7d bfeafc7af93b9e80 3fc3489fa31cbfc6
3ffa278df5d5bab2 bfc635d68ecbbd78 3fe02227f0d8a43f bfd422819a057958
3fcec1537b77f046 3ff1472140d7ea5f 40008ed524949f6b bf99fdd1366790b3
3ff7f536834ebb25 bfea1cde1cd35149 4005ba9acfff1110 4000c6a54b7cbe68
3ff783d76c9ca120 3fd63528bcfa6b4c 40030c348cae2913 3ffba3e2c97c3b2c
bfb19b4b47fe6bd6 3f8a7ac3f3d3efac 3fe17e46a9887e12 3ffc20878bd9dc36
bfeb776887dc82b7 3ff5844820a86175 3fe1156a22e735d8 bfd746c90ae4b370
3fdb3b2a25e7931b 3fdd7506dc61d34e 3fe56e565ecc4618 3ff3469dfdbec0b0
3ff822815dd5857f 4001fd3c47594041 3ff1d76187977a67 3ff01d6eecd1f064
bff18b352ffcc57f 3ffa47b9f82968d8 3fead1c0b305fb4a 3fee8d046ca3ef97
3ff9fa352697f186 3ffaded9b39bcc3f 3ff215e53103f6e0 4006a62e67369ac0
bfdf346fd86222f7 3ffdaf4a0abffad7 3ff6e15af2642dd7 bfc703603b8a69c5
40044c62064b74a9 3ff4fe8a243f7a7a 3fe32ba69d4199b6 bfeadb5add808f1c
3fef1411a7db0df2 bfc4b1b0949270aa 400033c42baf96bb bff3746ee72c1186
bfe36d63f292d08a bfe5b4acea89978b 3ff54f5e0943f180 bfdb09511b242563
3fc2ec30bf6ce52a bfed175168a67279 3fe50ece7640d517 3ffdea97f328a892
3ffdca43d3654148 3fea3b43b97d124b 3ff629f3e22fba00 bfbf8b020eeb27ad
bfe2d5720bc3c605 3ff48d49eb72b390 3fdf4bd49532d4d2 3fc2438806c0cacb
3fe47ab8aba02eb6 3fe3841a61f6371f 3ff853af63f001f6 3fdcf283677d76c3
4004452f09016d61 3fe749f109df9a5c 3ff81712bba0b8d0 bfd5e3d5fd23390e
3feaddf85bf479ec 3fda3f141917a932 3ff0535aaedc0d99 3fb1cadc1e9d8f01
3ff16ce0a5a576e3 3f8b9f770238b16e 3fe909a0640c8274 bfec5c6a509a1ada
bf7e8424b7498634 3ff910ba32c1bd20 4001ee7cc531a4b4 3feec41b61ea296f
3fe631dbe9e4049e 3fbd535fc2ffe14c 3fe119087718f9d9 3ff06939041cd06b
3ffa283ce9fa9381 3ff7b8e3adfe0b70 3fd750f0be8314af bfb7aeb8e19d53f7
40015f630af5ed26 4002c0534c76c7de bfe11c4506619fa8 bfb49b0fbfc11fdf
3fbbd6a221a0af2f 3fb53514e3c25b11 3fec8c5235b489dd bfc4c4707aad7fb0
bfd5ffbe2c4f8aa5 bfd927f5e4d887d7 3fea47dd75bd0958 bfe48f70ba747cba
bfe48a086c4ee3ac 3ff17e2b22db98e8 3ffddae29bf0eac4 bfd2249c2b909ab4
40084722737af626 3fe6fdd787940a07 3fecf38bcfccd8c1 3ffc64bd00674eb3
3f5072bf01874152 bfbae08ea9b38d35 3ff65a225d770724 4001de3cb007b9d2
4002d36dcaf34a0f 3ffc420ca256d990 3fd91f01ffd9acf3 3fe47ab8aba02eb6
3ff5be2067b2c08c 3ff03f2a6bb6d70b 3fe6cbf62b1eea1f 3ff1cf883d9992fd
3fddbd1b62b444c7 3fa93f59b0a5520d 3ff3625a664d7820 bfd87aff7f4391db
40009e5d42af42fb 3ffcb7c7c22b97e9 3ff81ed07eb17c1f 3fea5c46245b4af8
400592e86bd39133 3fca8dea7aa182a3 bfb3760fac011be2 3ff1812d8886e1c5
3ff1d7c7454ec3f7 3fdc2d9b97cc4dd3 bfe9704cb6cf055e 3ff2de1ed41db72b
bfcdacdf74aaa9a3 40065fb9988c10fd 3fa6e659312df969 3fe2501a9b24158c
3fe8a07c714e4137 bfd7d30550e30041 3fdf272a68f08043 3ff22ac2d6d8c91a
bff1c6ff3f49490e 4005ba9acfff1110 3ffc18040192323e bfe9704cb6cf055e
bf983d3d81fec8f6 3ff3d16c6d7a8b79 bff5b30482d548f9 bfd8f6a47c736a51
3ff8bec1fea780b2 3fd78eaafe2f2924 3ff1e9ccbee7185e bfc8c01a71e9f6e7
3fd19d18eec8611f 3ff41bbc1f0c057d 3fcba3a217955a96 3ff8b903e412e506
3fd0a9f9625e9e04 bfd66719dca30fa8 bff2e45a4a07087e bfdbcf1be4321ed7
3fd7ddbea24efd0f bfde9c37c7edf2b7 3ff32be48a3d71bb 3ff6fbc48bcafcb8
3ffa02e0d913af90 3ff6236cc1c81b1d 3fe51bd396d588f8 3ff7ab0f85fab9a5
bfe0d3f126ff5c48 3ff7eb7df625a383 3fff58f1bc8daebe 3ff454e16d2489b0
3ff36c537b4b21a0 3fb13969f2ddbcdf 3fe14d86749d4a1e 3ff8b29ec98b7887
3ff949b098457148 bfebd1ee8ce39d9e 3fe35029a0961fd3 bfe9704cb6cf055e
3ffcd1fc011a9e61 3fe4a0c4b3f6ecf4 bfd9d69062c7a910 3fe2abd8c12288fb
40006ea4a2346bb0 4005603707105e34 3fef53874574feaf bfe3da61193e75cd
3ff6d2bf4bb3af2c bfd458df5869cd06 bfee09d11acd9904 3ffad38320e7424c
3ff32eb1b8056ef0 3fe89fea7c2e591f 3ff3bde1410651ec 3fe36430a668bf96
3ffee6de2675d6bc bfd4aa587b7ac354 3fd79a41a952c63a bfed8c53cdfc7eb4
3ffd48f2436e7972 3ff9256da399c97d 4003f1dedb2c6cf8 3fe0bb31275166f3
3fe93ca55a135a34 3fe1e7a957464f21 3ff9930e1e59d716 3fe7c079b3d4f435
4001832e0c9c5b41 bfe87d2d71213812 3ff1956d049e05f6 40037e24a39fdd56
3fffbb1f30fac7b6 3ff2d229ed2a89c5 3ff4b4d828724f71 bfe26aaac393db45
4001efd8c816e1f8 3ff7ab0f85fab9a5 3fd9c3207803b946 3ff454e16d2489b0
bfe815ec5b151742 bfd5e6d78c798158 3ff4e40e1b7d48a5 3fd44f6c941fee52
3fecb6c40a0b35de 4001aa2160d9afb7 4000b6cde7c77d04 3fea5d0da3099c60
3fd3473809fc42e2 3fedf412723e9443 4003cae14a7568b2 3fe048d90e4bd8f0
bfbd073aae097674 400824f2dec21ab3 3ffa42e961caa97a 3fd0ded006eede27
3fbf48effbe3694c bfddd8b9696929f7 3feb53605a4a23f6 3fe5b14b5d430dc3
3fd963f57070d995 bfb1d2cd7d7fcc20 bff2d6a090bdd101 4005ba9acfff1110
3fee9f8d0d2ed061 3ff3a64301c8095e 3fd5a8cbc9884e19 3ffae5eece76af33
bfeaf9745ad2f813 3ffcdd061b428f48 3fc98fd0be6fe625 3ffc96a0e28f8876
3fe3d3fecb050a47 bfe4d6e859545e09 3ffa003b4a210622 3ffd8fa2760bb99c
40012c8382b5a2f5 3ff0f96b7c71b359 3fc820dda00b0ebc 3ff87d50afa351b5
3fd13d9f48d6f75a 3ff2d44db25a41bd bfeb3296f83d5317 bfe95f14bab05dfb
3ff963939b39e070 3feaf027eb424a45 3ffeee18cdf0b8b8 3fbc37706db3cf0d
3fb9b674757c8ba1 3ff113b38ee3da7c 3fffbb1f30fac7b6 40059bdc06767b50
40026481fa7c5fb0 3ff84738241221d9 3fefc614a5e4a21a bff14e204b7aaf9c
4000726d62885b6d 400653f19c0a815a bfe0f1b0d7a7a36f 3ff8a1d2a2f127a1
bfc46ebcd4aa190c bfd3a35afdfd2bda bfd355ac139a741a 3ff176b56cded6c0
bfc49bdc52071f3c 40010760b40380bc 3ff1c0f8970ba443 3fdfd81eb78a1a8f
3ff66d6da30b36ab bfce0037dec622ff 3fad6a8901d0e7be 3ffa6c53b63bb739
3ffcf1e9600228f2 bfed0c1d474430ad bfde940cd3e180ea bfca9355ec8f7de9
3fed8b0d2b578d7f 3fe2954a18184c3f 3fd4493b21dcc1ee 3fd7e0ddfd0d13cf
3fe5ba3e226b7136 4002e64ab064c3bd bfc31014165f76a5 3fed9ee0cafd2e47
3fd93303f5ee304c 3fd1b1a8a05defed bfe216cd7630ba46 3ff1495b92b09a63
bfe9d6065f6b5534 3fbeb87318173a3b 3fdb3fb4f626fb6a 400796c715633d39
3ff30dc96248a6e2 3ff69531be8ee5aa 4003213de43080d8 4000c50eb5e29271
3fdf64634b1af27e 3fffc4e6ae9661bd 4001f35d52c61a54 40004353702e1105
3fd3782f1285dbd8 bfc443c7ce7cd69f 3ff99093550cd90a 4001e0ea739b5907
4000ead3f96f059d 40000d27bd3dca94 40061f3e352b3a5f bfe355d8e63ad24a
4002005dd12b663a 3ffaa3f8d77fbc16 3fe676801c921859 3ff4b5841ac04ad4
3ff100a3780b0f00 3ffd7fde6e6d0369 3ff247eac5feb11f 3fdb6cb1547aef7d
bfd9c7b2a04a483a 4000dc3a4dae4d04 3fd57449f2b7db74 3fe9c4e6de673f07
3fd09880c2f8b845 3fd863bd7d4d51e0 3ffd4bf1a9750fde 3fd0ada9859657a7
3ff34cb0e9cc572c 3faa9e02ea2565cb 3fc2cc97ccac618c 40021b4587fcabd7
bfd2fba0f445b0bd 3fe78b0b7585964e bff56714f5096c26 3ffa3abdfa1d64c8
4002f949f278fc02 3ff2b5d16d876a95 3ffbe1e678ce71f2 bfd739c6f2e4d633
bfd3073518966a54 3fe0d63dac2b5d45 3fbc28ef4b483d09 40070f876b15af51
3fd6de9e81f2c62c bfe55d5fb35c8c6d bfe852862ed71768 3fcf9a2c6a0dbdc5
3ff92dc885afc18b bfe3f515ac620eab bfd949bb5febd9c4 3ff9874d999e0c86
bfc9891dbee66b29 bfba2830cc7d340a 3ff554ae2b5916cc bfafdc0cbb13ea5f
3fe94dc4adf7edae bfdaca37c2b8cf95 3fdbfafa4180cb62 3fed6e93c326eca1
bfe36d63f292d08a 4004318eef48a943 40032209d70b7946 40001df3c5e4c409
3ff8155606a86d13 3ff0a84b65597735 bfe16aa5953472f5 3fec015326e7d11e
3fa8e39faa79a053 3ff056db6a2599f7 3ff9ce747de75f47 bfd970a341278acd
3fdd4baa1d33313a bfe0182dbc601ef3 4006815ba1b6d1d7 3ff2ab6b8be58802
40034daadec77838 3ff8155606a86d13 3fff2645ee17705a 3ffc87f07b6a7220
4005ba9acfff1110 3ffeaccef8df2a5e 400135409ae0b14b 3fd9cf86c5cabf9e
3fe6be74ff4b75f3 3ff512b12e322046 3fd63f5db228a9fc bfebd3fc32d84b7e
3fcb515aa51fafda 3fdfc8eab69562ee bff5b30482d548f9 40036b40aeff4fe0
3fe3ceda70d7a422 3ff5f31dbf6e585c 40025ebbe8605f22 3ffadc102f3efb48
3fee72efec271738 bfe9bc44083ce33f bfdb09511b242563 bfe24a97015f1a61
3fff9eea746d5023 3fcaf6748137ccae bff3d280a676831f 3fd0ce716ecf65b4
3ffd06ccf480ddc5 3fe08182ac405e36 3ff9c89117945432 3fba66e8996fbc09
bfb11cc8185f11f9 bfd4a00c7644fae3 3fc2c9ce0694050d 3ffab64b8f01c019
bfd144331c0e45d9 bfb9104bcb089dca 3fe28db686a1d449 3ff0528a82016b0e
3fe25dd2c4505541 3ff6e15af2642dd7 3fec44763d46800d 3fe2d8a29e58f64d
bfc65920ad6acfcb bfbc5db74039848b bfa8ed12a11fda1c 3ff102206215f098
4001118fe3e2a432 bfc2e9ee479a2eea 3fd0d43532675ad4 3fe354ab27a71ace
4002023140a69f8c 3ffc1a490dd8ed0b 3fffbb1f30fac7b6 3fd8294621339f6d
3ff6b820888e32ff 3fe4bf07a1897445 3fe6c47cc5c988be bff18b352ffcc57f
3ff17dd3ca699ea4 3fead3e375d84a65 3fe779b8afbd4440 3ffc33d85d71f31c
bfd78b1562513dcd bff20d1640afbf9e bfddaabc5bd0f56d 3fd1bb52c52b2420
3fe08182ac405e36 3ff2025ba2015241 bff580f415f3f58c bfe1bedbdfe43ecb
3fe8e6fff80f9bd3 3ff02dafe194be13 3fe9f89fbf3b3c5e 3ff1eb60a094b442
3ffa6e54efeaffac 3fc6a4dcd680de99 3ff13b2e538a980c 3ff3deb5a7547575
3ffda8e3d7a187ae 3ffc1e81768cab11 bfd06950350f234c 4002bc554c929a1c
3ff0b751a477939c 3fec44763d46800d bfcdc317e3ade285 bfd2c3d237d912a3
bf914240e27f5824 3ff0fe865a956e59 4000a0722d2dac63 3fdf4bd49532d4d2
3ff9d79fb372ef1d 3ffa301214939700 bfb4f7ab9eb3bc0c 3fc23da13e6b5aaf
bfe15c68e4c6d905 3fd5240ab66a4135 3fd44f6c941fee52 4002c6475c5aaba7
3fc9f5439049585e 3fd03b848c30caf5 3fe2cda9d804e09e 3fed6e93c326eca1
3f9e07d07ff4679f 40018192150f375c 3fd1a3db7fff525c 3ffe3ed83bb18cbc
3fdb3fb4f626fb6a bff2d6a090bdd101 bf9561f576bdda4e bff1ff5c87668821
3fbddfbf82991e39 3fb950700d943511 3ff3e084d16d5775 4005d56de701dc76
bfc015d27a283803 3fe3dde070a5f651 3ffbd3fe9388ea9c 3fe835ee5c53d432
3ffba0d988ffa9fa 3ffdbd68eb3cd3ae 3fed9176c0d64d48 40004353702e1105
3ff0480dd3cf70c1 4002a8551cffca04 bfd10244a03b168f 3ffd9b6609de218e
3fec44763d46800d 3ff77e7d27e09c5b 3fdf4ef4286e913e 3ffdd0398ecce88c
bfd07356a75ec3b1 3ff06a14a926a824 bff09002742d4351 3fdea15bd93eb9e4
3ff61fed7232bff3 3fd0aa378ee3617b 3ffea2ca993746d0 3ff23c49c0929177
3fc4d0f2d9941b8e 3ff5bdf32958f782 3fc49fc4746e2c58 3fd0732b953aec55
3fe6bb6dae4d571b 3fe2def5c6308a88 3ff1874d9509f306 bf9fa683d5880a7a
40031f585fa479ca 400002a8d83bab91 3fdeb7ef193c8fb3 4004452f09016d61
3ffaacd5edf90d4c 3fe84d6c8d68c55a 4001de3cb007b9d2 3ff0948f9c085924
40061f3e352b3a5f 3fe7836b3eec2577 3fd732d40b6af080 40015b8bd940b12e
3fdd6bc139b0dcde 3ff2c7bd79f1a031 3ffe88a301d971ef 3ff6982c8644145a
3faed721e7f59ba5 bfc16d2f9cac403c 400599a3bbb273a9 3fe506a5397b5afe
3fef1411a7db0df2 3fe9622792f8704b bfbb8be3dbc860a3 3fdf4bd49532d4d2
3ffa003b4a210622 bfd1e81fca328632 3ff8169447ff1c7c 3ff925f3536d6d85
3fcb238f7db9d34b 3fdea3a3df0c27e0 4004452f09016d61 3ff3b89b4dad61f8
bfd55c5830d68dd8 3fe58e0320c5fe58 3fe89afd3f3f074a 3fbfcee6a000b2c6
bfec79988ef1612e 4001226e650f8b40 3ff474ad518b6d46 3fea427f9c8a15e2
4002f949f278fc02 3ff1ddd61afefbe0 bf972a92db7646a6 3fe362da419d6c37
3fe63b3e00f637ab 3ffcf7ffe3e7d5b9 bfe13908e4fb80c4 3fe92c24c7dae9fa
3ff8ff17929a69eb 3fe2abd8c12288fb 3ff9961b357ea512 3ff8155606a86d13
bfeb15aa606bfff6 40000c4eec87657f bfed8c53cdfc7eb4 3fe60c86d3ddf755
3fe98b540dbdffe1 3feb34b02919db32 3ff6cc38b7dd9ef9 3fffa17b751ac9fc
bfeadd89ace1f796 3fb9f425b85980ce bfe5c135465780b6 3fec333e7a434122
3fe1384588456284 3fdd6158b7185f1f 3ff2010c61d0fedf 3fd49ebf72a3e1d0
bfd7f55b3186755e 3fe1262d665ce166 bfd4e824973d4bef bfe0f95281183aec
3fede631c91d9546 3fd7231e0c633c1b 3fef3fe560a530d9 3fcb31ec84d61a35
3ffc461e12a082c8 3ffe176744c1e8a8 40012549eb6cec02 3feb1d2fd0d75210
3ffc0e5e7d188212 bfd27766acc4e94d 3fd8a2cdc156e7c1 3ff9746231a42889
3ffe29de15385178 bff2bb0db6db3203 3fc30fb5d38f5493 bfad7e9a40c74a4a
3fe6578ba9a9f6c2 bfeadbeaaad7d13f 3fe38784e2abaa99 3fdddc1a39589614
400470ea6100f89e bfec769261b68f02 3ff0528a82016b0e 3ff71db756b17544
3ff3625a664d7820 3fe4cae58ed4f676 3ff8825969375466 3fd63d0a5148b33e
3ff6574bd4456840 3fece6207ca2a6cf 3fc449cf61b6a533 3ff2d58a1b6c135d
4002c319a2540fc9 3fea0592e8d4d641 bfd9b626f92889f5 40002828af253072
bfc35a101dc93cfe 3ffa74986b113eda 3ff4332031106e6a 3fe16940e81dfdb4
3fec5976a6d4e6e7 bff331b37761fd96 3fb09c528d8fac46 3fd13d9f48d6f75a
3feec617be541c36 3fe3616721454fa8 40002359cfe4afce 3fe4423d56707e42
4004be9bdf6f8bd8 400109140e4dc69e 3fe7860da5dc0b3f 3ff17265867d8ebd
3ff549acee2887b0 bfe72b72efcd0398 bf94569b1d12a9d2 3fc0a116bb3f83c4
400104a19180d47e 3fdf93ff88994cf4 bff1d74f9a6dc078 3fedfa63139106c2
3fcadec9ae6019b5 3feb8a5a17f5332a 3fd248be46445b76 3ffd7a587cd79a80
3ff95a712c4b80e7 3fee9b6f9170c727 40000c4eec87657f 3fef090ba41746f3
3ff1431ecc9a4b3d 3fc898211c8e2f09 bfde9e9e1183b25d 3ff585167f6d5b48
40004353702e1105 3fe4d28ffc52e3de 400155b552441561 3fe1bfde5344b13b
3feb6db3d0f6985e 3ff5d872c05084aa 3ff7b9d504409837 4006046b1e286ef9
3ffec1bad4f87416 bf983d3d81fec8f6 3ffacdcef03d36f7 3ff403c8d03b821f
3fc98a2927f26f60 4001670740dceee1 3fde2814d5aa167d 3ff1895b1d0c5725
3fed60109fd01271 bfe0d3f126ff5c48 3fd1ef94363410e7 bfbf8b020eeb27ad
3ffa02e0d913af90 3fe46e4cd192a8bc 3ff37468ed18e0d5 bfd07356a75ec3b1
40026481fa7c5fb0 3fb38a406a2147ef bf93a8b09971bb63 3fb5cb6d14ed4199
3fe6f4f3e056ed55 3fd1423b21e5bb3c 3fd3bfa2a7e34710 3fd9d4f1bee53459
3ff95a3c15fdb104 3ff006d79db4776a 3ffd7a587cd79a80 3fda947a3db73acb
bfd63d2a9a8bbbbc bfe62539ea69e3c5 4002a4fcdb9d362a bfee300bbde52875
3ff9884a5046bbc4 3fffbb1f30fac7b6 4003763a76075ee0 3fbb987ad053134f
4005ba9acfff1110 3ff52d85b5802d59 3ff56526f71e8526 40017b28174fdd5e
3ff126087ee2122b 3fe239e47976ffba bfbd073aae097674 bff580f415f3f58c
3ff5ae2e32412c1c 3fe476e1780819e8 bfdd71da06bcc6a9 3ff272bd337d39a3
bfedee11f56d0175 3fe81c3c280e8d5a 40036434133323da bfc46bc68f23ac14
3ff0453116611488 3ff0cb9dd442e220 400293ded55330b2 3fe45c0aa8e44cc5
3fe894d778226a90 3ff8ef61efc4092b 3ff06939041cd06b 3fe1b83832cc33fc
3ff084ade477eac8 3fed3c5cf5b3d908 3ff056db6a2599f7 bfb4f7ab9eb3bc0c
3ff3cf4e31eb1684 3ffb9c0b9b564bd2 3fda222a6a9ffde2 bfe30f7f08eff47b
3ffbbf2fa930f197 40044acb9363f2d3 40041140edc213a1 bfed0c1d474430ad
3fdbfafa4180cb62 bfe3f9542db46928 3fdd3375490468a9 bfe0cd3f7cb34de4
3ff2d862a635ab61 bfc443c7ce7cd69f bff149876034b3cb bfe29b07db46c273
40002828af253072 4005ba9acfff1110 3ff061591cedfec1 bfec3370e863edec
3fe041f81b0f5559 4002b51a60156abb bfe0f88e7c9f5235 3feb7dae7b8a27d2
3ff8e8275337bcaf 3fbe84d9e1653c45 4001870ef5094707 3fef4d174f8e4e72
4005ee1900fd0cbd 4001ba3fca84fd0d 3f67d40cad9bddad 3ff3cdbe187fcea5
3fc97847ea465218 3ff56988a436d143 3fe21e04772983fb bfce0037dec622ff
3feb3d337fd867f8 3fd0a9f9625e9e04 bfe13908e4fb80c4 3ff05d8003c3757b
400496fe22c7dfa1 bfdee46da7d4b254 bfeb9919cb6e29e0 3fc4d0f2d9941b8e
3fe1462c08b89235 3ff71db756b17544 400626fad4f75bd1 3fd6f4a591667853
3ff0ea2e909c017a 3fddff65ac89a6a3 3ffa9e81ee30437b bfe153a436108e25
3fb817ced5163e6d bfcb539429953d45 3fb717bdeeabc66f 3fc85cec130d2151
3fe3f40641fa6231 bfe87d2d71213812 3ffacf6c3e3a4bee 3fdc9a3c3ecda0ca
3ffb41ddba359159 3ff26b4602890d4b 3ff8dc22c62868aa 3fffcc4bd27bd50d
3fdc2f9406252836 bfc766aa0667638e 3fdb1e88d32ded15 3fe32c661e5dc7bf
3ff16aae34119843 3ffbec30bea0785e bff247bd4cc1da06 bfdbb2a91c83ce93
3fd1ef94363410e7 bfde450c173cb013 3ff4f5f47230d5ee 3ff4acd7a7bb9dc2
40040bb5c31209e6 40029ef64ecb2c44 3ffb543533fad77f 3fe310026e3e3b2e
3fe45c0aa8e44cc5 3ff0a45488b36b28 3fba26a120f7ecd8 3ffbc31d012aad60
3fefc614a5e4a21a bfe0101880a25a5b 3ff454e16d2489b0 bfc68463d41a97a0
bfc83dcf6374c27e bfb127531991125c 3fcc6ee90d954ab0 bff3d280a676831f
3ffb71de9bea702c 3fdf61c0391d5d5e bfe95a61432c64b7 bfe0081410a8d1f3
3fe690aa879da93d 3fe4930ca72e9574 4004eddae48f5b7c 3ff88b9639c68128
bfde79e4793476e4 3ff541f6b57a4df2 3fe2554efcaa17ae 4004659b7d3fda46
bfeb08e955daf677 3fedaf968224f11d 4005ba9acfff1110 4001a99d63901957
400205ec566a62c1 bfe805896c619178 4000b51aa5432cef bff580f415f3f58c
3ffa39b4d0561de3 3fea6ddeae629af1 3fe57a062557d218 3fdd7027b1ec0004
40018de90820e4b5 bfd3b11fe7bf73ab 3ff45c72c0dd106b 400275e7c3c43cf3
bfc6b671133c1461 3fd8204135197939 3ffadcbb947319c7 3fe27e9e84899fe9
3ff4b2664d1370c0 3fdcf283677d76c3 4002bc23284b0594 40019ce1bd015a8e
bfe6ee8cb35cccc1 3ffe30f64fd6205f 400088a95389f4cb bf918b9adbc9316f
3fed2bdb1080126c 3ff33c4f5d60b817 400272aede587e7e 3fe6c907d3f6c6e2
3ff95dba0c6b51c7 bfd6cdbd0f9ab200 3fe31ce9e88b7663 bfea8104291e2232
bfca19c2d2233509 4001e074277ace14 3fff2b83014b676d 3ff65788efbf3d0c
3ff9472db2b8a300 bfd76a407a6fd291 3fd58ad9bfc4e89b bfe12d717afc45d6
40060db63783b8b9 3fee0d9bdc359858 bfe36d63f292d08a 3ff65ad431e2c929
3fcd4368bd1443ad 3ff925f3536d6d85 400238d53fa2e5de 3fe107dc3a0384c2
3fec632494d2efbe 3fcd87164f4c4177 3fe37dd9bbd062a5 4006ebc42d3dfbb4
3ff0ee2f6d5a2346 3ff68a520ef7dd57 bfe15c68e4c6d905 3fe1d0fe69048b42
3fe5750109055bb2 3ff0c7570b34692c bfb921a90cd2b043 3fe1c7e9bcdb2e67
bfd07b5a6ddafff7 3ff429565c52d22a bfe73d5a125e87f4 3ffbe30a601237f1
3ff90a7a11127362 bfe7c1c4a420739e 3fef090ba41746f3 3ff4258104ca041d
3ff015e85418d5de bfbccae45bf1f721 3fff5beb6ec62118 3fe35ff84fe43344
3fe690aa879da93d 3ff5f04aeaca4e88 400287bbde78d05a bff580f415f3f58c
3ffd45b5b9b2ed25 3fe495c4b991ee9d bfe40d963f144b2b 3ff03829038cb5da
3fd96208bd70424e bfd8308f6ff0ed5f 3ff66d73056516d7 3fed9176c0d64d48
3fe7c4afcf97fc2c 3feef09dee42152e bff35f11ed2fc5a5 3fe488254fc47071
bfec5f710e9c8c66 3ff58056c8ab398b 3ff937d1dffdc1e7 3ff07281c201a9d6
3febf8d11439c0af 3fea653a11823f09 bfef98ccb136cbdc 3ff3fbbe7e921bd1
3fe6c7e30565ad0d bfde007b0e96eca9 3fda947a3db73acb 3ff9a47720f97ed3
3fe53602304d2bee 3ff62d9a807d0e96 400420f2a569a026 3fe19e369665dd9c
3ffaba88f97ab244 bfc9d87bd612a743 bfed0c1d474430ad bff3d280a676831f
bfcbb4e5be797bae 3fe78f8e1855efbb bfc1b2c079c198fc 400287bbde78d05a
bfeb6636ce162056 40070f876b15af51 3ffece1663244d90 bfc5b8621641042d
3ff42fa8e14a03fc bfe852862ed71768 3fbe84d9e1653c45 3febc268869cc801
3ffe6c6dd9bff24a 3ff62af0f3d4d0db bfd3e43ae06b7bb8 3ffee992fa661bde
4004ab9d4e758abf 40002ff96da1e7bd 3ff98311062882a3 3ff76643da0e3614
3ff579a025fe22f5 3fffe4dea85d3791 bfbb40d8eb47c4d3 3ff046b7ccfe203c
3ff2ca195019b83e bfe25c9955f3ec8a bfb96c8855f338b3 bfe0ea0604ec353c
3ff107189f60680e 400354be17f8228c 3ff4f205fdc252bc 3fd51e0debec51d6
3ff2bcc0158ea707 3ffcb419008b2216 3ff3d9617908cbba 3ffc1aa0139dc2be
bff047a1aed1f16c 3f50606fe52a25a7 3fcd353f5e3d0196 3fdebe8703ab4d27
3ff42e675796e504 3fe6b4d062ae0879 3fff7148847efe17 3febc268869cc801
bfbfeef43a8b10da 3ff5610e31653cac 3fec885ea0c1638d bfd4e65f7f8578a9
3ff8c4d7a63ca21a 3fed7e7da35ebaeb 4000136530774467 40001e1d95c8e3fb
3fbf5d534d5d58ba 400539b57f543d78 4001062e75df1a53 3fea7db997acb7ed
3fda8bd7e61b5b8d 3fde4bf43e58682e 3fdefd3acd344c41 bf72cc056869db79
3ff9153817d067ff 3fee8d87a8fd5f80 3fc2d9f21ecc5a1e 4000b84f096a3cd5
3ffab3280dee046b 3fe49dab6996b557 4001ce9af244137b 3f9dbd5a123768c1
bfd27766acc4e94d 3fd7b6908355c37b 3fe912ab9247c35e bfe76f83311d0b3b
3ff27a4b27707a63 3ff7507c653c4a22 bfe2d6811d5b2080 3fa62edd33a2e84a
bff1396ffce05efb bfc8ed58993aef8b 3ff5f65a8dd1fb51 40017770b6269053
3fdb3fb4f626fb6a 3ff65c62ac58e68f bfdb448b532d2b70 bfd6b9687369cc42
4000b428b8e3eca4 4005d56de701dc76 3ff9033332916084 bf7e8424b7498634
3ff36c537b4b21a0 bfc03d7e3a9825a7 4004055c21b822a0 bfe03dd4d04e5340
bfd355ac139a741a 3ffa6e54efeaffac bfe9bf0a8da197ac 3fe3361932b38103
3ffe17065541d31c 3fe239e47976ffba bfeeb5efd446b034 bf8b5d6f68c02bdb
3ffee2064c0c72a2 3ff98549429c24bb 40030b1e0d2d8f69 bfd6141a2136d6b2
400498500f29f137 40058108ef73afea bfbd925c683427f4 3ff903acef234029
3f8c803a4e49e26c 3ff9fa352697f186 3ffa534d79df08db 3fe3357ddd47c44b
bff09002742d4351 3fef090ba41746f3 4006046b1e286ef9 4000a33441a2a751
bfe9fe026dd566b0 3ffacfc9553b3e08 3fdfa73b511e6737 3fe1462c08b89235
3fe55160186fa576 3fbc28ef4b483d09 3ff79d86da77ce4a 3fde2afe78202215
3fbdd2cd7c934db1 3ffd7fde6e6d0369 3ffd534f8eaa0c64 3fd963f57070d995
bfefb446c4c57e0f bfebd1ee8ce39d9e 3fbcf07bc8e1f137 3fe0962d410b4222
3ff73384517680fb 3fc1fe630ae78e9f 3fef171568ab5bb3 3fe2fb16563b96c3
3fd5d58a07d4d130 3ff6d8d168f9dab6 3fe2fe7a83eeb932 bfc2527d973f791e
400183f29146da44 3fec66b99a47497e 3ffe7a9dbf5142a8 bfe36d63f292d08a
bfd811e3750365fb bfdfb58fe94ec87e 3ff90b816bfe1bfa 3ffa31975023bb0e
bfd72ec99e63a437 3fd47920d69f0aa2 3fe005fdb2feac97 bf9026a2a83b11d7
3fe0f6fd5710f9ab 3ff3962590c3220e 3ffea66ab7f190e0 bfe7c85759bd4311
3ffb543533fad77f bfab9cc0fa0d16e3 3ff9884a5046bbc4 3fd5da8d4f0f96e3
3fdb93fa0caba8e0 3ff4f72e2444cd92 3ff4a89e179f2065 bfb9fdd97f6a70c8
3fec44763d46800d 3fef308b100baed0 3fe9c62984723be9 400482ae089f82e2
3ff5a6265cd8da94 3ff83e0f23834686 3ff60799951e02c7 3fd44977ec593ba1
4006b2e39909109f 3ff370efd387e979 4000e8e48ef96c0a bfcae14f06c1387f
3fa7a282ddb07995 40070f876b15af51 3ff60196fdda2694 3ffd71df5e25e6a4
40023fb7739f50ba bfd6b1d3ea159843 bfe8debc3acfb577 bfc9891dbee66b29
3ff0fbf86493af86 40006376759b26af 3feb23cd2cbbb6a9 3ff7ffa1e2b85e74
4004bd10785be7cc 4002f949f278fc02 40043ebd5ddb1130 3fe99ad20ba88578
bfd540112795604e 3ff4d3e280ac7481 3fd7b386860808aa 3ff46ec403cb4800
4002edc456c44bc6 3ff34149419c5570 bff522efe5b03ff2 3fc757719c578fc7
4002f6f1f7884ee7 3ff722204ebacedb 3fedef8244044e11 bff331b37761fd96
3ff7f73e5ee80afd bfa55cd143b30c88 3fdba0fe151c27a5 3fd43ff2999b21ef
3ff68114faee6d2d 3ff3d8ba5617d820 3fff4019c908367b 3ff0817edeb2af5d
bfe12ddf089a4850 3fd9575f19e052c9 3fe37dd9bbd062a5 3fd21a2de8cf4acb
3fe68d15608fe53a 3fe9cabf2b2a49ba bf8341b617bd42c6 bfec6892e10b63a5
3ff76f61237edc06 bfe26be811f1e019 3ff286a367a655c6 3fe00dd5cdb6e0a0
3fe5bcf61048c23a 3fbd3aa8e608d7b5 3fd9daec0ccbb1e3 3ff1eb60a094b442
bfd0d058ca53b5bb 3ff429565c52d22a 4001a99d63901957 bfceb5b785d142d5
bfd9eb20d4a3b3f2 bfd540112795604e 40015b8bd940b12e bff20f13fc57fa04
3ffb1ddf8d762ac7 bfeaaed6acfa9019 3ff1eeb52b73fe44 3f979971179fa5b0
3ff3cf539119dd45 3feaac7897af0bef bfeec5d97b550439 3fe2b632395ba3e7
3fe205b50de5c900 3fd54b2de75a6db9 3ff267a5a7254219 bfd746c90ae4b370
3ffe2cc83da5e48a 3fd3fa9b4ecee266 3fe58e8f3575ccaa 40051c5693a53a85
3fe9a33c063ecd54 3ff0cf96dccebc96 3ff36549ae3c64e4 3ff51463034ba2d3
bfd7970ce605cc31 3fddff65ac89a6a3 bfe3ae907cd96507 3fe0afd9a6ba8ca2
3ff0597fdaa49329 3fced67fc8692ae3 3fb74152f298b843 bff287cde3d87cc5
3ff4858cca66a1d4 3ffeab9e9d6e29ab bf981e658fe6707b 3ff10ca80641d6df
bfe04029796a998a bfd175a499fde504 400088cc75888b45 3ffc420ca256d990
3ff98da7f637aab4 bfd9d715564925de 40084722737af626 3fd950241d4ae5b2
3fee4e562e047646 3fede631c91d9546 bfc1990f260118f9 40058385cd7d9b61
3ffd25e8a4331610 3fe71ed7caca02b3 bfc46bc68f23ac14 4000a6692a10bfd1
3ff12b6a30bcea74 3feb1e42cab23496 3ff070818d0c063a 3feccae3f1723644
bfd79c2f15e66c37 3fe1262d665ce166 bfd13856200edbff 3fffe607f63b4f1d
bfad1bb8d5b1fc8a 400430888891d2cf 3fe2255f88eee5b2 3fe1a57914fd6a46
3fd03b848c30caf5 bfa96ea3b8b3c970 3f92b381260a4140 3ff651622127c240
4001ce9af244137b 3fc7fb63b3149a2f 40005c6ca2194ba4 3fe18a402da5a22b
4005d56de701dc76 4002fd8115f432d7 bfe7df12542afb1c 4002c7b6dd8aaf33
bfd6ec349ca3718c bfcd05b18274187d 3ffb36e53d115c4b 3ffa3c7f8b08cc2e
bff5b30482d548f9 bfe07ea453919f84 3fea5d0da3099c60 3ffb01d40104ed2b
3ff6631b84e0372c 3fdd4baa1d33313a 3fff9587b6f01f67 3ff783657f9e73e5
bff5b30482d548f9 3fcc6a1b1386383b 3ff4474aaf52c2c8 3ff4b2a2a87403f2
bfcdcda872155ba6 3fec29c7c041d191 bfa867c4913f6936 3fea4c41eaec7f3c
3fe46e4cd192a8bc 3ff066cac73978db bf9b6c6b8dd8583d 3fbb0eb3ee862703
3fe2f62970367df9 3fe1809d2303110b 3ff46a1085295f73 4000278dd2075513
3ff822e1eefb2557 bfe312b85c41075f 3fe59ae58664966e 3ff48eb0e6831922
3fea8c6a4cd478ce 3fbda88ce2a8f313 3ffa3c7f8b08cc2e 400112d74864fbff
bf493959b19b553c bfe7155a02eecc82 40001e1d95c8e3fb 3fbd70b32cfabce9
3ff4d3d05ad48e33 3ff7101dab9e21d2 3fe042bd9623aba9 3feb43f4e28e2f0b
3fbcd1ebafe94896 3ff5d825f82e8c6d 3fafe496cfda115d 3ff4fe8a243f7a7a
3ff4304aa57842ad 3ffcdd061b428f48 3ff26067f5570b20 4005c3dc03ca7234
3fffda07e4b812ce 3fe2eb5d20a483db 4003c332476fdf8e 3ff321de9716ffcc
3ff8a38a56aba761 3ffeab9e9d6e29ab bfc5edcef45586f1 bff2d6a090bdd101
bfef28d8017dd722 3ff6cc38b7dd9ef9 bfc8ed58993aef8b 3fe0626cab22def2
3fe0bb31275166f3 3ffa3abdfa1d64c8 3fe9fbd101cdc0a1 bfd55c5830d68dd8
3fcb238f7db9d34b 3ff8cac84715f186 3ff3d536d0cf8004 3fe6173ce7e59091
bfe6523423e2e8fe bfe41990133b7b73 3ffde158b6fc6b28 3fe98b540dbdffe1
40024c1d13337cef bfed8c53cdfc7eb4 bfad72236769388f 3fe72a7726087105
3ff95dba0c6b51c7 3fd63d0a5148b33e 3fe56309a395a977 bff5b30482d548f9
3ff4bad27bacb718 4002eb6aef01bb67 3fd52d87b542b2a5 3ff3623d81176964
3ffc74b3ddfc67ea 3fda4f8f9b7a57db 3ff376db2fbac79e 3fb391573edfd4f0
bfd768700b7aee62 4003c1e574e8abfd 400502d814609c89 3fe80fa8f3bf95b7
3ff6d48e75cc912c 400000a16faf886c 40022f8d0436ee12 40061f3e352b3a5f
bf918b9adbc9316f 3fe412319708d786 bfd7a2efc5c0bf6c 3ff65c62ac58e68f
3fe586ee07f97fde 400404902edd2a32 bfdefaed5d6c7ac7 3fec8d780483fc3d
3ff82520b8bc8de2 3fd6a5adb68de65a 3fe89afd3f3f074a 3fe8917cbd604d64
3fd79a41a952c63a bfc80635e2be4d34 bfeec5d97b550439 3fb53514e3c25b11
3fff09129695ff29 bfe0f25f7c11b61f 40001e1d95c8e3fb 400433aca9fdc7e8
3fea7db997acb7ed bff5b30482d548f9 3fe80389eed03c5d bfdfdf51a9e92141
3fd545a097abf163 3ffa1156efd9f960 3fe137cc04809b12 3ff7a855902e34b4
bf9a81a24031fc18 3ff1d3f0f9cd3c1b 3feb34b02919db32 3fdf8e95429fda7f
3fcc6ee90d954ab0 3fd3175d6c98d315 3fed6e93c326eca1 3ffb138dbe6b9d62
4007ddc96f9be195 3ffd9b6609de218e bfe721fce70f6708 3fe09dd58abd0b22
bfd3690719197959 3feaca7beafff988 bfd255d3d806be7e 3ff1eb60a094b442
40009ff45602375a bfee1100b1253881 3ff5559307abcafa 3fed50bf1a488c5f
3ffcaa51fe2a6e48 3ff93b160eb22c5a 3fe197543cd39e50 3ff4960f1afe2329
4002f627537691b1 3fc1b595c43fd0b5 3fe7d6e2d513f118 3fecaac64bc20281
3fe9511386519c11 bfd77a360c85a886 3ff0eacd71d11c4e 3fe37dd9bbd062a5
3ffcda3df0eba5ec bfaa4bcdb9ba81cf 3fffcaba5728b6f1 3ffea764e3df2927
3ff5418381d8d26c 3ff1fe453d4f5d02 40008d686799a786 bfc68dad15d15a67
3ff8f1348eb3d3bb 400631c20751a5e7 3ff348c18aa2de8e 3fd22b705d18c066
3ff1487b130483ec 3fcfad7f7d2b0a03 3fe3dde070a5f651 3ff59880cb73d9a9
3fd562ab18edbbb9 3fe729cd4868d889 3feae36a6abbb9c3 bfdb3fae02ec4596
3fe0257eb2deb47b bfd30e42472bb71f 3fee0d9bdc359858 3fceb3040f15deeb
3fd6f308183f7664 3fc099c5ea604209 3ff3d9617908cbba 3fc0e6f0943b957a
40031f934238aeac bfe22502ae8a1290 3fd4df99f2812f0b 3ff5c2edcf117a1b
3ffc69c8fe471ddb 3fdd9c18ae499bc3 bfd739c6f2e4d633 40006d17bdd625db
3fec21a436884730 3fcc6ee90d954ab0 3ff4875bf47f83d4 3fffb69ba2453905
3feb65fc6f8f2838 bff3425e7a4abe19 bff2751930e45ce2 bfd65e104f066c8d
3ffa919bdec34512 bfd536ac8ce7ab34 3fdd21538847319c bfe3793e6627b62d
3ff95a3c15fdb104 3ff0bee804a314b4 3fc6d3ff2447b7e9 3ff6e15af2642dd7
3fd39629c85364be bff1ff5c87668821 3fe1809d2303110b 3ffa49abbfad2cc7
3ff148c5122bf9fb 400147c25281e8b6 3ff599e31cd723c1 3fd6e6ae7270c6df
3fd5e9d39b7b7596 bf90412561a5bdd0 3fe0649b291a0d94 bfe49d0405ed616b
bfe5348433e3fb22 bfdfdf51a9e92141 3fd5f138a1b95934 3fe0abe4de1f8551
bfe62539ea69e3c5 3fc0a116bb3f83c4 bfe45749ce8bfeda 3fff000e7612ae1b
3fe6c14d23d7f4ce bfee251a746520c2 3fe3fc00b82216a3 bfec3370e863edec
bfe69fa7b68726f5 bfe852862ed71768 3ff6d48e75cc912c bfeca8d386c16db0
bfd1340cf2a50433 3fc2c9ce0694050d 3ff665d4f3c98d34 3fe1d8804b262ef4
3ff3326bb971da7e bfd8b1d262086bd0 3ff91d594e37d809 40020a720c17a26d
3ff8da14566f1858 3fb0e306bb254c9d bfd72ec99e63a437 bff2d6a090bdd101
3fc0e6f0943b957a bfc72dd2bf3d810a 3fa239054ca73020 400186f98887eaee
3ff990b4ab717a51 3fdae647f20c25b8 3ff3b5dda73ac9f7 3fd841263038369d
3ffcd3e71669f430 3ff8bc07e0c8a71f 3ff9ce747de75f47 3ff8dc22c62868aa
3ff406184b5d8be2 bfdcc2f3d9796b89 3fea05e86a3ff59d 3ff1d880cf0d4262
4003d70bc429a192 3fd7534b41eea428 40027b8260d91fce 3ffacd193fbe025a
3ff3625a664d7820 3ffa278df5d5bab2 40029a5111b42891 3fbda88ce2a8f313
400104a19180d47e 3ff2f1b8a7368e0e 3fe1544ac6ec2d5b 3fecd8abd25545cd
bfd4590f65e90396 bfc443c7ce7cd69f 3ff6916d54ed9c62 3ff4489ef5c5bf63
bfbd38c858921280 3fe6c7e30565ad0d 3fcfbd9846281be3 3ff822815dd5857f
3ffcdd061b428f48 3fe05d24628ffa1a 3ffb83c14d8b08dc bfceaa96ef9594c9
4002ea7cbb374421 bfa915b5d62b176d 3fed6e93c326eca1 3ff5d825f82e8c6d
bfd8a392c13766b8 3ff95dba0c6b51c7 bff0234f44bae28c 3fd258379b794401
3fb23a4f06524abb 3fe2255f88eee5b2 3fe98b540dbdffe1 bfdaa7a2c69d8c55
bf9bff48772a160b bfe2f975f37131b9 3ff7e9fa2015bc71 bfe2d5720bc3c605
3ff6b8c3b0b93970 bfeb0a0a9e560a0d 3f879bec4f2c9ebe 3ffe3deec0d854a2
3fe7c079b3d4f435 3ffbe1e678ce71f2 bfe4400349c83d49 3ff3ab6112f67e8e
3ff38bc75a40a8ad 3ff18d86331d292d 3ff8022af97bbe11 3ffaa3f8d77fbc16
3fe104a8dca542ec 400439db5bff7e96 3fab75a407f56010 3fec1287b1874b65
3ffce6b7e084deaf 3fe5b0b521bcd016 3ff34cf4c40c5c7f 4001ed5343dad88a
3fe4e92be764af06 3ff33fb44a45b9cd bfd8e09786e8c224 3ffd4f9b88458e9a
3fd25778786206c7 bfed1872b121860f 40063dfc6ad4679e 3fd6f8349c2c36f6
3fe9aac797bee874 3fd7b386860808aa bfd53ad3e3d87da0 40021ddf1981baed
3fe113331485fc62 bfbc5db74039848b 3ffb4e311d2ee095 bfd2fba0f445b0bd
3ff788950310e90d 4002c7b6dd8aaf33 3ff98547496b7ae2 bfc5578ce5bb7714
bfce09cd5f0b04c2 3fea5e1c2869cf06 bfd07b5a6ddafff7 3fd377e20ac08af7
bfec3f89e2be6fb5 bfe5dbdcc14698aa 3fe72a7726087105 3fff666d19617329
3ff62cd0540c53de 3fff8be39fefe4fc bfbc5db74039848b 3fe4a0c4b3f6ecf4
3fe80fa8f3bf95b7 3fcfbd9846281be3 3fee0d9bdc359858 3fd63869944f5c7d
bfd2efdc4e0a6910 3fd0e02d90328440 bfeadbeaaad7d13f 3fd7d2b21df1e732
40004f67aae9e139 3ffcd3e71669f430 4000c8c9e3c90eff 3fedae7f0410ec68
4002815c1dfd6052 3ffd43d8e92f633e 3fe5b14b5d430dc3 3fddff65ac89a6a3
3fe459dee9d6c2b6 3fbda88ce2a8f313 3fed773b99bb13f4 bfe65f4c2c48c295
bfcdc317e3ade285 3fff576c0c7fe7a0 bfe14cad79bd5c27 40041fd541635c4a
3ffd01ae2ac6d777 bfdcbe5b56ca03e0 3fe5a4a49be51891 3ffc9083ff749154
3ffb38b4672a3e4b 3f879bec4f2c9ebe 3ffb6a0e239c5a41 3ff4b2664d1370c0
3ffd5802fc6416c9 3ff2911a298a9395 3ffa9c116ac94aca 4000b6cde7c77d04
bfd69d9a36eec4cd 3fef48a9c3278aac 3fe9540236906e7e bfe1ab858d75a9d5
400022a562fd6e4c 3ff27301a5c72e7f 40074cf6c739df50 3fd6498abbafc84b
3ffb458f5b0ab3ce 3ff0bc4f21b1983a bfdb8d44df89d716 bfedadf23c68f7d9
bf99f4cd747f366c 3fdee928fa41caa5 3fe2e5375987e721 bfde5b3d1ffa080f
3ff15ff0c9da2259 bfc41ab3d989b051 3fd32108130a5342 bfcabe04e822262c
bfec79988ef1612e 3fe721232c6d30a1 3ff4489ef5c5bf63 3fe113331485fc62
3ff52b9cc252b0a9 400436f0fda2a342 3fc4d0f2d9941b8e 3ff73384517680fb
4001726f8402508d 4001efd8c816e1f8 3fe63ea5724f708a 3ffbc31d012aad60
bfd4dc470967a09d 4001dc3459cebf88 bfeb9919cb6e29e0 bfed3cd0068e69a4
4006378b538d73ee 3fefb78aa145e10b 3ff1bc215acbd87e 3fa028aff92f03df
3fffcdd9392e3b7d bfaaabb1a0198f7a 3fe77e10154fc597 3fca8dea7aa182a3
bff1ff5c87668821 3ff8f89d2a84b7d0 3fbd70b32cfabce9 3fd1cd0177ed34b1
3ff8709f43152875 3ff497113f324342 3fdcf2d951109ac2 3ff42affb0518376
bfe4e3107dade9fd 3fec1287b1874b65 3ffa196630d38c2c 3ffdb0deb0381d56
4005fdf9730212c8 3ffa5156bcc64834 3ff5a6265cd8da94 3fee21398a8bb569
3ff7c013ea448aff 3fea15fecd9fa9cc bfd69cc25ef82d7e bfd272fc11c4d8f9
bfb49b0fbfc11fdf 3ffc32374c3a4858 bfdab5c3c2d14910 40012867df2bd2db
3feb83b42b32d2d5 3ff56b47e9e1a4ed bfd2efdc4e0a6910 bfcfe9444cea982a
bfe4936e1eda4afc 3ffdca1db35145c1 3fd0bf2b58b7ab22 3fd9e4356037904f
3ff8cac84715f186 bfde4bbda4bd073f 3fe55c93c2cb8f18 3ff4b2ff743c2271
bfe3d0642c6ffcc4 3fe5a8580a747986 bf92675c7723302f 3ff2d44db25a41bd
bfe098d59b5f75bd bfe0f88e7c9f5235 3fe18a402da5a22b 4006378b538d73ee
bfae751d7ed0294e 3fdf4bd49532d4d2 3fd16079c7016c05 bff190cf40ade665
3ff6d2bf4bb3af2c 3fd382d51bcd9613 bfe9d9672227f8f3 3ff2cef496f87fea
3ffd1826f79711d0 3fd68427752dbc65 bfd78b1562513dcd 3fce29ad934ef906
3ff88cd527726ab6 3fdd49e4ccbc7385 3fffcaba5728b6f1 3fdcab5d33555bd9
3ff7346c6223d260 bfd6ec68d4e8abec 3fdfee34d0f0f4d7 bfd12a07c6f01904
400121c059bd9e61 3ff16683afc61bee 4002c3c67227776d 3ff5499d34a8ab68
3ff879c9f5a9af32 3ff69531be8ee5aa 3fdcd32ef74abde3 40037f9322fd553e
3ff2c61d9f4987fc 3fd8f7f6e02389ac 3fe539e36a0e32a4 3fc4d0f2d9941b8e
3fced67fc8692ae3 3fd43ff2999b21ef 3f66dcd9c48f9fc8 3ffbdfadb5b2a188
3ff454e16d2489b0 3fe5f11644d64d00 3ff8753d3e8800f9 3ff233d68671f661
3ff6158c860466cc 3fc566900a6450b6 3ffe7a9dbf5142a8 400093ee72fdea53
4006c99498852041 bfe0770a28ff11db 3fbc76cff7e8ff22 3fdd4baa1d33313a
3fef9cdde139d1af 3ff8b9b817100d6d 3fd3399950bcfad7 3ff5dac905a69805
3ff8b198d6c315fc 3fe19c7d348a1cd3 3fe8a655ea142193 4005bd17ae08fc87
3ffb38b4672a3e4b 3ff0221c67de9341 bfbf339d394d64d7 3fd5340bd5d4f130
3fecba0a85cb146e bfc9ab2d959576e4 3fd7b6908355c37b 3ff098638188c2ca
bfda54bda3056d96 3ff291b38f5dde51 bfd8a08ee8fdebea bfd8a392c13766b8
bfdb00bd2be3667c 400174c64de936c5 3fd24ed1ca3c322d 40037e52a7a402dc
3ffe2ee12b8287b1 bfa7391ddd16f094 bfd2a22111801f5b bff2658060425452
3ff5146e25404a6a 40010b40481f88cb 3fbab16cf3c791f7 4003d999dcdc93a9
3feae36a6abbb9c3 3fee130197ad52ff 3fe959b59bf7be7b 3ff7185840128b86
3ff3942a6b23a344 bff331b37761fd96 3ff4165bf4b911ac 3fe6f5417b350ea9
3fe7dc97c236105f 3ffe7bfd6f714c9d bfc0d7ebf928ffdd 3feaf25b48944b14
3fc54f80bb50623f 400003ca4edf349b 4000f2bf0857340a bff04ecf552a9f14
3ff993883da1fe4e bfba145bf20720e1 bfe0ac8f8eb859e0 bfb3e5ad33f1be52
3fee9b6f9170c727 3fe8c48ecd25bc5a bfdb8d44df89d716 3fef94a1748dab95
3ffbbdc4e9305f1c 3febb36407331acf 3fac9553dcfeb372 3fc98fd0be6fe625
3fe0bb31275166f3 3fffe5a8dd5824ef bf86fd380b4bb162 3ff69344e8fc6839
3fd968947137b022 4004f44c8fb5b7ad 3fe5b4b580a6d832 3ffbbaacd724eb50
bfc65920ad6acfcb 40041a1b3cc3badd bfd7cbe27cafad09 3ff4bcdf375a8eba
bfe36d63f292d08a 3fd4b44683de170d 4000dc3a4dae4d04 3fd5fea4fb40cfa4
3fea18727145351a 3ff05bc3dc3fb258 3fe8c258cd9ca19e 3ff7fe8ea33e43b1
bfe5a21d0bb269f4 3feb23cd2cbbb6a9 bfecfb924345daea 3ff4b2664d1370c0
3ff0163d637bf9f5 3ff59c6078a6c570 bfe62539ea69e3c5 3feffbb0ada09046
bfebd1ee8ce39d9e 3ff68a520ef7dd57 3ff4ea9a4b9cf808 4002d36dcaf34a0f
400193fe72ca7b33 3ff3ea608e9a8931 40015f630af5ed26 3fcb515aa51fafda
40047d7cf82725d1 bfe042ab910a4a18 3fc3d677cdc4a370 3fd42fdaeb9c04f2
bfd2a10e7d759839 bfeee6e848548eb3 bfeadb9553e5ec3c 3fe7dcf31c059513
3faff7fc960c5185 400041ccb8d3d737 3fd268abf7aa69e7 3ffbbffb3258fa92
3fe09dd58abd0b22 40022f30697f62b3 bff331b37761fd96 3fd2f94eb0f06a55
3fe17000aa0fad0c 3ff6574cd3e1d22e 3fb023869703ba2e bfbccae45bf1f721
bfc46bc68f23ac14 bfb11cc8185f11f9 bfd8382c800ad1ee 3ffd4d6ea1232825
3ffc60a00cb92d45 3fd1403ec0ef5e1a 3f9dbd5a123768c1 3ffcd3e71669f430
3fe0e1e61c2a6150 3ff06939041cd06b bfc1b2c079c198fc bfd2249c2b909ab4
3fd8a95ddc3e542e bfc1990f260118f9 bfe55b520d67de8d 3fee57eaaa3d57c5
3ffeab9e9d6e29ab bfdbe78406a4c491 3fffafc503875d3d bff404911357d68c
3fe38784e2abaa99 3ffddd1ec9bf6f6a 3ffc1c324ed58739 3fec885ea0c1638d
3ffda185aa8dae1c bfce09cd5f0b04c2 3ffd7416f5d8a3ee 3ff77e7d27e09c5b
bfec3370e863edec bfdacc0b73e81ce5 3ff7f71347840969 3ff2c5d257e94dec
bfe2e75a0cc101e1 3fd1217da13b937c 3fceb0380deaf505 4006412753db0635
40002c027ef945e5 bfcdb7ef074ecdc7 400282a58dcc43c7 3ff9472db2b8a300
3ff044a1cd6f50e5 bfdcbb780c6b832d 3ff80e9a816f7620 3fe14a422d1c3683
3fc341b92954e686 400174c64de936c5 bfd3e43ae06b7bb8 3fead3e375d84a65
3ffd6cebc7e3a284 3ff00508739b956a 3ff8eeb5d9a57bfd 3ff97f0f4af374eb
40075367de80d300 4003f14f1383cf26 3ff0cbf6415c7f65 3fc814437e518546
3ffc45bc2342e2cf 3fe631dbe9e4049e 3ff342a1be5a93fe 4005010831278566
3ff403c8d03b821f 3ff3fbbe7e921bd1 3ff4b2664d1370c0 bfdae5a29aa1be27
3fdaac62caa233d7 3fd63528bcfa6b4c 3ff6befdecdaddc0 3ffeab9e9d6e29ab
bfea01092bd7d83c 3fe553778cc9af04 bfb998a55bfb19b7 bff20a0f19c179aa
3fe8ad68e5c03c50 bfd1e6b77ee9f0a4 bff27dff994e8518 40075a6740e72286
3ffd7fde6e6d0369 3fd5baf2419d91b2 3fdd3574d21785bc 3ffd25e8a4331610
3ff7ea71aadc36af 3fe60f5ca9913d51 3fe608e87fac3354 3fda38d857c1632d
3fe8692f3b11b9b0 bfd67975cb331a32 bfcfd1d9fe744f46 3fe4cddcc99720b6
3fe92c24c7dae9fa 3fe919109ac202e4 bfc443c7ce7cd69f 3fdd00c91b8893d9
4005ee1900fd0cbd 3ff4b2ff743c2271 3ff627cc72109e22 3ff852ed7d11dd21
3fe51e493bdffa6c 3fff9b31d2133d25 3ffc3c30303c00b1 4002c0534c76c7de
3f919e046c33c600 3ffe8c691eea8bb7 3fc2438806c0cacb bff3746ee72c1186
3fb248dc8949e19d 3fc4c951c7c5f7b6 400179a0259d6e90 3fc41ee33f573551
3fdeeb31e970c4fb bff2d6a090bdd101 3ffce6b7e084deaf bfd47343e256e263
4001ba4fa3866447 4002611733ace0e5 3fbcd508999b269c 3fc8a97134519359
3fcbae6fcbc1840e 3fd772fa9b996d93 3ffcf7ffe3e7d5b9 bfba44a97e35bcb3
bff5b30482d548f9 3ffe7dd445e95ce6 3ffc6e3989241750 bfef1ceacfc61652
3f86e3a41c939380 bff0ee940189491c 40070f876b15af51 3ff0eaa802d51f15
3fd84996965697b9 bfd4362bd8d81bb0 3ff12f050bcb26f6 bfdb09511b242563
3ff7a0e14c1ad779 bfe99af3f9192608 3ff1e97f578ea73b 3fd238d8eddeb1fd
3ff2895f982d9657 3fdf065fde691f00 3ffe531054b9df6b 4001aa2160d9afb7
3fdd4baa1d33313a 3ffa0a264fd562a2 3fe894d778226a90 3fe6bdaecfa630cd
3ff5c59861502d1d 3ffe176744c1e8a8 3ffd7a587cd79a80 3ff1261918432b7c
bfa9e0de99987dd2 40015b8bd940b12e bfd3ec2fcf98abce 3feec617be541c36
3ff01e68b17c293d 4003d70bc429a192 bff0da52380a3b3f 3fe88ba4709d42b1
3ff0f96b7c71b359 400648ddc390ecdf bfdfdf51a9e92141 400135409ae0b14b
400205ec566a62c1 40068c253c60f042 3fef2cb3464d1314 3fe4d9fb2ea2cc05
3ff8e8275337bcaf bfd08d0076a41cdf 3ff0edc505895b77 3fc198d2419b8ae4
3fe9fb738028876b 3ff4a2270b5d81b5 bfe0120cd99e65b5 3ff6e15af2642dd7
3ffb3234bc5d2ae3 bfbfda6af7b85f0b 3ff6459f1c8972f1 3ff3a04e7fb8dedc
3fe8a03022f784d9 3ff4fcd919548b93 bfeb5f5df66a3872 3fb1cb16e6a429d4
3ff3c39182ae40a2 bfc6ecf57c0c1e8b 40017770b6269053 3fff210a6eb9fe6d
3ffd8fa2760bb99c 3feb23cd2cbbb6a9 3ff89d9d55d21c9a 3fd7bb5b0243c0aa
bfcc255ec3d3e7a3 3ff920a1676cda00 3ffad491ceff43aa 3fd39629c85364be
4002bc23284b0594 bfeb147f70d58396 bfe78b94a0032a73 3ff848277c477bd9
3ff1f5aabd010fcc bfe12ddf089a4850 3fd963e7e8e1c2ab 3fefd76b672fa9b4
3fff956029602683 4001b5ec1bee3d4a 3fe2dd3c5e54e9e5 3fdd1c9c064f4cb4
bff2789c607a1b67 40057c3bf052a314 3ff0eacd71d11c4e bfe6e7996341e489
4000c50eb5e29271 bff5b30482d548f9 bfe6654f007832d5 3fbc76cff7e8ff22
3ff05257f15fad4a bfe38aabaea70d90 3ffb62bd5c6ef8f3 400131c7a6a83b1c
3fec7fd0a573f2eb 3fe94ed6c8d1bbb8 3ff3ded27da0ad6e 4003e8d93e9fb570
3fab53a12f34ae26 400027585ccec48e 3ffade8502908423 bfa2b8116a573a95
3feff45839016af3 40009318aac1c98a bff3425e7a4abe19 3fd6f411aa1dc2e6
3ff0ab022d74fb71 bfdf8448df8d2851 3fd3566785a45be5 3fea76fcd2b43d04
3fef090ba41746f3 bfe579e09e7b272f 3ff006d79db4776a bfe912942e5e5d72
bfe30f7f08eff47b bfb883c0a76680e3 3ff737c173bf0bd6 bfe90b69f11b80f8
4003c1ccda4cae52 3ff1630d2a111bf8 400238d53fa2e5de 3fd7ae0f4e653374
4005d3c3b1358ac7 bf99fdd1366790b3 3fdf272a68f08043 3ffd7416f5d8a3ee
bf9a81a24031fc18 3ff78d0916d4923c bff268134a052f45 3ff18d546828041e
400824f2dec21ab3 3ff6916d54ed9c62 4003b7f3ea2f680d 3fbb1e7ca4e8dbf6
3fed8a588af37410 3fe9ca056724127c 3ff1d3f0f9cd3c1b 3ffa06b24315f567
400473673f0ae415 3fe9cabf2b2a49ba bfdf2ad7399628a8 400240c283bdb68b
bfe12ddf089a4850 3ffd276bdd214742 3ff76dd33f1adfd8 3fdf4ef4286e913e
3ff87acd01ca1bee 3fd63065e8e40275 3fffc01e809487a9 3fd2247d8b51bc8d
3fd63d0a5148b33e 3ffd534f8eaa0c64 bfed8c53cdfc7eb4 3ff068a15bc6ae17
3fb97c8b4389af1f bfe27ebd7e3950b1 3fe4180f7520437d 3fb2925895071a54
bfe2398c8b9e4a04 3f9dfb47b5b0d17a 3fef036bb3c775e9 bfe76f83311d0b3b
3fd1789a9f181ca6 3fea7db997acb7ed 4004139bf4132092 3ff2f16667f90d01
bfca7931eb51632c bfe98fa0e0fc8521 4000c9e1b4656b3b 3ffab3280dee046b
3fe4f6eeabd1f5cd 4000e395c44012e4 3ff722c7b27c4e7d 3fcadec9ae6019b5
3fe55c93c2cb8f18 3fde4dfa11232c38 3ff68a520ef7dd57 3fe6706957d810e6
3fc8f23a2f047814 3ff5b2dac534e233 3ff4096b3555c257 3fcfa65371fd6b33
3fd8590ec71b788d bfc39544ac1d9c83 3fe4146a72dc7544 3fcca592bb23d242
4006378b538d73ee 3fe37c13f0c36a12 3ffcded89f39b223 bfe13d314c006474
bfed0c1d474430ad 3fead10a833082dd 3fec154a7c6a618d 3fbaf8dcd1ddda28
3feaadfdf63d839d 3ff9078fcccbac9b bfe15c68e4c6d905 400250435e7c2756
bfe99ac49b5720ac bfdda514469d342c 3fee4a6cba2eb5f9 3fe6f5417b350ea9
3ff559d4e8cf7fea 3fcba3a217955a96 3febd92f86cc19d5 bfdfb58fe94ec87e
3ffe8742765ea4be 3fe2260c9b19e333 3ff2a03bf8ed9fb4 3fe3dde070a5f651
bfd6f05186e0c2a0 3fd837a657fb78cb 3ff102e274d5ae65 3ffff111f5e95799
3ff66d73056516d7 bfe0f1b0d7a7a36f 3fd44254186a2ecc 3ff4265e2e0bbc2c
3fe28cb2e3766ee6 3ffdaf4a0abffad7 3ff3528a6eeb6884 3ff259785e4dcb10
3ff83fa603caf8fe 3ff11f1d64921894 3fd963f57070d995 3ff1d8d60a947252
3ff8ce1307cb83aa bff1ff5c87668821 bfb48b49643fa8c6 3fbfc9fcd06784c2
3ff499c52beee4a7 3ff71db756b17544 40026323ee2f7d99 3feb3185231fa44c
3ff6236cc1c81b1d 4000b64651456839 3fefd46031ef49ea bfed8c53cdfc7eb4
3fdf8ae5284cf9a8 bfbb40d8eb47c4d3 3fbed7333c08bdbe 3fd1bb52c52b2420
3ff6eada19acc259 bfd87267fce251fe 3fd93303f5ee304c 400111935943d96f
3ff1c654f90d25fd 3f9965cca2cf5dfb 3ff06939041cd06b 3fd212baced53102
3ff2d3ebdd4e298f 3fdb96fa0d4695a2 3ffd25e8a4331610 3ff886d90f5dd8fd
3ffe07118b757fd2 3fd1b1a8a05defed 3fde1b2e669b74b0 3ff97f0f4af374eb
3fef2fd9a8e5e8b7 3fe59530548c7350 400130eed5f1d607 3ff3f86c9b3c6599
3ff9c89117945432 3ffcacae07929110 4005bd17ae08fc87 3ffd109bd5a6ebfb
bfcb16adddc21ab7 bfd76e292c67e946 bfe2b3c52596f4fd bf97c08f48202e90
bfc48d9383a6f5d2 3fff4b82b3ca8fdd 3fe4ad481cb3b391 bff522efe5b03ff2
4004452f09016d61 4001d8b3d2d3e210 3ffd71df5e25e6a4 4002f6f1f7884ee7
3fdbfafa4180cb62 3ff12c1d95bef51e 3ff119841e00a9a3 400089bdd1a84c80
40040bb5c31209e6 3fb3ef32cb4a8075 3ffefe31b98593ea 3fe986c702e695ae
400088cb92b6f0bf bfd932d9a3c9c9d4 3fd93303f5ee304c 3ff0a59ac7783181
3fd42fdaeb9c04f2 3fef4ea94ace5f37 3fe6b28428963add bf8180e3d588da06
3fdabc0e3d039c12 bfc213dcea50a751 3ff2e0c072990b7b bfe7ec6be66a9448
3ff9e0449f9b4ccf 3fff4019c908367b 3fa0373c812cf1b4 3fd6a5adb68de65a
3fda61292788b388 3fece65204e351dc 3fcd4368bd1443ad 3ffa0a77a4f5ae38
4003b3dc2ca9ca75 bfc443c7ce7cd69f 400238d53fa2e5de 3fff09129695ff29
3ff028ec71d77138 bfde768d69375dde bff43b07ed9a99e3 3fdecbcc4247c404
bfdab062aabe4964 bfa8170ad94913b0 3fe5ab089da66d88 3fec498bb7feb2a8
bfec5bfdc1703881 bfc28bbd61d10673 40033079a4b1aa9e 3ffd863abcec29ac
bfd07356a75ec3b1 bfec92200d6d153d 3ffaba94dcffda87 3ff403c8d03b821f
bfe852862ed71768 bfc8ca36e3c444e2 3fc7d2edda54dc5f 400003ca4edf349b
bfedadf23c68f7d9 bfeec1d4f04a61ea bfa55cd143b30c88 3f92b381260a4140
3ff7187b8679f7bc 3ff28fa30eb7982f 3ff173f676e90284 3ff3f33dd9bffd89
bfc1344d29bedbae 3fc058ce1b6bf638 bfd77abab57e043c 3fed168ff90ebbee
bfe648b48aa50dc2 3fe21f2360f37af7 3ff8b29ec98b7887 3ffb90a6748af9eb
3ff703a3e472786f 40044acb9363f2d3 bfe08ce869a7b4ee 4003c449b85699c9
bfd51693631ee6e5 bfeb08e955daf677 3fcad4a364e76f7f 3fe5184c78990f02
3ff019a6c32db3c4 bfd3b30432cd8dd3 40052f51b2a010d1 3fda4e5a2f2f63c6
3fb391573edfd4f0 bfe792135a7a5033 3ffe082ad3f06d92 3ffeb85b449e1d58
3ff41ae38fb49f0a 3fc76d2ff43666bd bfeb6636ce162056 400104a19180d47e
400022a562fd6e4c bfbf995b5d3a3d91 4007bb99dae30622 3ff4c6f66083bf42
3ff6eada19acc259 3ff28ee09403b1d1 bfe9d6065f6b5534 3ffac2139b98f7ab
3fedc3d694549838 3fe748f2d102c430 3feeb27e94d5f55c 3fe37dd9bbd062a5
bfc3cfa4f17edede 3ff60619991de521 3fe4d08ad5fe14c6 3ff08f78694db091
3feb8a5a17f5332a 3ff044a1cd6f50e5 3fe72a7726087105 4002815c1dfd6052
3fe8193aa65ff9d3 3fff2645ee17705a 3ffb4f623c5e09a0 bfeada3995057b86
bfce0037dec622ff 40016b8288b47676 bfca7931eb51632c bfb3e5ad33f1be52
3fee0d9bdc359858 3fee25cb72fa622b 3fbac5db9afc4bd9 40004092292b62e6
3fd9b9c983a239e9 3fd1cd0177ed34b1 bfc902caf090be73 3fddff65ac89a6a3
3fcf9a2c6a0dbdc5 3ff9033332916084 bfe5efa94b4560f2 bfd4aa587b7ac354
3ffd94764415e466 4004452f09016d61 400335b9162037ad 4001dc3459cebf88
40039dce7e8bfdfb 3ff0e696e6a93cb1 40070f876b15af51 3ffc9083ff749154
3fe2255f88eee5b2 40061f3e352b3a5f 3ff1db95a564fcc8 3ff323728dd17a0f
4001e6b196dabf44 3fe01369e95ef1a5 3ffb6c1a3d8b6eb1 3ff040552ff39092
3fe0bc27643fdead 3ff9472db2b8a300 bff5b30482d548f9 3ff5edf815cd5c49
3fd2af6281dbffb4 4005bd17ae08fc87 bfe36e490d3e868a bff123ab80201a0b
40007d2cb62045cb 3ff4cbbd1d1543ab bfda12df394214b3 3ffbbaacd724eb50
40022f8d0436ee12 3fe2250638feaded bfe03dd4d04e5340 bfdc8f1723c0f9f3
3ffb543533fad77f 3fd04e2ee6ae7196 bff10461989c2318 3ff7ffa1e2b85e74
3fea9959cc08587f bfb883c0a76680e3 bfeb9919cb6e29e0 bfd8004a173935b6
3ff8a3ec36427aa0 3fe6d88d8902f892 3fec2ae75f73ce43 bfdb1b38aff27291
3fc678467bbebfcc 3fdcff899738fb65 3fdbe4d6d58a3d92 3fdb3fb4f626fb6a
4003a56c05c03855 3fdcab5d33555bd9 3ff86056cf171db0 40084722737af626
3ff80e9a816f7620 3ff3f9b3a3d218a8 3fe61189ad0b08ce 3ffa8f4a33ffc9b3
3fb07d362fa7793e 3ff044796d67e358 bff14e204b7aaf9c 3f3d84217ac56168
bf7d27b803b959b5 3f94fd9774a178aa 3ffb14a539a732d1 3ff6e8c25c4b75e9
3fef368abb826017 3fecba0a85cb146e bfa59b2b13e262eb bfee981676c73358
3ff9907f3ed11ef6 3ff2aa9e995916fb bf86fd380b4bb162 3fdbe4d6d58a3d92
3ff1d7c7454ec3f7 40044c62064b74a9 4005d56de701dc76 bfd1df79dc72a9f7
3fdb0ea64944a1f2 3ff17b0241cae3ab 3fd5d4e0918d5ae6 3ff5f04aeaca4e88
3fea9a74250e3c02 3ffae5eece76af33 3ff28da2e45870d3 3fea1a6de1c91e55
3fdbb06ccf64aeba 3fecc6909b513a82 3ffa1156efd9f960 3fecae28b960655f
3fd84996965697b9 3fe96131e0378296 3fe72cb371c6c431 bfe829b822c58f0d
bff44babdc58f712 4007a190b00d5ba4 3ff50e55bc9a08ef 3fb6cb2eb6a78170
3fd31cee544b5c15 3feff93a21eca255 4002023140a69f8c bfa2b8116a573a95
3fed85da47ac2b67 bfb1701533246528 bfd48e706e6c02dc bff149876034b3cb
bfd02949da244abd 3fe2ac6efca8c6a8 3fe6c01e1b3bf991 3ffb0687c2bf11dd
bfe9bf2ba6a407f2 bf79449452f0f0ad 3fc34a58ab6ef3d0 bfd2a22111801f5b
bfdcdf4cdcde5b84 bfe24f9d6a9ae295 3fc447dd2c8616f4 3ff1e17fd5fb6cb6
3fec15f44c597910 3ff31810cce144ee bfd07356a75ec3b1 3ff6e4144430868c
3ffa4127759db0d1 3ff264f088b2256a bfed0c1d474430ad 3fdd4baa1d33313a
3febc268869cc801 bf79449452f0f0ad 3fe8ee46ff11cf9e 3fe470ff426e3d44
3fda8bd7e61b5b8d bfc55314be51993d 3fe8a655ea142193 3fd0a9f9625e9e04
3fdb4d8aa73608d7 40029ef64ecb2c44 bfe6b5b1b150b304 40009ff45602375a
bff5b30482d548f9 bfe30f7f08eff47b bfd3e43ae06b7bb8 4000d9295224410a
3f7d63e4abd69c34 3fe9bc337764cbd4 3ffaee1c17b9d509 3ff6da2918b04cf8
3fc33018fe5758a5 400498500f29f137 3fedace2e68d8100 bfe058fa0f9af858
400374fe8f2df9e4 3ffc7db12553c5dc 3fdfd682804b76cf 4000a4f903293788
bfd2f5ee92ac495e 3ff3bd807850afd7 bf9c61b2332d3cce 3fe882fabd764405
bff287cde3d87cc5 3fdd3375490468a9 3fb9296610948d4b 3ff0e26579c2ea92
3fe5130f377e7d63 3ff22699c8afd7a0 bfb30409cacbf956 3fedf1f5002f5add
40028a41aebc0978 40044e7fc100934e 3fd5e9d39b7b7596 3ffbe0fce6349fc1
bfe95b828ba7784d bfe4b53df9d17c17 3ff68c94db24986d 4007bb99dae30622
3ff5dac905a69805 3fe760a94db30bbc 3fe107dc3a0384c2 3fe6367cfbde25ad
3ff4b2ff743c2271 bff580f415f3f58c 3fe49423f7c56b2d 3fcfd62f629a8ecd
bff580f415f3f58c 3fbe84d9e1653c45 3fc2cc97ccac618c 4000ea08416488b9
3ffac91437ed44fb 3ff2cb66351dd1aa bfe280339a1c3388 3ffd8cd926cb2d15
3ff62aa4dbdbe88d bf983d3d81fec8f6 3fd50a4668dfbcfc 40043db2ba17d5d7
3ff194477cb1aad9 3fe1262d665ce166 3fe96e97c0259cfe 3ff52b9cc252b0a9
3fb3836cae2f9327 3feeb3e4ba1ba962 3fe2255f88eee5b2 3fe2b79be07d2cd5
3ff217c7bfc993f2 bfd555b236d68644 bfec3370e863edec 3ff48d49eb72b390
3febf1751a3654fb 3fd32c7870bc2e06 40053738a14a5201 bff44babdc58f712
bff2da543394762c 3fdf4651002ebff0 bfb4b6d16991b450 bfdb00bd2be3667c
bff09bfae3a58119 3ffb06366d9ec647 3fdd414ac8580cbd 3ff97fa8721c269c
3fc8a97134519359 3ff9247065c038ff 3fda29155ee9bc27 bfea23667a350bde
400131c7a6a83b1c 4002eb6aef01bb67 bfd2efdc4e0a6910 400016080f5256a7
bfd47d3e9bfd6c6f 3fb1cadc1e9d8f01 3ff6eada19acc259 3ff61023c0f92dec
3fdab22f927dd9a7 bfe91e0b43cb9cfb 40016c7a84eca126 3ffff1625fc427f5
3ff78d322309d280 4002815c1dfd6052 40059fdf16b58a55 3fb8e895b17b8659
3fe88ba4709d42b1 bfe0f95281183aec 3ff247519ed5ff6e 3febe50db52f3162
bfcf96e19e3cefa9 3fde0bb13ac2e017 3ff6e4a0e7aa8af8 3ff6b9e78c47cd9f
bff2658060425452 3fe7b8aee416a4ba 3fdfeb9563081ac6 3ffc15bb2038698c
3fe061923c6b8dc9 3ffac9f40628018e 3fd4cd5b4771d7fa 40013fe7137add54
3fd6a5adb68de65a 3ff82ec0e83c6502 bfd1340cf2a50433 3ff5dac905a69805
3fbc76cff7e8ff22 bfdab5c3c2d14910 bfedadf23c68f7d9 3fdff39fa9c2e6bd
4001d8c8a24e5d2e 3ffafb8dbf3f2624 3ff3112d89550087 bfed0d3e8fbf4443
3ff0d32fb0a8ee8f 3fef3fe560a530d9 3ff8393d7c80b06b 3ff4c53440e86235
bfd2a0ea963d0943 3fec856b0bc3385a 400067dceb453c8c 40015a767545de54
3ff7c3b1e2eb21c6 bfdd4d00eab293d0 3fc53214b08985b5 3ff6a14d219e483d
3fef260a62c04b54 bfc4eeca3fc08195 4002815c1dfd6052 3fe94ed6c8d1bbb8
3fe4b39cb78cc5fd bfe048b2ea982f93 3ff47b7b107bd1e9 3fe37dd9bbd062a5
bfe4b53df9d17c17 3fd7545e57d86742 3fe7673a038444e1 3fcfbd9846281be3
3fe425923dfd0568 bfdf63d568a49a44 bfb71d772879a470 3ffca67de8f4e3ed
bff268134a052f45 bfefed4d6a042dc2 3fd258379b794401 3fc609ab5b0e77c6
3fd39629c85364be bfc3fde9e195af00 4000c6a54b7cbe68 3fd258379b794401
3fe1262d665ce166 40031f934238aeac 3ffd73ab476d6a0c 3fe0e4b759006246
3fd5bc69b40ccf52 3ff470bb3821f04d bfc01a26209e29d8 3ffad01910ef1fe5
bfeaafb7290589af 3ff6c169858ad46e bfe815ec5b151742 bfd542c53cdb5514
bfc54fcdd162e558 bfdadb62cefc7c50 3ffdce0762252caf 3fdbb06ccf64aeba
3fe1aba7935b359d 3fce08729ca4bee7 3fe0d63dac2b5d45 3fc0bcf5dab79418
bff0d73c211e8ba4 3fe1f59d2f447fa4 3ff5aa8e1a6dd614 3fafe496cfda115d
bfd213ace0868419 4005a24db85cf181 40029ad9d7ae6c8f 3fd93303f5ee304c
3ffa859e66b49f4e 3ff5ae2e32412c1c 3fb53d0be165ad47 bfdf50afe434e9a3
3ff627317d808bd4 3ffd73ab476d6a0c 3fe9bc337764cbd4 3fdc055b11072eed
4002e29e9c524cc6 bfebd1ee8ce39d9e bfd7082bea50443f 40012327b8e831d0
40058b688959d3bc 3ffe8217d9577ab7 3ff853af63f001f6 3fc157996767d0ac
bfe62595127ac0cb 3fe2abd8c12288fb 3ff1c87bc1983edf 3fef803c5107d8be
3fe6ad50a2522647 3fe5184c78990f02 bfa7c876dcf25a69 3fd215aada664652
3ff3533dce73c437 bfb6b51bf63be02b 40065f2e5c026faa 3ff313ede9a0e483
bfe0120cd99e65b5 3ff822815dd5857f 3fe4146a72dc7544 3fdb0529ee377bbe
bff580f415f3f58c 3ff424c6fbec1f25 3fd0e40d78055e51 3fd571a88b87d188
bfd842d377df8a4d 3ff474ad518b6d46 3fc9f5439049585e 3ff4304aa57842ad
40015cbc3492efc9 3ff837f71c16e7d4 3ffacf6c3e3a4bee 3ff194477cb1aad9
bfe80dbbf58e517f 3ff18d546828041e 3ff1431ecc9a4b3d 4000136530774467
3ff225c613625626 40003bf92aea7a0f 3ffce6b7e084deaf 3ff2cb66351dd1aa
3fed2bdb1080126c 3ff07281c201a9d6 3fe7c1ed27ee35e9 3fe2954a18184c3f
4002e78effc8bf59 3ff2f1b8a7368e0e 3fc82f35e18064cb 400362d0999977d4
3ffca97898dd3ff7 bfaa5733b45aa40a bfbce7b954eb7e70 bfe13908e4fb80c4
3ff3ebb3cae8b9a1 3fd1df3dd9f18bfe 3ff088f517bf99c8 bff29d077106bb8a
3fe5184c78990f02 3fddd93b5d0b5976 bfcf5fe6880f8996 bfd55c5830d68dd8
4001071e6f8abff5 40008307ac4e1d1e 3fe2e415fe2178da 3ff3b3d88d122b3b
3fe6b4d062ae0879 3ff732b0ca73015d 3ff8091bf5d53e07 bfe0cd3f7cb34de4
3fe3e986e8a58faf 3ff2260473e1e461 3ff8a67376d3fc9b 400285936f5b5e9b
3fcaf6748137ccae bfcf0b49547a26ef 4002a347da1fda80 4004452f09016d61
3fe8a99d514e0f3c 400238d53fa2e5de 3ff6916d54ed9c62 3fe354ab27a71ace
bfe21f6b0a1034bc bfe09f92b8371ded 3fe8a03022f784d9 3ff323728dd17a0f
3fda9ac7e0d9b436 40044c62064b74a9 3fb8cae13cf3e8c0 3fd8f1a296a580d1
3ff069732c42bb29 bfd72ec99e63a437 3fe4881cd57f3ee2 bfe9bf0a8da197ac
3fea22eb0b522153 3ff044a1cd6f50e5 bfe05c132c02a725 bfe76440d69de24c
bfd0f3e090c2b891 3ffcd61837a243dc 3fedaeb93335c047 3fe08182ac405e36
3ff04502a9c06466 3ff8803d896c7c1d bfe3793e6627b62d 3fbf4aff00ffa9f9
3f66dcd9c48f9fc8 bfcef8cc366538f6 3ff31f87b2eee9eb 3febc268869cc801
40014312b03e0927 3fd5615e373235c7 3ff2f29d2be38a08 3ff6c927628d7146
3fe3b0820689fa64 40061f3e352b3a5f 3fe0a753869ba1e0 4000c9e1b4656b3b
3ff4fb4b848a30f9 4003e8d93e9fb570 3fe72a7726087105 bfeadbeaaad7d13f
3ff48d49eb72b390 3ffed75b6fc70de3 3ff953073093ac15 3feae201d1e7e8e7
3fcff57f5ff46031 3fd2a5e06137ca7e 4006be43a51ce432 bfe0f1b0d7a7a36f
3fd8a2cdc156e7c1 3ff34cb0e9cc572c 3fda74ea3276d577 3ffefe31b98593ea
bfc81d647172b575 bf9c61b2332d3cce bfa8f4df14f4d6d3 3ff454e16d2489b0
3fbd3aa8e608d7b5 3fd76205de27219d 3ff60f67315ef71d 3ffd2904e486eee1
40031f934238aeac bfeb08e955daf677 bfe65f4c2c48c295 bf7f51026242e604
40053738a14a5201 3feff93a21eca255 bfe2a728a0578251 bfec5bfdc1703881
bff3a12638532f25 bfc38ac12ddc0e2d 3ffd6cebc7e3a284 3ffc2de9591ca3c9
bfd6e9a74f45b37a 4003c1ccda4cae52 3ff429565c52d22a 3ff10a0defd7dae5
3ff9f9e36976f33d 3ff3625a664d7820 3fe09bd88af6af18 bfd8d4bbcd0bf98f
3ff8cb66f76856de 3fdf065fde691f00 bfe22c99223578f0 bfcffbdaa54efcb2
3fda3eaebccdfb8b 3fed9ee0cafd2e47 3ff22786018bc5df 4006be43a51ce432
3fc6d6a9a393ce7b 400259fa2ff5f362 bfdab5c3c2d14910 bfcf5a8e10793c56
3fdcb4c6e65254c3 3fff54899d4c99c7 bfc886183aec17b5 40016150f988105b
3ff65339d41b199e 3ff770a81b7de60d 3feae84a3f2563f4 3ffe8217d9577ab7
bfde768d69375dde 4006be43a51ce432 bfd932d9a3c9c9d4 bff18b352ffcc57f
bfa27bbc510fbe5c 3ff7a62740079c04 bf98f054ecd29ac9 4000bda0ac9ed9ac
3fd5d97754d3b6c2 bfe5d8d48774204f 3ff266bfb2cb076a bf983d3d81fec8f6
bfd175a499fde504 3fcd06279a8d90ae 3fdea3a3df0c27e0 bfd6b9687369cc42
3fe9c28b304077e8 3fe393d4dd23a0e1 3ff8535d2a6a397a 400183b6d2969f3e
bfdbce891622b03c bff3d280a676831f 3fdbfd57e3579164 3ff31ee9760b2381
3fc5962c2cb9e4e4 bfa90c9f31203ba4 bfbd07ebb44c527f 3fc98fd0be6fe625
4002a8551cffca04 3fb23a4f06524abb 3ff8e861d2b7c778 3fd80681d102652e
3f8b9f770238b16e 3ff92dc885afc18b bfeaf9745ad2f813 3fee7c83401fc371
3fd1055ba1b26808 3ffd82294ca28074 3fff4019c908367b 3fe2fe50baf627c4
bfc624f78fff0acf bfe24a97015f1a61 bfeb09bd2eb43f26 bff0a20760d2c1ff
3fb2a7920a16ddaf 3ff79aeb6e31c665 3fec5c16d350f2e0 3fef6eba086f7dae
3ff66cd96ccc93e8 bfad1bb8d5b1fc8a 4000d3940b970ccb 40033079a4b1aa9e
4000aeaa753d82e5 40047d7cf82725d1 3fe02acadf1afcf0 4005863a43155b98
3fe1bfde5344b13b 3fc83fbd4895bcbd bf8341b617bd42c6 3fd48acb452a8f82
3fd591fb8247fdbb 400209c2299fdeca 3ff04502a9c06466 3fe31681ce19a8b6
3fbe734ac8b8add1 bfdcbf60be6399e2 3fe2df9af0340fca bfe11c4506619fa8
3ffada703426f0b3 3fb9f31ed3f56e2d 3ff014210a65607b 3ffbfcf73b95c5f8
3ff2d44db25a41bd 3ff47ccf23d74e41 3fe0b15f6b8676b8 bfd72f705a9861ff
40043ebd5ddb1130 40010bbff5a689ad 3fe27b87af5c2911 3ff73d30fc5f67d3
40037cb4fee3e08e 3fe08182ac405e36 3ffb8c681d775537 bfceead3b2c9fbd3
4005d3c3b1358ac7 3fc5ad24a1ebc751 40021c761fd962b5 bf77e891a0c979f0
3ff13b2e538a980c 3fe38784e2abaa99 40005e299c698448 3fff9a597c57d775
3fdb2ffa6bab7a0d bfc443c7ce7cd69f 3fe239478331a819 3fffbb1f30fac7b6
bfed8c53cdfc7eb4 bfe374d0c7453651 3fef555e702fdb51 3fe0d14f5863d642
3fdb04c6f72cf6c2 bfe62539ea69e3c5 bfe048c4de34770e 3fc91fcf25fc94df
3ffd4d6a012e452b 3fc5ad24a1ebc751 40060e8702ffe27b 3fe143df5f2b44fc
3fe8d595803e5d69 3fe5331df1f934da 3fdbfafa4180cb62 bfe36d63f292d08a
bff18b352ffcc57f 3ff35416a0cb56fb 3fed51730e5b0f4d 3fbda88ce2a8f313
3ff6b066abb39a9d bfb030342b21d816 3ff6b8c3b0b93970 3fe42e7e9d164258
3ffa67f048786ac1 3fec3800e8600ec6 3fe5acbc0330bc2c bfe5d8f05c0770d7
3fea9d6a3b261821 3feb5c27e768d36e 3ff459cb38de169a 3ff61f6b838d65c1
bfdd4d00eab293d0 3fe944fecc3730e4 3fdf305b960acdfb 3fd6e6ae7270c6df
3feabbb1f6c9227c bfe08ce869a7b4ee bfd55c5830d68dd8 bfeec5d97b550439
bfed8c53cdfc7eb4 3fd5d4e0918d5ae6 3fdc3af5109fc2c9 3fe7673a038444e1
3fe9735a392a1383 3fed4a2bf0bc215d bfa8ed12a11fda1c 3ff4fe8a243f7a7a
3fa7dbfee30226dc 3ff440bbdec458a5 3ff1397dc8b5ce0f 4001b7175b1a04aa
400238f31ddddb0d 3fe4ecae4d1011d2 3fd8f1a296a580d1 3fea6ddeae629af1
3ff2d44db25a41bd 3fd72a0312f887a6 3feb9fb3d75aab66 3fd13d9f48d6f75a
bfdb5f3bf519173a 3fd91bda234cae8f 3fe894d778226a90 3fd09880c2f8b845
400287bbde78d05a 3fd80681d102652e 3ffa42e961caa97a 3ffa320b0414e36e
3ff8bcac22f94dd5 bfebaf504fc8d6c8 3ff9d800dff7db36 400498500f29f137
3ff2820faeb6c99e 3fa60842723428f9 3ff57859020f0992 3ffaedbcda073cd6
40031f934238aeac 4000de3670365f05 3ffea66ab7f190e0 3fdee5e574e7e5ac
bfef64ae5be1bf08 400502d814609c89 bfd09300f437afe9 3ffb4e311d2ee095
4003c1ccda4cae52 bfb6de35919b723d 3fd1b1a8a05defed bff2658060425452
bfbd07ebb44c527f 3fb23a4f06524abb 40050f1fa6b88313 bfdc211117c23f3c
3ffea764e3df2927 bfd30ca15bc39cee bfe21f6b0a1034bc 3fc02cb0d5d58671
3fba97781dd522f9 3fef3fe560a530d9 4002815c1dfd6052 3ffa1156efd9f960
3fedc6653332abae bfc6e46e8dabbdc3 3fe600bb1f756d8a bfcae14f06c1387f
3ffcd3e71669f430 3ff4e36ebc9b0f99 3ff37468ed18e0d5 bf1f1b12e51c1375
bfc5c5b345d350c6 3fc8719027de01cc bfa5d34385d55158 3ff03a2ddd865d04
3fe4811f47818ab2 3ff3581e6bb47a78 bfc65c44071a441e 3fe498950ae41f8f
3fe94ed6c8d1bbb8 bfeb15aa606bfff6 3fe9424e9eadade1 3feae36a6abbb9c3
40008ced0e7f0955 3fef0da86015d12c 3ff2183e96c91897 3fd9891c969733bd
3fc0ba5defe3bab4 400469a37b6d63cd 3ff45a6a71deed9c bfa947ebddde8c86
400022dc2b3270f9 3fdd1ee7a6362c83 bfeeaa1a55f46caa 3fec46a3a2da1036
3fc3736783da2014 4000ceb4d708b6bd bfe45749ce8bfeda 3fff41d6186b6fbe
3fe98b540dbdffe1 40022f8d0436ee12 3ff03668b9fa449d 3ff552238343a2f2
3fd5baf2419d91b2 3ffc2d7614dc9204 3fd873859c906958 3ff92f08d5b5f0cb
3ff474ad518b6d46 3fe1478b686bdea3 400343489ab622ed bf996ce081ddfbfa
3ff5aa8e1a6dd614 3ff52a94c14438b8 bfd1e81fca328632 3ff8091bf5d53e07
3fe8cc53b74f6fd6 bfeec1d4f04a61ea 3fed5146e0b641ed 3ff03dff1fb2eef2
3fec1e1fcfc5db08 3fd93303f5ee304c bfa075ae305b14b3 3ff214e39eaf95be
3fc92b583ed5e946 3fddafc3abfc0304 40025ebbe8605f22 40005b374e0755ac
3fdcc40aa4ab2258 bfad632d761cb5b2 4001c3690040e083 4004083a631b0b7b
bfb5cc80b85e2f1a 3fed158de1d1efed 3ffada703426f0b3 3ff5dac905a69805
3fd4005c3b255bf4 3fc875b8d6d43ffc bfd0fc3e256b196d 3f879bec4f2c9ebe
3ff3d9617908cbba 3fe68d15608fe53a 3ffed75b6fc70de3 3ff7a0476ddc6439
bfe4d15c105b7d93 3ffdd7cc7ee82ca3 4006e31bc740d00c 3ff06939041cd06b
3fbc28ef4b483d09 3ff7e6ea377625fe 3fef1426b86c8947 bfed8c53cdfc7eb4
3ff290b5a9613780 3ffc9cdd751af3a4 bfe755eb1f0706fe 3fddbd1b62b444c7
bfd25a394aa3765b 4004c2b97ac76ade 3fec1d1345bf522b 3fdebd33b3c48689
3fd6aef389f45828 4000b42dc5401d0b 3ff140a7d47066aa 4004eddae48f5b7c
40045da878d56c81 3ffb5bb9c8b1cbf0 3ffdae1521ae1fa1 bfc2558726e3ace3
3ff8cb66f76856de 3ffd020c0904cf76 40060e8702ffe27b 3ffe909952174607
40058108ef73afea 3ff73ea2eec9984a 3fed6f50d48f49db bf99f4cd747f366c
3ffdd2d531cc00f8 3fcbbdc9f7bad7ed 3fdc19955dd2f7a4 4000b428b8e3eca4
bfde768d69375dde 3ffecacc84616d44 3ff1060b175850bc 40024362c8e86500
3ff62ba9391bde44 3ff1aeab2edb148b bfcdc317e3ade285 3fc8719027de01cc
3ff9a47720f97ed3 3ff9c6b426a0a506 3fad509cfbfb0ff3 3ff5a34dfbb2505f
bfa8f4df14f4d6d3 3fda7300365f7e9d 3f879bec4f2c9ebe 3fbe6cbccae86554
3ff0c7fb112b92bb 3fe729cd4868d889 4002deaa8aeb385e 3fd0705413350baa
bfe4549d43294f01 3ff5c9758fd81f1b 3ff1247fd16f32af 3fffaf58de06ee47
4006a62e67369ac0 3ff52d85b5802d59 3fe8e6fff80f9bd3 3fc58ac8aad6a7b7
3ff6668909ccbad4 3fdb7c3370450a78 3ff37a7b713dca8f 3fddc00f82ca0183
3fed344e1a3a185e bfee5911ea8e3c56 4002bd71aeaec284 3fe27e9e84899fe9
400496fe22c7dfa1 3fe0cf5a8915c18b 40011543e34ee551 400533724fe4817d
bfc6e625da1b17a0 400496fe22c7dfa1 3fc2cd5b53fc5473 3fda61292788b388
3fcec1537b77f046 3ffcdd061b428f48 3ff6908f84bd59fb 4001d9f12aa05f6d
3ffda8e3d7a187ae bfbd92e2d523a481 3fed7a39c975820d 40019a2ab165ea6e
3ff1f88e2aa0e636 3ff2788af17df1f4 bfc81d647172b575 bfd05b9d5bde7054
3ffe31b26a7bded8 3ff224994b054288 3ff841c61c093d82 3fa6dedda75ec30e
3ff4db27f5f33975 3ff102e274d5ae65 bfe5b4acea89978b bfdef7051c5b5f18
4003ab92df170ba5 3ffb561e0fa79357 3fe6f5417b350ea9 3fe3ceda70d7a422
3ffeba332048ef42 3fd48acb452a8f82 3fe8a07c714e4137 3fddc362d1aa87b7
3ff2674cca045e18 3ff452c761fe6c51 3ffbd3fe9388ea9c 3fca9b5ba711c8eb
40068c253c60f042 40013a21f3337e27 3ff2f29d2be38a08 3ff044a1cd6f50e5
bff5b30482d548f9 3fd669cba3613d63 3fb5b8b1d089974f 3fdb48b98bb7c35e
3ffa1990694be66d 3fb3ef32cb4a8075 3ff979df5bb85930 3ff75196a47c4620
bfd2c77556648205 40021b4587fcabd7 bfb127531991125c bfd11a1f03aa982e
3fef5df13ce8da80 3ff9d150f9b13fc2 bfc990ef22d69891 bfe3ae907cd96507
bfec644d8990a756 3fffbb1f30fac7b6 bfe3bc9db248481f bfe402d7aeb72b48
3fe87dc42d073e46 3fe09dd58abd0b22 400104a19180d47e bfe9631455747f58
40017b2b8cb1129c 3ff454e16d2489b0 3ff799953635533f 3ff5514fc6fc454a
4006fb3d5cf284c7 40070f876b15af51 3fd1618b03fe6f62 3ffbce7baeb0fa2d
3fcf76ede4bf22a3 3ff040552ff39092 3fe0e2f974fc4d26 3ff5b95d3face0d6
3fe4bc318f7a9210 3fe37dd9bbd062a5 bf81d7687eaae840 3ff91bf28ec5d467
3ffd6faba6ca4e58 3ffcc50b8ba3980d 3fd4c6deacd8c686 400470ea6100f89e
bfeec5d97b550439 3ff96250d7e37c68 400496b62815228e bfdadf235cf9ee14
3fdeee5c699b2ecb 400357e9bfc707a2 3feb7d819c413988 4005713d13441a8b
3fe7c4afcf97fc2c bff580f415f3f58c 3ff141a042d372e2 3fd58ad9bfc4e89b
3fe9174101ea5ab1 3ff6edd22e6d33a7 3ff7afceb2f1600f bff1c5c367af72aa
3fd5e9d39b7b7596 3fd6bd33b898d5af 3ff454e16d2489b0 3ff65d2bb5763125
bfd746c90ae4b370 3fe104eb8fb7488a 3fc4d0f2d9941b8e 3fefc8cb95c8e82b
3ffe5a6b895ae3fe 3ff454e16d2489b0 3ffbab280cdf1935 bfe11bccd2bc7302
3ff3401ee883ec4f 4002e0b9877f408a 3fe95ade3d24f4fb 3ff63a903cbc6989
3ffb4f623c5e09a0 3ff8f89d2a84b7d0 3fe4a0c4b3f6ecf4 3fc9472729e0041d
bfc46bc68f23ac14 3fcfbd9846281be3 3ff7b7d6cbc0a3dd 3fe84d6c8d68c55a
bfdacb7b4f90095a bfebf95bdbddda32 3ffa8df8976ea4c3 3ffd77a0f7e9491b
3ff3ebb3cae8b9a1 3fe7e6e20ab16e0a 3ff215f895b0b1f2 3fe3970b5c24e1ce
3feee4b36ee8a458 3ff0302fffd776dd 3fe0997563490cef 3fc1ef7976d96eb9
3fd2754a68b760a9 3ff30f48421c62a1 bfed0c1d474430ad 3ffea22213fde824
bfed825aff249e9a 3ff49d27af81ca5c 3fd0d43532675ad4 3fd600732d4eb786
bfd25a394aa3765b 3feed6dd51d0f181 3ff7f02f40a5d159 3fe45e7d9cc6008e
3fd3782f1285dbd8 3ffe3ec414a6f59a 3fbd3aa8e608d7b5 3f9cd8b3a74a05c1
3fda37315a1b2b87 bfd6f05186e0c2a0 bf983d3d81fec8f6 3fee713c91090af6
40010ca1bad6c2d9 bff580f415f3f58c 3ff215f895b0b1f2 3fe894d778226a90
bfda1bf4e05cf699 3fe15f167d09fbf0 3fff08ffc8fef51e 3fbe5e644696f54c
bfb6566023ca03e9 3ffcaa51fe2a6e48 400470ea6100f89e bf983d3d81fec8f6
3fe709ae60397acf 3fe29b8b9af87f4d 3ff8d99607fee9ca 400228de902f2095
3fe1478b686bdea3 3fe44c1a967936de 3ff81712bba0b8d0 3fef1411a7db0df2
3fe5b4b976daa425 3ff0eacd71d11c4e 3fd8338dcc1df77b 3ff6ee95bd66ef10
3fcaf570d1123ed5 3ff4b2ff743c2271 bfec644d8990a756 3ffcf7ffe3e7d5b9
3fe6f5417b350ea9 3ffa389569d868c5 bfe8aba2bb9155e3 3f93f90be46c8711
3fe1bfde5344b13b 3feb5ecf310ec882 3ffa1156efd9f960 3ff4165bf4b911ac
bfb2635f2c6e7691 3fc898211c8e2f09 3fe5bcf61048c23a 4004f44c8fb5b7ad
3ffc377f6f2a9260 400152fabd3d8d3b bfe95f14bab05dfb 3ff01d6eecd1f064
3ffae5eece76af33 3fcbae6fcbc1840e bfd2efdc4e0a6910 3ff4b2ff743c2271
3fec29c7c041d191 3ff78141fc9a51ef bfcdb7ef074ecdc7 3feabbb1f6c9227c
3fedfc1eb38f4766 3fea4c41eaec7f3c 4001a35ac1fba4ed 3feb617a14f661b6
4005461ba97c29ed 3ffad8927eb6a809 3fdece0b6e2a94f9 bfc6e625da1b17a0
3fedc3d17ed0ecb0 3fefb7de8ea2aed2 3ffce18dbaf9718e 3ff81abac6577246
3fb09f2577339df4 3fece6207ca2a6cf 3fd571a88b87d188 bfd7d7f488440add
bff07b2548428ba3 3fea5d0da3099c60 3ff83e0f23834686 3fe1a57914fd6a46
3ff98311062882a3 400648ddc390ecdf 3feada5956a264c4 3fea15fecd9fa9cc
3fa8e39faa79a053 bfe9e6346790bd77 40076f265bf2bac3 bfa258da92d328d6
bfe68498783828b8 3fe75f3b258b57aa 3ff1e9ccbee7185e 40046eb525973731
3ff30a082d94283d 3ff06939041cd06b bfc4b1b0949270aa 3ff7c3b1e2eb21c6
bff03dc8922d9abb 3fbb8d651ce4435d 3ffada703426f0b3 3fd3f40f2bb44712
40021ddf1981baed bfa01c778db50182 bfe4ff0fd7ccbd3f 3fd9e7e8c1de4f6c
3fb248dc8949e19d 3ff008a22e04c772 3fa4d8255f09c7fb 40022f8d0436ee12
4002f949f278fc02 3ff3a64301c8095e 3fe6b4d062ae0879 4000b542435e74c0
3fe816dd03dc6f72 3ff27959345e43f2 3ffa320b0414e36e 3fc79429f7d48b79
bfe07ea453919f84 4005b22f247ed424 3fe3fe22c075c4bd 3fcc28fdaad759f6
4001d9f12aa05f6d 3ffd7fde6e6d0369 bfe4661d7808a40a bfdfdf51a9e92141
bfe1aa986025cc02 bfebcb3c36b994b7 3fd3f5e03356ad06 3ff3326bb971da7e
bfebf95bdbddda32 3ffe1841545e3cfc 3ffff9861b4a4624 3ff526d6bc925151
3ff2176f93ab0d79 bfdb09511b242563 3fe3616721454fa8 40059bdc06767b50
3ff4bc00dec1b4d5 4001f01b8046a0b5 3fee3099f55b1bb4 bfb4f7ab9eb3bc0c
3ff56019e27b9714 bfdb09511b242563 3fe1809d2303110b 4002611733ace0e5
3ff5d825f82e8c6d 3fe94c9f02d0a1bb 3faa9e02ea2565cb 3ff65f6b80043a1f
3ff62aa4dbdbe88d bfd9026769c03a3d 3ff3c06286461086 4000c9e1b4656b3b
3fe0249230de0f5b 40005059a503335b 3fe8692f3b11b9b0 3fd8f7d5a7294e75
3fb59b5e0b09e008 bfcb3502268390f1 3fcb238f7db9d34b bfa4a1dd2e3c5d98
3ff2d229ed2a89c5 3fcbae6fcbc1840e 3ff366377d856163 bfeabb3224c2a0f1
40007d9d5714e189 bfd8cddc6b3149a9 3ff5323548f6ea9d 3fe8692f3b11b9b0
3fe6485b557d450b 3fe3101d4b8c8243 bfce09cd5f0b04c2 bfecfb924345daea
3fec1287b1874b65 3fe2435f8c014c8a 3ff00d9498c176e4 3ffef7228811f44d
3ff58fcb41a6d70f 3fd3e2c0eb5f60c7 3fc70f45eabfee49 3fd3d971964d765f
bfcf0b49547a26ef 3fd93282cf12b4b8 3ff5dd970a053165 bfb9668ee38913fb
3ff247519ed5ff6e bfe852862ed71768 bfd72f705a9861ff bfe13908e4fb80c4
3ff658861f2f32fb 3fddd34de9ca5a2f 3fe8cc53b74f6fd6 bfb34ceceabfa9db
3ff08dcfec5ff8d9 3ffeab9e9d6e29ab 3ff115ca95bfbe8f 3fba117a35feaeb9
3fd63d0a5148b33e bfa4388365bb147d 3ff2f29d2be38a08 bff10461989c2318
3fe09c299fe4362c bff142c240e8e76d 3ff3f6f03802d250 3fc71f761390c882
bf79449452f0f0ad bfd7970ce605cc31 3ff1f7caaffc2ce4 3fca87d1b2c70d89
bfbc5db74039848b 3fdf4bd49532d4d2 3fa02086c009715c 3fe178a3f1038501
3fdeaf04e986c273 3ffec0fdb73856f3 bfeb9919cb6e29e0 bfe0101880a25a5b
3ff49598a897d443 bfc22ecf7770ab55 3fd3f40f2bb44712 4003c1ccda4cae52
3fbe2a01cc5458a4 bfbf37d799f14a93 3ffb0b5c0ff89d86 3fed6af56ef528a0
bfe5efa94b4560f2 3fd044a78e3196e7 3ff2cef496f87fea 3fe142f9d955b757
bfe0154c9b5f7b42 3fffdf98624501ae 4004feec4fd10dde 3ff45414a83ef069
3fbda88ce2a8f313 bff5b30482d548f9 3fecc8132443a80b bfea9261d4af406d
4001f1cf318a5654 4004df80c5cb1c57 3fffc747765b530d bfee5911ea8e3c56
bfe12d717afc45d6 bfb15f2d113fb680 3ff174bfbe01c12a 3ff3d16c6d7a8b79
3fe9e690f1502988 400374fe8f2df9e4 3ff2313c815fbd01 3fe894d778226a90
3fe6bee0b9c445be 3ff16a0282b44413 3ff9a484999acc9a 3fca7949e395f154
bfe65f4c2c48c295 3ff2d77887546927 3fe98d7668db7c05 bfc482dbf7cb96cc
3fdddc1a39589614 3ff655267408bf3f 3fece8f558f7816c 400131c7a6a83b1c
3ff99b3ef7d7fa9b bfaf1b35cd47b6a9 3ffe632a3cdc9ec2 3fec1e69dcc15ec9
3fcf837b58dfc840 bfea4ad3c92f4872 3fe0b2cbe33d3b5f 3fd0a9f9625e9e04
3ffd519212f98f9a bfd4e8006aed92da 3f7435ac295a8bd0 3fd3f5e03356ad06
3ff6d2bf4bb3af2c bfedadf23c68f7d9 3ffaacd5edf90d4c 3ff91112df2ecc14
40071364e6ae7e2a 3ff816791c0485c0 bfde768d69375dde 400196fa5f520748
bfd07356a75ec3b1 3fd5f7ee53984e11 3fe4180f7520437d 3ff527a0325db733
3fd39629c85364be 3fd93303f5ee304c 3fbc76cff7e8ff22 3fe5bcf61048c23a
3ff319918502700f 3fc4b8039c73d48d 4007ddc96f9be195 3ffd68a45680ef70
3faa84f61fb9c868 3fea8a120e3161fb 3fbda88ce2a8f313 3fef109ea6d04452
3fa028aff92f03df bfe158bfd81c27eb 3fe22b874705c0fa 3fe4146a72dc7544
3fd7d2b21df1e732 bfe78b94a0032a73 4001f1cf318a5654 3fb59b5e0b09e008
3ffc0e5e7d188212 3ffb2ced864ae2de 400439db5bff7e96 3ff55054592ac0f2
3ffb53a5f729c282 3fbfe7258c082a7a 3fec6cf3ac3870f1 bfa94466ffe578ab
bfd11a1f03aa982e bfb6fdfd2cbf4dc1 3ffc160ad77aac96 3fe1393d4259e218
3fff65943549e2f8 3fe1c7ef5188599f 3ffb4e311d2ee095 3ff6d01b6c8bae27
4002611733ace0e5 3ff2de1ed41db72b 3ff02d18a6236d0c bfcf0b49547a26ef
3faa84f61fb9c868 3fe7e6e20ab16e0a bf94730a2ecc0a6c 3ff896d2475c2d55
3ffb18faad3a6c62 bfd348be8b63d911 3fe28db686a1d449 3ff0f6dc7d02a91c
bfb75121795e892d 4004f6c96dbfa324 3ffbbffb3258fa92 4004ddb373949803
3fd48fc6a2442a79 bff2789c607a1b67 3fea1703baef3367 3fd4753eed89a870
3fdf272a68f08043 bfe321122a36bf6a bfe098d59b5f75bd 3fdd6599dbf621e3
400088a95389f4cb 3ff18f555d360b2d 4003f3cd2da3549a bfbd16cd1be9c3e3
3fd91f01ffd9acf3 3feb39f5a5dbe96e 3fdbaff064ec2c22 4003b7f3ea2f680d
3ff80fd7522d9330 3fec3ff039620456 bfc7388af6250aa7 3fe58e0320c5fe58
bfca7348c6c6235b bfd739c6f2e4d633 3fe22b9fa20cc94d 3ffa16463288af93
3fddaf94e4692c9f 3fe4a0c4b3f6ecf4 4000bb02c6bf6ac8 400111bb8b175345
3fe2533e2b85f422 3fe2abd8c12288fb 3fe8d414a48dbc7b 3fc0a116bb3f83c4
3ffe274a1b4209d7 3ffcb516dfa8e44b 3ffb2ced864ae2de 3fe0f0c0c18cd10d
3ffaacd5edf90d4c 3fead2a547a89941 bf82382b521ea934 bfa0c4150cd5dd8e
3ff5f2d9c5efe57c 3ff627317d808bd4 3fe74a74be347850 bfe721fce70f6708
3fd6bd33b898d5af 3fd6a08efcd34749 3fdef8a409edf064 3ff3b3d88d122b3b
bf9561f576bdda4e 3ffce6b7e084deaf bfe0f1b0d7a7a36f 3ffdccc180e7e3c4
3ff848277c477bd9 3fed9176c0d64d48 3fea653a11823f09 bfc531acef53ec2e
3ffd765393306996 3fc0a116bb3f83c4 3fe4562b0143ae59 4002186b406d7164
3ffde106e816029b 3ff21d6f9f9c3f69 400275e7c3c43cf3 3ff5f34fcd2e24ca
3ffc2034680f288c 4002195f36a006a5 3ffa66a4c217cbb1 bfb26f16fb43923c
3fe5750109055bb2 bff2d6a090bdd101 3fca052393b8d0d1 3fecac36914abfb7
bfc5a37531de0506 3fcc6ee90d954ab0 3ff8a0ab69491c4e bfd6f05186e0c2a0
3ffa1156efd9f960 3fc0a8a5fc3b82ab 4002299f4099df80 40009e75fcea868d
bfddb922fa08d99a 3ff75ce52ee5fb01 400498500f29f137 bff2789c607a1b67
3fcdc8596174e853 3fe4734323d655e7 3fc53214b08985b5 3fefd76b672fa9b4
3ffeda0eb4ad1404 3fee04cf6393a210 3fecb16ab67c41ca 3fbc76cff7e8ff22
3ffb4f623c5e09a0 4003bfcbe2375d27 3fe1ab0294118ea1 3ff91d594e37d809
3ff82ec0e83c6502 3fe690aa879da93d bfbd07ebb44c527f 3fd9971da7851266
bfcec58e12c176a6 bfa8ed12a11fda1c 3ffc1a490dd8ed0b 3fcadec9ae6019b5
3feb2313fb7986e5 bfec79988ef1612e 3ff7d03b9b056648 4004e320847f0e09
400062599f1d7a65 3fcaf945b088415f 3feeb78907a853fb bff4009a45ee3940
bfe21c1d7fe19fb2 bfad1bb8d5b1fc8a 3ffa6cb18ea876e5 bfed0d3e8fbf4443
3ff55e6edd7d6a1e bfe1aa986025cc02 3fddff65ac89a6a3 3fb324e944f0c1fd
bfe04029796a998a 3ffb15a59cc7d82b 3fdf305b960acdfb bfc2a4d65ee4d083
3fbe32370a51542d 3fdab0b7ca017d4a 3fe18b2716c2b06b 3ff0cb6bc68315b2
3fcbae6fcbc1840e 3ffafd0afadf82f4 bfbf1717aaa13b48 3fff000e7612ae1b
3ff7418a846a4e1b 3ff050af3beeb0c6 3fe0d0a59c272249 bfe1b69c235763b3
bfeaaa2e6d8eb522 3ff0ee2f6d5a2346 3ff2cb66351dd1aa bfc8c77199ea93fa
3fcec1537b77f046 4001de3cb007b9d2 3ff323728dd17a0f 3fe1b83832cc33fc
3fc878980b3bbe76 bfb9a8583f16a3aa bfe1402a4f00a73b 40054b36e742ff68
bfec6892e10b63a5 3ff95a712c4b80e7 40053738a14a5201 3fd39a532361d201
3ff38d9684598aad 3fd364fc742288cd 3fe09274525b0c78 40023b3ce4cb23ce
3ffcfa13d66bb3c6 bfe5abdcec8fc080 3ff47cc777683db3 3fe08d4d6275d6e6
3fe0abe4de1f8551 3f959b427960c545 3faf307b46060225 3ff7c4f11139fd63
bfdf58d2f3870324 40010961805c0077 bff23ea256a1998a 3ff59880cb73d9a9
3fbda88ce2a8f313 400041934c1ffd9b bfed8c53cdfc7eb4 3ff1f4af99241e67
3fcedc38b2a0a3cf 3ffaa9c4b58a8563 3ff6f434d4126a18 3ffff03c92ed69fb
3ff59880cb73d9a9 4005ba9acfff1110 4002683e18bf18c0 bfdcdd32c3b609c4
3ff215f895b0b1f2 bfeaf9745ad2f813 bfe13908e4fb80c4 3ff1df98f2d51a4e
3ff34d87f631415b bf9d2a78b01bafbb 4003763a76075ee0 bfcc1a02502f08c6
bfc28bbd61d10673 bfef4ff6901d8b95 3fde0bb13ac2e017 3fa955fb737acb11
bfd747e174409db6 3fea4c41eaec7f3c 3fb8e895b17b8659 3ffc7f0f5435f35c
3fc754c5cb2ce146 400111bb8b175345 3ff429565c52d22a 3febaeea46b257c8
40062341456a4964 bfd81919c9bf5b14 3ff1cec85a8ba08f bfde9c37c7edf2b7
bf4cca04a6a4b1d0 3fda914a442f17f1 3ff19d292fc5c59d 3ff8005dcd5725b1
bfe5b4acea89978b 4005466d6edf57c9 3fcd4368bd1443ad 3ff49c6ade38b75e
400151f944c696e0 40004353702e1105 3fedf78ac9a0d278 400088cb92b6f0bf
bfe42d4590e7e270 bfe07065c831c5ed 3ff8b286ad5051e4 bfe42d4590e7e270
3fcf838dc933dfcb 3ffd8cd926cb2d15 3fd0877afe4192f3 4003c1e574e8abfd
3fe178a3f1038501 bfdb8d44df89d716 bff5b30482d548f9 3ffe994e45ea7cc6
3ff90a4d9f27ce94 3ff01d6eecd1f064 3fadf89cd12860e1 bff408f780b94676
3ff8dc22c62868aa 3fd035b99a4c11fd 3fd1df3dd9f18bfe 3fddd93b5d0b5976
40016b8288b47676 3ff098638188c2ca 3fecc6909b513a82 3ff47d1a19c685f1
bfeaafb7290589af 3ff8d2062c31805d bfc958929d52eabe 3fe8b46a388b8690
bfe99af3f9192608 bfb9b96bf44f3de1 3fce0cc23b323ae2 3ff62b19d946369a
3fe894ada01c9f70 bfa2c9a546bba57f 4000904da14df77a 3ff075e827c2ef89
3fe94ed6c8d1bbb8 40000288635463d5 bff2d6a090bdd101 3ff4474aaf52c2c8
3f9a2f3b12336687 bff408f780b94676 3fed0a2822723c8f 3ff7d4461a8aec03
3ffe648efa76417d 3fe652d5a40ca09d 3fddd34de9ca5a2f 4004452f09016d61
4001ca9676051158 4002b94ef3671d29 3feb12f33cf95b17 bfdb00bd2be3667c
3ffec369a9a8e064 3ff08f78694db091 3f853ddbac29a108 3fbb77292688b60c
3fed291ac2d53489 3ffccb9bac28dbb4 3ff26f7ddb491fc0 bfef98ccb136cbdc
3fd47713b0c0b88b bfd418fb574bdfad 3fec74f62f990bbf 3ff9033332916084
3fbf3e17e0c41b0a 3fd0a9f9625e9e04 3ffca2430bc0433a 3ff454e16d2489b0
bfc2fe08195ff4b2 3fd03b848c30caf5 3fa4c1ae1cfa21e6 3fa97b21f06aed0e
3ffb4e311d2ee095 3fc90b0a417cb399 3ffcacde8d1caa38 3fff41d6186b6fbe
3fe43fa416f43db0 3fe37340304033bc 3ff788ea37241be6 bff0234f44bae28c
3fe5e99259d9aafd 3ff10b2f63283347 40029813367087bf 3fe7ad2314dbffc5
bfea1cde1cd35149 3fb23a4f06524abb 3fe6a22bba42f7ae 4000eada921278d1
3fb8e895b17b8659 3fd11df09c37b25b 3fe5fbf98facb2f0 3ffc43def5abafb3
3fef821ae29a0d9a bfe9b62cb35a9f3f bff2658060425452 3fe57052e9427f03
bff19b4bd9f591a1 3fe22c3262367fed 3faa84f61fb9c868 bfad6d0bc2f96057
bff580f415f3f58c 3febc268869cc801 3fe6f25fb11e6473 3ff9c6b426a0a506
3fe81dc87afd6408 3fdd40222c2007ef 3fe57c045cab6493 3fe835ee5c53d432
bff1c5c367af72aa 3fe4180f7520437d 3fdc2d9b97cc4dd3 3fe9a9e8b89d7c93
3ff088f517bf99c8 bfdd1de24b11eada 4002611733ace0e5 3fe583c56a907210
bff247bd4cc1da06 3ffac2a2a8a1179f bfd4e65f7f8578a9 4000c26b5a3e3e38
400003ca4edf349b 3fc443388d07e9f2 bfba5f68151c4d7c 4002f949f278fc02
bf86fd380b4bb162 40084722737af626 3ffbe0fce6349fc1 40009ff45602375a
3ff48ffa026e788a 3fdd94ba847a833d 3ff4d789e5d2756c 3ff025e7b563c10a
3ff3fbbe7e921bd1 3ff8753d3e8800f9 3fcca592bb23d242 3fd43afa1399a1ce
3ffb6a0e239c5a41 bfc225d26004a9a4 3fedb548a0efd0d1 3ff990dcb9024b95
3fed6e93c326eca1 3ff9e04d5055ea45 bfb127531991125c 3fe80fa8f3bf95b7
400240c283bdb68b bfa43b677b512265 bfd025666c381ba6 3fe1478b686bdea3
3fec44763d46800d 3ffbfa14e7fc521c 3ff6687a78eb68ab 3ffe994e45ea7cc6
3fe44514a9455341 3ff2bcc0158ea707 3ffe7d9d5d53f7eb bfe84f3cfbd94528
bfd0d4f1239e52d7 3fedeae9db3ec69a 3ff0564a87e7c130 4005520bb84d1d67
bfe53d3e3b973497 3ff541f6b57a4df2 3fb452a708b63dbe 4002dd5a25069ca4
3ff3ff66e7fef774 3fd55bc608273417 3ff96ee386204542 4000ef0d554d917e
bfe8023914cfba5f bfbd073aae097674 4001a2f527dcb893 3ff986bbfb8af2f5
3ff0e696e6a93cb1 400581822600beee 4002deaa8aeb385e 3fedc3d694549838
40084722737af626 3fff7148847efe17 3fa8950a4b43301c 3ff3d9617908cbba
3fdfeb9563081ac6 bfbd07ebb44c527f 3ff554ae2b5916cc 4002c7a699cfc729
3fe245b042a2430c 3ff7afceb2f1600f 400092fe5bc1e267 3fd545a097abf163
3ffd83468e4a86ef 3ff6b9e78c47cd9f bfdab5c3c2d14910 3ff4bad27bacb718
3fdd2eaa9f0dd389 40008ad44adbc55b 4007cefd04be2567 3ffb138dbe6b9d62
3ff21d5ddc79767d 3fc61a5b4bef32b0 bfbd07ebb44c527f bfa55a565b0f00d9
3ff2fb818b7f1804 3ff5b8abba0144a5 bfb93d91f8c8a3c8 40020215b2ea3a1e
bfa55cd143b30c88 3fdfa73b511e6737 3fc5501e3de6ef5e 3ff1c7fe1c8612fb
bfcb16adddc21ab7 3ff1f739b35f3f3b 40009ff45602375a bf8341b617bd42c6
3ff6a71bdbf4de52 3ff233d68671f661 3ff649e141174491 bfc3ed5f4c81101f
3ff499c52beee4a7 3ff31810cce144ee 3fe8ba6cf7681547 bfec6892e10b63a5
3ff8765be184a06f 3ff25c0e4deb16d3 3ff7dc41002399f5 3ffd8cc9e19c7d2b
3ff3571b0bfc9400 4005ba9acfff1110 3ff4096b3555c257 bfc7863672b65d66
3ff4956c6bbc1015 bff10c6895f99aa7 bff105bf861f5db7 3ff084ade477eac8
3ffa92172b88df72 3fc30b4bd34a068e 3fe843aebcc78265 3fc7e5414381e18b
bfe853297820c2d8 3ff3af0f67f13b62 3ff7aa07b4c5c56e 3ff11c4c58a9a49d
3fc2d9f21ecc5a1e 3fc80ecdb1f3e151 3ff3c39182ae40a2 bfc4c5ef59bc4376
3ff0d253fe60c2be 3fcd06279a8d90ae 4002deaa8aeb385e 3fda74ea3276d577
3ffb59eb8ff931fd bfdc3a0c8b5a60fc 3ff6631b84e0372c 40000f2a057acbf5
40030b1e0d2d8f69 3fd984d9b3b0d2c1 3ffbc31d012aad60 3fe94c9f02d0a1bb
3fc729431605141b 3ff1e9ccbee7185e 3fd65a7d1c963bb6 bfe54b1b368ff4e0
3fd51c26229b3078 3ff4fcd919548b93 3fcfbd9846281be3 3fded568547da88c
bfc5a37531de0506 3ff077020e6633f6 3fd254d859627ead bf983d3d81fec8f6
bfde9c37c7edf2b7 4002ee01201aaa67 3fd4ce265df70624 3fc8fb6c7d17ed48
bfe4339d77700bcc 3fe1709ded659419 3ff06939041cd06b 3fe8eef2897404af
bfc56161b632ef6d 3fe26d9e42739adc 3fdd35b4f1f77e23 3fe1db3403321f9f
3fd8d08df2daae72 3fe7a18f2b3d1927 3fdd4baa1d33313a 4001aead29a078b9
3ff2717aa433bd0a 3fcff57f5ff46031 3fdb48b98bb7c35e bfcf5a8e10793c56
bfb5128e612713c3 bfc443c7ce7cd69f 3fe894d778226a90 bfe0f25f7c11b61f
3ff73c0eeac8a14a 3ff438f11f02280a 3fffbafd8471c7ae 3fc6d6a9a393ce7b
3fff69684bbc4345 3ffabfef0d3754af 4004452f09016d61 400104a19180d47e
3fc15d26c90d8a75 3ff6601525f86520 3ff4e7c8e9cab217 bfd746c90ae4b370
bf9fe35340a2b047 3fdfc26a008ba7a1 40021ddf1981baed 3feb617a14f661b6
bfec902bb47109e3 3fad6a8901d0e7be 400099d7d64e9cd7 40000b30313c6113
3fea2f9647594504 bfd3e43ae06b7bb8 bfddc2ea92676ac0 3ff4bd5a1fdf5253
3fe1bfde5344b13b 3ff3118d2fef8d19 bfed8c53cdfc7eb4 3fdaa383034ad88a
40023778e5a5db20 bff44babdc58f712 bfd6b1d3ea159843 3ff8b6643a860a39
3feaedb561475081 3fd968947137b022 3fc67f603869b789 3ff221deb2e1e3f8
3ff61023c0f92dec 3feae201d1e7e8e7 bfba3dfacb48d343 3fe3f6586dac5a19
3febb469bca6c741 3ff6e15af2642dd7 4001e9519fb02571 400022a562fd6e4c
3ffb8bbeab463c54 3fb9af0a3393da43 bfeb776887dc82b7 bfdda18c2a1bc750
4002deaa8aeb385e 3ffa08e74da21cf6 3ff488d1f2da9408 3ff3571b0bfc9400
3fbc411ee9f435b1 3fee130197ad52ff 3fd57449f2b7db74 40006d17bdd625db
bfd2249c2b909ab4 3ffd2904e486eee1 3ff6eada19acc259 3ff0d8cc900c8b7f
3ff0089233b8bfdf bfb43ee803ec5090 3fcc5b3c3d161d1b 3ff54532807ea0b6
3ff9a47720f97ed3 3ff06a14a926a824 3ff2176f93ab0d79 3fec44763d46800d
bfec902bb47109e3 3fe944fecc3730e4 bfe46b848aee50d2 bfe6f9a3493e86a0
3ff93584ac2c73aa 3ffc15bb2038698c 3fdfa9dcd4c3ba6d 3ffee992fa661bde
bf820b8ff2a5394a 3fd25778786206c7 3fe5b4b580a6d832 3feb4d0b8398c329
40026257162a822c 40002b7aa913a5bb 3fd888e49a5eac53 3ff564029c950088
3ff0c629e24f3ecf 4000fb42994e6092 bfa4388365bb147d 3f914d43c48f67b5
40061f3e352b3a5f 3ff9078fcccbac9b 3ff4041ca3d2dc89 bfd07b5a6ddafff7
3fb73da1ba08a533 3fee5ccafa0e6148 3fc2757c7a9b817a 3fed6e93c326eca1
3ff0a575f00c57a2 3ff45c72c0dd106b bfe0f73a026905aa 3ff179ce3152f778
bff2da543394762c 3fd00684008f920e 3fec46a3a2da1036 3ffdb2d8bb63fad4
3ff53a094535f13c 400384d834664c6c bfd1e81fca328632 4000c6a54b7cbe68
3fe061923c6b8dc9 3fc21efafc67b0d4 400126517238b07c 3fda9497d3a628b2
bfd27766acc4e94d 3fdd2c50ad34cd1a bfe16744d36fd0ec 3fe4180f7520437d
3fddad9946ea29cc bfd21b7962f7ff2f 3ff96335813a87aa 3feca7ea8176580d
3fc0dc9e5631a54a 3ff7d65816192db2 3fbd3aa8e608d7b5 3ff8d2062c31805d
3ff10a0defd7dae5 3feee4b36ee8a458 3ff1fe453d4f5d02 3fe4146a72dc7544
3fd400cb28ed58c4 3ff2edbc01535eed bfd07b5a6ddafff7 3fe46e4cd192a8bc
3fb23a4f06524abb bfc84fc6a1901995 3fe54df1df0d2479 bfd99c7302aa5b1c
bfd746c90ae4b370 bfbd073aae097674 3f9a15960b8322e1 3fa0861c8d7ec0ac
3fd888e49a5eac53 400079aa0718d120 bfc8c77199ea93fa 3fefa0d846115563
3ff290184940ac6f bfd3a35afdfd2bda 3fffc2e2b2b917d9 bfdc417b1a8473c9
bff0899b37482863 bfd1f3f15604fc41 3ffd53ded74d34f0 3ff34d87f631415b
3ffd6edd59c40be4 3fdeeb31e970c4fb 400227b2dfd15104 3ffcd1fc011a9e61
3ffc6e27b3010636 bfdfb58fe94ec87e 3fef3fe560a530d9 3fc4d0f2d9941b8e
40053738a14a5201 3ff0e89dcd21dc00 3ff8b9b817100d6d bfc886183aec17b5
bfec2ad36b631372 3fe39c2726040dbb 3ff3f86c9b3c6599 3ff8bc07e0c8a71f
4000b428b8e3eca4 40034daadec77838 bfd2efdc4e0a6910 bfa5dfce2e04193e
3fd54cab543cf688 3fc3f9e56402c9c4 3fe7a4554e47b65b 3fed85da47ac2b67
400238d53fa2e5de 3ff2ca195019b83e 4001a1ab1c84f4c0 3fc71f761390c882
3ff300a02b6ac055 3fe98b540dbdffe1 3fe2c580272f3d8d 3ff9315f3da4144b
bfde940cd3e180ea bfe17acb9e943729 bfe5b4acea89978b 3ff3326bb971da7e
bf983d3d81fec8f6 40040b31bad6eab8 3fe80fa8f3bf95b7 bfbd07ebb44c527f
3fee87a287376417 3ff5870eb3b5d9d3 3fe92dd5e055d6bd 3ff50c309a1d681f
bfd8308f6ff0ed5f 3fb5c079f6a13d3f 3fc02cb0d5d58671 bfe1b69c235763b3
bfd2e0b01f991145 3ff2db430afd69d8 3fb5cb6d14ed4199 400031a678bf2c18
bfc54fcdd162e558 3ff3c90979de6375 bff1ff5c87668821 bfd6ec68d4e8abec
3fed6f50d48f49db 3fd9cf86c5cabf9e 3ff06939041cd06b bfd1372f20da1520
3fd200fead721276 3ffe3ec414a6f59a 400155b552441561 bfb2956fb0aaca0d
bfe0d4361ce78a27 3ffb8c681d775537 400227b2dfd15104 4004b3ee171d7707
4001ba4fa3866447 3ffd534f8eaa0c64 bfd2fc95edd4350d 3fdae647f20c25b8
3fc1874e2e46d3eb 40009e5d42af42fb 3ff722204ebacedb bfba3dfacb48d343
3fe95a27fdad0b38 3fd4172a2e1445dc 3ff0d58ac0c8e9ae 3fbb1e7ca4e8dbf6
3ffc001c2ec1f3f7 40059fdf16b58a55 3ff083ce44138021 3ff37bd191bec8a6
3fcb85da3157a066 3ff4b2664d1370c0 3fe1544ac6ec2d5b 3fddf6fc7120b69b
3fec154a7c6a618d 40018999a7c93849 bfdb6dca0156a10c 3fec2ae75f73ce43
bfb9c801493d3d8d 4001c5dcd1ccd399 bfea14d070456521 3fdd444ea5a7b3f6
3fe2c709c84b0c74 3ff121ece812bbf0 bfdb516e937eb351 bfa61f56f3ea195b
3ff6a71bdbf4de52 3ffb4e311d2ee095 3fef253f153e2665 3ff1aeab2edb148b
3fe2d80bdf92e088 bfe95f14bab05dfb 3fe19bd92c3082b4 3fd61d4e598ae9ad
3fd7d2b21df1e732 bfa8ed12a11fda1c 3fd230840f04b0a8 3fffa761e8ae709a
3fb07d362fa7793e 3fc9beff26a6a032 3fbc76cff7e8ff22 3ff6e2cb29e4e19b
bfdf82b8402fce21 40041fd32d4405e4 4001870ef5094707 3fdd19a3abb74285
3ff846947265cc1a bfe434aa840434ac 3feffbd54918e708 3feaf027eb424a45
3fd0705413350baa 3ff7f7505148b4b9 400622cc3881dbb1 4003a4f8cb3491c1
3fe754316e9b7a28 bfef47c27dd68587 3fcd58dd27f0af40 3fc53aca75ca8b60
3fd9fef14f0829c1 3ff6eada19acc259 400824f2dec21ab3 3fbb05a77bb85f20
3ffbbffb3258fa92 4005520bb84d1d67 400196fa5f520748 3ffa3a92fb56f492
3fcec6afa056be15 3fb46fd57a0e9b3b 40056252a3824660 3fe010ffbf523a2d
3fd963f57070d995 3fba9c7e5933111b bfc3fde9e195af00 3fe49fbc8064b27c
bfe8de4c05cec37d 3ffc1a490dd8ed0b 3feff81757e5b24b 3ffe7bfd6f714c9d
3ff160803d83596c 400796c715633d39 3ffb750b4aff26ec bfd6ec68d4e8abec
3fe49e8bd8234f36 3ff885a58fc391a3 bfd30e42472bb71f bfd4d1c720f21969
3ffb544f78c4eebf 3fe46e4cd192a8bc bfddaabc5bd0f56d 3ffed521250c8193
3ffffc4c6de85b93 4005ba9acfff1110 3ff3fbbe7e921bd1 4006046b1e286ef9
bfe99af3f9192608 3fd39776c75c5c34 40061f3e352b3a5f 3fe4c30e7e83436e
4004a24fbda6eeba 3ff452c761fe6c51 3ff32d680e839bc5 3fcb515aa51fafda
bfe3a2a34b21a437 3ff7cc4b6fe5e5dd 3fd748dc75818b2f 4004f44c8fb5b7ad
3ffd699ae8949c8b 3fda052cc88339b6 bff0724356dd6aeb 3ff73c0eeac8a14a
bfdb1f90d1b59eca 3ff8cfcca83af8f6 3ff3326bb971da7e bfbea73058d890f9
40022f8d0436ee12 3fe7b6311c352f1f bfc0610802ee2e6d 3ff02d18a6236d0c
3fcbae6fcbc1840e 3ff10d18b9827dcb 3fe39a370d15a5d3 3ff2b5d16d876a95
bfe19fe97dcae326 3ff1c70bc89c5ab4 bff18b352ffcc57f 3fd963f57070d995
bfdacc0b73e81ce5 3fe39da142fd24ac 3ff1397dc8b5ce0f 3fd39ee9e6a82ddd
bfc21751e3fa7ee3 3fdff059c60b37bd 3fd471973876abf1 bf79449452f0f0ad
3ff3275b135a9985 bfe42d4590e7e270 bfeada3995057b86 3ff7d354407660cd
3fe6d904a330c00a 3fbd3aa8e608d7b5 3fddff65ac89a6a3 3ff1eb60a094b442
3fee57eaaa3d57c5 3feefe927d37fb35 3ffb61946df4878d 3fe2435f8c014c8a
3fe3970b5c24e1ce 3fe1262d665ce166 3ff4d7599f30e19c bfa43b677b512265
3ff0a30d447d7f3e 3ff81712bba0b8d0 3ffd79c32bb35767 bfb6679e1d049fc3
3fe663b8ae3ac047 bfd6f05186e0c2a0 3ffcd3e71669f430 3fd9747cc897ad9e
3fdb6cb1547aef7d 3fc98fd0be6fe625 3febc268869cc801 400121d694d61fab
400496fe22c7dfa1 3ff40ccbc9c075b0 3fff2645ee17705a 4003d45ec601e3c0
3fbb2316167f3664 3feff07859f07e35 3ff98f0d8ee96995 3ffcde2cab0eee2e
3fe6494117c3ee11 3feb662b1bd5ee43 3fd03b848c30caf5 3feb23cd2cbbb6a9
bfe8589c088233f4 bfb5a887f3475743 3ffd7aec8f72bf84 40009318aac1c98a
bfbd07ebb44c527f 3fe048d90e4bd8f0 3fdd6aa6520a3210 3fc2ec30bf6ce52a
bfd72f705a9861ff 3ff30115884829f8 3fdb48b98bb7c35e 3ff36df408b2df49
3fd77ec8aa2abf63 bfad1bb8d5b1fc8a 4002bde4484898da bfeec5d97b550439
bfec5bfdc1703881 3fde8840e536de2f 4004452f09016d61 4001d7a19a1cc3d1
3fedd25bea6bda39 400196fa5f520748 3ff732b0ca73015d 3fd3e2c0eb5f60c7
bfea957bb0472937 3ff0340091b0ae50 bfbc5db74039848b 3ff0ab152fb562d3
3fea653a11823f09 3ff82c985745baea 3fcb12c6879a3d0a 3fd1707110973e3a
3fdda08a76d869ad bfec3f89e2be6fb5 3fd44bebb185f934 3fe58617497b22d4
3fd63d0a5148b33e 3fe9fc3e136d005f 3fccda411fe0860c 4003a4f8cb3491c1
3ff47cc777683db3 bfc72cb7652312fb 3fea33ea5f6465f0 bfe99af3f9192608
3fff95397175d88b 3fdf305b960acdfb 3ff79aeb6e31c665 bfdee142c91c2f61
3ff7185840128b86 3ff8efbfa194d981 40069c14106408bf 3ff1c7fe1c8612fb
bfcb591f8d4997dc 3fe583c56a907210 3ff0eacd71d11c4e bfcb539429953d45
3fc44d4231e3f97e 40049498b4666cc9 3fe8a1210f4700bf 3fef090ba41746f3
4003c6854d0d3e77 3ff0a45488b36b28 3fff2fb2da7fc1c9 3fd67b5f9b22ab1a
3fc46b3b64644bfd 3ff0b135e9ae3068 bfe87d2d71213812 3fee7c83401fc371
bfd3f5104db19f51 4003e9e52c0fb816 3fd74ca3449481c9 4005867b8a5f4784
bfa9ff6b54579df7 3fe600bb1f756d8a bfba145bf20720e1 40058108ef73afea
400147c25281e8b6 3fc5ad24a1ebc751 3ff9884a5046bbc4 3ffad18a8fbbc5de
3ff8710b4cad3faa bfcceb3f99e63e8b 3fe6d904a330c00a 3ffa0706a4705f98
3ff438f11f02280a 3ff9aac61f6faa50 400259fa2ff5f362 3fe50ece7640d517
3ffd8461d7259adc 3fc61a5b4bef32b0 bfe537569b336969 bfd3eb4c606674c0
3fe1262d665ce166 3ff8bad5561afb81 3ffc2d7614dc9204 bff2658060425452
3fd1463d55205fea 3feb7d819c413988 3ff2787bfd579863 3ffb7f8b7da4bb72
3fe6d88d8902f892 3feb5ecf310ec882 3fe36bfdede2e9e1 4005461ba97c29ed
bfd0fc3e256b196d 3fe299d5ee3c3aaf 3ffa6dfd932a0297 3fd7c02403c408d3
3fb0e306bb254c9d bfeaafb7290589af bfd1eb1e4656555a 3fea5d0da3099c60
3fed11b3c68a4d3e 3ffff6c653e5a8af 3fe2255f88eee5b2 3febc268869cc801
bfd70c81feb1703a bfed8c53cdfc7eb4 3ffd48d2fa65c0fb 400796c715633d39
3fd5baf2419d91b2 3ff319918502700f 4003f09b14cdf07b 3ff1c87bc1983edf
bfee251a746520c2 bfe0a0b400b23183 3ff2d44db25a41bd 3fecb1c70d922960
3f94c8ebe9113706 3fe6d88d8902f892 bfc635d68ecbbd78 3ffc43def5abafb3
3ff8753d3e8800f9 3ffdc9a55cdf841b 3ff05bc3dc3fb258 40022f30697f62b3
3fe2fe7a83eeb932 3ff85d02e1bcc17c 3ffbbab8f0bbb44e 4000d267417b6ad8
3fea1166434e7d84 3fd0e2c3cd8e646d 40021fe5ab6b1b71 3ff564c53dd02b05
3fdea5fa6a567818 3ffaacfff36559e3 3fe178a3f1038501 3fd76205de27219d
3ffeea8543dba88b bfbc5db74039848b 3fed56b3d39129ef bfdaa7a2c69d8c55
3fe4b2ada9c96451 3ff3fbbe7e921bd1 3fece6207ca2a6cf 3fe6a43bd6605f04
3ff4e66e3c13c526 3fe4e1496f7dd066 3ff937d1dffdc1e7 3fdb84aabfae1cd3
3fdeeb31e970c4fb 3ff53a094535f13c 4000d267417b6ad8 3fffcaba5728b6f1
3fe0249230de0f5b 400238d53fa2e5de 40059bdc06767b50 3fe8bf21b1edee49
3fc058ce1b6bf638 bfe18cb2c125bb83 bfaedc12701f85d2 40022ad24378ee0b
3fedcc517ff0b1d4 3ffdf03942ce6e0b 3fc1ced53c44e647 3ff28b9cb9480265
3f8a840e63a8f132 3fc88ef396017751 bfcb539429953d45 3ff6e4a0e7aa8af8
bf94730a2ecc0a6c 3ff0535aaedc0d99 3fe7c4afcf97fc2c bfd739c6f2e4d633
3ff3a64301c8095e 3ff87e84fe588822 3fe42ce80d30ef48 40005059a503335b
3ff00c76ec37d18b bfe46b848aee50d2 3febd2261a6679c0 bfb058ff7c52edca
bfd1e81fca328632 3ff2fdf33db33b73 3ff1d828692c2ddb 3fdb04c6f72cf6c2
3ff3c90979de6375 3fed49cb34f83ac7 3ff3942a6b23a344 3fe1b6d07f27c406
3fe84a3c9cb1ad50 3fefea8942067cf9 3fc585857adc4b5f 3fe36430a668bf96
3ff4acd7a7bb9dc2 3fd65cf7d899a6f4 3ffb2f74dd767f0f 40011a91c703b9be
3fd1cd1faaf928b9 bfe8d0143195424a 3ff006d79db4776a bfe1b69c235763b3
3fe7ad2314dbffc5 bfe7155a02eecc82 3ff5ae7987ec042d 3ffce094cbe68f45
3fc1c394278315cc 3fd2f6007a4a54f5 3ff1a83d8b0ab483 3ff696145e8ef4c8
3fd0ada9859657a7 3ff6a0ab7e134a43 bfd2efdc4e0a6910 400016b9fa7ca5ef
bfd39ac2a8ac0e18 3ffe5de6e701d255 40031816e7db5c6b 3fef24092fdd1384
3ff01d6eecd1f064 3fe539e36a0e32a4 3fddb206192ec61e 3ffc9c67150d68e8
bfd87267fce251fe bfe1aa986025cc02 3ff5c59861502d1d bfd19a6094cb5ea5
3ff73bf560b3e3d6 bfe0101880a25a5b 3ff2cb66351dd1aa 3ff6a71bdbf4de52
bfb7aeb8e19d53f7 3fe32a9e1846a42f 4002f949f278fc02 bfdd1e4976ef072b
bfd29ca67989f9b7 3fe817ef1d57769a 3fcb85da3157a066 3ffca67de8f4e3ed
40052bbbe02e9ad3 3ff8b29ec98b7887 40006ef681a8a003 3ff2bcc0158ea707
3fe88ba4709d42b1 3ff722204ebacedb 3ff8250342e385e8 3fdbbfecce0b559d
3ff82ec0e83c6502 3ffca95dcc3bcc0a 3fb86f203cedc5e1 3ff02b497c0a8b0c
bff04ecf552a9f14 bfd842d377df8a4d bfd932d9a3c9c9d4 4007e9062f93cfe5
3ff9c89117945432 3fe80f83fcb78176 bfecdf6c0be2d65c 3fd0c37cfbd08b08
3fe38171e08fbca8 bff04dc6d274c394 4000eada921278d1 3fdd3cf19a5b0625
bfbbb95c7c1e3826 bfdcdf4cdcde5b84 3fdd4987b58ecb97 3fe5184c78990f02
3fd63869944f5c7d 3ff4265e2e0bbc2c bfaadeeeb664c328 3fd7724f61b06551
3fc8f135867dc2eb 3ff55da10be173b3 3ffaedbcda073cd6 3ff253c7333f3e29
bfdcf192be5d748f 3ff694d9562e8cbe bff11ff3d12f4d15 400224228dc67eff
40007d9d5714e189 4002e6beca9d3528 3ff4875bf47f83d4 3fec45a217ee16fa
3fdaeca680e4a2e0 bfe609a160d3c4c7 3fd9bb61511583db 3ff81712bba0b8d0
3ff925f3536d6d85 3fec8c5235b489dd 40042b12015b4a9a 4000c9e1b4656b3b
bfe458edf5820df8 3fad509cfbfb0ff3 400101b7bd37a9e3 3ff2a03bf8ed9fb4
3fdfa73b511e6737 3fec1287b1874b65 3ffe1577bc7d113e 3fdd3cf19a5b0625
bfdfb58fe94ec87e bfdf8448df8d2851 3ffc001c2ec1f3f7 bfb5cc80b85e2f1a
3fd1bb52c52b2420 3fe68d032455070c 3ffd73ab476d6a0c 3ffbe7e98c4543ba
3ff7af8f28932297 bfd6dad143c3ea30 3ff1c654f90d25fd 3fc678467bbebfcc
bfe5b4acea89978b bfeec5d97b550439 3ff019a6c32db3c4 3fe52ad8c9324fc6
3fb8fe877dc2a9a7 3ffe96ba7841dd74 3fea9a74250e3c02 3fec0bd71614768a
3ff23d22fa01b32f 4000c9e1b4656b3b bfeb9b6dde1d3c21 3ff6044a6f050321
3ff4875bf47f83d4 bfeeaa1a55f46caa bfd8a392c13766b8 3ffe2eb28300130b
bfd348be8b63d911 3fd5440b01ee49d5 3fe4180f7520437d bfdd4d00eab293d0
3ff9b0bfdbaa2dc9 bfe464291f0c98b0 4003a4f8cb3491c1 3ffc324c46acb00f
3ff3d9617908cbba bfef29fa5517ab13 bfe38aabaea70d90 3ff8b9b817100d6d
4000a6e2cfc54298 3ff9e81bfe0be808 bf6206a65e5e7ee0 3fdddc1a39589614
bfcb3502268390f1 bfe71c2faf5002c1 3ff6459f1c8972f1 bfd927f5e4d887d7
3ff2a79c7ff4aef4 3fb377cf9fe870e7 3ff6b67463708a9f bfe30f7f08eff47b
3fc40c479f978e00 3ff85f6e992f866e bfbd073aae097674 3f8b9f770238b16e
3ff2f1b8a7368e0e 400539b57f543d78 3fc10c536d369070 3fe7cb5fa96f69bd
bf97c7bbf3e9d0e3 4003bfcbe2375d27 3ff0eacd71d11c4e 3fa7c37b5312069a
4002e0b9877f408a 3ff044796d67e358 3ff8ff17929a69eb 3fcbbdc9f7bad7ed
3ffde1303a0918a2 3ffaebd458c951fc 3ff46e531afe6013 3fc81b93ddd811ff
40009e5d42af42fb 3fc16eeed1fe7ff6 3fe2e042151107d8 4000928bb07d99bf
400496fe22c7dfa1 bfd27766acc4e94d 3ffe7a9dbf5142a8 3fe247e00408dbd4
3ff8d6cf0ba083bf 3ff1dea8bc0f4633 400502d814609c89 3ff608dc486e6ff6
3fe01326dcb65d82 3fe56309a395a977 bfde9c37c7edf2b7 bfdd4d00eab293d0
3fc3d677cdc4a370 3fff8cc60bf339bd 3ffe082ad3f06d92 3ff7d65816192db2
bfde0125885a295b bfaa84625ae4b72c 3fbc28ef4b483d09 3fe73ff024687ffc
3fdb3fb4f626fb6a 3fff184cc9476655 3ff4d7599f30e19c 3fe68efa28de4130
3fea653a11823f09 bfe12ddf089a4850 bfdd07085f7fe545 3f9651ef7c3c512a
400365448f1485a7 3ff04d71fedb92bd 3ff0f6dc7d02a91c 3ff16aae34119843
bff190cf40ade665 4000c51154b6f238 3ff2ca195019b83e bfe9ef0b98f6e362
3ff474b439a8f32b 3ffbeffb247d3de6 3fff51180ddf7198 40002359cfe4afce
3feb398c4d30fed0 3fe20bbb6056eef3 3ffd2d6d0bf96f35 3faa84f61fb9c868
bff3425e7a4abe19 bfeec5d97b550439 3ff53c3c960893d6 3fffbb1f30fac7b6
bfc911cef2f489f7 3fe8ee46ff11cf9e 3ff2cef496f87fea 3fe8e6fff80f9bd3
3fe022e962af296d bfdbad139b378f1f bfe6f414734de00b bfe426758b5ba26e
3ffc43def5abafb3 4002bc23284b0594 3feb4fed5d82dda4 3ff73384517680fb
3ffc43def5abafb3 3fc49fc4746e2c58 3ff5e45cd0437cb3 3feae201d1e7e8e7
3fddcf55547fec63 bfd81a62442d79a2 3ff653786a8db4a3 3ff91bf28ec5d467
3fdebee403408fe3 bfd932d9a3c9c9d4 3ff5014d8e19323d 3fd8f82a81ac08c0
3ff3dde0063dd366 3ff2fcfdf8ffe6c8 bfd11688b016d2b5 3fff16a4af0f84ad
3ffb4e311d2ee095 3ff0d7f3bd46458c 3fa93ea69129754f 3feb2313fb7986e5
bfe24f9d6a9ae295 3fc878980b3bbe76 3fa02086c009715c 3fe7539504745c84
3ff6ae91d32b8d8c 4005b42924d8b4df bff047a1aed1f16c 3fd3f5e03356ad06
3fa1c43fa9bde34d 3ffdfd3b84e7585c bfcc0e55ed8a986e 3fff8f910c71f523
3ff4894f31351902 3ff5edf815cd5c49 bff06603a05a9328 3ff557ededdd6388
3ffc1b04453dbecd 3fd65cf7d899a6f4 bfe86a4811dbd0ab 3fe9b9772c403844
bfe14cad79bd5c27 3fec93a433c45807 3fec93a5c9cacac7 bf983d3d81fec8f6
bfd88da909e6a8d0 3fda7300365f7e9d 3fe78d0b58799ae6 3fd888e49a5eac53
3fe98b540dbdffe1 40017b44ce7fe9ab 3ff86ba8dadef697 3fc3f9e56402c9c4
3fcffd30de279432 3fe7b6311c352f1f 400186f98887eaee 3ffed4f2d777a6f5
3ff5499d34a8ab68 bfd44eb373c1e787 3fdd0019025d5630 bfc9d87bd612a743
3fd66e6266a7993f 3fda97f9885299f7 3f7268892eebdb4c 3ff200432e0aa850
3ff9408e4ce5674b bf9003c20b1470bc 40029813367087bf 3fd9cf86c5cabf9e
3fc678467bbebfcc 3ff98549429c24bb 3fd41d874ff7773e 400282a58dcc43c7
40022c00176b7f8d 3ff52f5255660324 400275e7c3c43cf3 3ff0f41bdc5e6941
bfecccb3bace0a7f bff43b07ed9a99e3 4003c332476fdf8e 3ff1e9ccbee7185e
3fe9aac797bee874 3fe91cf83e33ff2d 3fc0a116bb3f83c4 400088cc75888b45
3ff65469f23bf4e5 bfb75121795e892d 3ff65214b8926468 3fee284247e73263
bfb6b51bf63be02b 3fdda74355e77078 3fc7ec9128c96636 3fff54899d4c99c7
3feef09dee42152e 400183b6d2969f3e 400470ea6100f89e 3ff4a579ae9a4da5
3fe47ab8aba02eb6 3fd5f7ee53984e11 3ffd6faba6ca4e58 3fe7c4afcf97fc2c
bfe86a4811dbd0ab 3fffec939bccee8e 3ff9078fcccbac9b 4005b0bf77f1baa1
3ff3fbbe7e921bd1 4001511b13a74b70 3fe894d778226a90 3ff44511284b1fcf
bfb20a023abcad23 bf86fd380b4bb162 bfd1eb1e4656555a 3fe6485b557d450b
bfe38dbff9b91750 bfa8ed12a11fda1c 3ffb369617e3a1a1 bfe99af3f9192608
3fd117001c13a715 bfd8b1d262086bd0 bfcf9042e8e267aa 3ff92f08d5b5f0cb
bfc24aac334f4bb5 3fe894d778226a90 3ff4f00d89ab1620 bfe36d63f292d08a
3ff2cb66351dd1aa 3fc33018fe5758a5 3ff121a6591a49fd 3ff2ca195019b83e
bff43b07ed9a99e3 bf83ce2e40ab39a0 3fc80d590941ed8b 3ff81afc8cfc4571
3ff9a535a6944693 400090f0903bfdba 3fe2c709c84b0c74 3fd65eecaf8d9180
40061f3e352b3a5f bfe2d6811d5b2080 4001d4db12546ed8 3ffbebc7718e3280
3fec5286f1bd08c5 3ff02192a765ed16 40020215b2ea3a1e 3ff15df4cba0f223
3fed6e93c326eca1 bff1486c5671c97e bfe62b228d59bf00 3fe8e9d6ca715795
3ffc70e964ce9978 40001e1d95c8e3fb 3fee92575386e120 bfe0d3f126ff5c48
3fd6e6ae7270c6df 3feb12f33cf95b17 3f9a2f3b12336687 3fd4cd5b4771d7fa
3ff92dc885afc18b bff4a9b00c9cacac 3fbc28ef4b483d09 bfe46b848aee50d2
3fd602035fdcde19 bff4a9b00c9cacac bfcad97da2d10b16 3ffb89f402c84f41
3fe5c3997869a376 bfdd86421c0528d4 bfe002674f9b87fb 3febcff0b24008ac
3fe7b6311c352f1f bf1f1b12e51c1375 bfd72ec99e63a437 3faa9e02ea2565cb
3ffdfd3b84e7585c bfd5fe8e57c80763 3fe6b131534d61ab bfe206ef32a12e56
3fdd35b4f1f77e23 3febc268869cc801 bfea1cde1cd35149 400383c66068d8b5
3fff000e7612ae1b 3fe709ae60397acf 3fe003ce578cde68 bfee300bbde52875
3fe1bd8b83ac0205 3ffc7151f1453d8b 40029813367087bf 3fb715a9d4432536
3fff71a97da93f65 40029e96334af666 3fe55160186fa576 40021d70f5320058
3fddc00f82ca0183 bfcc0c626defd8b3 40030b1e0d2d8f69 bf983d3d81fec8f6
bfb293e1a62881f5 3ff02ae39f4d6391 3fd5f138a1b95934 3fe0dd93809c2d35
3fe49423f7c56b2d 3fcdbeecf1826fba 3ffa06b24315f567 3ffb0c61623e0f63
4003047e5d199d4e bfc4c4707aad7fb0 3ffcfea56614e495 3ff3d4f936d2605f
3fe80fa8f3bf95b7 bfe7c85759bd4311 3fdf13eb88265a1d 3ff5e7dd50095aa4
3ff4d789e5d2756c 3ff5325125cfc6b5 bfe30f7f08eff47b 3fddb206192ec61e
bfec5bfdc1703881 40032196e4fde8f0 3ff21d5ddc79767d 3ff9c89117945432
3fe5bcf61048c23a 40061f3e352b3a5f bff1ff5c87668821 bfd272fc11c4d8f9
3ff9cda8f4bf564c 3ff5f6fb7c86edf5 3ffac9ba14885a41 3fe32a9e1846a42f
3fffcb05db2201e0 3ffbbf2fa930f197 3ff1431ecc9a4b3d 4000fb42994e6092
3ffaba88f97ab244 bfe14cad79bd5c27 3fd43c6eae605f7c 3ff25a32383c8ca8
3ff020b11ca2cd9a 3ff0b094c8936b0d 3ff3d43e466c93d4 bff522efe5b03ff2
3fd6ab5ab15c5a07 3fe8ca9742253560 4005520bb84d1d67 3ffdf185f2c318be
3fe39c2726040dbb 3ff98f0d8ee96995 3fde5db3c50e9daf 3fd81abebe61d5e3
3fe98b540dbdffe1 3ffc1a6be4ac8257 3fe2f62970367df9 bff4009a45ee3940
bfe0101880a25a5b 3ff990dcb9024b95 3ffc74b3ddfc67ea 3ffa6db61ecdf702
bfdb1f90d1b59eca 3ff06a14a926a824 3ff121ad9fda2f79 bfe5d8f05c0770d7
3fe877fb77590a2a 3ffd83468e4a86ef 3ff5f04aeaca4e88 3fe4a7e37aca54ea
bfbccae45bf1f721 4000c9e1b4656b3b bfe7ec6be66a9448 bfe573f7fb8b4bf4
3ff077aa4d6fd211 bfe426758b5ba26e 3fd766dc33e4b239 3fe20fea88a5af12
bfe98c1c921cad7f 3fbac5db9afc4bd9 3ff0ab022d74fb71 bfe16aa5953472f5
bfb7d2f94074c82a 3ff0b135e9ae3068 3fea5d0da3099c60 3fdfc8eab69562ee
3fe94ed6c8d1bbb8 bfd590884bc94cad 4001559043cfe796 3fe1bc2183c2fe24
3fe4e92be764af06 3fe9f8e2cb2b1725 3ffde757804dce27 bfe3c7a5ece9a8b2
3ffc1aa0139dc2be 3ff4b2664d1370c0 bfe9497c33475c1a bfbd073aae097674
3ff8ece18b48b821 400400926e085488 3ff7418a846a4e1b 3fd952ae5ad9c5d6
3ff732b0ca73015d 3fe7e74511a5cccd 3fec16c95995048f bff4a9b00c9cacac
3ff6d1febc8b94e1 bfcb59398fb40751 bfe42d4590e7e270 4001580e8bfc7907
3fe4f67496d36ce3 bfee893b4e27c79c 3fe9583482f07cb9 3ffb462734b9126c
3fcfb4708de25296 3ff1397dc8b5ce0f 3ff46b912f6c2c18 bf9ca1d76f3e0f3a
bfe8aba2bb9155e3 3fddb206192ec61e 3ff32ead274cce50 4000a0722d2dac63
3fee0aa43922d8c1 40019a2ab165ea6e 3ff5844820a86175 bfd927f5e4d887d7
3ff6b37d3115fcd5 3fd5baf2419d91b2 bfb8a5262672a27a 4001b7175b1a04aa
4008525f3372e476 3ff30f767caa8ae1 bfce63ab40054654 4002f949f278fc02
4001aa2160d9afb7 3fcf76ede4bf22a3 3ffb9698976cab82 4003f7dc8c970dda
3fdea5fa6a567818 3fe6c0f940b7385f 3ff06939041cd06b bfe537569b336969
3ff7a0476ddc6439 3ff145932b2f4aa0 3fd52c8e5cd718ef 3fe42e3ce54aff00
3fb68922480f0e39 3fd2754a68b760a9 3ff6fbc48bcafcb8 3fe87d289c92bf85
400160a09a4acd02 3ff459e95aa7d0fa bf9026a2a83b11d7 3ff4b2664d1370c0
3ffa5f42ab956f99 3fffae5fc18edc10 40029af20e83db06 400307b49671f6c3
3feb969330d6800c 3ffef494ae1a72bb 4000d001dd6e0394 3fe49fbc8064b27c
bfe48a086c4ee3ac 3ff014210a65607b 3fdd2eaa9f0dd389 3ffeef88c26d5bdc
bfcf9042e8e267aa 3feb5ecf310ec882 3ff22ea10b240a8b 4000ead3f96f059d
40014845a7373930 3fda592c14d4f62f 3ff2a4b87131f1ef 3ff6c12f07500162
400079755b5cc15e 3fd5de3cdb0c34af 3ff53a094535f13c 3ff2edbc01535eed
bfd8308f6ff0ed5f 3ffc9be05270f6e4 bfe7bfc7d431390a bfd4e824973d4bef
3fd9f3db10288da3 3fba117a35feaeb9 3ffbd3fe9388ea9c 3ff649e141174491
bff5b30482d548f9 3ff5ff4ea79a178e 3ff8bad5561afb81 3fcebd8acb1405e4
bfd769ec98d69e0a 3ffed901d565d62f bfa96ea3b8b3c970 40005084d00456eb
3fe49dab6996b557 3fe539e36a0e32a4 bfc9dafcb4721758 3fffcdd9392e3b7d
3ff70b41214877d3 bfe38f575305ed44 bfed8c53cdfc7eb4 3fe572273858ef3a
bfd29ca67989f9b7 3fe04661651c70ec 3fe6c7e30565ad0d bfcd660b4dcbbeca
3fc2438806c0cacb 3fe2260c9b19e333 bf6e647cc561e250 3ff92f08d5b5f0cb
3ffd188ecc90e9e4 3f905a56735bcb2e 3ffff889dd94dbe9 3fbdb946399a6323
3fd3f5e03356ad06 3f8f9f3b1b2e9b84 40023b20b97697b1 3ff4e21b804cdf29
3ff59880cb73d9a9 3ffeb13d0f35890d bfe9899f647c0302 3ffbaf578e34c062
3ffae5eece76af33 3ff1e2440bc114b9 bfbd07ebb44c527f 3ff5b3bbd0740acd
3f879bec4f2c9ebe 3fea301673a9d488 3ff497113f324342 3ff3118d2fef8d19
bfd4e824973d4bef 3ffca6b8ada9e881 3fdf4ef4286e913e 40006376759b26af
3fea5d0da3099c60 3ff98f0d8ee96995 3fffbb1f30fac7b6 400172c0f2637d1b
3fdfb7914c10b2e5 3fe2afc7aadb3324 bfd3f5104db19f51 bfd369e74ee492ec
3ff0d166d88482ee 3fe6c7e30565ad0d bff2524ecd44ea56 3ff3326bb971da7e
3fdab22f927dd9a7 bff0c407d09e48c2 3ff7c63984293edb 3fd3399950bcfad7
3fd76a5a5b637031 3fce18d8b41e4e90 3fe72a7726087105 3fe1db3403321f9f
3fe2586b0c1336f3 bfd69d9a36eec4cd 3ff0be334f578c36 3ff22786018bc5df
3ff6b9e78c47cd9f 3ff7fc96d1f3e9d9 3ff5eac2917ddd84 bfd2f5ee92ac495e
bfd7d51f6a0b5201 3fe7bce50d19e276 3ff2bcc0158ea707 bfe27ebd7e3950b1
3ffd8cd926cb2d15 bfe2b3c52596f4fd 3fcd4182131ea4a8 bfeeaa1a55f46caa
3fb9af0a3393da43 3fda592c14d4f62f 3fe8d414a48dbc7b 3ffff1625fc427f5
bff3746ee72c1186 3ff7cb45e2ac6ba3 400278bbd910cad5 3fd6f3e2e28aec81
4001dfcef2708197 3fc9472729e0041d 3fd93303f5ee304c 3ff3326bb971da7e
3ff55fbeb8547d2e 3fef30f97ea50ff6 3fff8c4294874b7a bfe90944e94d3294
bf92a744d5ac7cc2 bfae045eda0fbc06 4005c7207ddcba6a 3fe1f52baa811795
3ff724d8f57d7948 40047c89a931782b 3fd9747cc897ad9e 3ff1d3f0f9cd3c1b
bff20d1640afbf9e 3fe80389eed03c5d 3f641cd2ac942e70 400502d814609c89
3fff58f1bc8daebe bfd25a394aa3765b 3fec160deb999ab7 3fe9fa4670fed15a
400231cee79ef63c 3feb39f5a5dbe96e bfd9eb20d4a3b3f2 3feda649edd91ea4
3ff9218a6b048ddb bff5b30482d548f9 bff5b30482d548f9 3fdb7b3df27aee14
3ff6a571f56c2983 3ff34a4e1cf53f3f 3ff0f4a5c0bd5df6 bfe2c95b331dbc44
3ff47f3ff8da67a0 40044e7fc100934e 3fec673126b4fb5a 3fcdfb1b450c2999
3ff5883114e485fd 3fc0a2fd9c7dc9cc 3ff121ad9fda2f79 3fee130197ad52ff
3fe2d80bdf92e088 3ff119841e00a9a3 bfd7f06bbfc53a90 3ff33bee8a8bc026
3ffb2ced864ae2de bfe08dec8cb3494c 3fc2ec30bf6ce52a 4005ba9acfff1110
3fda61292788b388 bfdfb58fe94ec87e 3fff305a6c06602a 4002ae4ce3fa9e4d
3fef5786e2a46996 4002b17fbba426eb 3fccda411fe0860c bfd4a00c7644fae3
3ff325bc82ad05ca 3ff290b68e05f53f bff287cde3d87cc5 3ff36040a952c3b4
3ffd79c32bb35767 3fcbae6fcbc1840e bff2d6a090bdd101 400824f2dec21ab3
3ff2f29d2be38a08 bfd8a08ee8fdebea 3ff0f2d696a47bf6 3ff454e16d2489b0
3fef3fe560a530d9 3fef5a1f88fa404d bfe235063c1596d2 3fef86410f82d807
3ffd534f8eaa0c64 bfe9704cb6cf055e 3fee79d31296406d bfe491663a0bd777
3fd8a4205bcd6614 4003f09b14cdf07b 4000978d6770275d 3fe1d0fe69048b42
3ffc43def5abafb3 bfa5d34385d55158 3fd3015162d95d65 bfe03dd4d04e5340
3fe49e8bd8234f36 bfec703edac40ff5 3ff604f03713e11b bfe4cc5704d152fd
bfdd3ea1f85acda7 bfe525446555941c 3ff3af0f67f13b62 bfe4f228ec3efb6d
3fa6e659312df969 4000914aa6e3e639 400318cae6913b16 bf9d381c27cc725f
bfb98b3c61bf4d3e 3fb90f0b63ab6aa9 3fdd6aa6520a3210 3fe3b6f4c6fef29a
3ff4db3325d07e5c 3feba22ac601f559 3ff1e9ccbee7185e 3fd3566785a45be5
3fe72cb371c6c431 3fc7760f0fd66d05 bfdd86421c0528d4 bfcec58e12c176a6
4006a62e67369ac0 3fd31cee544b5c15 3fc7d2edda54dc5f 3ffce7906583b493
3ff175464603c829 3ff9552dec3ade61 3fee284247e73263 bfed0c1d474430ad
400196fa5f520748 3feb523a75295378 bfd300fe849831c4 bfa27bbc510fbe5c
bfd255d3d806be7e 3ffa0a264fd562a2 40001599c7e7ca18 3fec0d42d93b64be
3fd475c3e5017be6 bfc1f62bc7631081 3fe959b59bf7be7b 3fdd9c18ae499bc3
3fe7c079b3d4f435 3fd3015162d95d65 3fe2435f8c014c8a 3ff191e2f5447ca3
bfd5958dae723070 3ffa3abdfa1d64c8 bfd2486439774b58 4005ba9acfff1110
3ff7320b0ec13094 3ff4ffc9528e5617 3fee39eacd1f15f1 3ff47ccf23d74e41
bfe0f25f7c11b61f 3fffa70f40812a75 3ff7f536834ebb25 3fc35bc9a33c36d4
3fe022e962af296d 3ffd8cd926cb2d15 bff179d3d9e395d7 3ffc0e5e7d188212
3fe15dce1a804642 3ff9ea289497aabb 3ff2176f93ab0d79 3feec617be541c36
3fec877078426fd1 bfe6a7090aa7bf6e 3ffddb151d4bcfb4 4001efd8c816e1f8
3fea7db997acb7ed 400439ae49a728bd 3ff529a751a221f6 bff0234f44bae28c
3fd1df3dd9f18bfe 3fd1cd01880915e1 3ff4fe8a243f7a7a 3ff4ebb6d8444d0f
bf946d671252ee48 bfd170e4d1718291 3ff3b3d88d122b3b 4004083a631b0b7b
bfc9d5d2d55462e3 3fe178a3f1038501 3fe5bf7a70f602a4 3ff10178ebcdf64b
3fe9f8e2cb2b1725 40033079a4b1aa9e 3feef09dee42152e bfda3487b3d4dd52
3ff1d7c7454ec3f7 3fdfc26a008ba7a1 4004452f09016d61 bfe2319c387eade7
400482ae089f82e2 3fea2f9647594504 bfebd1ee8ce39d9e 400272aede587e7e
3ff8155606a86d13 bfdfe5ecdbafa7f7 3fd00849f311945f 4005bd17ae08fc87
3fe27afcc93ce3fe bff20d1640afbf9e 3ffada703426f0b3 400404a289e15f92
3ffe723aeb4b290e 4001de3cb007b9d2 400824f2dec21ab3 3fe600bb1f756d8a
3ff076214ce8a631 3fe94c9f02d0a1bb bfd1dbf3b790811c bff5b30482d548f9
bfe26be811f1e019 400000acd3867eab bfe7c1c4a420739e 3fc9472729e0041d
bfebaf504fc8d6c8 3fd8d1de8f3a512b 3ff8e8275337bcaf 3f879bec4f2c9ebe
3fcbae6fcbc1840e 4006a62e67369ac0 bfe27ebd7e3950b1 3fdfeb9563081ac6
3ff0702e5cbff886 bfd72ec99e63a437 bf94730a2ecc0a6c 3ff2cef496f87fea
bfe80dbbf58e517f bfd216053a9dd6de 3ffa92172b88df72 3fed6af56ef528a0
bfc1a72bbefad255 3fdb3fb4f626fb6a 4005d56de701dc76 3ffb5826d53b76af
bfc886183aec17b5 3fdb2e7119d69ad4 3ff937d1dffdc1e7 3ffa31975023bb0e
bfedadf23c68f7d9 3fe94dc4adf7edae 3f919e046c33c600 3fc754c5cb2ce146
3ff24b624eaa58af 3feaec39f48f6abf 3fe9da1eef12655e 3ff65339d41b199e
3fe395e2e824f74c bff3425e7a4abe19 bfee02393a08ba8e 3ffddd4e25ffcdcb
3fe904cec9385e95 bfe5b4acea89978b 3ff9efeb655c6a8c 3fdcf283677d76c3
3fd63d0a5148b33e 400147454c67ded4 3ff26c44f0f3704a 3fec39f04b3eea0f
3fe6c0f940b7385f 3f92d34c0e6e8420 bff18b352ffcc57f bfd6a06d589f21eb
4001832e0c9c5b41 bfd77abab57e043c 3fead2a547a89941 3ffed4f2d777a6f5
3ff4a3a2cf8a5eaf bfc3706c5435ddf9 bfb4af10288fc7f1 3fe7caeee8acf588
3fe539e36a0e32a4 3fe18851e53156b4 3feb62b878a3c629 3fcffd30de279432
3fd382d51bcd9613 3ffc1e81768cab11 3ffdb46bb630014a 3ff3cf539119dd45
3ff10a0defd7dae5 3ff2a03bf8ed9fb4 bfd4e198c2a5c174 4003a77ec55a713d
3fe437439646814e 3fe2c52936406dbc 3ffc39d72d8c4039 3fdc913c09b3114d
bfe5009806d6f414 40029813367087bf 3fd708ff4552d90d 400030e76c394584
3ffbc31d012aad60 3fec39f04b3eea0f 3ff370efd387e979 3fef604af2f479ab
3fe35b03470c7624 3fe0856004e75ecc 3fe4cddcc99720b6 3ff65d2bb5763125
3ffd71df5e25e6a4 bff268134a052f45 40041237a01f1553 3fd015605cbfdcc8
3fe27b87af5c2911 bfdcbf60be6399e2 4003c8a9caaa21aa 3fe8a03022f784d9
3fef0d175672b60c 3fecce89050dc26a 3ff98f0d8ee96995 3fe2fe7a83eeb932
3fe9540236906e7e 3ffb4257d97db891 bfe0f1b0d7a7a36f 4000c34697a14cd9
3fcffd4c1829fd5c 3ff5a049697d8e21 3fe66fe2bc9f70d7 3ff47cc777683db3
3fb68922480f0e39 3fe709ae60397acf 3fffb755d2a24245 bfccd79785801462
3fe48d0f1219493e 3ff71db756b17544 3ffe7bfd6f714c9d 3fdf305b960acdfb
3fe9511386519c11 bfd2efdc4e0a6910 3fdd73213c9c1c89 bfe5dc33f9abca83
3fde4da407469cfc 3ff8a3ec36427aa0 bfced2b2412cf608 bfd5cae5bf7b6458
bfd7f06bbfc53a90 bfe743007261d1bb 40010a06f718bd55 bfe2a85edafe2513
3ff4db3325d07e5c 4002f2d847529fd1 bfd55c5830d68dd8 3fe6c14d23d7f4ce
3fb2b8870a7caeb6 4005aa042812d82b bf9dbc32888d4643 3fdebe8703ab4d27
3fcadec9ae6019b5 3ffe909952174607 3ff7f71347840969 40047d7cf82725d1
bfe27ebd7e3950b1 3fe14ba3f5515f3f bfdbef974eaf38e0 3ff9f929b06f1e21
4005520bb84d1d67 3fc1ced53c44e647 bfef29fa5517ab13 3ffa9edda5ba558a
3fe608e87fac3354 3f879bec4f2c9ebe 3fd5615e373235c7 3ff5abdc6589c9a9
3feec874fc94d2d9 bfe8023914cfba5f bfdc4d5f1a2882ad bf9bff48772a160b
3ffd4d6ea1232825 3fd5340bd5d4f130 bfda440588a94b18 3ff217c7bfc993f2
40054b36e742ff68 3fd65eecaf8d9180 3ff6a71bdbf4de52 3fe709ae60397acf
4004f44c8fb5b7ad 3ff42affb0518376 3ff9247065c038ff 400121b23a8b188d
3fdd00c91b8893d9 3ffae5a210ba26d3 3ff4cd0c818e1822 3ff12c1d95bef51e
3fc2ec30bf6ce52a bfeec6fac3d017cf 3fef351a9e87f2f7 3fcf86d325f8154a
3ffadc7a790c5918 40017770b6269053 4001d86029a7295f 3fc1ea59455f72a6
bfa8ed12a11fda1c 3ffb0b1c5ffff287 3ff4332031106e6a bff331b37761fd96
3ff348c18aa2de8e bfc28bbd61d10673 bf1f1b12e51c1375 3fc92b3fdc0538ef
3feae201d1e7e8e7 3ff51754b3ce8867 3fe5cffa6dfee8e6 3fddee860d49309d
3ff1291c9a5f3571 3fe95c9e6720a953 bf983d3d81fec8f6 400282a58dcc43c7
3fd5fda015f45876 3fd77ec8aa2abf63 bfec88c1a7b40d46 3fffc4e6ae9661bd
bfe01dfa0b562685 3fe037eccd6b3516 3fecc48fcc622d73 4002186b406d7164
3ffef7228811f44d 3fef260a62c04b54 bfd11260059c216f bfe04029796a998a
3fef5df13ce8da80 3febd460b0efec48 3ff46b912f6c2c18 3ff56f3654046477
bfe773d1656eb9af 40036566a419aada 3fef85a443740da3 3ff48074326ac8b3
3fb5bb50f188411f bf983d3d81fec8f6 3ffe7a9dbf5142a8 bff18b352ffcc57f
3fe8fc3769774f72 bff081e0b9c42932 bfdefeadebd76d22 40056682b333cd18
3fde7fc38867b5ee bfe5efa94b4560f2 3ffc160ad77aac96 bfc28bbd61d10673
3ff070f22dbf423b 3ffeffbff8581608 40022f8d0436ee12 3ff640e49d059566
3ffb4cdae5326d6f 3ff9f51602639493 3ff5870eb3b5d9d3 3ffde757804dce27
4001aa2160d9afb7 3fedbc3a8139af9c 3feab7c9a81c9c9f 3ff4ee5d48e24150
3fe59ae58664966e 3ffcc269c1c59991 bfd977b06fbcd796 3fef41153d4ed19a
3fc344e53fd8a741 3ffe987fdf010b5c bfaaa88167b0b7ed 400485f36b43cd76
bfe180ac5644b252 3ff1d3f0f9cd3c1b 40068c253c60f042 3fe7cb5fa96f69bd
bfce0037dec622ff 400131c7a6a83b1c 40006b84556cfeb3 3ff527a0325db733
3fe01b3cfe558637 bfd7f55b3186755e 3ff5499d34a8ab68 3ff3fbbe7e921bd1
bfe8d7ad8a1092f0 3ff4cdf00f894a7e bfc326144ef7bc4d bfd7d469bc1a4245
3fd01c4adfe69267 40011543e34ee551 3feeb5fb06a26abb 3ff945bebd11cc7d
4001bc1c084104d8 3ffb6c2a0367328a 3ffc5a32a9377190 3ff76f61237edc06
3ff9d6f7f419683e 3ffd25e8a4331610 4002611733ace0e5 bfed075e8731b8a1
3fe0f83255f72c1d 3ffb14a539a732d1 3ff5a26add264808 3ff9a47720f97ed3
3fb90f0b63ab6aa9 3fdfa73b511e6737 3fb53d0be165ad47 bfe9704cb6cf055e
3fe27e9e84899fe9 4003bd0fc8c5c220 3fe894d778226a90 3fec78e3a2954933
40000e26e6551eb3 40003bf92aea7a0f 3fd4856c64d84416 3fcd0d06b943d9a8
3fecaac64bc20281 3ff6cc38b7dd9ef9 3fdf4569887e465f 3fe2435f8c014c8a
3fd475c3e5017be6 400049ca2e43810a 3fd3826b51514267 bfe426758b5ba26e
3fe76b5423f87c3b 3ff910ba32c1bd20 3fb300805bbab12e 3ffc420ca256d990
3fe58e0320c5fe58 3ffc420ca256d990 3ffe7a9dbf5142a8 bfd008115a080e40
3ff434c95bc3b691 3fd87435ded69d7a 40015b8bd940b12e 3fdd7027b1ec0004
3fed6af56ef528a0 3fe7c3be00c8fdba bfe9291efa752964 3feca701841269f3
3fd748dc75818b2f 3ff083ce44138021 3ff5c8e48ec66f73 3ff088f517bf99c8
3fecdbf1476db1e0 3fde67791322796f 3feae201d1e7e8e7 3fe7e74511a5cccd
3fec856b0bc3385a bfd4a00c7644fae3 3ffa1156efd9f960 3fe9511386519c11
3fda052cc88339b6 3ff4fc8292069e25 bfd7970ce605cc31 3ff7a0e14c1ad779
40074cf6c739df50 3fed85da47ac2b67 bfe153a436108e25 40004a82bda9e19a
3ff0361a4e8136df 3ff459e95aa7d0fa 40003a9217da1a9d bfa61f56f3ea195b
3ffd7df9cb4cd947 bf92a744d5ac7cc2 bfdbb94801fe7bda 40062f059b077f68
3fd2ae0967c4c3a1 40008307ac4e1d1e bfb482f3a298c541 3ffb2ced864ae2de
3fe27e9e84899fe9 3ffe88a301d971ef 3ff7fa3ae2bc6feb bfd8e64161dc15ed
3fc53214b08985b5 bfef1ceacfc61652 3ff25f5b196ee81e 3fd117001c13a715
bfcb5e433a489814 bff46d77f3c241bc 3fcad4a364e76f7f 3ff18c44dac91319
bfdab062aabe4964 400796c715633d39 bfd29083e20b5e55 bfd927f5e4d887d7
3ffa988876c82370 3fda61292788b388 bfc31a11152744ed 3ff643557ad8c813
3ff7524b8f552c22 bfe95f14bab05dfb 3ff959f0ffc196ef 3fe894d778226a90
40040bb5c31209e6 3ff7be04d19c8298 3ff7dd9360beb4d1 3ff4a9bf3ac66bb9
3fec29c7c041d191 3ff0c24b9280d3ef 3ff8eeb5d9a57bfd bff3ed15c61f0bf8
3ff0e46147d595b2 3ff0e666e8f652a0 bff3d280a676831f 3ffc2d7614dc9204
bfcd41431f7c72f6 3ffe5c4dc889455d bfcdfa09e1fdf035 bfc911cef2f489f7
bfec319e8ac01e7f 3fefc8cb95c8e82b bfe5d8f05c0770d7 3ff8155606a86d13
3ff81ed07eb17c1f 3ffeac792cb57aaa 3fe7b9938403bf12 3fe6a6b746700838
3fe35ff84fe43344 3ff8525df98a4941 3ffe051e051a7021 3ff0d41073a50f4c
3fe6f5417b350ea9 3ff6fbc48bcafcb8 3fe1f59d2f447fa4 bfe36d63f292d08a
bfd2efdc4e0a6910 3ff8b29ec98b7887 bfc55256da0bc5da 3fe354ab27a71ace
3fe44c1a967936de 3ffa859e66b49f4e bfa55cd143b30c88 3fea4c41eaec7f3c
3fe88ba4709d42b1 3ff04d8a433336be 3ff3e084d16d5775 bff5b30482d548f9
3fd13c257ca39e00 3fee284247e73263 bfe1c9065899f357 bfb7aeb8e19d53f7
4003a77ec55a713d 3fe009d772b7c527 3ff3d3df61a6d4b0 4005a24db85cf181
3ffc9a4b3542909f 3fe875ff9e0cae11 3ffb08c6ff63963e 3ff77c2401dc42cb
bfd81f6718f8dad6 bfe24f45fb3d22aa 3ffb15a087154a12 3ffe07118b757fd2
3fc678467bbebfcc 3ff07deaea2453c2 bff2d6a090bdd101 3ff3cf539119dd45
3ff83e0f23834686 3ff7c4f11139fd63 3fd3566785a45be5 3ff02b497c0a8b0c
3ff33cffcdc6c2ac 3fd47920d69f0aa2 3fdeb03629e088e9 3ff6c7e7aee52e4d
3fcc6ee90d954ab0 3ff1d4dcadd5ef24 3ff75365ce952820 3fb481538d306b14
3fe768cb3d03ce71 3fd873859c906958 3fdd35b4f1f77e23 3fe0249230de0f5b
3fbe3e305141d0d1 bfca400efd964ebf 3fdadd9b0130242c 3ff529a751a221f6
3ff2671ed0bbfddd 3fe2f2e020bd6f73 3fd9c2cfc565a226 3fff5be688836a6c
bfe4bd433e50eaa1 3fceddc27fa01d56 bff5b30482d548f9 3fd60cbebb87f0fa
3fdd2e34c3863dd9 bfeb776887dc82b7 bfd746c90ae4b370 3ff3c3a5647c85c1
3fed85da47ac2b67 3fd39776c75c5c34 bfe25191c396edef bfd17f36e40a7d1e
3ff2d44db25a41bd 3fe178a3f1038501 bfed175168a67279 3feb617a14f661b6
4001ed5343dad88a 3fe25611b66b47c4 3ff73680ca41d466 4002e64ab064c3bd
3ff8a3ec36427aa0 3ff1ac62ce6d536c 400043720ff277c3 3fd4361c0d5eaeb6
3ff0089233b8bfdf bfc6245793c40ec3 bfba98f5966fa95e 3ff044a1cd6f50e5
3fea5d0da3099c60 bfe07ea453919f84 3ff8b286ad5051e4 3ff9935546874ff4
4001a35ac1fba4ed 3feb3a1388ec2da2 3fd7534b41eea428 bfbb6a1f8012334d
3ff2591040a15907 400160a09a4acd02 40009e5d42af42fb 3fff8cc60bf339bd
3fe66f472085e0ce 3fd6f411aa1dc2e6 3fcc6ee90d954ab0 3fd117001c13a715
3ffd5b5677e34e3c 3ff97fa8721c269c bfd2efdc4e0a6910 bff05e101867863a
3fece6207ca2a6cf bfc07fbf1f8451d7 bff07b2548428ba3 bfbccae45bf1f721
3fe59ae58664966e bfec644d8990a756 3ff8709f43152875 bfeec5d97b550439
3fef604af2f479ab 3ff00273b769a795 3fe10856d245e00e 3ff30fc129b6f98d
bfbf5b1c71cc2386 4003ae0c3e6daff1 bfe15c68e4c6d905 bfce6a289e42cc5d
bfe9ef0b98f6e362 3ffa1189c9348bc2 3ff1efa0c981e1fb bfcad97da2d10b16
3ff1bae009e9b26b 3ff32fa48c4bdec0 3fea7db997acb7ed 3ff1b7ca7b202e5d
3fdd19a3abb74285 3ff3cdbe187fcea5 3fe652d5a40ca09d 3ff3326bb971da7e
3feb969330d6800c 3fbf0c635a78b54d 4005ba9acfff1110 bfe12ddf089a4850
3fbadf5e735a4eb9 3fd145e81080ce27 3fd49ebf72a3e1d0 3fa1a58d4b196c55
3ffa301214939700 3ffb2f74dd767f0f 3fd600732d4eb786 bfb048ea817a87a3
3fdf065fde691f00 3fd7fef05d787ef3 3fd33d022c0c2da7 bff2d6a090bdd101
3fce0637c5e19214 400033c42baf96bb 3ff9455e889fc100 3feea27236343e9e
40056682b333cd18 3ff6c46ffdf017d4 bfe0f1b0d7a7a36f 3ffb15a087154a12
bfedadf23c68f7d9 bfc8c62ccb5d1b6a bfb7d2f94074c82a 3fee4a6cba2eb5f9
3ff069732c42bb29 3fb09c528d8fac46 3fdeffc3995d8829 bfe30f7f08eff47b
3ff84ac9bf84d721 3ffd7a587cd79a80 bfe87cf2fabbdaf2 3ff2875f71d33d03
bfd07b5a6ddafff7 3fee57eaaa3d57c5 3fe1ab0294118ea1 3ff4474aaf52c2c8
3ffee0f3796e604d 3fff68999acae97f 40020a720c17a26d bfd4ffb0f9abd65a
3feefda4aeb1d500 3ffc39d72d8c4039 3ff4bd5a1fdf5253 3fc12f992afc2f55
3ff6427a25f652a0 40060e8702ffe27b 3fa5bd421846c0a4 3fd6f21c037ef5fd
bfc482dbf7cb96cc 3ffed4f2d777a6f5 3fe684e5b77d17d5 3ffae01f70735204
3ff3165d2a9803dc 3ffd59d5cccbccd6 3ff4826ac648d992 3fce26073e426d89
bfed8c53cdfc7eb4 3fded87ab2cccb0c 3feb3673c7d9faa8 3fdf065fde691f00
400405fb2a4c776b 3fe6c7e30565ad0d 3fed6ec4bfdf7b8a 3ff235d2c7265733
3ff79d86da77ce4a 3ff2788af17df1f4 3ff65214b8926468 bfd3b11fe7bf73ab
3ffcded89f39b223 3fe894ada01c9f70 3ff413998d47afe1 3ff688033501edbc
4005025aeb8ceb48 400473673f0ae415 3fc82f35e18064cb 3fd51e0debec51d6
3ff29b91e75ee398 bff35f11ed2fc5a5 3fe94ed6c8d1bbb8 3ffed75b6fc70de3
3ff6f10c20a2bc75 bfc8706cf0dec0d2 3ff2d44db25a41bd 3fddb206192ec61e
3ff97fa8721c269c 3ffac4793e06d0a0 4000629ea12316d7 3ff20bf29cdcb8fa
400244e3a8562ee5 3ff0bc4f21b1983a 3ffae5eece76af33 3ff8f17885f0ba27
bfbfda6af7b85f0b 3ffbea605491796f 3ff879c9f5a9af32 3fe61aae1239208b
3fe55657bfee459f 3ff2d862a635ab61 3ff47a35d0f98bed bfe098d59b5f75bd
bfeec5d97b550439 3fef105497de2fda 40004ede419ad80e 3ff3e7751101af55
bfe6ee8cb35cccc1 3fdcecb8f9d42763 3ff0221c67de9341 3fe3c2d74c4f59d1
3ff0523824cf6bdd 3fdd7027b1ec0004 bfe05c132c02a725 3fe1be81da9c1dae
3fe033d45551c04f 3ff52a94c14438b8 3fe5fbf98facb2f0 bfb21423bc8eef57
3fde8840e536de2f 3fa6bf2217d108a8 3fec45a217ee16fa 3fdc61c84332ad0d
bfd8d6a486fc6da9 bfdb8d44df89d716 3fdeeb31e970c4fb bfed0c1d474430ad
bfe95f14bab05dfb bfd5e6d78c798158 3fff4ffc43ea2a1f 3fe3885303e8dc05
4003cae14a7568b2 3fd8a95ddc3e542e 3ff306979a48bacc 3fff9b31d2133d25
3ff028540eb38970 3ffd109bd5a6ebfb 3ff7ccef6cdd3826 bfd8a392c13766b8
40002b7aa913a5bb 3fbac5db9afc4bd9 3fb1cadc1e9d8f01 3ff9147a5295c226
4000afd2414f2697 bfc886183aec17b5 3fd6c51ffd96b834 bfba98f5966fa95e
3fc6fe942408f6de 3fd0ce716ecf65b4 4003a4f8cb3491c1 3ffed190fc5e6dfa
3fec29c7c041d191 3fb09f2577339df4 3fe1393d4259e218 bfed0c1d474430ad
3ffb89f402c84f41 bfdc4eefb985dcdd bff1a1585722d287 3ff40e5d06066e0f
bfc01a26209e29d8 3ff1f523795857c9 3fef351a9e87f2f7 bfe11c4506619fa8
3fe640a533214d25 3fe51e493bdffa6c 3fc22dd6883eca2d 3fb25b379663510e
3feef93cbfcda682 3ff6eada19acc259 3ffefe31b98593ea bfe57f6663c5059c
3fe8b126739c7e5b 4004bfec02cc0235 3ff91112df2ecc14 3ffcaa51fe2a6e48
3fdbfafa4180cb62 3fd7bb5b0243c0aa 3fea5d0da3099c60 40004c0b37d037bf
3fbba19754786b63 3fe1b6efbb56cd0d 3fe2a8acf697032d bfe852862ed71768
bfe9704cb6cf055e 3ff53a094535f13c 3fe944fecc3730e4 40028978a90b888d
3ff62ed4c639ef3c bfa59b2b13e262eb 3ff56b47e9e1a4ed 3fd033ba709e8321
3fe3b0164c112a99 bfbed681e9ecb055 3fe58e0320c5fe58 bfdfdf51a9e92141
4001d86029a7295f 3ffa4f76285b70f6 3fc8043a35e32c63 bfd932d9a3c9c9d4
4002a347da1fda80 3ffc95734fbac4ca 3ffb4e311d2ee095 3ff6f9756cfdf763
3fd4b44683de170d 3fc2cc97ccac618c 3ffa513e0a044206 3fd39ee9e6a82ddd
bfe5a21d0bb269f4 bfb127531991125c bfefe06f89bee586 3fe80389eed03c5d
3fe1bfde5344b13b bfb7d2f94074c82a 3facbf0f9d87daa8 3fdd4baa1d33313a
bfe434aa840434ac 3ff5addeeac3648d 4003477f1bc0a9a3 400008253912e0cf
3fd3afdb6f4175fe 3fb852cdd6b4b5b4 40018eb0dcb6164a bfe07c0dbb34dbfe
4000864f977e9fce 3fd39776c75c5c34 3fe94ed6c8d1bbb8 4001c62095b01cc5
bfe65f4c2c48c295 3ff5c8e48ec66f73 400147454c67ded4 3ff34d630ee15a93
400581822600beee 4002a8551cffca04 3fefc2139143bdb1 3ff84bb335d313cc
bfe69fa7b68726f5 3ff7b385078de89c 3ff898270880ee21 3febbc5676d2ffb9
3ff2c8a062497fcc 3fc051721dd266be bfbea73058d890f9 4002205bf78ba664
3fe817ef1d57769a 3fc5ad24a1ebc751 bfe08ce869a7b4ee bf9d381c27cc725f
3ff3571b0bfc9400 3fef02f48b39e0b2 3fbddfbf82991e39 3fd5f04ac31de9ac
400592e86bd39133 bfeca8d386c16db0 bfdfb68bdb3ebdfc 3fe70ed346414175
400041934c1ffd9b 4002c7b6dd8aaf33 3ffeab9e9d6e29ab bf9f85ccd7024ce9
3ff65c62ac58e68f 3ff860fc73e8247e 3ff5ed61f78ab998 3f9df3049c42985b
3ff1eb60a094b442 3ff38cecf9e6bd1c 4005010831278566 3fbda88ce2a8f313
bfafdc0cbb13ea5f bfb33d273fb993c5 3fecf912496fa6e9 3fea259fb9367e80
3ff67501e6c72b14 3fe959b59bf7be7b 3ff5d8f9db8db605 bfe87f21ca1d436c
3fda7110e896aad0 400278bbd910cad5 3ff057e8ae7ccfe3 3fe8193aa65ff9d3
3ff30b921c8c7561 3fe92baa1e3f72d3 bfd3ec2fcf98abce 40002828af253072
40031f934238aeac 3ff35aed9de4a79b bfa8ed12a11fda1c 3fe4f67496d36ce3
3fe1393d4259e218 40037292935b1da7 bfe42d4590e7e270 3fee532f17bd294a
3ffa2771dbb849f0 3ff824e67700247a 40037292935b1da7 3ff656b6f51650fb
bf98e6c431efcbde 3fddb206192ec61e bfbd38c858921280 3fecff0f32709edd
3ff61b23a66587b3 3fe33ee9f54b7091 3ff65214b8926468 3ff476782175e644
bfe434aa840434ac bfeb776e77afb979 3ffa31975023bb0e bfe29b07db46c273
3fe0a89d363a171e 3ff6ee95bd66ef10 3ff3a58b94752ccc 3fe29130def40ca0
3ff6735498a06c75 bfd536ac8ce7ab34 3fe32a9e1846a42f 3ff88a9c013a46e7
bfe84f3cfbd94528 3fd380e05b2895c8 bfc9891dbee66b29 3ff438f11f02280a
3fd31cee544b5c15 40016150f988105b 3ff24cea7360a4e8 bfeb3296f83d5317
3ff6b5aee4ae29d6 bfe0770a28ff11db bfd71f05fb8fb596 3fe90602c78d2f4c
3ffbbffb3258fa92 4003c1e574e8abfd 3fff2645ee17705a 3ff098638188c2ca
3fed9ee0cafd2e47 3fe4180f7520437d 3ff06939041cd06b 3fd8b174470ef24b
3ff47cc777683db3 3ffeb488549c3f97 3ff51875a78f68b0 bfebc6b291a06c98
bfe9704cb6cf055e bf9cf8c4c1c3083b 3fe9366d5f7aaf10 3ff89c1a81e11c8e
3f9ce6e9eef3cc51 400296e93955e2a1 3ff90a28b7e55415 bfd750310dc6b6f6
bfe2398c8b9e4a04 3ffd8fa2760bb99c 3fdd3cf19a5b0625 3fff000e7612ae1b
bf983d3d81fec8f6 400824f2dec21ab3 3fdaf382b2435c5e bfae7362cf2a2177
4000eada921278d1 bfe4d6e859545e09 3fd1789a9f181ca6 3fdf61c0391d5d5e
3fe652d5a40ca09d bfc0f0b452c90065 3fe32a9e1846a42f bfd833e41bbf1a49
3fbac5db9afc4bd9 3fe6b29786da6abb 400238d53fa2e5de 3ff1fe453d4f5d02
3ffd7a587cd79a80 bfd76ebd7e01a42a 3ffef98e0abf7566 3fd5d711bac29f0d
3ff70232efd3444a 4003058c4cff0973 bfd8d2dcb8575716 bfa7391ddd16f094
3fe9da1eef12655e 3fb3e3bc6e4d6253 bfeaaed6acfa9019 40031f934238aeac
3ffce6b7e084deaf 3ffa05e2f01f988b 3ff1feeb428da81e 3ffcf9bfbbd595f6
3ff9bdb388c9aede 3fdcd32ef74abde3 bff56714f5096c26 3fe600bb1f756d8a
3ff2a03bf8ed9fb4 bf99f4cd747f366c 3ff49c6ade38b75e 3feb200bd02d823d
400088a95389f4cb bfeca8d386c16db0 3ffd188ecc90e9e4 3ff61b11d26f2c83
3ffad01910ef1fe5 3ff61015bb0f80c3 3ffd51f1ff3e5c13 bff268134a052f45
bff0724356dd6aeb 3fecd8cea2b8867a 400631c20751a5e7 bfd255d3d806be7e
3ff60f67315ef71d 3fd6f21c037ef5fd 3fe724d5a7898aec 3fd047e9c0b508c0
3fed9176c0d64d48 3fe86f4ce8507f95 bfda4ae2047f7c53 bff43b07ed9a99e3
4002500282556f12 3fe409ae7769b765 bfde9c37c7edf2b7 3ff6a0ab7e134a43
bfe0f1b0d7a7a36f bfec6892e10b63a5 3fe63e0c6e4a957a bfd2e9e4b1b7e6fb
3fc2438806c0cacb 3fb1bd25b94844b5 3fdd2e34c3863dd9 bfd6185ce4babde2
bfba81a15521d3f9 3ff5cf393f30aa87 3fe2c41ad40244e8 3ffd7bec6139d063
400030d56e6d49a5 3feb43f4e28e2f0b bff43b07ed9a99e3 bfe15c68e4c6d905
400475a312f1c2e5 4000a6e2cfc54298 3fc16eeed1fe7ff6 bfd6b3f9694137d4
3ffec0fdb73856f3 3fe887a65b2532e0 bff18b352ffcc57f 3ff11f426c795322
bfc742b1d881a3cf 3fde6dc9acbf32d7 3ff052be770198ce bfd55c5830d68dd8
bfe36e490d3e868a bfe9704cb6cf055e 3ff454e16d2489b0 bfeec5d97b550439
3ff598c938652919 bfe7a946a89c1295 40047c89a931782b 40024851177102bf
4002b51a60156abb 3ff6eada19acc259 4002f949f278fc02 3ffd25e8a4331610
3fea05e86a3ff59d 3ff6163d1d9302c2 bfc0ee93c4bb4208 bfd9a6b9f8c0836f
3fdbb6fea0f44931 3fdccae51ec51eca 3fe1262d665ce166 bfe7343fcb489697
bfe55d6457cf6eeb 3ff5d8f9db8db605 bfdab062aabe4964 3fd32a3a04f7ae55
4004f44c8fb5b7ad 4005fdf9730212c8 bfe4cc5704d152fd 3ff95b5430c0d224
4002bc554c929a1c 40058108ef73afea 3ff454e16d2489b0 3fe17000aa0fad0c
3fe25dd2c4505541 bfed0c1d474430ad 3ffba50a53a8eae0 400796c715633d39
bfc4fb2979b23845 3ff25a32383c8ca8 3ff4a73e3d98c8c0 bfbfe212e0fb4dea
3fef4b1817be7acd bfc4d06443fc8209 400224228dc67eff 3fe3ed8566e99e1d
bfec89e2f02f20dc 3fdbc4f832871ab9 3ff4bd5a1fdf5253 3fd94479c5605085
4004139bf4132092 3ff1eb60a094b442 bfdd4830fa00d771 3fee47ab9536c550
3ff4459318194816 3ff4a73e3d98c8c0 3fdef2ec487226e1 3ffb59d79af69769
3ff22ac2d6d8c91a 3ff3c4af4d470f0c bfdb237983adb57f 3ff1de97b8ed11fc
3fcfbd9846281be3 40022f8d0436ee12 bff247bd4cc1da06 3fe944fecc3730e4
3fefdc7a22a7be73 bfd55c5830d68dd8 3fc0a116bb3f83c4 3fe7fa00d94768d9
bfd09300f437afe9 3fe49ef464ab7f40 3fb689a3d3d96cbd 4004e320847f0e09
40009bac5062cce5 bfcb0f1cdfb35e7a 3feb5ecf310ec882 400306aceea0956e
3fbe32370a51542d bfe852862ed71768 bff268134a052f45 4003c1ccda4cae52
3ff95b5430c0d224 3fa0373c812cf1b4 bfbd07ebb44c527f 3ff3fbbe7e921bd1
bff2d3af471e47fc 40084722737af626 3feef03e612c90b3 3fc330cbcd753748
4000f106631d2371 bfd72ec99e63a437 3ffb3e2db587eacb bf7040e2f69049c8
3feaf25b48944b14 bfcaeaaa775cab50 bfd927f5e4d887d7 3ffb4abb296dca06
bfeb6636ce162056 3ff2d6cd5a29f3fd 3ffafa7a13dd2bd1 3ff61fd6a30401a8
3ff79317ce34f1ca bfe5b4acea89978b 3fb300805bbab12e 3ffcacde8d1caa38
3fb1cb16e6a429d4 3fee6df4879bb5d4 4003c1e574e8abfd bfea2b7b0d6a6cb2
bf93b0d3c4619767 3fe37dd9bbd062a5 bfeebbf4e7bcd648 bfd8a392c13766b8
bfe87f21ca1d436c 3fe88ba4709d42b1 40015f630af5ed26 3ffae5eece76af33
3fd076377467bb0a 3f810429dd892828 bfda539c99818771 bfd94b4033f56b30
400470ea6100f89e 3ffb79b07d05e570 3fe5bcf61048c23a 3fd4ce265df70624
3fecaac64bc20281 3ff05f7a0bd99ac2 bfdcaf2f2858ac15 3fcf1f60a9d12d02
bfcb18d291952359 3fe013f7e7fc8302 4003c1ccda4cae52 3fd3c40a92ab7e6d
4002815c1dfd6052 3ff3bf84b03fe505 3fe13c1df1df044e 3ff73c0eeac8a14a
3ffd25e8a4331610 3ff9ab4a00174a9d bfde69b83fca562e 3ff57594cbe03f94
bfe2b5c43940b65a 3fa97b21f06aed0e 3fe5eb8f8790f458 4000a33441a2a751
bff5b30482d548f9 bfd2efdc4e0a6910 bf6f1a2432f80b5f 3ffb2ced864ae2de
3ff241757719e6f6 3fec44763d46800d 3ffb5a2f7ff13c8f 3fe1478b686bdea3
3feae36a6abbb9c3 bff07b2548428ba3 3ffb01d40104ed2b 3feb5ecf310ec882
3fecfe4b1f036a78 4001ae6cb8ea372d 3ffcb08d6946ef24 3fbc28ef4b483d09
3ffb475d924aa002 40015f630af5ed26 3ff03348ccf3e804 3feaa89fac991cd3
4003262ba9f9824a 3fd86f253647984a 3fd748dc75818b2f 3fe49ef464ab7f40
3ff479710abc329c 3fffa17b751ac9fc bf7d6e599807043c 3feb28d1a6ffffe1
3ffd49ad89ad11fa 3feae2a73dcc397a 3fe418acdb5af9fe 3fee9888acceeae9
3ff0e46147d595b2 3ff696145e8ef4c8 3fb481538d306b14 3ff57c959eb5eebe
3fe709ae60397acf 3fd258379b794401 3ffb3478e61b8dc8 3fe1640d4ce4f946
3fe86f4ce8507f95 40012867df2bd2db 4000f2f701e09c1e bfa056d547da1553
bfd1a883e877e0cf 3ff875c136d0faf3 3ff80e9a816f7620 bfec3370e863edec
bfe18cb2c125bb83 3fea5d34e129efaa bff149876034b3cb 3ff7d7b804d01482
40022657569177a2 bfec3370e863edec 3fed4613da4324c7 bfc15e1df717750d
3fd560bc1627b621 3fe74c09a6303b08 3ff9884a5046bbc4 bfe8d7ad8a1092f0
bfc3cfa4f17edede 3ff29ca6fe32a487 3fe5c35eaaeb125c 3ffe723f0654a081
3fe68d15608fe53a 3fe197543cd39e50 3feb200bd02d823d 3fed180cf2ba1a44
3ffd758b9db16cd6 3fed60109fd01271 3ff1c462d312f133 3fd8ae1f8d17b40a
3ff6496395f2709a bfd5cae5bf7b6458 3fe5b4b580a6d832 bfe68498783828b8
bfa4388365bb147d 3ffb51f13d02e59b 3fffa5ae6657888e 400307b49671f6c3
3fe627ec2c06d47e 3ff18ce382d60b93 3ff7550dc0017716 3fd189747124f7f7
4001a99d63901957 3fdea5fa6a567818 3fec594e3df77e93 3fedef8244044e11
3fe58617497b22d4 bfcbb4e5be797bae 3ff73bf560b3e3d6 bf97c0e17a8df1ac
3ff7389f8eb7eb5e 3fe2283e488c02c5 3ff56ec1569a166a 3ff4015946c749c0
40074cf6c739df50 3fd6a5df0b0f7467 bfc83dcf6374c27e bfeb15aa606bfff6
3fe2abd8c12288fb 3fea4edb19e9c454 3ff3fbbe7e921bd1 40061f3e352b3a5f
4005c5d6fc1796df 3ff7090d8bf82b40 3fd5d58a07d4d130 3fe467d85bf52157
3fb1c6142c188695 3fd63869944f5c7d 3fd11c07ab904d45 3fddf527ade9a681
3fd800badc3966b3 400275e7c3c43cf3 bfdb09511b242563 4000495af1983de8
3fe0257eb2deb47b bfc70a88a09e6eb7 bfe98ecb68a7dd3d 3ff0ab022d74fb71
3fd4493b21dcc1ee 40030df2421177d4 3fe47ab8aba02eb6 bfd7970ce605cc31
3ff081f1ab2278c9 3fea4376aad86630 bfd555b236d68644 3fe859f2d0748167
bfd8308f6ff0ed5f bfd1e81fca328632 4001fef5a022f4bc 3ff98c8d8f98c42a
3ff816791c0485c0 bff1000b91167799 40065fb9988c10fd 4006d55f615c2b2a
3fec39f04b3eea0f 4001ae6cb8ea372d 3fe71003edb9684e 4003687921475b87
400092fe5bc1e267 3fe31681ce19a8b6 4002f949f278fc02 3fea7db997acb7ed
3ff0fbf86493af86 3fe73b272768863b 4000e53f878737bb 3ff4b2ff743c2271
3fda8bd7e61b5b8d 3ff1c87bc1983edf 3fdf4ef4286e913e 3ff0f6ddd15bec83
4002c7b6dd8aaf33 3ff3165d2a9803dc 3ff5014d8e19323d bfd7517db34e6ff4
3f9ecd34fdc28369 bfa3075e6fc711f3 bff3aaf3507590dc 400284ddbb69e504
400041ccb8d3d737 3fdc19955dd2f7a4 3ffa96702bfcd0bb 3fe4734323d655e7
3ff429565c52d22a 3fdb7c3370450a78 3fcea6ae95cd5ade 3ff4c894f3e97d15
40033a66593b7a12 3fcd4368bd1443ad 3ffbc83962f62df6 3fee39eacd1f15f1
bfea4ad3c92f4872 4006046b1e286ef9 3ff9218a6b048ddb 4000e5ce3e7dceb5
3ff0b52d7d346b74 3ffc420ca256d990 bfefd9b0d4f03e69 3fe5167f686cef3e
3ffdb46bb630014a 3fea653a11823f09 bff132cb106a30cb bfdab5c3c2d14910
3f923fa5b4960095 3fbb9b1b82398da9 3ffa3a41a636a8fc 4000e7b61821eb1d
400003ca4edf349b 3ff55e6edd7d6a1e 3feacdb49fd99e4a 3ff1c654f90d25fd
3ffec7c522ff9b8a 3fe536f4b9cf6037 3fe47ab8aba02eb6 3fd252f0a65feadc
bf683b09820f3d03 3ff0b094c8936b0d bfd4ffb0f9abd65a bfe4c1406eef991f
bff522efe5b03ff2 3ffe515edd4355b3 3fb74152f298b843 3ff9131c38058fb6
bff11c74adb3b65b 3ff753e0731df965 3ffb78f20aa45020 3fcd68e04a58faf2
bfd4e824973d4bef 3ff2d64b56ff6ab7 4001aa2160d9afb7 4001401011116d78
40025ebbe8605f22 3fcffe0ffa018d8a 3ff2df42060cbcc7 3fd883922873510d
bff43b07ed9a99e3 bff1a1585722d287 3fe5590d5b242046 3fb23a4f06524abb
3fe9c28b304077e8 3feb59965b7b04f3 3ffddd1ec9bf6f6a 3fed6e93c326eca1
3fd883922873510d bfcc0e06ce17e682 3fedb5e2823b395b 3fe3b18a5c7043e3
3fbeb79ed26c3137 3fe32c25f2a8df03 bfe30f7f08eff47b 3feb40e4b18db036
3ffd8cd926cb2d15 3f8b9f770238b16e 3ff0340091b0ae50 bfd707ca5c989204
4005dfe026ed415d 3ffb0d92bf874ab0 3fe4180f7520437d 3fd8f1a296a580d1
3ff54532807ea0b6 bfe5894282afd178 bfab9cc0fa0d16e3 bf7319b0c958cc80
3fef821ae29a0d9a 3fee6ece904dc587 bfdd03ca1e1da6ee 3ff4287550e6dbf9
3ff2cb66351dd1aa 4007a190b00d5ba4 3feee4b36ee8a458 3fe0b5d88069a3c0
3fd7bb5b0243c0aa bfd67b1244448c14 3ff27959345e43f2 3ff5b7e8451164e9
3fa6ade62717bdf1 3ffb475d924aa002 bfe27ebd7e3950b1 3fdc6ac06fb079f4
3fcff76ba01528fd 3fe0afd9a6ba8ca2 bfe5c135465780b6 3ff0db308616a92d
3fe94ed6c8d1bbb8 bfcad97da2d10b16 3ff5dc4901024237 3fd867d09a1d6c01
3fc2276bce9b01a8 3fd5baf2419d91b2 3ff853af63f001f6 3ff969a7adeba16a
bfd679ca9a06d69e 3fef07396fc3793a 400235ecc4b25e1a 3ffa5d2f8f5091e2
bfd06950350f234c 3ff44fd95d66d070 bfebaabbd291a0b0 3ffb01d40104ed2b
3fe539e36a0e32a4 400003ca4edf349b 3f9ecd34fdc28369 3ff1ccefe428b6de
3ff6b9e78c47cd9f 3fd3afdb6f4175fe 3ff977308f1a2d82 bfe0f1b0d7a7a36f
3ff3b98bfc3586cb 3fdb206635cf34ee 4003ff4bd9eac3d3 3fe01b3cfe558637
3fed9a7ed74df17c 4003327e5ce88772 bfde5700373d9c43 3fd86bf8f46e8d2c
4002611733ace0e5 bfeeaa1a55f46caa 3fe851f6426243e5 bf9f85ccd7024ce9
3ffb15a59cc7d82b 3fc0e6f0943b957a 3feb0c325c5577d2 400238f31ddddb0d
bfcdc317e3ade285 40001a287e532094 3ffd68c0594d4b8c 3ff5bc198acb37ac
3ff80ee4c1d41f63 400104a19180d47e bfe4d6e859545e09 3fe5b8706a0f22a4
bfe0f25f7c11b61f bfd3a35afdfd2bda 3feb6a1dd9c20f66 bfe2c95b331dbc44
3fe72cb371c6c431 3fe94dc4adf7edae 3ff1c9039b0b642b 3ff87accf4c31800
3ff81712bba0b8d0 3fd65260006dbb2d 3ff6ee95bd66ef10 4005ba9acfff1110
bfcb6c9ce7022677 bfc326144ef7bc4d 3ff7afceb2f1600f bfc81d647172b575
3fc31597403ca2d9 3ff5ae2e32412c1c 3ffa859e66b49f4e bfc8b02e21d808fe
3ffc77350dffdc50 bfe7c85759bd4311 400103acc65487f8 3ff72e59e249db77
3fcadec9ae6019b5 3ff816e8709353fc 3ff7511e73dd0847 3fe734e0e05d66e4
3fe5ba3e226b7136 400000a16faf886c bff43b07ed9a99e3 bfed0c1d474430ad
3fd1bbf205e44b73 4002ee01201aaa67 3fec1e1fcfc5db08 bff331b37761fd96
bfcdc317e3ade285 40059bdc06767b50 3fe1dba9935bc96c 3fb74152f298b843
bfe0101880a25a5b bfeadb9553e5ec3c 3ffa3788bb97417c 3fe84660ca67a069
3fe476e1780819e8 4002611733ace0e5 3ff7d03b9b056648 3fb74152f298b843
3ff8aa84e71da5d3 3fecd8abd25545cd bfe4661d7808a40a bfb752bee88fcffb
3fe3a64b4ac00e6d 3fd1bb52c52b2420 3fda61292788b388 bfcfbd2272cbf390
bfb8a5262672a27a 3ffed75b6fc70de3 4002815c1dfd6052 3ff2d44db25a41bd
3ffa278df5d5bab2 40002985a0ef5a6e 3fe4ff65b36c2cf2 4006fb3d5cf284c7
3fec74f62f990bbf 3fc3f9e56402c9c4 3ff65c62ac58e68f bff1ff5c87668821
bfa35a3d0ac2936f 3ff38d9684598aad bfa0fb31a242481b bff5b30482d548f9
3fda698927fc3ae4 3ffe2944266de755 bfdacb7b4f90095a bfe581efa70486a6
3fe72aeb802b8357 3ff3528a6eeb6884 3f9a6496c7f9dfb2 3ff717414401c456
bfe9704cb6cf055e 3fe5130f377e7d63 400824f2dec21ab3 4001d8c8a24e5d2e
bfceaa96ef9594c9 3fed9ee0cafd2e47 3ff18d0b399ae85a 3fb97c8b4389af1f
3ff06939041cd06b bfdd28d87a0da81c 3fe5184c78990f02 3fd6ae6da0accda9
3ff599e31cd723c1 3fe24bf58eeae9cd bfd977b06fbcd796 bfc31a11152744ed
bfdcd9343c7c3d5b bfed8c53cdfc7eb4 bfea2b7b0d6a6cb2 3ffe29de15385178
3ff2da77f275a614 3ff05f510060c279 bfd5ffbe2c4f8aa5 400383c66068d8b5
3ff91112df2ecc14 3fe6a6b746700838 bfebd1ee8ce39d9e 3ff6c12f07500162
3fea7dd77abcfc21 40009e75fcea868d 3ff45e41eaf5f26b 4005ba9acfff1110
bfc9a7e32fca810f 3fa46fd70e46c648 40031251143832b9 3fdf3f2f46215850
3ffbe0fce6349fc1 3fd5c2e01b39bf75 3ffced0a58a7530e 3ffdca43d3654148
40041a1b3cc3badd 40053738a14a5201 3fe335a6849cb7c6 4000c51154b6f238
bfd255d3d806be7e 3fe7c079b3d4f435 40036b40aeff4fe0 3ff290b5a9613780
3ff723419b49ef79 4004eddae48f5b7c bfe316fde75dcdca 3ff25400c2531dfd
bff408f780b94676 3ff6e39a81cdec08 3fc35bc9a33c36d4 3ff89d9d55d21c9a
3fd32108130a5342 3fed7db8ab107c63 40017547251688b0 3fbe7b2252b14cf2
3ff9875789ac63bb 400205ec566a62c1 3fe061923c6b8dc9 bfd4b5254f1c7c97
3ff15df4cba0f223 3ff6e9a803481cf6 3ff6c9d73f3b0243 bfeb147f70d58396
3fdbaff064ec2c22 bfbd073aae097674 3ffd765393306996 3fe5772966d9e1d6
4000dbdb45a15525 40053738a14a5201 bfe8b89f921291d6 4000b84f096a3cd5
bfd998b7c109817f 3fe63cef4d557b1f 40002985a0ef5a6e 3fd1ef94363410e7
3fd3e76061f4ffea bfed0767c1981e74 3ff783657f9e73e5 3fdf4ef4286e913e
bfb51aa2d09dd15e 3ffb0ce53730903c bfeb5f5df66a3872 3fe03d88d9707816
3ff73384517680fb 3ff2de88866b3a16 bfbd07ebb44c527f 3fee72efec271738
4005ba9acfff1110 40019ce1bd015a8e 4003763a76075ee0 3feb3a121c0bbea9
4005ba9acfff1110 3ff751400fc0ec79 bfdb68df9d8af975 4003a4f8cb3491c1
3fd883922873510d 4001e57a20624036 3fe78f8e1855efbb 3ffcf7ffe3e7d5b9
4000f0022079de03 40042b12015b4a9a bfc48fd87507fc99 bfe36d63f292d08a
bfafdc0cbb13ea5f bff27c5e2dd6d22e 3fffc378d66bd22e 3fb23a4f06524abb
3fef2a89194c2f75 3fd377e20ac08af7 3feb471af8747f18 3ff10a0defd7dae5
3ffef27ddd0e8e15 bfc3cfa4f17edede 3ff65316583aa25e 3feef60706c1c9a8
3fd1bb52c52b2420 3ff91112df2ecc14 3ff8b198d6c315fc 3ff008a22e04c772
3fd32108130a5342 3ffabfef0d3754af bfc5dd72db52e11c 3fde4da407469cfc
3ff331ed1e46957e bfde9c37c7edf2b7 3ffe11b52fc7d2a9 3ffaf025d3df91e5
3fedbc3a8139af9c 3ff28de038c7df39 3fd77ec8aa2abf63 3fcad4a364e76f7f
3fff2645ee17705a bfdc211117c23f3c 3fdf205fe798019d bfb3e5ad33f1be52
bfbd604773505a18 4001e9519fb02571 3fd602035fdcde19 3febf8d11439c0af
3fef1c39a0c96142 3ff1d828692c2ddb bfe5ce8aa1921ecb 3ff0e6d35405d9a2
3fd377e20ac08af7 3ff119841e00a9a3 3fe71003edb9684e 3ffa2b62169f4d18
3fd3d971964d765f 4005b42924d8b4df 3ff60f67315ef71d bfe755eb1f0706fe
3ffbfa14e7fc521c 3fda61292788b388 3fe6d88d8902f892 3fd42fac24092e8d
3fba27dd98c43ed1 3fbc28ef4b483d09 3ff26067f5570b20 3fe6b28428963add
3ff4897c41df9ecf 3fda38d857c1632d 3fc9b56c56ccca06 3fe09dd58abd0b22
3ffd7fde6e6d0369 bfe9704cb6cf055e 3ffedb515125209e 3ff3aa389606e38a
3fe69e80f0d68c1d 3fe47ab8aba02eb6 3fe64c51a54879b9 3ff35e4631ef47f0
3fec92ed69035020 bfe426758b5ba26e bfda13356b562b4b bfca01ad543eb988
bfed0c1d474430ad 3ff3a57117343ff5 bfde9c37c7edf2b7 bfcadd328fa14465
40043ebd5ddb1130 3fed7c3c4a8e094c 3fecc933bf2bdbd1 3f68b2d514950731
3ff73d8dabc48157 bfdb448b532d2b70 3ff6dab5a80bacb2 3febf8d11439c0af
3ffa2136bc8aa53b 3ff2cb66351dd1aa bfd72ec99e63a437 3ff822815dd5857f
400151f944c696e0 3fcb9c2cf8fbd7c0 3fd76205de27219d 3fd5298b436392e6
40009e5d42af42fb 3ff977308f1a2d82 3ff8fbbb5ce9b55b 4001b7175b1a04aa
3fe7b08714ca9382 4000b9dbdf3dc8e3 3fa975e2e0d5eb59 3fe2d8a29e58f64d
3ff3d16c6d7a8b79 bfe80dbbf58e517f 40036566a419aada bff4a9b00c9cacac
3fd1bb52c52b2420 3fcfbd9846281be3 3ffbf3249ad46f46 3feeb78907a853fb
3fd81579eaf24992 3ff718211f359254 bfe95f14bab05dfb 3ff79ecaff0d1c91
3ff0f8a0d09bb0f0 3ff083ce44138021 3ff55da10be173b3 3fc98fd0be6fe625
3fe4ecae4d1011d2 3fd22f3b179ee360 3ff29fc6c9a42062 bff0724356dd6aeb
4006fb3d5cf284c7 3ffb08c6ff63963e bfd6f05186e0c2a0 3fe87d289c92bf85
bf94730a2ecc0a6c 400306515aff3652 40040bb5c31209e6 3fe652d5a40ca09d
3ff9d1692f0e2767 3fd392a82d5123a8 bfcea970ac6b5205 3ff2e34692413980
4004cea9f7235770 bfdd86421c0528d4 3fe9bc337764cbd4 3ffc97e7845963a4
bfb0879900bf9aec 3ff050af3beeb0c6 3ff2263e7afd4f66 bfcfbd2272cbf390
3ff1c7240b99407e bfbb134f07851f42 400498500f29f137 3ff84c87ef5ceeb3
4003c1ccda4cae52 4005863a43155b98 3fd4700665ab51c5 4005aa042812d82b
3ff53a094535f13c 3fe7c079b3d4f435 3ff08f78694db091 3feef60706c1c9a8
400470ea6100f89e bfb3bdc9b450cb2a 3ff7b385078de89c bff3aaf3507590dc
3f879bec4f2c9ebe 4005520bb84d1d67 3ff454e16d2489b0 bf918b9adbc9316f
3ffe994e45ea7cc6 3ff7b7d6cbc0a3dd 3fffae5fc18edc10 3ff732b0ca73015d
3ff53a094535f13c 40015a767545de54 bfd4b4b46da1df64 400103acc65487f8
3fd47920d69f0aa2 3fe6f5417b350ea9 bfe4bd433e50eaa1 40052bbbe02e9ad3
3ff3c90979de6375 bfd7970ce605cc31 3ff7a0476ddc6439 bfeeaa1a55f46caa
bfe8505e445458be 3fdfa8719ac7f088 3fea7fa11beb61b7 3fe02acadf1afcf0
3fe9c62984723be9 40002828af253072 3fe04661651c70ec 3feb34b02919db32
3ff1424dac5ea7af 3feffbb0ada09046 400275e7c3c43cf3 40034b190c3f6e42
40017639ee43fc06 3ff03812ac0e2f59 3fb46fd57a0e9b3b 3ff822815dd5857f
3fdab10f14b50a06 3fe5fb1c33272321 400062599f1d7a65 3fa93ea69129754f
3ff06939041cd06b 3ff65469f23bf4e5 3ff1d2a06bcfff61 3fe1afe0b2fe5480
3ffd73ab476d6a0c 3fd9b056ae59b27f 3fc6fd74ba3d4e19 bfd5f4f6a09ee7f9
bfe0081410a8d1f3 3fcc6ee90d954ab0 3ff3219f7879633c bfd1f3f15604fc41
3ff1fb75084639b2 3fdddc1a39589614 3fe56b2abed8e71a 4001c62095b01cc5
bfe12ddf089a4850 bff1ff5c87668821 bfc886183aec17b5 3ffb543533fad77f
bfe2b1c77bd2c7ff 3ff0ab152fb562d3 3ffaba88f97ab244 40029ef64ecb2c44
400475a312f1c2e5 bff3d280a676831f 3ff94f1709b81e7c 3fe6eeb6f9f9ee26
3fe7dc97c236105f 3fe418acdb5af9fe 3ff90b816bfe1bfa 3ff649e141174491
3ff2717aa433bd0a 3fda45b7183068e7 bff19b4bd9f591a1 400041ccb8d3d737
3fd8f7d5a7294e75 3fd4361c0d5eaeb6 3ff55686863aeea7 4002de07b73b43c3
bfbfe212e0fb4dea 3ff789153a9bf833 bfdfdf51a9e92141 3fe4bf07a1897445
3ff4897e99af2c94 3ff3bf84b03fe505 3ff7027ff10a19f3 bfd9b22cb3f4f989
4000864f977e9fce bff5b30482d548f9 3fb23a4f06524abb bfecc88a164b2b11
3fe22619c1092aaa 3ff19a3e4d4334b6 3fe919109ac202e4 3ff3563aeb512f46
3fe88ba4709d42b1 400340c245b1acd0 3ffc567b8efb00ca 3ff274fc66e357d9
bfd4ca24dbfbe288 3fd0ada9859657a7 3fdb48b98bb7c35e 3fef171568ab5bb3
bfd213ace0868419 3fd5d711bac29f0d bfe6ee8cb35cccc1 bfc6b671133c1461
3fe1c7e9bcdb2e67 3ff2cef496f87fea bfc1990f260118f9 3fe8cc53b74f6fd6
3ff6916d54ed9c62 3fd14709967e7a29 bf98e6c431efcbde bfbd073aae097674
3fe01369e95ef1a5 400130eed5f1d607 bfe5d0fbc3376636 40016f0149d8a3da
3ffa96702bfcd0bb 3fec89efe98c85a7 3ff87f759d67d78a 3fea05e86a3ff59d
3fddf6fc7120b69b 4002b94ef3671d29 3ff333762b53202f 40060e8702ffe27b
bfd422819a057958 4001a2f527dcb893 3fa228cb8dd56357 3fea9d6a3b261821
bfd2414b07ebeeec bff580f415f3f58c 3fe5c7af904a13ec bfd07356a75ec3b1
40013b4fe3bc965c bfe36d63f292d08a bfc2b35ffcc8adf5 400152fabd3d8d3b
3fc5f5c05c8c7ebb bff247bd4cc1da06 bfa4388365bb147d 3fff716522de9eb4
bfd70c81feb1703a bfdb45c42082d494 bfe0f88e7c9f5235 3fcca592bb23d242
3fa8e39faa79a053 3fe80fa8f3bf95b7 bfee14076f27aa0d 3ff73756f9779403
3ff56b8f7705c3ff 3ff1bfd8745d0da3 3ff0e696e6a93cb1 bfcec58e12c176a6
3ff4cd8b5c1faef3 3fc4d0f2d9941b8e 3ff4db3325d07e5c 40000f0348a3d2a2
3ffab64b8f01c019 3fbc28ef4b483d09 40048bd21653d7d4 3fd0a9f9625e9e04
4002c7b6dd8aaf33 3ff6c9e323a1a123 400183b6d2969f3e 3ff7b583be2ffe32
3fde4dfa11232c38 3ff1c532738aee2e 3fe6a43bd6605f04 3ff993883da1fe4e
3fea8e9a28805ffe 400246ab2ee3191f 4000e5ce3e7dceb5 3ff7a0476ddc6439
3ff8005dcd5725b1 3ff1cf883d9992fd 3ffbdc1959622936 3fad509cfbfb0ff3
bfdcd9343c7c3d5b 3ff92dc885afc18b 3ff2a03bf8ed9fb4 bfe3c08c9b54db7c
3fee842376b93448 3fd28d29d81f98fa bfb45a745cecbcf8 400218fc8d232315
4006046b1e286ef9 3fd354dc8f7baf46 3fbdabe0f074ef36 3ff06939041cd06b
3fb9f31ed3f56e2d 3feea46128afea5b 3ff7914d554bf35b 3fdba588d18228f8
bfe08ec63129e967 3ffc34e5580a6d5f bfa056d547da1553 3fd1cd0177ed34b1
4005ced6ed92a114 3ff76d077beb31ac 3fe236339f8fbfc3 40021f7de0966db0
bfd4a00c7644fae3 3ff1eb60a094b442 bfed9f4a332e84ab 3f905a56735bcb2e
bfd4fe39e2f4fa48 3fd0d1d24c5048f2 bff27c5e2dd6d22e bfe07c0dbb34dbfe
3fe53ea9c6fdbfb4 4003e9e52c0fb816 3ff30f767caa8ae1 3fee284247e73263
3ffdd0cc0f1fa7e7 3ff63ced7230f6ae bfc4eeca3fc08195 3fe32ba69d4199b6
400041ccb8d3d737 bf918b9adbc9316f 3fe760553ce6a085 3fdb9b5af1e19e51
bf996ce081ddfbfa 3fe42e7e9d164258 bfa8ed12a11fda1c 3fd8204135197939
3fe4f76fe1f0ebed bfd71f05fb8fb596 bf9a947b946f2a0a 3ff55054592ac0f2
bfa8ed12a11fda1c 3ffdb46bb630014a 3fdabc0e3d039c12 bfd927f5e4d887d7
4001c62095b01cc5 3ff65dff8bfef922 4002deaa8aeb385e 3fe51d1b4e753c8f
4001aa2160d9afb7 3ffa3abdfa1d64c8 40009ff45602375a 4000b42dc5401d0b
3f5f1aa202d5768f 3ff79aeb6e31c665 3fdb0a845749345c bff18b352ffcc57f
4003f7dc8c970dda 3fe754316e9b7a28 3ff70649da7fe0c6 3fdf36329014ed56
3fc38fd348bbdb65 400008d52f1b2240 40033840f30f9c1a 3ffda85f7e4b3421
3fe041f81b0f5559 3fe27f1edc01fc04 4000b64651456839 3ff655267408bf3f
bfe7e64ea69414ec bff04ecf552a9f14 3ffe96ba7841dd74 3ff45bd6c5c7b1cb
bfc4883afe9cda7a 3ff5e0d4ac422dcd 40039bd633926a6c 3fddb206192ec61e
3fd600732d4eb786 3ffdea33a996a115 400261713668865a 4007a190b00d5ba4
3fd91f01ffd9acf3 bfeec1d4f04a61ea 4003b8e480bb4699 4000027bf3d1255b
3fe0f83255f72c1d bfe30f7f08eff47b 3ff3a61ac895deff 3fe3ea467acaee4e
bfdd07085f7fe545 bfe9aa4f6bdeb7a9 3ff40e46837c6824 3ffa0793b3a0ca6f
3ff6e15af2642dd7 3fff2bfbbae3f775 3ffc612c53a67711 4005b0bf77f1baa1
3fc2ec30bf6ce52a bff5b30482d548f9 bfd07b5a6ddafff7 3fc82f35e18064cb
3ff6601525f86520 bfe3e258a687144b 3ffd51f1ff3e5c13 bfe7018cc8cc7ef9
bff3f44c988b6fa8 bff29a23b6e682b2 3fe6c4fc3a4042cb 3fda80ce5337a861
bff580f415f3f58c 3ffc567b8efb00ca 3fea05bc73f16723 3ff5cd2c8f7974d4
3fda0b1d969f1bcf 3feccae4b78117d1 400088a2bbf946f9 3fed56b3d39129ef
3feeb78907a853fb 3fe26d750d15b209 bfdc3a0c8b5a60fc 3ffd42fd6c42d7fb
3fc1a6e41024056b 3fb42ebc9177dde7 3ffc2441e701b223 3ff235d2c7265733
3ff1ad2b381042e6 40018540bd183cae bfde1e8f9a20468d 3ff96004ca988185
3ff5addeeac3648d bfba98f5966fa95e bfe55d6457cf6eeb 3ff102e274d5ae65
3ff7abb639c37ff4 3fe1bfde5344b13b 3ff9874d999e0c86 3ffb42a1a54ec845
3feab11496dbf4bf 3feec1df6643e19b 3ff30115884829f8 40031f934238aeac
3fdf4ef4286e913e bfe51e1ffac39026 400089bdd1a84c80 3fed1279062a4ba5
3fee03b3d3a3f8b4 3ff7d5a384f1ab48 3fe5167f686cef3e 400104a19180d47e
bfd959af95781982 3ffb4e311d2ee095 4000c6a54b7cbe68 3ff4260bf07d9365
3fc79429f7d48b79 bfd4444eca7fb259 3fbe2a01cc5458a4 3fd86f208ab8ef11
3ff696145e8ef4c8 3ff2260473e1e461 3fe934fbb176ba8e 3ff7b965fa426df2
3fd1d32e994ff0cf 3ff0e696e6a93cb1 bfd2f5ee92ac495e 3ffe1bf4d035197c
bfc08a3ff67c820c bfeadb9553e5ec3c 3ff4acd7a7bb9dc2 3ff7d4461a8aec03
4006046b1e286ef9 3ff2717aa433bd0a 3ff990dcb9024b95 3ffebc76998550ba
3ff81712bba0b8d0 3fe2fe7a83eeb932 3fd52d87b542b2a5 3fefe885cf03273f
3fec707a4d756902 3fae0f7017771ced 4005466d6edf57c9 3fecbaefb2d4b804
3ff84ac9bf84d721 3fff305a6c06602a 3fe6b4d062ae0879 3fef7b57bb387f83
3fdd9c18ae499bc3 3fe4a8ade646cbac bfd255d3d806be7e 3ff4ba8eb4e3b0eb
3fc2cd5b53fc5473 3fe3012c2aa52abd 3ff24269840104c2 3fd44bebb185f934
3ff49c6ade38b75e 3fd25778786206c7 3fec154a7c6a618d bfe72b72efcd0398
3fd6e9df4bac790d 3ff0535aaedc0d99 3fee39eacd1f15f1 3ff4474aaf52c2c8
3ffb2f74dd767f0f 3ff4b07a96206904 3ff6b84b1ebf5afc 3fd842b51553267c
3fe8fdf23d61f882 3ff6574bd4456840 40068c253c60f042 3ff57bcba937d16f
3fe655db834908cd 3fd39776c75c5c34 3ff35b673a81f82e 400404902edd2a32
4006be43a51ce432 3ff65c62ac58e68f 3fed2bdb1080126c 4004f84fc1006a0a
3fd65eecaf8d9180 3ff40b6549d9e43a bfe506d0b4ed9f75 bff580f415f3f58c
3ff22c854983cea1 3fe80fa8f3bf95b7 3fefaf529f71f7ee 3fdf8e95429fda7f
4003eb4181d6b450 bfeeaa1a55f46caa 3ff49afd1a56023c bfec6892e10b63a5
bfd5e6d78c798158 bfdd30316a8c8761 3ff89d51a5881828 3ff3deb5a7547575
bfe49d0405ed616b 3ff9eaf32ac29f0a 3ff3a8de84240547 bfc6245793c40ec3
3febf4236693276e 3ff1ef361d4e3148 3ffd68c0594d4b8c 3fca8dea7aa182a3
3ffb39d5056852c1 3ffa1156efd9f960 3ff044a1cd6f50e5 3fc8719027de01cc
3feaadfdf63d839d bfd4e65f7f8578a9 bfc8c77199ea93fa 4001d86029a7295f
4004ddb373949803 3ff3fbbe7e921bd1 bfbd8791b96d3ca9 bfee598a148e9f78
4005a24db85cf181 3fd0add720c05b51 bfced579a2ed0809 3ffd9cbe9d1f3da5
3fdf79229e3bc1f4 3fd61342135de4c5 bfe0f88e7c9f5235 3ff629f3e22fba00
3ff2cb66351dd1aa 3fd1bb52c52b2420 3fef3fe560a530d9 3fff58f1bc8daebe
3ff4b2664d1370c0 bfbd073aae097674 3fe2ed0e0ef9d0ea 3fe31ce9e88b7663
bfeec6fac3d017cf 3fc5ad24a1ebc751 3ff5708efdcf634c 3fe80fa8f3bf95b7
3ff4165bf4b911ac 3fdf74bc9ab0dba7 bfbfc33cacd02cbe bf9d2a78b01bafbb
3ffa103aed9d565e 3ff9828447ebd71d 3fbc76cff7e8ff22 4007c3c044c63717
3ff06cfc30aead33 3ff8ff17929a69eb 3ff078785f733076 3fefc19ca51c9ef7
3ffc7c6bb76b1875 3fd4ce265df70624 3fd7d2b21df1e732 3fb25b379663510e
bfa35a3d0ac2936f 3fe359ca3964991e 40015a767545de54 3ffafb8dbf3f2624
40058108ef73afea 3ffb62302f89770b bfd6b1d3ea159843 bfc990ef22d69891
3fc8fb6c7d17ed48 3ff5c6c1133688bd 3ff02b497c0a8b0c 3fb23a4f06524abb
3ffe3ec414a6f59a bfe506d0b4ed9f75 3ffe3ed83bb18cbc 4003ab92df170ba5
3ffaedbcda073cd6 3fe259a241b6cb06 3fe709ae60397acf 3fda74ea3276d577
3fd1ef94363410e7 3fed9176c0d64d48 bfed0c1d474430ad 3fd1bb52c52b2420
3ff5fa52b83e89b4 3ff88451f5c23f1c bff5b30482d548f9 bfec88c1a7b40d46
4005c3dc03ca7234 bfb0879900bf9aec 3fd6f0e7633f49eb 4001e51969fcf4f6
40029813367087bf 4000154f1e7ce64c bff268134a052f45 400365448f1485a7
bfe4924cd65f3766 3fd38c18f12326d7 3ff7514654b4941b 3feb7f2797648182
3fe104eb8fb7488a 3fb9f5101c03804d 3ffbc31d012aad60 3ff737c173bf0bd6
bfdef7051c5b5f18 3fc223e87c9aebe4 3fdef2ec487226e1 bfd739c6f2e4d633
bfe4dcd73c03f467 3ff822815dd5857f 3faff7fc960c5185 bfe12ddf089a4850
bfe3d0642c6ffcc4 3fe438e68be1d1e3 3fd1ef94363410e7 3ffd87ee40408f25
3fe600bb1f756d8a 3ff1543771025702 3faa71782c8aa763 3fef59f2de04fe73
3fff8f910c71f523 3fd644049ebe1d6b 3fde0144e44b4d97 3ffcf7ffe3e7d5b9
4003763a76075ee0 bfec3370e863edec 40059bdc06767b50 3ff4b2ff743c2271
bf94730a2ecc0a6c bfe8aba2bb9155e3 3fee8f4c644d75d2 3ff2911a298a9395
3fe8a03022f784d9 400176f6fce49a63 3ff66a454d2bf3b4 4001b29366bf6f47
3fd9ffd3e6296464 3ffe449dcc67092b 3ff4b2ff743c2271 3fb2b8870a7caeb6
3ff49c6ade38b75e 3f9fe44a405a24e9 3ff8de587869c71c 3fdeaf04e986c273
3fe57c7e0872a866 3ffd8461d7259adc 3fd50a4668dfbcfc 3fd5b3a2850cca6e
4005b42924d8b4df 3fd883922873510d 3ffbe5ef8119663c 400440648c836d87
40054b36e742ff68 3fedfc1eb38f4766 3fc886f512580560 4002bea00654f10b
3ffa8df8976ea4c3 3ff88f3935bfc29f bfe73d5a125e87f4 bfe5dbdcc14698aa
3feb23cd2cbbb6a9 3ff18d86331d292d 400092fe5bc1e267 3ffc001c2ec1f3f7
3fe9f50926ad7317 3ff9cd39635f74ff 400282a58dcc43c7 3ff47ea50851e6b3
bfd69cc25ef82d7e 3ff19ebc4252023f 4005d56de701dc76 3fe66828ec9ee478
3fd74bfe2da4f258 3fdafc8757647861 3ffec7c522ff9b8a bfc03287b2c55432
bfcd660b4dcbbeca 3fd3bfa2a7e34710 3fdbdf9d81ae5629 bfdd4d00eab293d0
bfc3987e28e2da1b 3ff9552dec3ade61 3fef48b592aa146b 3ffa737192aae2bc
bfd2c77556648205 3fe89fea7c2e591f 3ff174a1623d3380 bff580f415f3f58c
3ff2d44db25a41bd bfdde0c232782cba bfc0610802ee2e6d 40010bbff5a689ad
3ff5abb70ec5674c 3ff132fd3cfe6f46 bf9c4bf1202d0d87 bf79449452f0f0ad
bff03dc8922d9abb 3fe747bddebdac95 40020215b2ea3a1e 3ff06939041cd06b
3ffd58a7054f0f04 3ffa0a77a4f5ae38 bfe7155a02eecc82 3fe212839a45e592
4001ff458f3983c4 3ff3c06286461086 bfc1990f260118f9 3fa8968087d9e345
bfdfdf51a9e92141 3fe009d772b7c527 3ffba3e2c97c3b2c 3fb19978d75d8f0b
3fe14ee4119cb290 3ffc74b3ddfc67ea bfe929e5b760829e 3fa6bf2217d108a8
400088cb92b6f0bf 3fe44c1a967936de bfd2efdc4e0a6910 3ff57dd0abf63303
3ff7b57e8d47a6cc 40000d27bd3dca94 bfde4cafe346b505 bfab05c88e3ad535
bfe56b10bf73a27e 40044c62064b74a9 3ffcb08d6946ef24 3fe21e9eddaba2fc
4003c1e574e8abfd 3ffb42a1a54ec845 3ff747fe93cb24d7 bfc1920a77bb0c0e
3feb5a7893261fc5 3ffe2622c6caa07f 3ff8f85413643cd8 bff2d6a090bdd101
3ff38bc75a40a8ad 3fe7c4afcf97fc2c 3feb0d056395b6a9 3fd3473809fc42e2
4000cb6e98c3b110 bfe42d4590e7e270 3fd5d711bac29f0d 3fd963f57070d995
3ff44b77be9c6a5a 3ff732b0ca73015d bfd55c5830d68dd8 3fffdcf1fd8473b2
3fe5b8706a0f22a4 3ff1e9ccbee7185e 3fe9a9e8b89d7c93 3fecb17871014713
3fe7a18f2b3d1927 3f5a09f973d04467 40009fb22b88360b bfe1b69c235763b3
3ff3a04e7fb8dedc 3ff781ea6618902d 3fe197543cd39e50 3fefad09f3fa7d90
3fe29b8b9af87f4d 3ff076214ce8a631 3ff81ed07eb17c1f 3fea7db997acb7ed
3ffe02111e1e34e3 3fe89fea7c2e591f bfdfb58fe94ec87e bfcc0c626defd8b3
3ff306979a48bacc 3fcff76ba01528fd 3ff4b2ff743c2271 3fef21912dfde0ba
4005461ba97c29ed 3fd76205de27219d bfe86a4811dbd0ab 3ffb6a0e239c5a41
bfd886585cadf81c 3feccae3f1723644 bfdba62e14383180 3ffcc91454fd3f83
bfeec1d4f04a61ea 400496fe22c7dfa1 3fe25dd2c4505541 bfe5d8f05c0770d7
3fe46e4cd192a8bc 3fe32a9e1846a42f 3fc1b595c43fd0b5 bfb3760fac011be2
3ff7fa56fffefa89 3fe53ea9c6fdbfb4 40009fb22b88360b 400062599f1d7a65
bfe25191c396edef 3fdb3fb4f626fb6a 3ff34a4e1cf53f3f bff2d6a090bdd101
bfc33097160e3ec3 bff1ff5c87668821 3ffac00a7d7e4b8b bfb717f394ddf3cf
3ffd7a587cd79a80 3ff724d8f57d7948 3fe2fe7a83eeb932 3ff5e9c0ef74a8b4
40058108ef73afea 3fda8bd7e61b5b8d 3fdd1e322b1e23c1 3fd4361c0d5eaeb6
3fff2645ee17705a 4000f82b2d7b0c40 3fe183ca4cdb87cc 40024fc53ef36e86
4000d6aac7a894f7 40010fe641198059 3fdcab5d33555bd9 40058778b04a9164
3ff19fb95f4f0d4e bff2f5a28c6e1958 bfb45a745cecbcf8 3ff73d8dabc48157
3fde2afe78202215 bfd8d9e2c85eac00 bfdbad139b378f1f 3ff2e41a9617da3f
3fd77ec8aa2abf63 3fc058ce1b6bf638 3ff048883c38ae88 3fef0da86015d12c
3ff1caeaa25b4487 4001870fdd62e35e 3ff2d3590cc8325d bf9164838dda7ba2
bfde5700373d9c43 3fff58f1bc8daebe 3fef368abb826017 bf72cc056869db79
bfdf2cc4289197bc 4000c6beab2bae7b 3fffbb1f30fac7b6 3fd5340bd5d4f130
3ff0ab152fb562d3 bfa58b2f7309393e 3ff6801ad73316e6 3fba076908988a5b
3fe6d88d8902f892 3fb9af0a3393da43 400195698e64217b 3fe74a74be347850
3fd9be328dd36fcd 3fd5186aef5abd75 40009e75fcea868d bfccead351192d5e
3fd99510acb2f1ae bfe431ffe7db8ef2 3fe9366d5f7aaf10 3ffa09974c19fcaa
3ff2e5c58a33528b 3ff7e6ea377625fe 3fcce48209086f33 3ff2d876b765fb1c
3ffbbffb3258fa92 3fc5dec2ffd34d82 bfeb9919cb6e29e0 3ff0b52d7d346b74
3fdebe8703ab4d27 3fe6fdd787940a07 3fef7d5a909a7711 3ff25adf6aba3b07
bf9d2a78b01bafbb 3fffc01e809487a9 3ff849ae9f4868d8 bfdbad139b378f1f
bfbaf37b1bcfddf7 bfe6d1d4f81a5fc7 3ff50e55bc9a08ef 3fe616de60520a7d
3ff1feeb428da81e bfc35050d993ddaf 3fa8e39faa79a053 bfa27bbc510fbe5c
3fddf527ade9a681 3fef24092fdd1384 3fdaea9d395aba63 3fdd4baa1d33313a
3fe3ceda70d7a422 bfcaf5f464e79a90 3ff0cf4c6ce44525 3ffee3a93538525d
3fddafc3abfc0304 3feeb0f543cdf568 3fdd853f6b947fd0 bfc83dcf6374c27e
bfd2c3d237d912a3 40046eb525973731 3fe894d778226a90 3fbd55932327d41d
3fe527050269149f bfdf2cc4289197bc 3fec44763d46800d 3fdd3cb85a444d6e
bfe0770a28ff11db 3fede631c91d9546 3ff194477cb1aad9 3fd600732d4eb786
3fdbaff064ec2c22 3ff4cdf00f894a7e bfce8151f61dae66 3fffbb1f30fac7b6
40037cb4fee3e08e 3fbda88ce2a8f313 bfe25191c396edef 3ff459e95aa7d0fa
3ffd06a3ca84ea2b 3fc4611a13464199 3ff53a094535f13c 3ff07bacff8b1eb9
400287bbde78d05a 3ff92dc885afc18b 3fddda44dcc13868 bfd34f24034400e1
3fe7e6e20ab16e0a 3ff824e67700247a 3fb1c9128a0d1af1 3fb74152f298b843
3fd3e2c0eb5f60c7 3fd9971da7851266 40002353385401fc bfe7db12ab50e3e5
3ffe449dcc67092b bfdda18c2a1bc750 3fbab16cf3c791f7 3fb90f0b63ab6aa9
bfe09f92b8371ded 3ffe0bf2991e1da2 bfd07b5a6ddafff7 3fe2255f88eee5b2
3fc3b7c8355e25bc 3fdd414ac8580cbd bff5b30482d548f9 3fc90b0a417cb399
3fc9036ff27d1572 3fbe84d9e1653c45 3fc86986bbd63189 bfeec6fac3d017cf
3fe4d9fb2ea2cc05 bfdf0d5e93f539f2 3ff3cf539119dd45 3fd258379b794401
3ff9247065c038ff 4007c3c044c63717 3fd5298b436392e6 400147454c67ded4
bfe0b64f5d552dda bfe4e3107dade9fd 3fe139bc6ead80c8 3feaf7b8467e364c
3ffa103aed9d565e bfad632d761cb5b2 3ffd94764415e466 3fffab9280b8ea47
3ff9d46948ac5151 4001cc3daa5afa3f 3ff44575c5d350a4 3ffa06b24315f567
3fdfd7675c2fe7eb bfbb5540ddea9b86 3ff751b605bcdb96 3ff0d90066966b91
3fec44763d46800d bfd977b06fbcd796 3ff08f735e0fb251 bfe9704cb6cf055e
4004f44c8fb5b7ad 40059bdc06767b50 3fedcc517ff0b1d4 3ffc937ed0dbbeda
3ffd7c95eda7d9d7 4005ba9acfff1110 3fec62dcf5fea00b bfd5b1573e8f6212
3fbdc7d196982bcd 40019bba7a21dbba 3ff9b3e35929db9b bfc2637fbf022efb
3fe37340304033bc 3fdf1e30a428cfa8 bfea183258d2ca12 3ff3581e6bb47a78
3fe5a4a49be51891 4001efd8c816e1f8 bfbb134f07851f42 3fdecbcc4247c404
3fee5080b65f5a34 3ffbd270abbe8664 4000904da14df77a 3fedfc1eb38f4766
3ff1632dd888feb9 3ff247519ed5ff6e 3fdbaff064ec2c22 3fdebb05d73928af
bfec40e9e75dd22d 3feb39f5a5dbe96e 3fccd89656b5caf6 3ff9247065c038ff
bfe4c1406eef991f bfc22ecf7770ab55 3fddbe51a767ff92 bfeec5d97b550439
4000904da14df77a 3feefe927d37fb35 bfc36818bded6e8c 3ff11b5348198ba3
3ffe1e48848a4c64 3fffe607f63b4f1d 3fd0ee896c42fa87 3fe80389eed03c5d
3fca707498630c7d 40010aed1acc7d89 4004139bf4132092 3fdfeb9563081ac6
bfdd4d00eab293d0 3fe8c258cd9ca19e 3ff08f78694db091 3ff0565baa3bcd47
bfe136f21209181a bf9c61b2332d3cce 4000c50eb5e29271 40022f8d0436ee12
3fd09f28b172bd03 bfd47d3e9bfd6c6f bfe31e58b8b70a98 400161c360e8456d
3ffed190fc5e6dfa bff1ff5c87668821 3fc4d0f2d9941b8e 3ffa1990694be66d
3fd51c26229b3078 3ff2cc75655d884f bfa647dc386242ed 3fe898aaf23426ce
3fd615256f956699 3feb19f8f96f4af6 3fcf4844612f4b39 3ff33153c71d92b9
3ff459074323b6e4 3ffae5a210ba26d3 3ff7c57b974834a2 3fea3b43b97d124b
3ff33138c0952970 3fdd21538847319c 3ff6a0ab7e134a43 3fe88ba4709d42b1
bfe2c0d96079ef92 3febf8d11439c0af 3fe8bd7cf902757a bff2978558c9eea8
4002b51a60156abb 3ffba3e2c97c3b2c 3ff0564a87e7c130 3ff9552dec3ade61
400282a58dcc43c7 bfe20bd89a885b52 3feb7382851d6cbd bfae501f12d00e45
3fc3c4677d29a201 bff2d6a090bdd101 400498500f29f137 3fe5ba3e226b7136
3ff36c673f1c026d 3fe9e710f17cb665 3fddad9946ea29cc 3ff545653836afd5
bfeabb3224c2a0f1 4006815ba1b6d1d7 bfc68463d41a97a0 400212a620261237
3fda7ff9139bd628 3ffb4e311d2ee095 3ff83fa603caf8fe bfe46b848aee50d2
3ffb0aa474ff99b3 4000ef0d554d917e 3ff82ec0e83c6502 bf97c7bbf3e9d0e3
3fb53514e3c25b11 bfdcbe5b56ca03e0 3fd79a41a952c63a bff2889db0b38a2a
4001e0ea739b5907 3ff37f6d6e2d1e17 3ff24884978fdac1 bfe3bb85932c4bfd
3ff727cfb7e0e11d 3fd84996965697b9 3fe4a7e37aca54ea 3ff719b1e53f5133
3fe85b4cd35f8b8e 3fefc8cb95c8e82b 3fc0a116bb3f83c4 bfd932d9a3c9c9d4
bf9e05d3b8b9cb65 4004b3ee171d7707 bfe36e490d3e868a 3fee6cbf61c9c46d
3ff910ba32c1bd20 3fecb16ab67c41ca bfdab062aabe4964 bfa9ff6b54579df7
4004452f09016d61 3ffe389c9f359be0 40040bb5c31209e6 4000278dd2075513
4007cefd04be2567 3fbe32370a51542d 3fe5c50d23b9d9b9 40047d7cf82725d1
3ff62aa4dbdbe88d 3fea9f05e8d640e0 3fe104eb8fb7488a 3ff6ee95bd66ef10
400135485f25e53a bfcf2b909992b3fd 3fd9ffd3e6296464 400622cc3881dbb1
bfe21f6b0a1034bc 400622cc3881dbb1 3fd9b1bc8af5662d 3ff7459ea0d33f90
3fbb8562791a437f 3fe6c7e30565ad0d 3ff2260473e1e461 3fe1439fc8e92466
3fc4cb2c2e8d2ff8 3fd1423b21e5bb3c 3ff078785f733076 bfc8c77199ea93fa
3ffd25e8a4331610 bfe0792aa158cba9 3ff81712bba0b8d0 3fddb206192ec61e
3ff2025ba2015241 40053738a14a5201 3fd7534b41eea428 3ff47e69423155cb
3ffe3af6928140ac 3ff0e59cd1c21c53 bfdcdf4cdcde5b84 bfc2fe08195ff4b2
3ffe4f223780df80 bfdd7e340ca21b96 bff07fdcdd51043a 3fc8ebe640254a2f
3ff3a57117343ff5 40009152029e8ae2 bfeadbeaaad7d13f 3fe266a255cb1058
3febf8d11439c0af 4006b32fcca34fb7 3ff4459318194816 3ff2d876b765fb1c
3fe663b8ae3ac047 3ff10a0defd7dae5 3fcff57f5ff46031 40044e7fc100934e
3ff14a95f4372be8 3fe572273858ef3a 3fdb3fb4f626fb6a 3ff126087ee2122b
3ff2cb66351dd1aa 3ffca67de8f4e3ed bff1ff5c87668821 400371625bff20ec
3ff32c436a2ea82b 3ff906a5021605cf 40008307ac4e1d1e 3fe3c2324fcbbefa
40000054c85feaed bff1486c5671c97e 3ffc6e27b3010636 bf9642e91128149d
3ff1cd48a2861364 4002c7b6dd8aaf33 bfe4d30daed02e6d bfbd4546195c90c5
4006c99498852041 3ff73593a0be98ae 4001eec2f972ba97 bfd5958dae723070
3ffa47c4e3dba304 bfed175168a67279 bfe87d2d71213812 3ff8753d3e8800f9
bfd99133160bf15b 3fdc38e0ae68fa36 3ff1895b1d0c5725 3fd254d859627ead
bfb26f16fb43923c 3ffe519afe3ab048 40020e131998bddd 3fcd7aa263e472fc
3ff8eeb5d9a57bfd bff4a9b00c9cacac 3ff92dc885afc18b 3fecfbf5c0d33dfc
3ff74f33793529c6 3fe87417b5e1464a 3fe80fa8f3bf95b7 3ff2d44db25a41bd
3ffd9b6609de218e 3fc32f9db9461bdc bfdf63d568a49a44 3ff2723578e17379
3ff724d8f57d7948 bfc28bbd61d10673 3fd7724f61b06551 3feefe927d37fb35
3fe19bf4ca1fc39c 3fffa92b204c02be 3fdae647f20c25b8 400198cd93d9872c
3ff3f5f5dff8ef17 3ff5c2edcf117a1b 3fb994ad4194b2c3 3fca28e234327360
3ff1150d7ca16c1b 3fc41ee33f573551 3fbab16cf3c791f7 bfd72ec99e63a437
4004c2b97ac76ade 3feb1d2fd0d75210 3fe9c884534f6fc9 3fddafc3abfc0304
3ffae5eece76af33 3fe6b7763c66c87e bfd07027af7ff75e bfe431ffe7db8ef2
bfe84ccfc496563c 3ff66d73056516d7 3fe99b61d78f5266 3fe79c47fd0fb256
3fee3099f55b1bb4 3f94c8ebe9113706 3ffcf1e9600228f2 3ff4b2664d1370c0
bfc61f17d9439c42 bff5b30482d548f9 3ffeae5eda61188d 3ffceffdffc6c89f
bf985b8fe3ded5d2 3f9fe44a405a24e9 bfa8170ad94913b0 3fef5df13ce8da80
3ff7576fb2a6659f 3ffd82294ca28074 3fdc0bc1f6595639 4003a89f6cb1dc26
bfa61f56f3ea195b 3fc898211c8e2f09 3ffd109bd5a6ebfb 3fdf98209254be2b
3fefc2139143bdb1 bfe6d78c9568a836 3fe09bf2242e31ef bff0973266b8a395
bfbe9103f4cf6816 3ffa16463288af93 4002a1e5fdffba7b 3fe7dcf31c059513
bfe58b36dbabdcd2 4001bf8deda889fc 3fb74152f298b843 3ff6c12f07500162
400371625bff20ec 400631c20751a5e7 3fd1fa168c918463 3fe19bd92c3082b4
3ff31caf8aff054f 3febcb401ed99181 3feaef299843478a 3ff5bf4ea14b1a8b
3fe4a8ade646cbac 3ffc555f2410bc19 3fff0a17d060ae77 40017770b6269053
3fe4ad481cb3b391 3ff697c1d884e737 40037cb4fee3e08e bfeec5d97b550439
bfd008115a080e40 3ffc37b2f201c013 bfe3a291a8938edd bfe1aa986025cc02
3fd4b44683de170d 3fe7e6e20ab16e0a 3fd117001c13a715 3ffed4f2d777a6f5
3fc46b3b64644bfd 40008d686799a786 3ffd9b6609de218e 3fc41ee33f573551
bfe1b69c235763b3 3fd206f5dc032fcb 3ff08f78694db091 3fe7b8cf6d0aced3
bfdbc4edfea710be bfe80044bbd3af05 3ff657ed59fb298f bfe1aa986025cc02
bfe7155a02eecc82 bfed0c1d474430ad bfcf5e61781beb85 3ff81a026940ef2a
3fee21398a8bb569 3ff5499d34a8ab68 40013c4256b4e1b9 40029af20e83db06
400244e3a8562ee5 3ff5ae2e32412c1c 3ffb51f13d02e59b 3ffba3e2c97c3b2c
3ff47e27c54f045f bfe076dacb3d0c7f bfbb8be3dbc860a3 3fe279976bb045fd
bfd395aa31a85f80 3fae0f7017771ced 3feaddf85bf479ec bfd9b22cb3f4f989
3ff5426d09ef99d7 bfef8a9628ee59e3 3fddd93b5d0b5976 3fd80681d102652e
3febfc7a8bd526e4 4003b78107219a93 bf9a81a24031fc18 3ff8a3ec36427aa0
3fef86410f82d807 bfed79c449682677 3ff24269840104c2 3fc341b92954e686
3fd13d9f48d6f75a 3fe0b4dcf652daa6 3ff05c6746d0d415 4000d08aea5f7e16
400498500f29f137 3fe5c6bf61e46965 3ffd46afdd6e436c bfa01c778db50182
3ff549acee2887b0 3fe0dd93809c2d35 3fdabc0e3d039c12 40033079a4b1aa9e
3ff15e1fcd70ad14 3facbf0f9d87daa8 bfee251a746520c2 bfcd1f9844f02bc1
3ffcf7ffe3e7d5b9 3fe6f5417b350ea9 3fe975f1f0888cd2 3fd87435ded69d7a
3ffcc50b8ba3980d 400824f2dec21ab3 bfe974a7f299fd0b 3fffd2482ab2036d
3ff272fef8994233 3febd9487f3d3eb2 bfe457e0e8ede518 3ffd53ded74d34f0
3fe44c1a967936de 3fcf59e08ae2e0ae bff0899b37482863 3ff28f22db60e6c5
bfeba540fb8c2760 3ff78fd9960853dc bfc44340ddaadbd4 3fb2b8870a7caeb6
3ff7c10804b442c4 3ff681c8858d8a36 3fe7774155b03d0b 3ff9552dec3ade61
4002642cd114004a 3fd21a2de8cf4acb 3fd13c257ca39e00 3ff60f67315ef71d
3ff45bd6c5c7b1cb 4006c99498852041 3ff45c72c0dd106b 3ffab83609d95c1f
3fe0c2e26a218f1b 4000fc584ac86bf1 bff0234f44bae28c bfd932d9a3c9c9d4
4002e8ae0ebf2989 3feef09dee42152e 3fe3b57c41691ddd 3fe4f76fe1f0ebed
3ff1c692b6f3d35b 3ff454e16d2489b0 4002a141483679fc 3ffc9601753b028e
3f9e385b0990e6d1 bfdee35ce2448121 3fee130197ad52ff 3fefc8cb95c8e82b
3ff2a4b87131f1ef 3fec10b574873750 3ff845a71eca122a 3fd033ba709e8321
bfe0101880a25a5b bfc24897b854b0d4 3fe0a6c2193e410f 40040bb5c31209e6
bfddd8b9696929f7 3fe6a031a623e7f8 3ff39275bcaebe7a 3ffe542d175789c4
3fe8cc53b74f6fd6 40029ef64ecb2c44 3fef1411a7db0df2 3fd6e6ae7270c6df
bff43b07ed9a99e3 40014e6595430d6e bfcf5a8e10793c56 bfe09c0eeeb088a2
bf725d3a8b229f02 bfe7ec6be66a9448 3ff25400c2531dfd 40033079a4b1aa9e
3fffcaba5728b6f1 3fdf065fde691f00 bfdc9ccf45758223 3ff10ed9c1f2c100
bfe11c4506619fa8 3fe2435f8c014c8a 3ffee3720256dc37 3fe7c4afcf97fc2c
bfe91e0b43cb9cfb bfedadf23c68f7d9 3fee39eacd1f15f1 bfe9e55453dfb9c3
bfd19a6094cb5ea5 4000cbb02bb8ac0e bfb6b51bf63be02b 3ffe7a9dbf5142a8
3ff0cfebc79609c8 3ffb4e311d2ee095 3fcffd4c1829fd5c 3fedfbc002074086
3ffd71df5e25e6a4 3ff98c8d8f98c42a 3fe2fb16563b96c3 3fc2c9ce0694050d
4004f44c8fb5b7ad 4006378b538d73ee 3ff898270880ee21 bfe0770a28ff11db
3ff306e9389ec50e bff2658060425452 bfbfe212e0fb4dea bfc069c0727d53cf
3fbda88ce2a8f313 bfd746c90ae4b370 3fe131784a2aac49 bfe7df12542afb1c
3fdd00c91b8893d9 bfe91e0b43cb9cfb 3fe587e9158d0a4f 3fff2645ee17705a
3ff5b7e8451164e9 3ff6a0ea24839511 bfcb3502268390f1 3feb43a1177e4676
3ff44b77be9c6a5a 3ff3a64301c8095e 3fd5d97754d3b6c2 3fe3a4d20d6971a1
3ff5c823855b5fbb 3fe60a92c271949b 4004d17ffb20441b 3fbd3aa8e608d7b5
3fff346436e3f6dd 3fe8692f3b11b9b0 3ff23d22fa01b32f 4000ffb9e0185a69
3feecf48c9c1d610 3ff4578a56d64c23 3fe010ffbf523a2d 3fceb3040f15deeb
3ffaba88f97ab244 3fe495c4b991ee9d 3fa6f45a38c6ca1f bfe402d7aeb72b48
3ff5627eb1771b06 3fe65ff845b283a7 bfa8ed12a11fda1c 40014d5b068c47eb
3ffe1841545e3cfc 3ff6601525f86520 3ff748b2b8b2de04 3fdd414ac8580cbd
3ffe7a9dbf5142a8 3feb8a5a17f5332a 3fdccae51ec51eca 40058108ef73afea
40001847c0757363 3fe6f5417b350ea9 bfdd07085f7fe545 3ff6b64a5271e1ca
3fee5080b65f5a34 3ff89c1a81e11c8e 3ffb62e0758b8a29 3fe27b87af5c2911
bfe1aa986025cc02 3ff8a67376d3fc9b 4000fb42994e6092 3ff32be48a3d71bb
bfe81b5214cca7b6 3fcff57f5ff46031 3fcadec9ae6019b5 3ff1bae009e9b26b
bfe0101880a25a5b 3ff35c7271fe8d90 bfebcb3c36b994b7 3ff4bdb01425aba5
3ff4d26846eadbb1 3ff60f67315ef71d 3fcaf945b088415f 3feffd17d164307f
3febe50db52f3162 3fee03b3d3a3f8b4 3ff6916d54ed9c62 3ff45a6a71deed9c
3ff8e3ec37815d86 3ff323728dd17a0f 3fd5d711bac29f0d 3fe9511386519c11
3ff007ba28d0d190 400383c66068d8b5 3fe4381f4be56f65 3fee5080b65f5a34
3fd1217da13b937c 400631c20751a5e7 4005713d13441a8b 4003ae0c3e6daff1
3fe0840810436778 bfc580dbeb71d386 3ffcdabe4238ca97 bfb96c8855f338b3
3fddd8f8a60f5dfb 3fdb752ce598bb09 3fdbfafa4180cb62 bfd959af95781982
3ffa0a264fd562a2 40022e8999316464 3ffb267521a69c6e bfc49bdc52071f3c
3fddb206192ec61e 400357e9bfc707a2 3ff7d6bf79efc1d9 3fd7534b41eea428
3fe1809d2303110b 4001ed5343dad88a 3ffd83a918df92a2 3ffa7ee213f776aa
3fe5772966d9e1d6 3ff53a094535f13c 3ff0df267021d20f bfa20e5eeaf56bc5
3ff2653135a678e8 3ff642b82e680623 4007e9062f93cfe5 3ff17a8f466d4377
3ffa93886e751cae 3fe5b0b521bcd016 3fe0b2a9feae60ef bff1ff5c87668821
3fd2ea03e788e7f0 3ff1c532738aee2e 3fbafb9d186e7fd9 40020215b2ea3a1e
3ff594febe76aaf2 3fd68e64967666bf 3ff2c9ed1b952a6a 4002104a869961f0
3ffae3802cea50bf 3fe4604c4fa72cfb 3ffea764e3df2927 bfe2e75a0cc101e1
3ff3a452832185cb 3fd963f57070d995 40020ae807994add 3fd4361c0d5eaeb6
3fd0b743f5979019 3ff7a0476ddc6439 bff0735ee70542f0 3ffa859e66b49f4e
3ffa16463288af93 bfebaf504fc8d6c8 3ffb0a8b8a88370d 3ff9218a6b048ddb
3feb51868344e80d 3feea46f9761aee7 3fe80fa8f3bf95b7 bfe03dd4d04e5340
bfec3370e863edec bfd3b8d3a9ab7e78 3ff448d9f24a03d2 3ff649e141174491
3ff476e72c1cf9fe 3ffb4e311d2ee095 3fedbc3a8139af9c 3ff7b385078de89c
bfe36d63f292d08a 3ff3ba44422d19e2 3fe8ba6cf7681547 3fe2ca435165e650
3ff5ed61f78ab998 3ff2cb66351dd1aa 400365448f1485a7 3ffad39cf12e90d3
bfd338d335800b1d 3ffae7459be11f3b 3f7d63e4abd69c34 400054a5aec128a4
3feacfa9479d7956 3fdb7b3df27aee14 3fe0840810436778 3ff454e16d2489b0
3ff7cb7f02f0a456 bfd2719983d2a86b bfedadf23c68f7d9 3fd382d51bcd9613
3ff74a0b2a56e78a 3fc0a116bb3f83c4 bfcbb4e5be797bae bfd338d335800b1d
3ff80897fd8c440d 40000d27bd3dca94 bfeb55085fae6c0e 3ff098638188c2ca
3fb4a29c7f89993e 3fe4d9fb2ea2cc05 bfe36d4b8c8dde06 4003c1e574e8abfd
3fd93303f5ee304c bfd1528749a81b9d 3ff7eb64833d0785 3fe2435f8c014c8a
3fc1970998105b91 bf7e8424b7498634 3ff04d8a433336be 3ff3a8de84240547
3fdff808793d77b4 3ff02b497c0a8b0c 3ff4cdb911e6ddc4 3ff5325125cfc6b5
bfdaa36ae939c92a 3ffec52bb3394b86 bfaf65684844882b bfa947ebddde8c86
3fc65d1bce486123 3fe247e00408dbd4 bfdda18c2a1bc750 3fe02acadf1afcf0
3ff47139e5a06721 400033c42baf96bb 3fe3dde070a5f651 3ff1c8d09d4fcc83
bfe9d6065f6b5534 3fb324e944f0c1fd bfe0d3f126ff5c48 3fb248dc8949e19d
3ff0be334f578c36 3ff48074326ac8b3 bf9dc233312710c7 bfd445f3febb1c39
400329c9253ddf88 3fd9382e14a68af8 bfe47899ddafbc11 3fff9b31d2133d25
3f8a840e63a8f132 3fd5baf2419d91b2 3fd7d2b21df1e732 bfe99ac49b5720ac
3ff7d345d25ddbf2 3fef8b84d7f10791 bfe73d5a125e87f4 3ff59644acdd6998
40010760b40380bc 3fc08e3604f7aaeb 3febdf663dab5b3c 3fd4005c3b255bf4
3fffc01e809487a9 bfbc2ca7b2e5a295 3fd15d219ff015cf 3ff82bfdaa5c4646
3fedcc517ff0b1d4 bfdcdf4cdcde5b84 3ff9078fcccbac9b bfdb6dca0156a10c
bff26b1a70f37539 3fc08f3a845d3b7c 3fe9553023b2b19e bfd3ec2fcf98abce
3fd8a2cdc156e7c1 3f9995e712fe9bf8 bff43b07ed9a99e3 bfa5b3d3f905b15a
3fc5ad24a1ebc751 3ff9a47720f97ed3 3fd76205de27219d 3ff65214b8926468
3fe9540236906e7e 3fdba588d18228f8 3ff173f676e90284 3fdbbfecce0b559d
3ff7185840128b86 4003bfcbe2375d27 bfe648b48aa50dc2 bff1e925bc76adb6
4000890447ee8955 bfdcdf4cdcde5b84 3fe781ae69818998 3fddafc3abfc0304
3ff4ab9309fffa67 3ffb543533fad77f 3ff339808afd87d9 3fe43bac3933997a
bfe79242b83c558f bfc9891dbee66b29 3fefd3cd12fde5b3 40018540bd183cae
3fcd7aa263e472fc bfd51693631ee6e5 3fc315975e3cfca6 3feb523a75295378
3fb4de5103b0b343 3ff59ab6079859a9 3ff73225be115fa5 40052f51b2a010d1
3f823ba165d7f96a bfe6a7090aa7bf6e 3ff3275b135a9985 3fc2438806c0cacb
bfc326144ef7bc4d 3ff00508739b956a 400135409ae0b14b 4004c21f8c8fd03f
bfde768d69375dde 40029ad9d7ae6c8f 3ff80ee4c1d41f63 3ff8bec1fea780b2
3fc425b0db26a5cd 4004659b7d3fda46 3ff9ccccc7f7ec80 3ff3ba44422d19e2
bfb15f2d113fb680 400246ab2ee3191f 3ff02075566bf4ab 3fccd89656b5caf6
3fec692abe3a6a9e 3fe64829d43b6da2 3ffd8cd926cb2d15 4004f44c8fb5b7ad
bfcb591f8d4997dc 3fa93ea69129754f 3fedeae9db3ec69a 40030e248e259764
3ffe176744c1e8a8 3ffb4cdae5326d6f 3ff8b7cb24c47c13 3fdfde31ddb4f49a
bf983d3d81fec8f6 400275e7c3c43cf3 4002edc456c44bc6 3ffca95dcc3bcc0a
3fed49cb34f83ac7 3ff6801ad73316e6 4000aa9c3e93a0e0 bfe0f1b0d7a7a36f
3fc7d2edda54dc5f 4001a99d63901957 bfe0ff07fceafb12 3ff9247065c038ff
3ff06939041cd06b 3fd0b62576fb028e 3fb950700d943511 3ff723419b49ef79
3fd883922873510d 3ff53a094535f13c 3fd6f21c037ef5fd 3fe6e6eb6eca6d81
3ff7fa3ae2bc6feb 3fdcdd4d7e35812d 3ffabe97a3a8f992 3ffdb32c0e17a59f
3ff30fc129b6f98d bfe68dc1f96edbeb 3fb391573edfd4f0 bfd5cae5bf7b6458
3feecca3507b60ef 3fe79ebf42d93027 bfa55cd143b30c88 3ffd763b64c44636
3ffb0fbec7509f42 bfe2ea7517d4c513 3fc98fd0be6fe625 bfefb88340c0893e
400383c66068d8b5 3feaf25b48944b14 3ff2ca195019b83e 40006d17bdd625db
3fe94725d6a5a191 3ff77220e413661c bfeba540fb8c2760 3fe7754e4367a09e
3fd72a0312f887a6 3ffbc31d012aad60 bfebff6a29ba6619 4006046b1e286ef9
3ffe52fbeabdbe8a 3ff6163d1d9302c2 3fe6494117c3ee11 bff14e204b7aaf9c
40009e5d42af42fb bfd4e798ff7e8425 3ff1a90e981f8590 bfc4c4707aad7fb0
4002b340502ef89a 3ffc43def5abafb3 3f9a6496c7f9dfb2 bfe98c1c921cad7f
4002bde4484898da 3ffea2ca993746d0 400259fa2ff5f362 3fe09dd58abd0b22
3fc5015d714a6d8a 3ff3aa389606e38a 40041237a01f1553 3fdea54cf61c5823
3ff4cdb911e6ddc4 3fe54ed67acf8df5 3ff4e81532a857d9 bfeb0a0a9e560a0d
3ff406184b5d8be2 3fe74fb3c7f437e9 3fef1411a7db0df2 3ff724d8f57d7948
3fd117001c13a715 3ff910ba32c1bd20 3ff1eb60a094b442 3fd39629c85364be
3fe1700a90b3328a 3ff7d65816192db2 3ff1291c9a5f3571 3ffd83468e4a86ef
bfe63e93ddec15bb bfbeeed9645e3fca bfa3075e6fc711f3 4001e7fea3b144ea
40022f8d0436ee12 4006046b1e286ef9 3ff47ea50851e6b3 400002b63df36bca
3ffa5d2f8f5091e2 3fd2247d8b51bc8d 3ff1529cd3482211 3febb586fe3429c8
3ff19c53f87e13f0 3ffd58a7054f0f04 3fb248dc8949e19d 3fff35bd2886e93d
bfd1f6b634040f70 3ff1c654f90d25fd 400275e7c3c43cf3 bfb9fdd97f6a70c8
bfdcdf4cdcde5b84 3fff54899d4c99c7 3fd1ef94363410e7 400121d694d61fab
3fd5b72464bdf0e8 40015f630af5ed26 bfe374d0c7453651 3fd5d711bac29f0d
bfdc39e8a421d206 3ff8a67376d3fc9b 3ff881fb5e27bcca 3fee26d22aecc543
bfa5242f5f58f120 3fe3101d4b8c8243 3fe8fdf23d61f882 3ff87d50afa351b5
3fdb3fb4f626fb6a 3ff7c2a17916b5ff 3fd4956d0953cd5d 3ffee992fa661bde
3ff08f78694db091 3ff0528a82016b0e 3ffb15a087154a12 4002e4d3651d4569
3fdddc1a39589614 bfce0037dec622ff 3fd5f37d250a1bb7 3ff9428026d93c93
bfdd7ef45b4c88af 3ff9218a6b048ddb 3ff8f17885f0ba27 bfc16d2f9cac403c
3fe3929e27f05c3a 3fe94ced4ad2e272 3fe39a5324010383 3fd8b8b262862fd1
bfed0c1d474430ad 3ff2e7be8ccd8800 3fe5167f686cef3e 3fbb9b1b82398da9
3ffa2ea896df3f9d 3ff0f1255fd54a88 3ff436fd8e8ea1c3 4000278dd2075513
3ff7324f68551686 3ffa1990694be66d 3ffd49ec603bad8d 3fe6f5417b350ea9
bff4d70057e4631f bfdb3fae02ec4596 bfdf0420fedd16f5 bfb49b0fbfc11fdf
3ffe04d5705a4a88 3fd72a0312f887a6 3ffeee060059aead bfc31014165f76a5
40074cf6c739df50 bfdff8209aee7486 4003c6854d0d3e77 40002ff96da1e7bd
400261713668865a 3fc754c5cb2ce146 bfd28ebf272c714d 3ff5870eb3b5d9d3
3febabb46c502da0 bfda440588a94b18 3ff0ab152fb562d3 40022f8d0436ee12
3ff2010c61d0fedf bfd09300f437afe9 3ff6a71bdbf4de52 bfe5b4acea89978b
3ff76163403f2464 400470ea6100f89e 40023615fb5c807f 3fe37dd9bbd062a5
3ffbb513ed24730c 3ff4fb65fd04e007 3ff088f517bf99c8 3ffa17e839a398a4
3fe47cc776f5a6f5 3ffb79442c35d822 3fed2a7c19d0c59b 3fd6a5adb68de65a
40015ab1168e60f3 3fffbafd8471c7ae 3ff9e6849cb4161c 3fc83c389259094d
bfbf39f1cf7c3715 400496fe22c7dfa1 3faa84f61fb9c868 3ffb6c1a3d8b6eb1
3ff2e5c58a33528b bfa4a1dd2e3c5d98 bfb75121795e892d 3fd883922873510d
3fc49fc4746e2c58 3fe08e0c2b06634d 3ff6044a6f050321 3ff2347b8f722f23
3ff6a71bdbf4de52 bfec6892e10b63a5 bfceaa96ef9594c9 4004a24fbda6eeba
3febbc2600f4832e bfe852862ed71768 3fe0f0c0c18cd10d 3ffdc9a55cdf841b
3fe97ccf29bc8059 3fff441ab4f20da0 4003262ba9f9824a bfd32934828bcd9d
bfeb15aa606bfff6 3fe5bcf61048c23a 3feb5a7893261fc5 3ff4c42f97d2090f
3fd5c2e01b39bf75 3ff7459ea0d33f90 3fedace2e68d8100 3fe3e986e8a58faf
3fe9c28b304077e8 3ff4c839cf765aa6 bfec7fa24ec1c27e 3fd133402b4ccb30
3ff343724b919d69 3ff627317d808bd4 3ffbcd99956ef8c9 3ff993883da1fe4e
3fd4005c3b255bf4 3ff2b217d52b1168 400071bd6c8019fb 3ff7de6b36cedf83
bfc56161b632ef6d 3ff8a1d2a2f127a1 3ff3a04e7fb8dedc 3ff65214b8926468
3fd118e6cf0da220 3ffd87ee40408f25 bfdfb68bdb3ebdfc 3febf21668a40934
bfd8e09786e8c224 3fc7ec9128c96636 3ff2e1119d1eaa77 3ffa392b8bf70046
bfe6f9a3493e86a0 bfe2e75a0cc101e1 3fe67744a700972c bfdb1f90d1b59eca
3fe495c4b991ee9d bfd6cdbd0f9ab200 3fac9553dcfeb372 bfe431ffe7db8ef2
3ff291a931130e66 3ffd5c6e9382a77c 400275e7c3c43cf3 3fc14760b0841705
4002815c1dfd6052 3fcd03261c131df6 3ff8fdf80f5063e3 3ff788914252fe39
3fd5d711bac29f0d bfc65c44071a441e 3fe15f167d09fbf0 3ff40f2c330084b4
3fe3dde070a5f651 3ffb78b8598e2550 3fee9b6f9170c727 40084722737af626
3ffca9bcbe4af76a 3fedbc3a8139af9c 3ffe7a9dbf5142a8 3fdeb7ef193c8fb3
3fb3b1139599a943 400022a562fd6e4c 3f879bec4f2c9ebe 3fa6140340455291
3ff9552dec3ade61 3fd38a014adf6aca 40047c89a931782b 3fe8a03022f784d9
3fec74f62f990bbf bff580f415f3f58c 3fd273b463bf44f9 3ff7f9a289e79611
3fe0997563490cef bfbd073aae097674 40008326ff6b6fa5 3fead2a547a89941
400677db2e3dc5b8 3fef07396fc3793a 3ffaf1941fc08e76 3fcad4a364e76f7f
3fdc2c62c13b2aa0 3ffed75b6fc70de3 3fd238d8eddeb1fd 3ff06054c7b9a1d8
3fe32ba69d4199b6 3ffd71df5e25e6a4 3ff459e95aa7d0fa 3feb04716f172b2d
3fd7101128661c69 3ffcacde8d1caa38 3ff323728dd17a0f bff11f88c38f3f92
3fc3840da0267541 3fe5cba8130c200d bff247bd4cc1da06 3fe1dba9935bc96c
4001739f2a85f681 3fde7fc38867b5ee bfe15c68e4c6d905 3fe6a621276ffcf4
3fee57eaaa3d57c5 3fd7d78328d3ed1a 3ff66341a28ee57d 3fd5f0e958fb508e
bfd5f8570bcf3e5e 40051c5693a53a85 3fec707a4d756902 3fd254d859627ead
40021b4587fcabd7 3fdd19a3abb74285 3fdf065fde691f00 3fed7db8ab107c63
3ffd82294ca28074 3ffafd0afadf82f4 3fb637cf4f6154e1 bff5b30482d548f9
3ffe771a0d0fa5f9 3fbf0c635a78b54d bff05c91e2d7f535 3fd32014be0f83c0
3fc6a4dcd680de99 3fce29ad934ef906 bfe852862ed71768 bfde6687854d270a
3fb09f2577339df4 bfe76470345fe7a8 3fe178a3f1038501 3ffe5c063d15d1e1
3fb253eadaa0aae7 3fd77ec8aa2abf63 3fe42e7e9d164258 3f87b0ed73ff5dc4
3fea6953d228c2d6 3fe42e7e9d164258 bf983d3d81fec8f6 3fdc2c62c13b2aa0
3ff8753d3e8800f9 3fdcc236070df0dd 3fcd4182131ea4a8 40034daadec77838
40035b76d71bd6ae bfe21f6b0a1034bc 3ffc4f23d6a887e4 3fee29a5b76009d4
bfe9526cc6c45802 bfc567ded152512b 3fdfa73b511e6737 bfeada3995057b86
4001eab2feb02a15 bf9e05d3b8b9cb65 3ffcd61837a243dc bfd1eb1e4656555a
bfbccae45bf1f721 40001e1d95c8e3fb 3fea4c41eaec7f3c 4000ead3f96f059d
3fc56020af73b8b4 bfebd1ee8ce39d9e 3ffd8cdbbc3f1143 3ff4c839cf765aa6
bfe426758b5ba26e 400430888891d2cf 3ffdbb47076cfd9a 3fd8f2ac110b6598
3fd382d51bcd9613 bfec88c1a7b40d46 3fd6d0514708aa4e 3fe5b4b976daa425
3fd33d022c0c2da7 3fd39776c75c5c34 bfde1e8f9a20468d 3fad8580569331a3
3fdd93442abcb179 3ff06939041cd06b 3fd5baf2419d91b2 3fd65ba71ce63cea
3fe6b131534d61ab 3fe8a07c714e4137 3fc058ce1b6bf638 400470ea6100f89e
3ffb12641904587a bfef47c27dd68587 3ffbdcdd40a9f394 3fe0bb31275166f3
bfdacb7b4f90095a bfef64ae5be1bf08 3ff02075566bf4ab 3ff68a1bce8a4f96
3fc7a414046e75b9 3fb3836cae2f9327 3fe6ea0474cde949 bfd2b25b9d27ede3
3ffeab9e9d6e29ab 3fecc8132443a80b bfd72f705a9861ff 3ffc15bb2038698c
3fd9c2cfc565a226 3fcff76ba01528fd 3fedb1b4dc47f695 3fcfb9ad1c0f60d8
400318cae6913b16 400088cb92b6f0bf 40001a287e532094 bfd3e30e64734484
3fe2260c9b19e333 3fdec0c77397fa27 3ff7c4f11139fd63 3ff841c61c093d82
3ffba95ba417de92 3f979971179fa5b0 3ff4a3a2cf8a5eaf bfd16b7c667237cf
bfd55c5830d68dd8 bfe81b5214cca7b6 3ffdca464aaf1684 3fc86986bbd63189
3ff214e39eaf95be 4001ef7c2d5f5699 3ff440bbdec458a5 3fe5bc011462cd5b
3fe236339f8fbfc3 3fefe885cf03273f bfe1b69c235763b3 3ffd62375fa6dafe
3ffa919bdec34512 400033beb079a20c 3fda61292788b388 3fcbae6fcbc1840e
bff0899b37482863 bff308b0fd9f246e 3fee4a6cba2eb5f9 bfbd07ebb44c527f
bfe12ddf089a4850 3f9fe44a405a24e9 3ff27301a5c72e7f bfe4bbee1afc6a80
3ff066cac73978db 3ff554ae2b5916cc 3fe5cf8bc07fe1f7 3fe08182ac405e36
4001cc3daa5afa3f 3fc46a165b46d590 40019a57542d22ed 3fd43c8e82e83c51
40013908b8e94215 3ff6e9a803481cf6 bfd833e41bbf1a49 bfca7931eb51632c
3ffc0bb127219459 bfab9cc0fa0d16e3 3fedef8244044e11 bfafdc0cbb13ea5f
3ff9455e889fc100 3ffc2d7614dc9204 3fe47ab8aba02eb6 bfee5911ea8e3c56
3ff8a1d2a2f127a1 3fee9b6f9170c727 3fe9f96ebe69235d 3feef47a3cb8c2fa
3ff5851559e9be3b bfe6a9dfd0ec783e 40006376759b26af bff0ef88bb2f834c
40008d686799a786 3fdf272a68f08043 bfa8ed12a11fda1c 3ffe2cc83da5e48a
3ff13fe216c8f837 3ff68a520ef7dd57 3ff80a89b96c0abf bfef64ae5be1bf08
3ffe7a9dbf5142a8 3ff96e55e44bbd38 3ffcb8337096d1cb bfe9704cb6cf055e
3ffe9beace58d72e 3fd815f65f594cc7 bfd72ec99e63a437 3feb5ecf310ec882
3fd76205de27219d 3ffb8c681d775537 bfeadb9553e5ec3c bff408f780b94676
3fcff76ba01528fd 3fe4bd3d2895f6ad 3ff3590c78014419 3ffc5a3735b26018
3fddb206192ec61e 3ffae2d2ce0d5ce1 3ffc2cf4c6bd8b3c bfe12ddf089a4850
3ff3dc945bc55d57 3ff476782175e644 3ff9f6be2fe0f271 bfe7df12542afb1c
3fd81579eaf24992 3ff81712bba0b8d0 3ffdca464aaf1684 3fdb48b98bb7c35e
3fe5bc011462cd5b 40008d686799a786 3faff7fc960c5185 3fb09c528d8fac46
3fea5d0da3099c60 3fe04179ea807693 3fc76bd50bc226d1 3fda7bf032cf99c6
3fdd82fb00bf7b3d 3ff7d7b804d01482 bfec6892e10b63a5 3fe8e6fff80f9bd3
bfeb6636ce162056 bf6e647cc561e250 3ff190c7c9ce29c5 3ff92590381ab09b
bf820b8ff2a5394a 3ff61ab0fe87cf68 bfbd8791b96d3ca9 3fe3970b5c24e1ce
3ff068a15bc6ae17 3fd142cfbbcb0d57 3fd4f4bacd866e9c bfd4a339f1dc1237
bfdcbe5b56ca03e0 3fcc6ee90d954ab0 3ff8f018b2807827 3ff6801ad73316e6
3fff71a97da93f65 400581822600beee 3ff03348ccf3e804 bfdb1f90d1b59eca
3ffcd1fc011a9e61 3ffa1156efd9f960 3fdf065fde691f00 3fe32a9e1846a42f
3ffee6de2675d6bc 4005ba9acfff1110 bfe7c85759bd4311 3fec44763d46800d
400003ca4edf349b 3fccd89656b5caf6 3fead3e375d84a65 40005963f3794968
3fffc0b28a0a3b05 bff4009a45ee3940 4003f7dc8c970dda bfe1163acb24f60d
40033840f30f9c1a 3fd7d78328d3ed1a 3fe1262d665ce166 40021ddf1981baed
40022f8d0436ee12 bf996ce081ddfbfa 3fdd19a3abb74285 3ff28f22db60e6c5
bfeecd0911aca3b6 3fdc053c2af528dc bfd5f8570bcf3e5e bfccb0445ec918a3
3ff61ab0fe87cf68 3fe4b2ada9c96451 bfeb9919cb6e29e0 bfb0b846c3f2b10d
bff1ff5c87668821 3fda2d5d88038ed6 400796c715633d39 40029ef64ecb2c44
bfd2efdc4e0a6910 3ffae3802cea50bf 3ffe073bbb255eea 3ff30afd96150ab7
bfeadd89ace1f796 3ff44b77be9c6a5a bfc9d87bd612a743 400370a36411992a
3ff0a493269f873e 3ff30afd96150ab7 3fc85cec130d2151 3ff20f1d22f55c9a
40058108ef73afea 3faa84f61fb9c868 3fff2bfbbae3f775 3fe5b0b521bcd016
3ff8b29ec98b7887 3ffbfe8557f2f65a bfe56b10bf73a27e 3fe1478b686bdea3
3fd3fa9b4ecee266 3ff25ed8d243d334 3fe2d8a29e58f64d 3fdcfba124d952bb
4000c26b5a3e3e38 bfe19fe97dcae326 4000fb42994e6092 3fe17000aa0fad0c
3fc1b2b8c2cf5261 3fec6f8b4e349164 4004452f09016d61 3fb7f34e3d3cdaaa
3ff454e16d2489b0 3fe0997563490cef 400088cb92b6f0bf 4001a2f527dcb893
3fbdd2cd7c934db1 3ffc612c53a67711 3fe6c7e30565ad0d 3fd31cee544b5c15
3ff306979a48bacc 3ff1efa0c981e1fb 3ff78e061c0d7b34 3fe7e7a9c723151d
3ff52f5255660324 3fe9ca056724127c 3fec29c7c041d191 3fddee860d49309d
3ff0cf96dccebc96 3fe41fa6b021ba5e 3fbd3aa8e608d7b5 3ff2717aa433bd0a
3fe2ca435165e650 bfe5b4acea89978b bfbf02b8712ca9d1 3fbf4aff00ffa9f9
3ffaaf7edf52c15d 3f95327152c038cd 3fcbae6fcbc1840e 3fff9587b6f01f67
40040bb5c31209e6 bfe1ac758212ce79 3ff73225be115fa5 bfe2a728a0578251
3ff36df408b2df49 3fdc2f9406252836 3fe690aa879da93d 3ffc092dac528a51
3ffdd1375c1271f5 3ff168a1ca7ca20a 3ffc34e5580a6d5f bfd30ca15bc39cee
3ffb08c6ff63963e 3ff81e9e65a3726f 3ffafa9bcac60588 3fd7433d7d6ffb53
40029a5111b42891 bfd3eb4c606674c0 bfe41990133b7b73 bfdfdf51a9e92141
3ff24cea7360a4e8 bfb53e383d9a7ac6 3fc4d6ed1017b036 bfb1e31e69230c8f
3fda74ea3276d577 3fb23a4f06524abb 3ffd7f0e745cded0 3fff2bfbbae3f775
3fb9f425b85980ce 3fe1bfde5344b13b 3ff9b57aaa50315c 3ff3581e6bb47a78
4005d56de701dc76 4002683e18bf18c0 bfdb09511b242563 bfeeaa1a55f46caa
4006e31bc740d00c 3ffe9beace58d72e 3ffc6789c3a90a03 3fb97c8b4389af1f
3ff750ca4b48d0c7 3ffee4f9c9181da0 3fea7da3a63d9e41 3ff40e46837c6824
40029ad9d7ae6c8f 3ff78d322309d280 3ff69344e8fc6839 3ffa9e3274e1b043
4002815c1dfd6052 bfe7c1c4a420739e 3fe2e5375987e721 bfe6ee8cb35cccc1
3febd460b0efec48 4002f6f1f7884ee7 3ff0c7570b34692c 40060dbb34acd56c
40000f2a057acbf5 bfe42d4590e7e270 3fe7e6e20ab16e0a 3ffd570036f07bc6
3fea9a2676370fec bfee09d11acd9904 3fd863bd7d4d51e0 3ff258d4d9d9e214
3fcf4844612f4b39 bfdd4d00eab293d0 bf983d3d81fec8f6 bfb4af10288fc7f1
40029ef64ecb2c44 3ff7afceb2f1600f 400084e755e45ee8 3ff81712bba0b8d0
40058108ef73afea 3ffa06b24315f567 3ffd0632adfc1fcc 40006d17bdd625db
40004a82bda9e19a bfdd4d00eab293d0 3fcf838dc933dfcb bfc03287b2c55432
3fd963e7e8e1c2ab 3ff01d6eecd1f064 3fdfa73b511e6737 bfd8a392c13766b8
3fc4d6ed1017b036 bfe211b6bc02b795 4002f949f278fc02 3fe60c86d3ddf755
bfd75dd7717335cb 400498500f29f137 4004e320847f0e09 3ff01d6eecd1f064
3ffe0bf2991e1da2 bfa55cd143b30c88 bfe929e5b760829e 3fbd86dc1366b2a6
bfee598a148e9f78 3fe4f73a11e7c333 3fe5167f686cef3e 3f9eaa52ca978e36
3feae201d1e7e8e7 3fd873859c906958 3ff89608fba1b63d bfe4d30daed02e6d
bfdb5f3bf519173a bfc8c62ccb5d1b6a 3ff2b5d16d876a95 3fd84996965697b9
bfeeb5efd446b034 3ffa859e66b49f4e 3fe749f109df9a5c 3ffb4cdae5326d6f
4001aead29a078b9 3fddbe51a767ff92 3fc9b56c56ccca06 4000c6a54b7cbe68
3fbcd1ebafe94896 3fe76d34cdc2ed0e 3fe0af03177ba469 3fec856b0bc3385a
3fc7d2edda54dc5f bfabd95f6c412120 bfdff8209aee7486 3ff860fc73e8247e
400089bdd1a84c80 3fbd70b32cfabce9 3fe600bb1f756d8a 3ffbf58043f704d6
bfba2830cc7d340a bfe48f70ba747cba 3ff2cef496f87fea 3ffd25e8a4331610
3fe80389eed03c5d 4004650d2046a4f0 bfdd1de24b11eada 3fe1478b686bdea3
3fe72458951b114c 3ff985a616bd89eb 3fc3c4677d29a201 3fcfbd9846281be3
3ffa26c2cfeca38c bfece44250fb23a1 3fffcaba5728b6f1 400240c283bdb68b
40055aaa2027be45 3fd61d4e598ae9ad bfd967b3e4fb9ff3 3fff71a97da93f65
3fb74152f298b843 4001d8b3d2d3e210 bfc8b628ae3ed18d 3fe894d778226a90
bff408f780b94676 3fe7e7a9c723151d 4001c5037bda1edf 3fee0d9bdc359858
3fe63cef4d557b1f 3fe164808b657275 3fb74152f298b843 bfea65ab9cf08645
3ffaa51845a02a57 bfdef7051c5b5f18 400362d0999977d4 3ffbaa048dcb2edf
3fdf65f2d6da60db bfedd159561a5fe5 3fb23a4f06524abb bfe62539ea69e3c5
bfe5a21d0bb269f4 3ff215e53103f6e0 3ffd7bec6139d063 3fd48acb452a8f82
3fd6f411aa1dc2e6 3f6c91af93678fc6 bfceaa96ef9594c9 bfeebbf4e7bcd648
bfd5525c109ae430 3fd6ae6da0accda9 bfb38f951811e0b4 3ff73bf560b3e3d6
40018ac2f8e7ed2a 3ff18b3adaf15b01 4000c34697a14cd9 3fd19d18eec8611f
3ff696145e8ef4c8 3fc344e53fd8a741 3fe4f67496d36ce3 bfde940cd3e180ea
3fe134537fb742c4 bfe0120cd99e65b5 3fd5fb6cdeb4c18a 3ff247519ed5ff6e
3ff5e015f20f0403 3fdf205fe798019d 3ff58fcb41a6d70f bff2889db0b38a2a
4004c2b97ac76ade 3ffa1156efd9f960 3fdfa8719ac7f088 3ff606e17f3a4a4b
3fdeeb31e970c4fb 3ff31ac669522ed2 3feb3ce4561abbdb 3ffa25285dcb446b
bfe6fd116e7c5e51 40047d7cf82725d1 40019a2ab165ea6e 3ff9ea289497aabb
4006378b538d73ee 3fed6af56ef528a0 3fcd0300345f4f93 bfd30e42472bb71f
3fd3e2c0eb5f60c7 3ff81712bba0b8d0 3ffa06b24315f567 bfe506d0b4ed9f75
3fe1bd8b83ac0205 4000eada921278d1 3ffb41ddba359159 3ff2d44db25a41bd
3ff29cbf55e13e36 3fda8bd7e61b5b8d 3ff0da7c5ee8d739 3fc3ed175dec4979
3ff130117bcc0d8a 3fff000e7612ae1b 3ff7e6ea377625fe 3ff3ba44422d19e2
3feeb3e4ba1ba962 3ff483259af69001 bfd7d469bc1a4245 3fef9048ab8abced
3fd13b8a9ff92cca 4004c21f8c8fd03f 3fec29c7c041d191 3ff163b0e9ae275f
bfd65e104f066c8d bfbccae45bf1f721 3fed6af56ef528a0 bff268134a052f45
3fe638f77d31f57b bfa43b677b512265 3fe5f411e2d021a3 400172c0f2637d1b
bfe948ec44580b30 3ff30fc129b6f98d 3ffabe97a3a8f992 3ff4b2664d1370c0
bff5b30482d548f9 bfddaabc5bd0f56d 40061f3e352b3a5f 3fd3f5e03356ad06
400037fa7ee4f929 bf983d3d81fec8f6 bf94730a2ecc0a6c 40014845a7373930
3ffcdd061b428f48 3fd51893a3c9d8da 3ffa5d29d9d4d060 bfb8a5262672a27a
3fddaf94e4692c9f 3fc22dd6883eca2d 3fea20937748eb69 bff4a9b00c9cacac
3ff6636d3acfd911 3fcfa10714e5ee60 3fecf38bcfccd8c1 3ffae871b9c5f29b
bfe17f1ca1e7654d 40044c62064b74a9 3feeeb294fb2beb7 3ff3f66e5b1ba089
3ff08f78694db091 bff1ff5c87668821 3feb20a981bf18ec 3ff37bd191bec8a6
bfe72fc3f32031bd 3ff068a15bc6ae17 3ffe7a9dbf5142a8 3fd54b2de75a6db9
bfdee35ce2448121 3ffa19b0e08e0ad7 3fc678467bbebfcc 3ff89e816bb13a14
400796c715633d39 3ffc9083ff749154 3ffa0a264fd562a2 bfdf0d5e93f539f2
bfd60e3e29c75e79 40024851177102bf 3ff506ef74717bff 3fce30a342f2fa43
3ff945bebd11cc7d 400088cb92b6f0bf 4006412753db0635 3ffb38b4672a3e4b
bfd2efdc4e0a6910 bfdb186b9c9a67fc 3fe3da37b9818004 3ff1c532738aee2e
3fbf48effbe3694c 3f68b2d514950731 400293ded55330b2 bfe84f3cfbd94528
3fed9abc88cbcdab 3ff263a53525c452 3ff2e41a9617da3f 3fab53a12f34ae26
3fd7b6908355c37b bfdf24f074b68454 bfeca8d386c16db0 3ff0093fd43eb3a9
bff29d077106bb8a bfc28bbd61d10673 3ffee3a93538525d 3ffa0e2a16566e31
3ff26ff4e8cc5303 3fc3f9e56402c9c4 bfcba5a47ad676b3 3fe7754e4367a09e
3fb9c47f4d5a6958 3ff25400c2531dfd 3f94c8ebe9113706 40013e23b720f55b
3ffd109bd5a6ebfb bfe53d3e3b973497 bfcf5fe6880f8996 3fcf2fb7d3e7353a
3ff0b094c8936b0d 3fed85f57fded7a3 3fef95914d819328 3fb411f648c4d473
3fd6580f81fe1ed7 3ffbe4335fe8f454 3fcbb8b188b72524 3fe8a03022f784d9
bfbd38092d0d5e9a bfbccae45bf1f721 3ffdda66c760b13f bf96e6f253e38c08
bfe36d63f292d08a 4001c2e05e027a85 bff43b07ed9a99e3 3fe6f5417b350ea9
3ff4a1ed13ff578e 3f9ba9b1b399b20b 3ffaa51845a02a57 3fd5baf2419d91b2
bfe62f59403cfe3e bfcbe383551f4227 3ff6574cd3e1d22e bfeab3fe61d974f6
3ff6beaef25737a1 4005ba9acfff1110 3fd93303f5ee304c 400631c20751a5e7
3ff08d0167080aa0 bfd55c5830d68dd8 3fb9b674757c8ba1 3fdd4baa1d33313a
3fb9c0d61480aba1 3ffca95dcc3bcc0a 3fd611f68faee686 3fdbfd57f3819a65
3ff824e67700247a 3ffce094cbe68f45 3ff1d5a73086e753 4003d45ec601e3c0
3fc1ea59455f72a6 3fedaeb93335c047 3ff1c654f90d25fd bfd144331c0e45d9
3fe074fb7ff3145e 4004e8eb3c3723c6 3fea63a6522fd4b9 4000b51aa5432cef
3ff4b2664d1370c0 3fe5b4b976daa425 3ffb6a0e239c5a41 3fdf1e30a428cfa8
3fd020a3279021e9 3fc341b92954e686 3fe9eac08f0dae01 3feb65fc6f8f2838
3fdf4ef4286e913e 3fe5987cb90559fd bfd927f5e4d887d7 3fe4e2df884ca47a
3fd2ae0967c4c3a1 bfa5b3d3f905b15a 4003ff4bd9eac3d3 bfe6fd116e7c5e51
bfd0eed59a023038 bfdb448b532d2b70 bfe07ea453919f84 3ff02cfb786c7a63
400172c0f2637d1b 3fcca592bb23d242 3ffc80581f298161 3fe99b61d78f5266
3fcc70f64630dd61 3ffe1e48848a4c64 40024851177102bf bfc81d647172b575
3ffe3ec414a6f59a 3feac570be44cbbc 40017770b6269053 bff247bd4cc1da06
4002a61697a65ade 3feeeb294fb2beb7 bfe58153be8a57ab 4000a33441a2a751
3fe88ba4709d42b1 3ffed9557af2eb61 bf493959b19b553c 3ff6fc8dcfdbb627
3ff2d44db25a41bd 3fb8e895b17b8659 bfcfcd42a6ae936b 3ff0528a82016b0e
bfe21f6b0a1034bc 3fec29c7c041d191 bfd927f5e4d887d7 400200e7a9436293
3ff48d49eb72b390 3fbcd567578002ba 3fe49aa038a094f9 bff580f415f3f58c
3fe9e25d1828ea2f 3ff1d6d7a88581b6 3ff3726fc6a62fb1 3ff95dba0c6b51c7
3fe8cc8bf0e45b07 bfdd4d00eab293d0 3ff01d6eecd1f064 3fe46e4cd192a8bc
3ffe39424d2f92d5 4001b5ec1bee3d4a 3ffa3abdfa1d64c8 3fd41d874ff7773e
4003c332476fdf8e 3ff2a10b7521fc81 3fcf76ede4bf22a3 bfeec5d97b550439
3fe5dbf68cf88ee4 3ff1e9ccbee7185e 3ff8091bf5d53e07 3fed6df62517b078
3ff8005dcd5725b1 3ff624859d435467 3feec874fc94d2d9 4003c8a2c0bbf43c
3fce0cc23b323ae2 bfd0ecf0fa809c1a bff5b30482d548f9 3ff91112df2ecc14
bfc457c462b640e9 4002f949f278fc02 bfe49e44b2025c46 3fec8bb5e2840cb5
3ffb4e311d2ee095 3ffd6cebc7e3a284 bfecdf6c0be2d65c 3fd5baf2419d91b2
4001559043cfe796 3fe87e5e0e52a6ce 3ffbb513ed24730c 3fd0ada9859657a7
3ff90ce4ecbd3e66 3fbb35ff86559ff4 bf7e4341d54329c3 3fffec939bccee8e
3ff21b6bb67f41b3 3ff637dd7cd94219 3fea653a11823f09 bfdfdf51a9e92141
bfbe1e47e80286ac 3fef3fe560a530d9 3fe1f0fe80f26dbf 40015f630af5ed26
bfd72ec99e63a437 bff3d280a676831f 3fec1287b1874b65 4002c7b6dd8aaf33
3fd1463d55205fea 3fe944fecc3730e4 3fe6955d3f74d1d0 3ff8c0de713a8a2b
3ffc937ed0dbbeda 3fe1544ac6ec2d5b 3ff25a8d80f0ce3a 3fd3e76061f4ffea
3ff5c2edcf117a1b 4004e29e24976d2c 3fc064ac1e235c40 3fff2645ee17705a
bfcb591f8d4997dc 3fb90f0b63ab6aa9 bfdef1b634e2505d bfeabb3224c2a0f1
3fe9540236906e7e bfd5f8570bcf3e5e 3feba256ed038a8b 3fb97c8b4389af1f
bfecfb924345daea 3fb74152f298b843 3fe1c7e9bcdb2e67 3fdddc1a39589614
4005ba9acfff1110 3ff233d68671f661 3ffd699ae8949c8b bfb33d273fb993c5
4005520bb84d1d67 3fee87a287376417 3ff24cea7360a4e8 3fe1bc2183c2fe24
3ff470bb3821f04d 3fd5e9d39b7b7596 3fa97b21f06aed0e 3ff2bcc0158ea707
3fe70ed346414175 4002c7b6dd8aaf33 3f96bfb31386abda 3feea46128afea5b
3ffafa9bcac60588 3fe6a43bd6605f04 3ff25400c2531dfd bfd7299bcb2af7f5
4000904da14df77a bfeadbeaaad7d13f 4001d8b3d2d3e210 bfef47c27dd68587
3fe56309a395a977 3fd600732d4eb786 3fea5d0da3099c60 3fc443388d07e9f2
4005fdf9730212c8 bfdefeadebd76d22 3fe652d5a40ca09d 3fff65943549e2f8
4004c9c98b6a5d71 bfdaa7a2c69d8c55 3ffe987fdf010b5c 3ff47a35d0f98bed
3fe2255f88eee5b2 3feaca7beafff988 3fff1a3fa112ad97 3fe2f3c6b4e0cf51
3fe005fdb2feac97 3fd0a9f9625e9e04 4003c1ccda4cae52 400131c7a6a83b1c
3fff5c61f76677ad 40047b5f3d72072c 3fe1a2b5f816bacf bfe30f7f08eff47b
3ff3c39182ae40a2 3ff5c41c39a60f64 3fa7dbfee30226dc 3ff040552ff39092
3fd6f411aa1dc2e6 3fe9c8e86ecf08a5 bfd2a10e7d759839 bfdd07085f7fe545
3ff38cecf9e6bd1c 3fff716522de9eb4 3fb481538d306b14 3ff4b53fb50dfb34
3ff64467d210fb18 3fed55e08a44abb6 3ff5db76843a0314 40009ff45602375a
3fdba588d18228f8 400482ae089f82e2 3ff65214b8926468 3fef01d6f4c3a1a6
3fc566900a6450b6 3ff30f767caa8ae1 bfe99ac49b5720ac 3fe28cb2e3766ee6
3fe9174101ea5ab1 400135409ae0b14b bfc1990f260118f9 3ff381581b20a83a
3fdf4ef4286e913e 3fd7da61b0b16e54 bfd2fba0f445b0bd 3fea3c55d7a960a0
3ffbe30a601237f1 bff2da543394762c 3febb586fe3429c8 3fc757719c578fc7
3ff4096b3555c257 3fff4ffc43ea2a1f 3fe1c7e9bcdb2e67 3fe09c299fe4362c
bfe0f1b0d7a7a36f 40004a5245e9d6d4 3ff173f676e90284 3ff81712bba0b8d0
3feee4b36ee8a458 3fd6e9df4bac790d bfce114c183cd3dc bfde9c37c7edf2b7
4002b51139bc13fa 3feacdb49fd99e4a 400131c7a6a83b1c bfcb539429953d45
3ff3da99f0c390ca bfde573d351de23f 4001b05e9df476a4 400089bdd1a84c80
bfe153a436108e25 400022dc2b3270f9 3ffc7e344d382714 400325b5fa45cd51
3ff448d9f24a03d2 3ff47f3ff8da67a0 3fe6f5417b350ea9 4003bd0fc8c5c220
bfeb55085fae6c0e bfdb448b532d2b70 3ffcbaf27843df5f 3fc31597403ca2d9
3fd86f208ab8ef11 3ff8155606a86d13 bfe46b848aee50d2 3faacfcdb01f0750
3fa5cd348f7cd26f 3fe8a07c714e4137 3ff30907cad4cd73 bfa2097f0b6d8f82
3ffd8cd926cb2d15 bfd6575b4d875b82 3fdfa73b511e6737 bfd0449a06778755
bfe9d09a35432f8c 3ff4d93399263379 bfe2398c8b9e4a04 3fe7c3be00c8fdba
3fb23a4f06524abb bfe581efa70486a6 bfed8c53cdfc7eb4 bfdd911a8d77ce7b
bfd379b49f640aa0 bfd4de6ace6796eb 3fe9e2a2eca2dcc7 3fff58f1bc8daebe
bfedcbc41c4e4805 bf9259dce2e95d6f 3fe19bc257f07cbf bfc6fdc6fd6cdbca
3fed11b3c68a4d3e 3fdba606533cb1df 3fe24eee5bd29068 bff1a1585722d287
3fdfeb9563081ac6 bfbf1717aaa13b48 3ffbbdc4e9305f1c 3ffd94af5d825115
bfca01ad543eb988 400631c20751a5e7 3feb23cd2cbbb6a9 3feb1df3043540d0
3fe88ba4709d42b1 3ffe4fd7627d2bf2 4000e9c4a42b9543 3ffa737192aae2bc
4000b42dc5401d0b 3ff8eb48cf8d77e4 3feae201d1e7e8e7 3fdf065fde691f00
3ff08f78694db091 3fe5a3d0213c68d3 3ff19c53f87e13f0 4000629ea12316d7
3fd5b3a2850cca6e 3ff7ea2123149fde bf98e36515320447 3feeff2a96d68fe1
bfe76f83311d0b3b 3faa84f61fb9c868 3ff540d33b176c0d 3ff6ee95bd66ef10
3fd5440b01ee49d5 bfe648b48aa50dc2 3fe851fc6bf75b93 3feb12f33cf95b17
3f9a22e86aeb0afb 400272aede587e7e 3fffc0b28a0a3b05 3ff070818d0c063a
3ffacdcef03d36f7 4007c3c044c63717 4003b78107219a93 400165e5ba3db3b5
bfd1ad704d41af5d 3fcf9a2c6a0dbdc5 3fca8dea7aa182a3 3fd1bb52c52b2420
bfd18c4359ba7944 bfdb9bfd6061bf15 3ffcdd061b428f48 3fe019a2e8ce09d1
3ff524ccd51f5169 bfd3ec2fcf98abce 3fea9a3ec5723947 3fd27e784491edac
400549181daa334a bfe22502ae8a1290 3ff1feeb428da81e 3ffe53695d442997
bf98e6c431efcbde bfe2e75a0cc101e1 3fe2954a18184c3f 3fe59ae58664966e
bfc0345513ba78b9 3ffda2b9db6aba9e bfd51693631ee6e5 3ff58072b2d4cf83
3fef036bb3c775e9 bfd5fe8e57c80763 3ffeda0eb4ad1404 3fe4274f46dac450
3ff077aa4d6fd211 3fdfd7675c2fe7eb 4000629ea12316d7 4000904da14df77a
bff5b30482d548f9 3fffa17b751ac9fc bfd10244a03b168f 4003a37d5cd3b174
3ffa097bc772c65e 3fff58f1bc8daebe 3f94c8ebe9113706 4004a24fbda6eeba
3ff9078fcccbac9b 3ff34cf4c40c5c7f 3fe2b632395ba3e7 bfe6f9a3493e86a0
bfd81ff4fc127084 bfd6b3f9694137d4 bfe97a979214b6eb bfd274f550000bcc
3fd963f57070d995 3ff454e16d2489b0 3fdf4ef4286e913e 3fc886f512580560
bfe2dcfca7439c55 40047d7cf82725d1 3ffd48d2fa65c0fb bfd3ce5ee1a9e6b4
3fcf0d31321de47a bfd0923a16ef96d5 bfe0f1b0d7a7a36f 3ff8b198d6c315fc
3fd252f0a65feadc 3ff4320a2948c62e 3fdc19955dd2f7a4 bf9e324709f602dd
3fffae5fc18edc10 3fffb755d2a24245 3ff8155606a86d13 bfce510cf8f275a9
3ffb40ee975f5ca7 3fddbd1b62b444c7 bfbe7654f9eb4b23 40033079a4b1aa9e
3ff9fa352697f186 3ff621a6cb5868e7 3ff1c872d43f9fdc 3fc364dbdb026106
3fc83c389259094d 3ffca2ae91716e82 bfd2e9e4b1b7e6fb 3fbd8dd25b960c25
3fb5cb6d14ed4199 3fe0f0c0c18cd10d 3ffb2f74dd767f0f bfe815ec5b151742
3fd8193d1437384c 3fd888e49a5eac53 40027b8260d91fce bfc5eb4dd25a1dd4
3ff47cc777683db3 3ff696145e8ef4c8 3ff4099d43158ec5 3ff1dda7fa30d8ad
3ff81ed07eb17c1f bfecfb924345daea 3ffc1aa0139dc2be 3fc5dec2ffd34d82
3ff0cf4c6ce44525 3ff3d189c9ed9cb3 400269f470d1eb43 3ff1312e12e58d46
bfd679ca9a06d69e 3fb0201088e05b38 bfe4524ef2f74633 bfe458fc359b0980
bfed8c53cdfc7eb4 3ff2416e49a29074 3ffdbd356865a5fb 3fb1038e29cbdc9b
bfd55b7fd615f046 40009318aac1c98a 3ff00508739b956a 3fd9b9c983a239e9
bfc096dd6f5ce8a9 3fddc4da4a240c57 3ff0340091b0ae50 40047fe74cc8d3b5
40044e7fc100934e 3fefa7b0ef421ce5 bfa8ed12a11fda1c 3ffba7c7a54652c7
3ff5400c7a8411dc 3fe849cef5632b6b 3fe5ba3e226b7136 3fec856b0bc3385a
3ff90669384387e1 bfd72ec99e63a437 3fe91ec97d6f554a 3ff6bf8f7df22f5e
3fdcb2c06a43bba5 3ff4096b3555c257 3ff75196a47c4620 3fff9b31d2133d25
bf77e891a0c979f0 bfe951e5c6d03e10 bfd833e41bbf1a49 bff132cb106a30cb
3fd54cab543cf688 3ffe39dd508eab6a 3fe0bc27643fdead 3fca7633f7449816
3ff71af5700e08b3 3fef18963c52021f bfd8219ce424cdf1 3ff5a6265cd8da94
bfe36d63f292d08a 3fcb2603d468fd9b 3fdf1e30a428cfa8 3ffb4abb296dca06
3ffd77a0f7e9491b 400043720ff277c3 3ff03f5bc10197be 3ff8b6643a860a39
3fd07b1a35a0a2f2 bfa45e9646039162 bfd833e41bbf1a49 bff4a9b00c9cacac
3ff48d49eb72b390 3ff5d1d06cafab12 3fa46fd70e46c648 3fe98b540dbdffe1
bfe1ac758212ce79 3ff01e68b17c293d 3ff361926a411178 bfd6ec349ca3718c
3ffe519afe3ab048 40001206deebbb21 bfde0ebf5758031e 3ff3aa9563042280
bfcdacdf74aaa9a3 3fd0d8693bce6f3b bfe87cf2fabbdaf2 3fd44f6c941fee52
bfd4e824973d4bef 3fed11b3c68a4d3e 3ff345ecd4c93bd0 3fdfdd5ffff60ec3
3fc8719027de01cc 3ffd4d6ea1232825 3fe1478b686bdea3 3ff73d8dabc48157
3ff1d29f872b41a2 3ff2d26d4c91d318 bfdcdf4cdcde5b84 3fc2438806c0cacb
4002bc554c929a1c 4003c1e574e8abfd bfd09300f437afe9 3ff554ae2b5916cc
bfe2b3bbd4ced359 400354be17f8228c 3fe9553023b2b19e 3fe94dc4adf7edae
bfeec5d97b550439 bfeadbeaaad7d13f 3fe6f25fb11e6473 400354be17f8228c
bfe08dec8cb3494c 3fe92d4eaff96973 bf983d3d81fec8f6 3fc27f6d22d10e1b
3fd9575f19e052c9 3fe5eb8f8790f458 3ff1aa08fe982f16 bfd5e5d443c3bcf3
3ffa6aa1e551d1f9 bfe0f73a026905aa 3fc820dda00b0ebc 3fe53910c1a49bc8
3ffd6f97affd3a64 3ff0f8a0d09bb0f0 3fc6a4dcd680de99 3fdc3a4c0cacc0e1
bff4a9b00c9cacac 3fd8d3a5ef9d2634 bf9529c3fcb0d3e0 3fe1f3fefd64eab1
3ff2d89c8442d5fd bfa35b17320a33a2 4002bd71aeaec284 3ff2be6e18d39e77
3ff649e141174491 3f9f5ab0312d343d 4004f44c8fb5b7ad 4003f14f1383cf26
4001efd8c816e1f8 3fbddfbf82991e39 3feefe3b94dbeb6c 40034b190c3f6e42
3ff8155606a86d13 4004847692702f33 4007a190b00d5ba4 3ff68c94db24986d
3ffb440cd58e47f4 3fe6c01e1b3bf991 3ffcf1e9600228f2 3ff1ef7f6e0443a5
3fe5cba8130c200d bff18b352ffcc57f 4001b0674426949b 3fd968947137b022
4003c1ccda4cae52 bfce8333a62311b1 3fe4f67496d36ce3 3fe79ebf42d93027
3ffc9da0f8b3d51c 3ffc420ca256d990 bfcb6e9922617bae 3fc2438806c0cacb
40008d686799a786 bfecccb3bace0a7f bfe17acb9e943729 3fe7c4afcf97fc2c
bfd48e706e6c02dc 3ffe632a3cdc9ec2 3fbfba0d59e333e7 3fe495c4b991ee9d
3ff247519ed5ff6e 3ff965562f280a82 bfec88c1a7b40d46 bfe68498783828b8
3fd84a8e082eb0d6 bff2d6a090bdd101 3ffdca1db35145c1 bff190cf40ade665
4003c1ccda4cae52 4001fed91d39b86d 3fe0f83255f72c1d bf82382b521ea934
3ff3e7751101af55 3fedd42605667bba bfb15f2d113fb680 3ff0a47b4a0cbec8
4003e915e40e1db2 3fe49f0f065a75ba 3fdf4651002ebff0 bfc302f4cbb85cec
3fe775b2e046766d 3ff4db3325d07e5c 3fe1ed814bf36fde bfdff9faf58e302a
bfe45749ce8bfeda 3ff83cc1f5a265d9 3fddd3134a3d6bc4 3fefaf1c52687930
3fdfaf9b075b2f6f 3fae40d63d6af0b3 3ff06939041cd06b 4002023140a69f8c
bf9bff48772a160b 3ff35819f73650c9 40006a88cb6ac933 4004ae7f72d70c21
3ff00969c0632f9f 3ff1c929736c0455 3fe7860da5dc0b3f 3fdd2eaa9f0dd389
3ff4d93399263379 3fe381cc5feca3d6 bfe1627da133fd12 3fecba0a85cb146e
4003e9e52c0fb816 3ffdf185f2c318be 3fe5b2692ea8e187 3ff9428026d93c93
3fd1450f51c497ac bfe6f9a3493e86a0 bff3d280a676831f 3fe437439646814e
3fe56309a395a977 bfe7c85759bd4311 3ff142ba8e924f26 bff2789c607a1b67
3fed09d985a87425 3fdfa73b511e6737 bfd39ac2a8ac0e18 bfe95e4a0e403f98
3fdb48b98bb7c35e 3ff025e74586bdcc 3fe8193aa65ff9d3 3feaca94a266687e
"""

comptime _LARGE_PRED = """
3ff4d0c9b40fccae 3ff1d3f0f9cd3c1b 3fe0840810436778 3fe5130f377e7d63
3ff04d8a433336be 3ff015e85418d5de 3fe0ac16eccfb6c2 3fc94a087e17e2b8
3fc82f35e18064cb bff0234f44bae28c 4000624daf22be56 3ff1defbc338de10
bfb43ee803ec5090 bfe12ddf089a4850 3febabad97cb4dfb bfcb36db202e0672
3fdb67bc863c2e07 3ff8bde67777f514 3ff5c6c1133688bd 3ff488d1f2da9408
3fd60cbebb87f0fa 3ff1cd48a2861364 3feee4b36ee8a458 3fee130197ad52ff
3ff73384517680fb 3ffd4d6ea1232825 3fff9f710314fd2f bfdb224f06b20602
3fce025a63f7acbe bfd8308f6ff0ed5f 3fecce89050dc26a 3ff194477cb1aad9
3feb523a75295378 bf983d3d81fec8f6 bfe6523423e2e8fe bf918b1cd8cd05ec
3fef6c7322225744 3feed6dd51d0f181 3ffac00a7d7e4b8b 3ff21e2a45f7e6d4
3ff4b2ff743c2271 3ff91c2b684db4ca 4005b7bb31535a1c bff53504882818b9
3ff7dd270653d8f0 3fc7ec9128c96636 3ff01d6eecd1f064 4001d8b3d2d3e210
bfe024c8a6821372 bfd949bb5febd9c4 3ff6befdecdaddc0 3ff27ef5eccf11ed
3fd1f486ad381053 3ff0a493269f873e 40037cb4fee3e08e 4001ed5343dad88a
3fe56309a395a977 bfdb1f90d1b59eca 3fdf4ef4286e913e bfde768d69375dde
3fe4b2e294aadad6 bfde79e4793476e4 3fea1393a3d5e840 3fe09dd58abd0b22
3f853ddbac29a108 3ff0cb6bc68315b2 40029ad9d7ae6c8f 3ff1005d1fb71d69
3ff80edcc0960039 4003bd0fc8c5c220 3fd39629c85364be 3ffddb03111acbca
40002ff96da1e7bd 3ffe618b012e4819 bff21e59fd931c94 3ff5aa8e1a6dd614
3fece6a7acff1419 bfe07ea453919f84 3ff1ed8cf740cfbc 3fe88ba4709d42b1
3fff41d6186b6fbe 3ff553e1fcc0dc13 bff5b30482d548f9 4004c2a1ec77711c
bfcdc317e3ade285 3fe55da541d6ce35 4007cefd04be2567 bfdd4d00eab293d0
3ff4e241b34d8e4f 3ff6dbf169339b9a bfd959af95781982 bfec88c1a7b40d46
400165e5ba3db3b5 3fe2f2f3d91bea8d 3fffa761e8ae709a bfe8a816ac63edd5
4005c3dc03ca7234 3feb91c58eab6358 3fde1a7c39672320 3f6e17ee71565c39
3fe8a655ea142193 3ffa05e2f01f988b 3fea6953d228c2d6 3ff8fc61a815c6cb
3fffcaba5728b6f1 3ff454e16d2489b0 3ffa3ba4c34e24f9 3feae201d1e7e8e7
4004c2a1ec77711c 4002da43dce3815a 40029ef64ecb2c44 3f879bec4f2c9ebe
40028c0d28eb0333 bfc12db1526a4a14 bfd48e706e6c02dc 4002299f4099df80
3fd117001c13a715 3fe449fe7635c3b9 3ffb3e700b8cdc23 bfe4571a70c9f97e
3feb59965b7b04f3 3ff3469dfdbec0b0 40033079a4b1aa9e 3fea9d6a3b261821
3ff788914252fe39 3fd382d51bcd9613 bf9561f576bdda4e 3ffdad73c732b509
bfaa4bcdb9ba81cf 3ffd276bdd214742 40034b190c3f6e42 40016c395b3d1318
3fefe4a6a57121b2 3ff4e955c11035ea 3ffa4f76285b70f6 4000600cc45f801a
40034b190c3f6e42 bfecaf03b816f5d8 3ff3c39182ae40a2 bfc22ecf7770ab55
bfc9f4e498a6fe5a 3fbc76cff7e8ff22 4001d8b3d2d3e210 3ff910ba32c1bd20
bfa0249cdeb19631 3ff010ebf3e57d9b 3ffe0faa302512e8 bf90412561a5bdd0
3fe68d19a26af196 3ff07232967f090d 3fe64667b863a228 3feb1f6eba616712
bfe0101880a25a5b 3fe178a3f1038501 bfe537569b336969 bfdf5a092c732c23
3fdddc1a39589614 bfdfb58fe94ec87e 3fe8d4ce23f0dac5 bfb871093bd4581d
bfd27766acc4e94d 3ffa3a41a636a8fc 40010a32583a1ca6 40031f934238aeac
3fbd86dc1366b2a6 3ffc7397df6d2c11 3ff7bd1880ef88ca 3ff9ce747de75f47
3ffe4ecc6b572fcc 40037ace7d6a1468 3fd9c4ab483f585b 3fff9b54b97c0de2
40006b84556cfeb3 3ff5ed05b801f9aa 3ffc567b8efb00ca 4006ebc42d3dfbb4
4005010831278566 3ff81367c9f6e32c bfea84e8d5b55c2c 3fb336e10d86eec9
3fe80fa8f3bf95b7 bfcb539429953d45 3fef35c1e61819a4 bfe62539ea69e3c5
3fee2aeeb9d9f020 400041ccb8d3d737 4002611733ace0e5 3fd364fc742288cd
bfc4dd560eef9b5e 3ff164e7b8f159dd 4000742daa6f16a4 400824f2dec21ab3
3ff7d7b804d01482 3ff0535aaedc0d99 3fd9747cc897ad9e 40008ced0e7f0955
bff3aaf3507590dc 3fdcb4c6e65254c3 3ff6a71bdbf4de52 bfd72f705a9861ff
bfea2b7b0d6a6cb2 bfeb147f70d58396 3fbc76cff7e8ff22 3ff425ccc8f26d18
3ff56f3654046477 3fd0bf2b58b7ab22 bfc6fdc6fd6cdbca 3ffc43def5abafb3
3ffbe344b61df6f5 3feffc9899b54f00 3ffcbaf27843df5f 3febe7b5b9d5b45d
3fe5b1172c751431 3fecb23319161d23 3feb1a29092fb1bd 3fe456e35712506e
3feb12f33cf95b17 3fe7ad2314dbffc5 4006fb3d5cf284c7 3fcb88203490af72
3fe9540236906e7e 3fbe3e305141d0d1 3ffbdb4fa156d6d6 3ffcf9bfbbd595f6
bfc4b50c4ed7c341 3ffa288854de79d2 400089bdd1a84c80 3fe72b1e3149ccd5
bfe3793e6627b62d 400278bbd910cad5 bfd975ed39a847f2 3ffda3b7c5c68ca1
3fdbf851e6d35312 3fefc2162a3617ee 3ffb58bb1b654e05 3fd0af717f1bf177
bfcb3502268390f1 bff18b352ffcc57f bfe03dd4d04e5340 3fe7620e37aadef1
3fcbae6fcbc1840e 3ffd0632adfc1fcc 3ff2cb66351dd1aa 3ff853af63f001f6
bfe03d22adc74ef6 3ff36772b58c552d bff3425e7a4abe19 3fb8de2c928c2731
bfe12ddf089a4850 3fdfae55b2b69a43 3fdd2eaa9f0dd389 bfcdc6a7b58eaf23
3ffb96de7e6a2a95 3ff1009d0dda7091 3ffbc7155efb8182 3fcbfc1adbdf17d0
bfcb3502268390f1 3fe7c4afcf97fc2c bfa59b2b13e262eb 400064abaa5c6409
bfc19583e5d86770 3ff29b91e75ee398 3fe6b4d062ae0879 3ffd6cebc7e3a284
3fda7ff9139bd628 3fed15f1fd5188c9 40004353702e1105 bfbbded6e7d75b7a
bff18b352ffcc57f 400001ced020096d 3fed6ec4bfdf7b8a 3fe5c7a35af1bf98
3fe3101d4b8c8243 3fd7703538ab2e7c 40053738a14a5201 bfe6a9dfd0ec783e
3feb9fb3d75aab66 3ffaad6f9ea1a7a9 3ff9ff54ce1d3c52 4000e9c4a42b9543
3fe5a3d0213c68d3 3ff94d195e115f72 3feec41b61ea296f 3ff3a452832185cb
bff580f415f3f58c 3ff8bc07e0c8a71f 3fe4f3cd67761d80 bfd4c460c0a2c8e8
4000ead3f96f059d bfbe3824033a2faa 3ff2b5d16d876a95 bfe7018cc8cc7ef9
3ffece1663244d90 3ff217c7bfc993f2 bfee300bbde52875 bfe80044bbd3af05
4003cf0fa66eaf05 bff14715f43bd3b4 3fcd4368bd1443ad 3fe5184c78990f02
3ffc87f07b6a7220 bfe84f3cfbd94528 3fe98fa39e3a1b53 3fec74f62f990bbf
3fe2e5375987e721 3ff15ff0c9da2259 3ff2cef496f87fea bfe1b69c235763b3
3ffe83229fb588e5 3fe2aeb77129be5a 3ff7c0e2543fc73e 4004452f09016d61
3ff1cf883d9992fd bfe16aa5953472f5 3fe3388d176d898a 3fe63e0c6e4a957a
bfe9704cb6cf055e 3fffc554c7d71ee5 3ffca67de8f4e3ed 3fe677a7a6c92453
3fd0b0bcf4ebc8c1 400496fe22c7dfa1 bfcf136a730819ec bfc624f78fff0acf
3fe4180f7520437d 3fefc19ca51c9ef7 bfe2208c528b4852 3fd6339795cb23b1
bfe6f414734de00b bfdee46da7d4b254 3fe85200168adcd3 3ff2cef496f87fea
3fe3970b5c24e1ce 3fea7db997acb7ed 3ff1e9ccbee7185e 3ff9e6c8640a9581
40058108ef73afea bfc48fd87507fc99 3ffcd3e71669f430 3f94ce70d6562d4d
3fff8be39fefe4fc 3fc2c9ce0694050d 3ff3a4ef5a911c4a 3fe6e8c9f28f76a0
4003213de43080d8 4001e51969fcf4f6 bfe5b4acea89978b 3fd1f4691f39cc3b
3fe690aa879da93d 40067823bbeda75c 3fdf3c0d627d9b8c bfc554906ce44848
3fd61d4e598ae9ad 3fe3d3fecb050a47 3fecb1c70d922960 bfd175a499fde504
3fe94ed6c8d1bbb8 3fa6bf2217d108a8 3fbb688190129b2d bff247bd4cc1da06
bfbcfec1a4a0663d 3fe5a5db6f435b03 4000928bb07d99bf 4001aa2160d9afb7
3fe2ebdb4ee6d2d3 3fea7dd77abcfc21 400205ec566a62c1 bfd4e824973d4bef
3ff73384517680fb bfe7155a02eecc82 3ff0cf4c6ce44525 3fd5d711bac29f0d
3ff102206215f098 bfd32057048b6119 3ff6459f1c8972f1 3fecdbf1476db1e0
40033079a4b1aa9e bfe11c4506619fa8 3ff20664d169386e 40015a767545de54
3ff03829038cb5da 4002f949f278fc02 3ffbcd99956ef8c9 4002815c1dfd6052
40061f3e352b3a5f 40058b688959d3bc 3ff9746943398640 3fae40d63d6af0b3
3ff1f51d63d20f8e 3ffb3a5353d8f755 bfdda18c2a1bc750 3fdd9c18ae499bc3
3ff4b5841ac04ad4 3fc341b92954e686 bfc443c7ce7cd69f 3ff4c7f0f324f8ce
3fec1287b1874b65 3fe91ec97d6f554a bfb45a745cecbcf8 3ff56019e27b9714
3fb74152f298b843 3ffbad132c9f81c2 3ff0e59cd1c21c53 3fef1411a7db0df2
4005e52a77ccc515 3ff737c173bf0bd6 4002f949f278fc02 bfc17e30a7d034b9
3fe2c2531b0ebd2d 3ff59cc79ec8af19 3fc3736783da2014 4000b42dc5401d0b
bfc20c1cf1a5d315 bfbd073aae097674 3fdb48b98bb7c35e 3ffb12641904587a
3fc2cd5b53fc5473 bff3d280a676831f 3fefd7e2430bb3bd 4000d0d3d91243b1
3fced67fc8692ae3 3ffb4e311d2ee095 3fda4e5a2f2f63c6 bfe853297820c2d8
3ffd11047f06c9d9 3fa7dbfee30226dc 3feab287d0adb6ba 3ff7a0ee24a8f186
3ffc6c6e6472c700 3fd63d0a5148b33e 3fe677a7a6c92453 3fe00bdc990c943f
3ff454e16d2489b0 3ff86056cf171db0 3ff7e9fa2015bc71 4005db7a20eb7171
3fea3be18017d85d 3ffd5b5677e34e3c bfd1e81fca328632 bfdb00bd2be3667c
3fdef2ec487226e1 3fd3455b14cbee12 40037ef0ed5410b3 3fbb2316167f3664
3ff43ff2fa9d10af 3fe03ff2b432ef77 3ff65c62ac58e68f 400631c20751a5e7
4006fb3d5cf284c7 3fd6f8349c2c36f6 3fb8b6e24c6f44a1 3fee284247e73263
3ff87f759d67d78a bfd30ca15bc39cee bfbd073aae097674 3fddee860d49309d
3fd3e8996b8a8f09 3fced012708f9890 3fe3afb46af34b3f 3ff8cb1802c9d362
bfe3bb85932c4bfd 3ffc2d7614dc9204 3ff47a35d0f98bed 40017547251688b0
3febb469bca6c741 3fc5962c2cb9e4e4 3fed6f50d48f49db bfdce2e3f3c43aaa
400235ecc4b25e1a 3ffb440cd58e47f4 bff580f415f3f58c bfe0720f4d6a5934
3fdeee5c699b2ecb 3fe8861192a3badd 3fdaff3a39b0cda0 3fdca95aa31f352f
3ff121ece812bbf0 4001a2f527dcb893 3ffeab9e9d6e29ab bfd16b7c667237cf
bfd977b06fbcd796 3fe39da142fd24ac 3fd6842dbd539002 3fe6a6b746700838
3fef0bee1f29ee31 bfaaa88167b0b7ed 3ff2787bfd579863 3fe8cc53b74f6fd6
3ff36bd75b29b981 bfe42d4590e7e270 3ff48f8f252cc517 3fe09dd58abd0b22
bfe0d4af4bc20a33 3ff10b2f63283347 bfa5dfce2e04193e 3fc49fc4746e2c58
3ff9033332916084 3ff06939041cd06b 4001ed5343dad88a 3fe4a0c4b3f6ecf4
3fdccae51ec51eca bfe4f228ec3efb6d bfd46fb0c74110bd 3ff675a37604dac3
bf9fd618e1412c5d 3ff42c554e2b0c24 3fedbcb82c120c15 bfc326144ef7bc4d
3fe425923dfd0568 3fecba0a85cb146e 3ff2d93ac7adc2ae bfe0101880a25a5b
3ffd25e8a4331610 3fe0eb35a387e9e9 bfe31e3a39c6e79a 3fea38661ca8ada8
bfd542c53cdb5514 3fe5bcf61048c23a 40019526d63462be 3fe42262a84e8183
3ff1e90580e6b583 3fed9176c0d64d48 3ffc83e3221cc32d 3ff3b3d88d122b3b
3fb5cb6d14ed4199 bff1396ffce05efb 3fd5baf2419d91b2 3fd0add720c05b51
3fb23a4f06524abb 3ff436fd8e8ea1c3 bff29a23b6e682b2 bff29d077106bb8a
bfe0a5049d50bdd3 3fe9511386519c11 3fac729cd6049050 3fe652d5a40ca09d
3fec673126b4fb5a 3fa6bf2217d108a8 3fffb90e482f9dae 3fd3c1c1737844ad
bff3425e7a4abe19 bff4a9b00c9cacac 3ff28f22db60e6c5 3ff01d6eecd1f064
3fec1e1fcfc5db08 3ffa445b5555a4c0 bfcceb3f99e63e8b 3fd76a1d10c0d8ae
3feb23cd2cbbb6a9 3fb260affa233702 3fb77085ae1df27d bfbccae45bf1f721
3ff0ab022d74fb71 bfeadb9553e5ec3c 3ff57c959eb5eebe 4003097ceb579361
400196fa5f520748 3ff8005dcd5725b1 bfc6c5b2eb1b2018 bfe7c1c4a420739e
3fc4ccf63bc6971e 3ff27a4b27707a63 3f9a6496c7f9dfb2 3ff0cf1034d6dafc
3fdbc4fc2d311713 3ff627317d808bd4 3fe5e99259d9aafd bfe08ce869a7b4ee
3fe09b413a740769 bff2d6a090bdd101 3ff86ba8dadef697 3ff587c4393c76a1
bff45dc07ed0cfd9 3fd3f5e03356ad06 3fea235286b9af97 3ffa06b24315f567
4002815c1dfd6052 bfb0df910a0f939c 3ff15df4cba0f223 3fb2dfc3fba4065e
400176fc1e5d0254 3fdae17cabd02f07 3fec6437e5af56be bfd19a6094cb5ea5
3ff0453116611488 bfc56161b632ef6d 3fd3826b51514267 3f5f1aa202d5768f
3fc3f9e56402c9c4 bfd0ecf0fa809c1a 3fccc90afdea0f5a 3feb7d819c413988
3fe5167f686cef3e bfd44eb373c1e787 bfda54bda3056d96 bf310b60afcf1dc3
3fe3eefe7a6cfb5f 3fedf412723e9443 bfdfb58fe94ec87e 3ff72e0a0829d846
4004a24fbda6eeba 3ff15075d84478b9 3fdfee34d0f0f4d7 3fd873455963a041
3fd562ab18edbbb9 3ff98da7f637aab4 40053738a14a5201 3fe4c5ad7f860525
3ffa811f11ea7c66 3fe47ab8aba02eb6 3feba55931acece4 3fdc2f9406252836
3ff55346f4908e9e 3ff303d6db2c8262 bfebf95bdbddda32 3ff55fbeb8547d2e
3ff69344e8fc6839 3fe495c4b991ee9d bfdc3a0c8b5a60fc 3fc53aca75ca8b60
bfe81b5214cca7b6 3ffb2f74dd767f0f 3ff7e0a97a1db0e6 3fdccbb0f068eedb
3fe4a0c4b3f6ecf4 40022f8d0436ee12 3fd827329dcc0328 3fe2c52936406dbc
3fd750f0be8314af 3fefe0dc6c76015b 3ff17282bf8842da 3ff9247065c038ff
3ff0ab152fb562d3 3ffe9ff7572ef2e1 bfb167b9b838084c 4002cab704009ebd
3fea7db997acb7ed 3ff1c87bc1983edf bfdc5d3e2fec0c1d 3ffec0fdb73856f3
bfc40e9dde9931af 3ffc1e4e0d0a3a09 3fd600732d4eb786 3ff8005dcd5725b1
3fec498bb7feb2a8 bfbe1e47e80286ac 3fe9e2a2eca2dcc7 bfd09300f437afe9
4004e320847f0e09 3fe87abfba20e2cd 3fe2cda9d804e09e 3fa6dedda75ec30e
4002b51a60156abb bfe0cd3f7cb34de4 3feec07d0db8656e bfa4388365bb147d
3ff0975e135fe112 3fd2ad71cd061395 3ff5e9c0ef74a8b4 bfef48f030f20e8b
3fffcaba5728b6f1 bfe1050a6017e694 3ff8709012e7d5f4 bfdd71da06bcc6a9
3fed9176c0d64d48 3fb66eb09ac2fd61 3fed48d8debd8a4f 3fff09129695ff29
bfe139d7bd37e0a7 3fdf74bc9ab0dba7 3fdf8e95429fda7f 3ffb4e311d2ee095
3fe1478b686bdea3 bfe1aa986025cc02 3f92b5438a738052 3fee583cbae46b11
3feb03be2974f51f 3fc886f512580560 3fe8eef2897404af 400218fc8d232315
bff4a9b00c9cacac 3ff2dc3ff29114a5 3fe02acadf1afcf0 3fd5baf2419d91b2
4006b32fcca34fb7 3ff41fd87e63dd7a 3fc794551dfaf6a2 3fed9176c0d64d48
3fd7545e57d86742 3fd3473809fc42e2 3fe3dd2551aa2a93 3ff24f970e0d36b6
bfccf9e2ade8b062 bfa9e0de99987dd2 3ff31caf8aff054f 3ffd7a587cd79a80
3fb8de2c928c2731 3fe918cb7d5fcf31 3fd6a08efcd34749 400365448f1485a7
bff5b30482d548f9 3fe2255f88eee5b2 bfdf8448df8d2851 3ff784d6fa1eefc9
3ff4c6f66083bf42 bfd72ec99e63a437 bfe098d59b5f75bd 4000f7d6366ef22c
3ff73bf560b3e3d6 bfcc1a02502f08c6 3fd873455963a041 bfc99eca19ddd549
bfd2e0b01f991145 3ffe39424d2f92d5 3fe4a8ade646cbac 3ff006d79db4776a
3fdae647f20c25b8 3ffaf14f1281cfa4 bfe07c0dbb34dbfe 3fea15fecd9fa9cc
40030aedc9cec4a3 3ff0412aeb1a79be 3ff264f088b2256a 3ff9fb8d7d071d1a
3fe4423d56707e42 3ff0937087cae9bd 40001847c0757363 3ff33fb44a45b9cd
3fe3f554feb811dc 3fe86fd99e24a653 3fe98460cbad3f9c bff29a23b6e682b2
40012f31380ab038 bfa258da92d328d6 3fffa761e8ae709a 3ffe08211b31ef94
3ffb440cd58e47f4 bfe42d4590e7e270 3fce0b0551a613da 3fdbbfecce0b559d
3ff07281c201a9d6 bfdb802a2eea4337 3fd86f208ab8ef11 3ffe9b72b778e299
3ff5addeeac3648d 3ff179ce3152f778 40015b8bd940b12e 4000c9e1b4656b3b
3fddbd1b62b444c7 40004092292b62e6 3fd8ae1f8d17b40a 3ff4e0e5d8b96f87
3ff2787bfd579863 3fdcd7bc079d20dc 3ff0a493269f873e 3ff60860b55a5ec6
3fdbc008043284f1 3fed1279062a4ba5 3ffc43def5abafb3 bfe852862ed71768
3fd43165ca8cf419 3ff79340621ce38c 3ff6d3c4113e4e5e 400003ca4edf349b
3fc5ad24a1ebc751 3ffaaf2d8a3275c7 3ff4bd5a1fdf5253 3fa6e659312df969
bfe84ccfc496563c 3fee39eacd1f15f1 3fd80da393857a4e 3fea65319d80e7a0
bfd379b49f640aa0 3ff108e7e66dd1d9 3fe70ed346414175 bfd4188fc5b56c82
3fc98fd0be6fe625 3fd051f3623bb03d 4001e57a20624036 bfe800413967712b
3fdb3fb4f626fb6a 4005010831278566 3fea915f55ac35ee 3ff91112df2ecc14
bff4a9b00c9cacac bfaeffbf3e8aa7d2 3ff14a95f4372be8 bfe30f7f08eff47b
bff18b352ffcc57f 3ff6bf73e2d6ccdd bfe7c85759bd4311 3ff294f06a853f2a
3fe42e3ce54aff00 3fffe39ce03ce5f0 3ffa13e5cff96be5 3fd87435ded69d7a
3fcb515aa51fafda 4003058c4cff0973 bfdfdf51a9e92141 3fb9af0a3393da43
bff1d74f9a6dc078 3fcd2458442708a1 bf6e366449f2a54c bfe2b3c52596f4fd
3fed8b0d2b578d7f 3ff93b160eb22c5a 3fd72a0312f887a6 3ffe274a1b4209d7
3fe1ae7782b2ae4c bff5b30482d548f9 bff29d386cdb84a4 3ff48f8f252cc517
bfc9e2e2641f189d 3ff3273241088154 bfc8faad085861de bfca7931eb51632c
bfa55cd143b30c88 3fdf205fe798019d 3ff4332031106e6a 400433aca9fdc7e8
3fd7275f5efebd1b bfe99ac49b5720ac bfe4f228ec3efb6d 3fc344e53fd8a741
bf9c61b2332d3cce 3ffe30f64fd6205f bf4cca04a6a4b1d0 3ffa9e3274e1b043
bfc5b8621641042d 3ffada703426f0b3 40007020c1d23522 3fdb48b98bb7c35e
3fed707d739872e6 3feccae3f1723644 3fefe500d31994c2 3ff9abde7e2ece67
3ff3ff66e7fef774 4003c8a2c0bbf43c bfdfb58fe94ec87e 3fea7db997acb7ed
3ff06939041cd06b 3fdebe8703ab4d27 3ffc43def5abafb3 3fe09c299fe4362c
bfb98b3c61bf4d3e 3ff9552dec3ade61 bfe99ac49b5720ac 3febc50745fd1f08
3fe46e4cd192a8bc 3feab03486ef8aa4 4005863a43155b98 40002828af253072
3ff161b9c8ece3eb 3ff2a5ebf51fe4f7 3fe0bc27643fdead 4001a99d63901957
3ff470bb3821f04d 40015c3dfaeca6c9 3fcce48209086f33 3ffb39d5056852c1
3fe3388d176d898a 3fd7545e57d86742 40031f934238aeac bfe15c68e4c6d905
3fe11e687c332dea 3ff6916d54ed9c62 3fe72a7726087105 3ff59ab6079859a9
40024dc680723bdf 3fd5a8cbc9884e19 3ff56019e27b9714 3ffe4ecc6b572fcc
3fe104eb8fb7488a bfaec59d0008b619 3fd3f75472c39931 4005ba9acfff1110
bfef4ff6901d8b95 bfbf995b5d3a3d91 40036dbd8d093b57 3ff92dc885afc18b
3ff68a09ae3bb994 3ff60860b55a5ec6 3ff9a24ac9919e9b 3ff5dac905a69805
3fe7fa00d94768d9 3fb23a4f06524abb 3fdf4ef4286e913e bfd746c90ae4b370
3ffd25e8a4331610 3ffd83a918df92a2 3ff0e89dcd21dc00 3fd87435ded69d7a
3ff627cc72109e22 3fdebee403408fe3 40058108ef73afea 40004071c6ba2e5d
3ffd276bdd214742 40020370b4f96ba4 3fe80fa8f3bf95b7 4002f949f278fc02
40045da878d56c81 bfce09cd5f0b04c2 4000a33441a2a751 40068c253c60f042
bfe9c2d817249bd2 bfe1627da133fd12 4005c3dc03ca7234 3fff5be688836a6c
bfb4f7ab9eb3bc0c 3fb5abd117d8e5fd 3ff18ce382d60b93 3fb2baa478ad28b7
3ff9348b851bd8c0 bfdc5373900b52b9 3fd1ef94363410e7 400238f31ddddb0d
3fec7d7449e91be2 40012f31380ab038 3fe2cda9d804e09e bff46d77f3c241bc
3fe0d63dac2b5d45 4002deaa8aeb385e 3fa6bf2217d108a8 bfdd0c255425875e
3fd93303f5ee304c 3ff1c87bc1983edf 3fef4ea94ace5f37 3ff7fa3ae2bc6feb
bfee02393a08ba8e bfc46bc68f23ac14 40009e5d42af42fb 3fc6a4dcd680de99
bf4cca04a6a4b1d0 bfe19fe97dcae326 bfb0b846c3f2b10d bfb45a745cecbcf8
3ff0eacd71d11c4e 3ffe14827605c2d1 3ffb4e7742e24802 3ff44b77be9c6a5a
3fe86f4ce8507f95 400121b23a8b188d 4002f949f278fc02 3ff90a82dae3e154
4001a99d63901957 3fcedc38b2a0a3cf bfd7d469bc1a4245 bfab9cc0fa0d16e3
40060e822a5eb127 3ff57f2df7ae0ced 3fc98fd0be6fe625 bfd8e09786e8c224
3fe5167f686cef3e 3fc4d0f2d9941b8e 3ff7593bd33f94ca 3ffe1577bc7d113e
bfae061ad4bae6c6 3ffb53a5f729c282 3ff846947265cc1a bf8b5d6f68c02bdb
400439db5bff7e96 3f7a05eb932fd6a4 3fec20b73caa86b9 4001efd8c816e1f8
3fc7667dc827a2ec 3ff102206215f098 40017639ee43fc06 bfe853297820c2d8
3f8b9f770238b16e 3fda2404b65ab6f7 3fbb1e7ca4e8dbf6 3fe8ba0179fcfff7
3ff849c5b2e44cd9 3ff4b2664d1370c0 3fe5130f377e7d63 3fea4c41eaec7f3c
4003c8a2c0bbf43c 3fe7b1836e8e95de 3ff4d9e9f797a5d2 bfebc6b291a06c98
bff43b07ed9a99e3 3ff5610e31653cac bf9c61b2332d3cce 3fea915f55ac35ee
bfd7d7f488440add 40020512a46a47ba bfe10473be01200d bfd0ac345e980b20
bfec644d8990a756 bff07b2548428ba3 40040bb5c31209e6 3ff63e371738ccf9
3ff8b40fdb84ef12 3fe8cc53b74f6fd6 3fc2438806c0cacb 3fcc6ee90d954ab0
bfb3411e4f0668d9 3ffb0775b1be7fb9 3fdf4ef4286e913e 3fc9613712642a20
3fdc93bd844c7d91 3fbda88ce2a8f313 3ff94881695d0aa7 bfee251a746520c2
3ffe2c33e22ff788 3febc0ad7d465cae bf9def03b9c10767 3fe8fdf23d61f882
4000eb9f1b23e4cb 3ffdb2d5e0ef3c5b bfbccae45bf1f721 3fc31c5d96ca6ca4
3fec707a4d756902 3fdb6cb1547aef7d bfeafc7af93b9e80 3fc3489fa31cbfc6
3ffa278df5d5bab2 bfc635d68ecbbd78 3fe02227f0d8a43f bfd422819a057958
3fcec1537b77f046 3ff1472140d7ea5f 40008ed524949f6b bf99fdd1366790b3
3ff7f536834ebb25 bfea1cde1cd35149 4005ba9acfff1110 4000c6a54b7cbe68
3ff783d76c9ca120 3fd63528bcfa6b4c 40030c348cae2913 3ffba3e2c97c3b2c
bfb19b4b47fe6bd6 3f8a7ac3f3d3efac 3fe17e46a9887e12 3ffc20878bd9dc36
bfeb776887dc82b7 3ff5844820a86175 3fe1156a22e735d8 bfd746c90ae4b370
3fdb3b2a25e7931b 3fdd7506dc61d34e 3fe56e565ecc4618 3ff3469dfdbec0b0
3ff822815dd5857f 4001fd3c47594041 3ff1d76187977a67 3ff01d6eecd1f064
bff18b352ffcc57f 3ffa47b9f82968d8 3fead1c0b305fb4a 3fee8d046ca3ef97
3ff9fa352697f186 3ffaded9b39bcc3f 3ff215e53103f6e0 4006a62e67369ac0
bfdf346fd86222f7 3ffdaf4a0abffad7 3ff6e15af2642dd7 bfc703603b8a69c5
40044c62064b74a9 3ff4fe8a243f7a7a 3fe32ba69d4199b6 bfeadb5add808f1c
3fef1411a7db0df2 bfc4b1b0949270aa 400033c42baf96bb bff3746ee72c1186
bfe36d63f292d08a bfe5b4acea89978b 3ff54f5e0943f180 bfdb09511b242563
3fc2ec30bf6ce52a bfed175168a67279 3fe50ece7640d517 3ffdea97f328a892
3ffdca43d3654148 3fea3b43b97d124b 3ff629f3e22fba00 bfbf8b020eeb27ad
bfe2d5720bc3c605 3ff48d49eb72b390 3fdf4bd49532d4d2 3fc2438806c0cacb
3fe47ab8aba02eb6 3fe3841a61f6371f 3ff853af63f001f6 3fdcf283677d76c3
4004452f09016d61 3fe749f109df9a5c 3ff81712bba0b8d0 bfd5e3d5fd23390e
3feaddf85bf479ec 3fda3f141917a932 3ff0535aaedc0d99 3fb1cadc1e9d8f01
3ff16ce0a5a576e3 3f8b9f770238b16e 3fe909a0640c8274 bfec5c6a509a1ada
bf7e8424b7498634 3ff910ba32c1bd20 4001ee7cc531a4b4 3feec41b61ea296f
3fe631dbe9e4049e 3fbd535fc2ffe14c 3fe119087718f9d9 3ff06939041cd06b
3ffa283ce9fa9381 3ff7b8e3adfe0b70 3fd750f0be8314af bfb7aeb8e19d53f7
40015f630af5ed26 4002c0534c76c7de bfe11c4506619fa8 bfb49b0fbfc11fdf
3fbbd6a221a0af2f 3fb53514e3c25b11 3fec8c5235b489dd bfc4c4707aad7fb0
bfd5ffbe2c4f8aa5 bfd927f5e4d887d7 3fea47dd75bd0958 bfe48f70ba747cba
bfe48a086c4ee3ac 3ff17e2b22db98e8 3ffddae29bf0eac4 bfd2249c2b909ab4
40084722737af626 3fe6fdd787940a07 3fecf38bcfccd8c1 3ffc64bd00674eb3
3f5072bf01874152 bfbae08ea9b38d35 3ff65a225d770724 4001de3cb007b9d2
4002d36dcaf34a0f 3ffc420ca256d990 3fd91f01ffd9acf3 3fe47ab8aba02eb6
3ff5be2067b2c08c 3ff03f2a6bb6d70b 3fe6cbf62b1eea1f 3ff1cf883d9992fd
3fddbd1b62b444c7 3fa93f59b0a5520d 3ff3625a664d7820 bfd87aff7f4391db
40009e5d42af42fb 3ffcb7c7c22b97e9 3ff81ed07eb17c1f 3fea5c46245b4af8
400592e86bd39133 3fca8dea7aa182a3 bfb3760fac011be2 3ff1812d8886e1c5
3ff1d7c7454ec3f7 3fdc2d9b97cc4dd3 bfe9704cb6cf055e 3ff2de1ed41db72b
bfcdacdf74aaa9a3 40065fb9988c10fd 3fa6e659312df969 3fe2501a9b24158c
3fe8a07c714e4137 bfd7d30550e30041 3fdf272a68f08043 3ff22ac2d6d8c91a
bff1c6ff3f49490e 4005ba9acfff1110 3ffc18040192323e bfe9704cb6cf055e
bf983d3d81fec8f6 3ff3d16c6d7a8b79 bff5b30482d548f9 bfd8f6a47c736a51
3ff8bec1fea780b2 3fd78eaafe2f2924 3ff1e9ccbee7185e bfc8c01a71e9f6e7
3fd19d18eec8611f 3ff41bbc1f0c057d 3fcba3a217955a96 3ff8b903e412e506
3fd0a9f9625e9e04 bfd66719dca30fa8 bff2e45a4a07087e bfdbcf1be4321ed7
3fd7ddbea24efd0f bfde9c37c7edf2b7 3ff32be48a3d71bb 3ff6fbc48bcafcb8
3ffa02e0d913af90 3ff6236cc1c81b1d 3fe51bd396d588f8 3ff7ab0f85fab9a5
bfe0d3f126ff5c48 3ff7eb7df625a383 3fff58f1bc8daebe 3ff454e16d2489b0
3ff36c537b4b21a0 3fb13969f2ddbcdf 3fe14d86749d4a1e 3ff8b29ec98b7887
3ff949b098457148 bfebd1ee8ce39d9e 3fe35029a0961fd3 bfe9704cb6cf055e
3ffcd1fc011a9e61 3fe4a0c4b3f6ecf4 bfd9d69062c7a910 3fe2abd8c12288fb
40006ea4a2346bb0 4005603707105e34 3fef53874574feaf bfe3da61193e75cd
3ff6d2bf4bb3af2c bfd458df5869cd06 bfee09d11acd9904 3ffad38320e7424c
3ff32eb1b8056ef0 3fe89fea7c2e591f 3ff3bde1410651ec 3fe36430a668bf96
3ffee6de2675d6bc bfd4aa587b7ac354 3fd79a41a952c63a bfed8c53cdfc7eb4
3ffd48f2436e7972 3ff9256da399c97d 4003f1dedb2c6cf8 3fe0bb31275166f3
3fe93ca55a135a34 3fe1e7a957464f21 3ff9930e1e59d716 3fe7c079b3d4f435
4001832e0c9c5b41 bfe87d2d71213812 3ff1956d049e05f6 40037e24a39fdd56
3fffbb1f30fac7b6 3ff2d229ed2a89c5 3ff4b4d828724f71 bfe26aaac393db45
4001efd8c816e1f8 3ff7ab0f85fab9a5 3fd9c3207803b946 3ff454e16d2489b0
bfe815ec5b151742 bfd5e6d78c798158 3ff4e40e1b7d48a5 3fd44f6c941fee52
3fecb6c40a0b35de 4001aa2160d9afb7 4000b6cde7c77d04 3fea5d0da3099c60
3fd3473809fc42e2 3fedf412723e9443 4003cae14a7568b2 3fe048d90e4bd8f0
bfbd073aae097674 400824f2dec21ab3 3ffa42e961caa97a 3fd0ded006eede27
3fbf48effbe3694c bfddd8b9696929f7 3feb53605a4a23f6 3fe5b14b5d430dc3
3fd963f57070d995 bfb1d2cd7d7fcc20 bff2d6a090bdd101 4005ba9acfff1110
3fee9f8d0d2ed061 3ff3a64301c8095e 3fd5a8cbc9884e19 3ffae5eece76af33
bfeaf9745ad2f813 3ffcdd061b428f48 3fc98fd0be6fe625 3ffc96a0e28f8876
3fe3d3fecb050a47 bfe4d6e859545e09 3ffa003b4a210622 3ffd8fa2760bb99c
40012c8382b5a2f5 3ff0f96b7c71b359 3fc820dda00b0ebc 3ff87d50afa351b5
3fd13d9f48d6f75a 3ff2d44db25a41bd bfeb3296f83d5317 bfe95f14bab05dfb
3ff963939b39e070 3feaf027eb424a45 3ffeee18cdf0b8b8 3fbc37706db3cf0d
3fb9b674757c8ba1 3ff113b38ee3da7c 3fffbb1f30fac7b6 40059bdc06767b50
40026481fa7c5fb0 3ff84738241221d9 3fefc614a5e4a21a bff14e204b7aaf9c
4000726d62885b6d 400653f19c0a815a bfe0f1b0d7a7a36f 3ff8a1d2a2f127a1
bfc46ebcd4aa190c bfd3a35afdfd2bda bfd355ac139a741a 3ff176b56cded6c0
bfc49bdc52071f3c 40010760b40380bc 3ff1c0f8970ba443 3fdfd81eb78a1a8f
3ff66d6da30b36ab bfce0037dec622ff 3fad6a8901d0e7be 3ffa6c53b63bb739
3ffcf1e9600228f2 bfed0c1d474430ad bfde940cd3e180ea bfca9355ec8f7de9
3fed8b0d2b578d7f 3fe2954a18184c3f 3fd4493b21dcc1ee 3fd7e0ddfd0d13cf
3fe5ba3e226b7136 4002e64ab064c3bd bfc31014165f76a5 3fed9ee0cafd2e47
3fd93303f5ee304c 3fd1b1a8a05defed bfe216cd7630ba46 3ff1495b92b09a63
bfe9d6065f6b5534 3fbeb87318173a3b 3fdb3fb4f626fb6a 400796c715633d39
3ff30dc96248a6e2 3ff69531be8ee5aa 4003213de43080d8 4000c50eb5e29271
3fdf64634b1af27e 3fffc4e6ae9661bd 4001f35d52c61a54 40004353702e1105
3fd3782f1285dbd8 bfc443c7ce7cd69f 3ff99093550cd90a 4001e0ea739b5907
4000ead3f96f059d 40000d27bd3dca94 40061f3e352b3a5f bfe355d8e63ad24a
4002005dd12b663a 3ffaa3f8d77fbc16 3fe676801c921859 3ff4b5841ac04ad4
3ff100a3780b0f00 3ffd7fde6e6d0369 3ff247eac5feb11f 3fdb6cb1547aef7d
bfd9c7b2a04a483a 4000dc3a4dae4d04 3fd57449f2b7db74 3fe9c4e6de673f07
3fd09880c2f8b845 3fd863bd7d4d51e0 3ffd4bf1a9750fde 3fd0ada9859657a7
3ff34cb0e9cc572c 3faa9e02ea2565cb 3fc2cc97ccac618c 40021b4587fcabd7
bfd2fba0f445b0bd 3fe78b0b7585964e bff56714f5096c26 3ffa3abdfa1d64c8
4002f949f278fc02 3ff2b5d16d876a95 3ffbe1e678ce71f2 bfd739c6f2e4d633
bfd3073518966a54 3fe0d63dac2b5d45 3fbc28ef4b483d09 40070f876b15af51
3fd6de9e81f2c62c bfe55d5fb35c8c6d bfe852862ed71768 3fcf9a2c6a0dbdc5
3ff92dc885afc18b bfe3f515ac620eab bfd949bb5febd9c4 3ff9874d999e0c86
bfc9891dbee66b29 bfba2830cc7d340a 3ff554ae2b5916cc bfafdc0cbb13ea5f
3fe94dc4adf7edae bfdaca37c2b8cf95 3fdbfafa4180cb62 3fed6e93c326eca1
bfe36d63f292d08a 4004318eef48a943 40032209d70b7946 40001df3c5e4c409
3ff8155606a86d13 3ff0a84b65597735 bfe16aa5953472f5 3fec015326e7d11e
3fa8e39faa79a053 3ff056db6a2599f7 3ff9ce747de75f47 bfd970a341278acd
3fdd4baa1d33313a bfe0182dbc601ef3 4006815ba1b6d1d7 3ff2ab6b8be58802
40034daadec77838 3ff8155606a86d13 3fff2645ee17705a 3ffc87f07b6a7220
4005ba9acfff1110 3ffeaccef8df2a5e 400135409ae0b14b 3fd9cf86c5cabf9e
3fe6be74ff4b75f3 3ff512b12e322046 3fd63f5db228a9fc bfebd3fc32d84b7e
3fcb515aa51fafda 3fdfc8eab69562ee bff5b30482d548f9 40036b40aeff4fe0
3fe3ceda70d7a422 3ff5f31dbf6e585c 40025ebbe8605f22 3ffadc102f3efb48
3fee72efec271738 bfe9bc44083ce33f bfdb09511b242563 bfe24a97015f1a61
3fff9eea746d5023 3fcaf6748137ccae bff3d280a676831f 3fd0ce716ecf65b4
3ffd06ccf480ddc5 3fe08182ac405e36 3ff9c89117945432 3fba66e8996fbc09
bfb11cc8185f11f9 bfd4a00c7644fae3 3fc2c9ce0694050d 3ffab64b8f01c019
bfd144331c0e45d9 bfb9104bcb089dca 3fe28db686a1d449 3ff0528a82016b0e
3fe25dd2c4505541 3ff6e15af2642dd7 3fec44763d46800d 3fe2d8a29e58f64d
bfc65920ad6acfcb bfbc5db74039848b bfa8ed12a11fda1c 3ff102206215f098
4001118fe3e2a432 bfc2e9ee479a2eea 3fd0d43532675ad4 3fe354ab27a71ace
4002023140a69f8c 3ffc1a490dd8ed0b 3fffbb1f30fac7b6 3fd8294621339f6d
3ff6b820888e32ff 3fe4bf07a1897445 3fe6c47cc5c988be bff18b352ffcc57f
3ff17dd3ca699ea4 3fead3e375d84a65 3fe779b8afbd4440 3ffc33d85d71f31c
bfd78b1562513dcd bff20d1640afbf9e bfddaabc5bd0f56d 3fd1bb52c52b2420
3fe08182ac405e36 3ff2025ba2015241 bff580f415f3f58c bfe1bedbdfe43ecb
3fe8e6fff80f9bd3 3ff02dafe194be13 3fe9f89fbf3b3c5e 3ff1eb60a094b442
3ffa6e54efeaffac 3fc6a4dcd680de99 3ff13b2e538a980c 3ff3deb5a7547575
3ffda8e3d7a187ae 3ffc1e81768cab11 bfd06950350f234c 4002bc554c929a1c
3ff0b751a477939c 3fec44763d46800d bfcdc317e3ade285 bfd2c3d237d912a3
bf914240e27f5824 3ff0fe865a956e59 4000a0722d2dac63 3fdf4bd49532d4d2
3ff9d79fb372ef1d 3ffa301214939700 bfb4f7ab9eb3bc0c 3fc23da13e6b5aaf
bfe15c68e4c6d905 3fd5240ab66a4135 3fd44f6c941fee52 4002c6475c5aaba7
3fc9f5439049585e 3fd03b848c30caf5 3fe2cda9d804e09e 3fed6e93c326eca1
3f9e07d07ff4679f 40018192150f375c 3fd1a3db7fff525c 3ffe3ed83bb18cbc
3fdb3fb4f626fb6a bff2d6a090bdd101 bf9561f576bdda4e bff1ff5c87668821
3fbddfbf82991e39 3fb950700d943511 3ff3e084d16d5775 4005d56de701dc76
bfc015d27a283803 3fe3dde070a5f651 3ffbd3fe9388ea9c 3fe835ee5c53d432
3ffba0d988ffa9fa 3ffdbd68eb3cd3ae 3fed9176c0d64d48 40004353702e1105
3ff0480dd3cf70c1 4002a8551cffca04 bfd10244a03b168f 3ffd9b6609de218e
3fec44763d46800d 3ff77e7d27e09c5b 3fdf4ef4286e913e 3ffdd0398ecce88c
bfd07356a75ec3b1 3ff06a14a926a824 bff09002742d4351 3fdea15bd93eb9e4
3ff61fed7232bff3 3fd0aa378ee3617b 3ffea2ca993746d0 3ff23c49c0929177
3fc4d0f2d9941b8e 3ff5bdf32958f782 3fc49fc4746e2c58 3fd0732b953aec55
3fe6bb6dae4d571b 3fe2def5c6308a88 3ff1874d9509f306 bf9fa683d5880a7a
40031f585fa479ca 400002a8d83bab91 3fdeb7ef193c8fb3 4004452f09016d61
3ffaacd5edf90d4c 3fe84d6c8d68c55a 4001de3cb007b9d2 3ff0948f9c085924
40061f3e352b3a5f 3fe7836b3eec2577 3fd732d40b6af080 40015b8bd940b12e
3fdd6bc139b0dcde 3ff2c7bd79f1a031 3ffe88a301d971ef 3ff6982c8644145a
3faed721e7f59ba5 bfc16d2f9cac403c 400599a3bbb273a9 3fe506a5397b5afe
3fef1411a7db0df2 3fe9622792f8704b bfbb8be3dbc860a3 3fdf4bd49532d4d2
3ffa003b4a210622 bfd1e81fca328632 3ff8169447ff1c7c 3ff925f3536d6d85
3fcb238f7db9d34b 3fdea3a3df0c27e0 4004452f09016d61 3ff3b89b4dad61f8
bfd55c5830d68dd8 3fe58e0320c5fe58 3fe89afd3f3f074a 3fbfcee6a000b2c6
bfec79988ef1612e 4001226e650f8b40 3ff474ad518b6d46 3fea427f9c8a15e2
4002f949f278fc02 3ff1ddd61afefbe0 bf972a92db7646a6 3fe362da419d6c37
3fe63b3e00f637ab 3ffcf7ffe3e7d5b9 bfe13908e4fb80c4 3fe92c24c7dae9fa
3ff8ff17929a69eb 3fe2abd8c12288fb 3ff9961b357ea512 3ff8155606a86d13
bfeb15aa606bfff6 40000c4eec87657f bfed8c53cdfc7eb4 3fe60c86d3ddf755
3fe98b540dbdffe1 3feb34b02919db32 3ff6cc38b7dd9ef9 3fffa17b751ac9fc
bfeadd89ace1f796 3fb9f425b85980ce bfe5c135465780b6 3fec333e7a434122
3fe1384588456284 3fdd6158b7185f1f 3ff2010c61d0fedf 3fd49ebf72a3e1d0
bfd7f55b3186755e 3fe1262d665ce166 bfd4e824973d4bef bfe0f95281183aec
3fede631c91d9546 3fd7231e0c633c1b 3fef3fe560a530d9 3fcb31ec84d61a35
3ffc461e12a082c8 3ffe176744c1e8a8 40012549eb6cec02 3feb1d2fd0d75210
3ffc0e5e7d188212 bfd27766acc4e94d 3fd8a2cdc156e7c1 3ff9746231a42889
3ffe29de15385178 bff2bb0db6db3203 3fc30fb5d38f5493 bfad7e9a40c74a4a
3fe6578ba9a9f6c2 bfeadbeaaad7d13f 3fe38784e2abaa99 3fdddc1a39589614
400470ea6100f89e bfec769261b68f02 3ff0528a82016b0e 3ff71db756b17544
3ff3625a664d7820 3fe4cae58ed4f676 3ff8825969375466 3fd63d0a5148b33e
3ff6574bd4456840 3fece6207ca2a6cf 3fc449cf61b6a533 3ff2d58a1b6c135d
4002c319a2540fc9 3fea0592e8d4d641 bfd9b626f92889f5 40002828af253072
bfc35a101dc93cfe 3ffa74986b113eda 3ff4332031106e6a 3fe16940e81dfdb4
3fec5976a6d4e6e7 bff331b37761fd96 3fb09c528d8fac46 3fd13d9f48d6f75a
3feec617be541c36 3fe3616721454fa8 40002359cfe4afce 3fe4423d56707e42
4004be9bdf6f8bd8 400109140e4dc69e 3fe7860da5dc0b3f 3ff17265867d8ebd
3ff549acee2887b0 bfe72b72efcd0398 bf94569b1d12a9d2 3fc0a116bb3f83c4
400104a19180d47e 3fdf93ff88994cf4 bff1d74f9a6dc078 3fedfa63139106c2
3fcadec9ae6019b5 3feb8a5a17f5332a 3fd248be46445b76 3ffd7a587cd79a80
3ff95a712c4b80e7 3fee9b6f9170c727 40000c4eec87657f 3fef090ba41746f3
3ff1431ecc9a4b3d 3fc898211c8e2f09 bfde9e9e1183b25d 3ff585167f6d5b48
40004353702e1105 3fe4d28ffc52e3de 400155b552441561 3fe1bfde5344b13b
3feb6db3d0f6985e 3ff5d872c05084aa 3ff7b9d504409837 4006046b1e286ef9
3ffec1bad4f87416 bf983d3d81fec8f6 3ffacdcef03d36f7 3ff403c8d03b821f
3fc98a2927f26f60 4001670740dceee1 3fde2814d5aa167d 3ff1895b1d0c5725
3fed60109fd01271 bfe0d3f126ff5c48 3fd1ef94363410e7 bfbf8b020eeb27ad
3ffa02e0d913af90 3fe46e4cd192a8bc 3ff37468ed18e0d5 bfd07356a75ec3b1
40026481fa7c5fb0 3fb38a406a2147ef bf93a8b09971bb63 3fb5cb6d14ed4199
3fe6f4f3e056ed55 3fd1423b21e5bb3c 3fd3bfa2a7e34710 3fd9d4f1bee53459
3ff95a3c15fdb104 3ff006d79db4776a 3ffd7a587cd79a80 3fda947a3db73acb
bfd63d2a9a8bbbbc bfe62539ea69e3c5 4002a4fcdb9d362a bfee300bbde52875
3ff9884a5046bbc4 3fffbb1f30fac7b6 4003763a76075ee0 3fbb987ad053134f
4005ba9acfff1110 3ff52d85b5802d59 3ff56526f71e8526 40017b28174fdd5e
3ff126087ee2122b 3fe239e47976ffba bfbd073aae097674 bff580f415f3f58c
3ff5ae2e32412c1c 3fe476e1780819e8 bfdd71da06bcc6a9 3ff272bd337d39a3
bfedee11f56d0175 3fe81c3c280e8d5a 40036434133323da bfc46bc68f23ac14
3ff0453116611488 3ff0cb9dd442e220 400293ded55330b2 3fe45c0aa8e44cc5
3fe894d778226a90 3ff8ef61efc4092b 3ff06939041cd06b 3fe1b83832cc33fc
3ff084ade477eac8 3fed3c5cf5b3d908 3ff056db6a2599f7 bfb4f7ab9eb3bc0c
3ff3cf4e31eb1684 3ffb9c0b9b564bd2 3fda222a6a9ffde2 bfe30f7f08eff47b
3ffbbf2fa930f197 40044acb9363f2d3 40041140edc213a1 bfed0c1d474430ad
3fdbfafa4180cb62 bfe3f9542db46928 3fdd3375490468a9 bfe0cd3f7cb34de4
3ff2d862a635ab61 bfc443c7ce7cd69f bff149876034b3cb bfe29b07db46c273
40002828af253072 4005ba9acfff1110 3ff061591cedfec1 bfec3370e863edec
3fe041f81b0f5559 4002b51a60156abb bfe0f88e7c9f5235 3feb7dae7b8a27d2
3ff8e8275337bcaf 3fbe84d9e1653c45 4001870ef5094707 3fef4d174f8e4e72
4005ee1900fd0cbd 4001ba3fca84fd0d 3f67d40cad9bddad 3ff3cdbe187fcea5
3fc97847ea465218 3ff56988a436d143 3fe21e04772983fb bfce0037dec622ff
3feb3d337fd867f8 3fd0a9f9625e9e04 bfe13908e4fb80c4 3ff05d8003c3757b
400496fe22c7dfa1 bfdee46da7d4b254 bfeb9919cb6e29e0 3fc4d0f2d9941b8e
3fe1462c08b89235 3ff71db756b17544 400626fad4f75bd1 3fd6f4a591667853
3ff0ea2e909c017a 3fddff65ac89a6a3 3ffa9e81ee30437b bfe153a436108e25
3fb817ced5163e6d bfcb539429953d45 3fb717bdeeabc66f 3fc85cec130d2151
3fe3f40641fa6231 bfe87d2d71213812 3ffacf6c3e3a4bee 3fdc9a3c3ecda0ca
3ffb41ddba359159 3ff26b4602890d4b 3ff8dc22c62868aa 3fffcc4bd27bd50d
3fdc2f9406252836 bfc766aa0667638e 3fdb1e88d32ded15 3fe32c661e5dc7bf
3ff16aae34119843 3ffbec30bea0785e bff247bd4cc1da06 bfdbb2a91c83ce93
3fd1ef94363410e7 bfde450c173cb013 3ff4f5f47230d5ee 3ff4acd7a7bb9dc2
40040bb5c31209e6 40029ef64ecb2c44 3ffb543533fad77f 3fe310026e3e3b2e
3fe45c0aa8e44cc5 3ff0a45488b36b28 3fba26a120f7ecd8 3ffbc31d012aad60
3fefc614a5e4a21a bfe0101880a25a5b 3ff454e16d2489b0 bfc68463d41a97a0
bfc83dcf6374c27e bfb127531991125c 3fcc6ee90d954ab0 bff3d280a676831f
3ffb71de9bea702c 3fdf61c0391d5d5e bfe95a61432c64b7 bfe0081410a8d1f3
3fe690aa879da93d 3fe4930ca72e9574 4004eddae48f5b7c 3ff88b9639c68128
bfde79e4793476e4 3ff541f6b57a4df2 3fe2554efcaa17ae 4004659b7d3fda46
bfeb08e955daf677 3fedaf968224f11d 4005ba9acfff1110 4001a99d63901957
400205ec566a62c1 bfe805896c619178 4000b51aa5432cef bff580f415f3f58c
3ffa39b4d0561de3 3fea6ddeae629af1 3fe57a062557d218 3fdd7027b1ec0004
40018de90820e4b5 bfd3b11fe7bf73ab 3ff45c72c0dd106b 400275e7c3c43cf3
bfc6b671133c1461 3fd8204135197939 3ffadcbb947319c7 3fe27e9e84899fe9
3ff4b2664d1370c0 3fdcf283677d76c3 4002bc23284b0594 40019ce1bd015a8e
bfe6ee8cb35cccc1 3ffe30f64fd6205f 400088a95389f4cb bf918b9adbc9316f
3fed2bdb1080126c 3ff33c4f5d60b817 400272aede587e7e 3fe6c907d3f6c6e2
3ff95dba0c6b51c7 bfd6cdbd0f9ab200 3fe31ce9e88b7663 bfea8104291e2232
bfca19c2d2233509 4001e074277ace14 3fff2b83014b676d 3ff65788efbf3d0c
3ff9472db2b8a300 bfd76a407a6fd291 3fd58ad9bfc4e89b bfe12d717afc45d6
40060db63783b8b9 3fee0d9bdc359858 bfe36d63f292d08a 3ff65ad431e2c929
3fcd4368bd1443ad 3ff925f3536d6d85 400238d53fa2e5de 3fe107dc3a0384c2
3fec632494d2efbe 3fcd87164f4c4177 3fe37dd9bbd062a5 4006ebc42d3dfbb4
3ff0ee2f6d5a2346 3ff68a520ef7dd57 bfe15c68e4c6d905 3fe1d0fe69048b42
3fe5750109055bb2 3ff0c7570b34692c bfb921a90cd2b043 3fe1c7e9bcdb2e67
bfd07b5a6ddafff7 3ff429565c52d22a bfe73d5a125e87f4 3ffbe30a601237f1
3ff90a7a11127362 bfe7c1c4a420739e 3fef090ba41746f3 3ff4258104ca041d
3ff015e85418d5de bfbccae45bf1f721 3fff5beb6ec62118 3fe35ff84fe43344
3fe690aa879da93d 3ff5f04aeaca4e88 400287bbde78d05a bff580f415f3f58c
3ffd45b5b9b2ed25 3fe495c4b991ee9d bfe40d963f144b2b 3ff03829038cb5da
3fd96208bd70424e bfd8308f6ff0ed5f 3ff66d73056516d7 3fed9176c0d64d48
3fe7c4afcf97fc2c 3feef09dee42152e bff35f11ed2fc5a5 3fe488254fc47071
bfec5f710e9c8c66 3ff58056c8ab398b 3ff937d1dffdc1e7 3ff07281c201a9d6
3febf8d11439c0af 3fea653a11823f09 bfef98ccb136cbdc 3ff3fbbe7e921bd1
3fe6c7e30565ad0d bfde007b0e96eca9 3fda947a3db73acb 3ff9a47720f97ed3
3fe53602304d2bee 3ff62d9a807d0e96 400420f2a569a026 3fe19e369665dd9c
3ffaba88f97ab244 bfc9d87bd612a743 bfed0c1d474430ad bff3d280a676831f
bfcbb4e5be797bae 3fe78f8e1855efbb bfc1b2c079c198fc 400287bbde78d05a
bfeb6636ce162056 40070f876b15af51 3ffece1663244d90 bfc5b8621641042d
3ff42fa8e14a03fc bfe852862ed71768 3fbe84d9e1653c45 3febc268869cc801
3ffe6c6dd9bff24a 3ff62af0f3d4d0db bfd3e43ae06b7bb8 3ffee992fa661bde
4004ab9d4e758abf 40002ff96da1e7bd 3ff98311062882a3 3ff76643da0e3614
3ff579a025fe22f5 3fffe4dea85d3791 bfbb40d8eb47c4d3 3ff046b7ccfe203c
3ff2ca195019b83e bfe25c9955f3ec8a bfb96c8855f338b3 bfe0ea0604ec353c
3ff107189f60680e 400354be17f8228c 3ff4f205fdc252bc 3fd51e0debec51d6
3ff2bcc0158ea707 3ffcb419008b2216 3ff3d9617908cbba 3ffc1aa0139dc2be
bff047a1aed1f16c 3f50606fe52a25a7 3fcd353f5e3d0196 3fdebe8703ab4d27
3ff42e675796e504 3fe6b4d062ae0879 3fff7148847efe17 3febc268869cc801
bfbfeef43a8b10da 3ff5610e31653cac 3fec885ea0c1638d bfd4e65f7f8578a9
3ff8c4d7a63ca21a 3fed7e7da35ebaeb 4000136530774467 40001e1d95c8e3fb
3fbf5d534d5d58ba 400539b57f543d78 4001062e75df1a53 3fea7db997acb7ed
3fda8bd7e61b5b8d 3fde4bf43e58682e 3fdefd3acd344c41 bf72cc056869db79
3ff9153817d067ff 3fee8d87a8fd5f80 3fc2d9f21ecc5a1e 4000b84f096a3cd5
3ffab3280dee046b 3fe49dab6996b557 4001ce9af244137b 3f9dbd5a123768c1
bfd27766acc4e94d 3fd7b6908355c37b 3fe912ab9247c35e bfe76f83311d0b3b
3ff27a4b27707a63 3ff7507c653c4a22 bfe2d6811d5b2080 3fa62edd33a2e84a
bff1396ffce05efb bfc8ed58993aef8b 3ff5f65a8dd1fb51 40017770b6269053
3fdb3fb4f626fb6a 3ff65c62ac58e68f bfdb448b532d2b70 bfd6b9687369cc42
4000b428b8e3eca4 4005d56de701dc76 3ff9033332916084 bf7e8424b7498634
3ff36c537b4b21a0 bfc03d7e3a9825a7 4004055c21b822a0 bfe03dd4d04e5340
bfd355ac139a741a 3ffa6e54efeaffac bfe9bf0a8da197ac 3fe3361932b38103
3ffe17065541d31c 3fe239e47976ffba bfeeb5efd446b034 bf8b5d6f68c02bdb
3ffee2064c0c72a2 3ff98549429c24bb 40030b1e0d2d8f69 bfd6141a2136d6b2
400498500f29f137 40058108ef73afea bfbd925c683427f4 3ff903acef234029
3f8c803a4e49e26c 3ff9fa352697f186 3ffa534d79df08db 3fe3357ddd47c44b
bff09002742d4351 3fef090ba41746f3 4006046b1e286ef9 4000a33441a2a751
bfe9fe026dd566b0 3ffacfc9553b3e08 3fdfa73b511e6737 3fe1462c08b89235
3fe55160186fa576 3fbc28ef4b483d09 3ff79d86da77ce4a 3fde2afe78202215
3fbdd2cd7c934db1 3ffd7fde6e6d0369 3ffd534f8eaa0c64 3fd963f57070d995
bfefb446c4c57e0f bfebd1ee8ce39d9e 3fbcf07bc8e1f137 3fe0962d410b4222
3ff73384517680fb 3fc1fe630ae78e9f 3fef171568ab5bb3 3fe2fb16563b96c3
3fd5d58a07d4d130 3ff6d8d168f9dab6 3fe2fe7a83eeb932 bfc2527d973f791e
400183f29146da44 3fec66b99a47497e 3ffe7a9dbf5142a8 bfe36d63f292d08a
bfd811e3750365fb bfdfb58fe94ec87e 3ff90b816bfe1bfa 3ffa31975023bb0e
bfd72ec99e63a437 3fd47920d69f0aa2 3fe005fdb2feac97 bf9026a2a83b11d7
3fe0f6fd5710f9ab 3ff3962590c3220e 3ffea66ab7f190e0 bfe7c85759bd4311
3ffb543533fad77f bfab9cc0fa0d16e3 3ff9884a5046bbc4 3fd5da8d4f0f96e3
3fdb93fa0caba8e0 3ff4f72e2444cd92 3ff4a89e179f2065 bfb9fdd97f6a70c8
3fec44763d46800d 3fef308b100baed0 3fe9c62984723be9 400482ae089f82e2
3ff5a6265cd8da94 3ff83e0f23834686 3ff60799951e02c7 3fd44977ec593ba1
4006b2e39909109f 3ff370efd387e979 4000e8e48ef96c0a bfcae14f06c1387f
3fa7a282ddb07995 40070f876b15af51 3ff60196fdda2694 3ffd71df5e25e6a4
40023fb7739f50ba bfd6b1d3ea159843 bfe8debc3acfb577 bfc9891dbee66b29
3ff0fbf86493af86 40006376759b26af 3feb23cd2cbbb6a9 3ff7ffa1e2b85e74
4004bd10785be7cc 4002f949f278fc02 40043ebd5ddb1130 3fe99ad20ba88578
bfd540112795604e 3ff4d3e280ac7481 3fd7b386860808aa 3ff46ec403cb4800
4002edc456c44bc6 3ff34149419c5570 bff522efe5b03ff2 3fc757719c578fc7
4002f6f1f7884ee7 3ff722204ebacedb 3fedef8244044e11 bff331b37761fd96
3ff7f73e5ee80afd bfa55cd143b30c88 3fdba0fe151c27a5 3fd43ff2999b21ef
3ff68114faee6d2d 3ff3d8ba5617d820 3fff4019c908367b 3ff0817edeb2af5d
bfe12ddf089a4850 3fd9575f19e052c9 3fe37dd9bbd062a5 3fd21a2de8cf4acb
3fe68d15608fe53a 3fe9cabf2b2a49ba bf8341b617bd42c6 bfec6892e10b63a5
3ff76f61237edc06 bfe26be811f1e019 3ff286a367a655c6 3fe00dd5cdb6e0a0
3fe5bcf61048c23a 3fbd3aa8e608d7b5 3fd9daec0ccbb1e3 3ff1eb60a094b442
bfd0d058ca53b5bb 3ff429565c52d22a 4001a99d63901957 bfceb5b785d142d5
bfd9eb20d4a3b3f2 bfd540112795604e 40015b8bd940b12e bff20f13fc57fa04
3ffb1ddf8d762ac7 bfeaaed6acfa9019 3ff1eeb52b73fe44 3f979971179fa5b0
3ff3cf539119dd45 3feaac7897af0bef bfeec5d97b550439 3fe2b632395ba3e7
3fe205b50de5c900 3fd54b2de75a6db9 3ff267a5a7254219 bfd746c90ae4b370
3ffe2cc83da5e48a 3fd3fa9b4ecee266 3fe58e8f3575ccaa 40051c5693a53a85
3fe9a33c063ecd54 3ff0cf96dccebc96 3ff36549ae3c64e4 3ff51463034ba2d3
bfd7970ce605cc31 3fddff65ac89a6a3 bfe3ae907cd96507 3fe0afd9a6ba8ca2
3ff0597fdaa49329 3fced67fc8692ae3 3fb74152f298b843 bff287cde3d87cc5
3ff4858cca66a1d4 3ffeab9e9d6e29ab bf981e658fe6707b 3ff10ca80641d6df
bfe04029796a998a bfd175a499fde504 400088cc75888b45 3ffc420ca256d990
3ff98da7f637aab4 bfd9d715564925de 40084722737af626 3fd950241d4ae5b2
3fee4e562e047646 3fede631c91d9546 bfc1990f260118f9 40058385cd7d9b61
3ffd25e8a4331610 3fe71ed7caca02b3 bfc46bc68f23ac14 4000a6692a10bfd1
3ff12b6a30bcea74 3feb1e42cab23496 3ff070818d0c063a 3feccae3f1723644
bfd79c2f15e66c37 3fe1262d665ce166 bfd13856200edbff 3fffe607f63b4f1d
bfad1bb8d5b1fc8a 400430888891d2cf 3fe2255f88eee5b2 3fe1a57914fd6a46
3fd03b848c30caf5 bfa96ea3b8b3c970 3f92b381260a4140 3ff651622127c240
4001ce9af244137b 3fc7fb63b3149a2f 40005c6ca2194ba4 3fe18a402da5a22b
4005d56de701dc76 4002fd8115f432d7 bfe7df12542afb1c 4002c7b6dd8aaf33
bfd6ec349ca3718c bfcd05b18274187d 3ffb36e53d115c4b 3ffa3c7f8b08cc2e
bff5b30482d548f9 bfe07ea453919f84 3fea5d0da3099c60 3ffb01d40104ed2b
3ff6631b84e0372c 3fdd4baa1d33313a 3fff9587b6f01f67 3ff783657f9e73e5
bff5b30482d548f9 3fcc6a1b1386383b 3ff4474aaf52c2c8 3ff4b2a2a87403f2
bfcdcda872155ba6 3fec29c7c041d191 bfa867c4913f6936 3fea4c41eaec7f3c
3fe46e4cd192a8bc 3ff066cac73978db bf9b6c6b8dd8583d 3fbb0eb3ee862703
3fe2f62970367df9 3fe1809d2303110b 3ff46a1085295f73 4000278dd2075513
3ff822e1eefb2557 bfe312b85c41075f 3fe59ae58664966e 3ff48eb0e6831922
3fea8c6a4cd478ce 3fbda88ce2a8f313 3ffa3c7f8b08cc2e 400112d74864fbff
bf493959b19b553c bfe7155a02eecc82 40001e1d95c8e3fb 3fbd70b32cfabce9
3ff4d3d05ad48e33 3ff7101dab9e21d2 3fe042bd9623aba9 3feb43f4e28e2f0b
3fbcd1ebafe94896 3ff5d825f82e8c6d 3fafe496cfda115d 3ff4fe8a243f7a7a
3ff4304aa57842ad 3ffcdd061b428f48 3ff26067f5570b20 4005c3dc03ca7234
3fffda07e4b812ce 3fe2eb5d20a483db 4003c332476fdf8e 3ff321de9716ffcc
3ff8a38a56aba761 3ffeab9e9d6e29ab bfc5edcef45586f1 bff2d6a090bdd101
bfef28d8017dd722 3ff6cc38b7dd9ef9 bfc8ed58993aef8b 3fe0626cab22def2
3fe0bb31275166f3 3ffa3abdfa1d64c8 3fe9fbd101cdc0a1 bfd55c5830d68dd8
3fcb238f7db9d34b 3ff8cac84715f186 3ff3d536d0cf8004 3fe6173ce7e59091
bfe6523423e2e8fe bfe41990133b7b73 3ffde158b6fc6b28 3fe98b540dbdffe1
40024c1d13337cef bfed8c53cdfc7eb4 bfad72236769388f 3fe72a7726087105
3ff95dba0c6b51c7 3fd63d0a5148b33e 3fe56309a395a977 bff5b30482d548f9
3ff4bad27bacb718 4002eb6aef01bb67 3fd52d87b542b2a5 3ff3623d81176964
3ffc74b3ddfc67ea 3fda4f8f9b7a57db 3ff376db2fbac79e 3fb391573edfd4f0
bfd768700b7aee62 4003c1e574e8abfd 400502d814609c89 3fe80fa8f3bf95b7
3ff6d48e75cc912c 400000a16faf886c 40022f8d0436ee12 40061f3e352b3a5f
bf918b9adbc9316f 3fe412319708d786 bfd7a2efc5c0bf6c 3ff65c62ac58e68f
3fe586ee07f97fde 400404902edd2a32 bfdefaed5d6c7ac7 3fec8d780483fc3d
3ff82520b8bc8de2 3fd6a5adb68de65a 3fe89afd3f3f074a 3fe8917cbd604d64
3fd79a41a952c63a bfc80635e2be4d34 bfeec5d97b550439 3fb53514e3c25b11
3fff09129695ff29 bfe0f25f7c11b61f 40001e1d95c8e3fb 400433aca9fdc7e8
3fea7db997acb7ed bff5b30482d548f9 3fe80389eed03c5d bfdfdf51a9e92141
3fd545a097abf163 3ffa1156efd9f960 3fe137cc04809b12 3ff7a855902e34b4
bf9a81a24031fc18 3ff1d3f0f9cd3c1b 3feb34b02919db32 3fdf8e95429fda7f
3fcc6ee90d954ab0 3fd3175d6c98d315 3fed6e93c326eca1 3ffb138dbe6b9d62
4007ddc96f9be195 3ffd9b6609de218e bfe721fce70f6708 3fe09dd58abd0b22
bfd3690719197959 3feaca7beafff988 bfd255d3d806be7e 3ff1eb60a094b442
40009ff45602375a bfee1100b1253881 3ff5559307abcafa 3fed50bf1a488c5f
3ffcaa51fe2a6e48 3ff93b160eb22c5a 3fe197543cd39e50 3ff4960f1afe2329
4002f627537691b1 3fc1b595c43fd0b5 3fe7d6e2d513f118 3fecaac64bc20281
3fe9511386519c11 bfd77a360c85a886 3ff0eacd71d11c4e 3fe37dd9bbd062a5
3ffcda3df0eba5ec bfaa4bcdb9ba81cf 3fffcaba5728b6f1 3ffea764e3df2927
3ff5418381d8d26c 3ff1fe453d4f5d02 40008d686799a786 bfc68dad15d15a67
3ff8f1348eb3d3bb 400631c20751a5e7 3ff348c18aa2de8e 3fd22b705d18c066
3ff1487b130483ec 3fcfad7f7d2b0a03 3fe3dde070a5f651 3ff59880cb73d9a9
3fd562ab18edbbb9 3fe729cd4868d889 3feae36a6abbb9c3 bfdb3fae02ec4596
3fe0257eb2deb47b bfd30e42472bb71f 3fee0d9bdc359858 3fceb3040f15deeb
3fd6f308183f7664 3fc099c5ea604209 3ff3d9617908cbba 3fc0e6f0943b957a
40031f934238aeac bfe22502ae8a1290 3fd4df99f2812f0b 3ff5c2edcf117a1b
3ffc69c8fe471ddb 3fdd9c18ae499bc3 bfd739c6f2e4d633 40006d17bdd625db
3fec21a436884730 3fcc6ee90d954ab0 3ff4875bf47f83d4 3fffb69ba2453905
3feb65fc6f8f2838 bff3425e7a4abe19 bff2751930e45ce2 bfd65e104f066c8d
3ffa919bdec34512 bfd536ac8ce7ab34 3fdd21538847319c bfe3793e6627b62d
3ff95a3c15fdb104 3ff0bee804a314b4 3fc6d3ff2447b7e9 3ff6e15af2642dd7
3fd39629c85364be bff1ff5c87668821 3fe1809d2303110b 3ffa49abbfad2cc7
3ff148c5122bf9fb 400147c25281e8b6 3ff599e31cd723c1 3fd6e6ae7270c6df
3fd5e9d39b7b7596 bf90412561a5bdd0 3fe0649b291a0d94 bfe49d0405ed616b
bfe5348433e3fb22 bfdfdf51a9e92141 3fd5f138a1b95934 3fe0abe4de1f8551
bfe62539ea69e3c5 3fc0a116bb3f83c4 bfe45749ce8bfeda 3fff000e7612ae1b
3fe6c14d23d7f4ce bfee251a746520c2 3fe3fc00b82216a3 bfec3370e863edec
bfe69fa7b68726f5 bfe852862ed71768 3ff6d48e75cc912c bfeca8d386c16db0
bfd1340cf2a50433 3fc2c9ce0694050d 3ff665d4f3c98d34 3fe1d8804b262ef4
3ff3326bb971da7e bfd8b1d262086bd0 3ff91d594e37d809 40020a720c17a26d
3ff8da14566f1858 3fb0e306bb254c9d bfd72ec99e63a437 bff2d6a090bdd101
3fc0e6f0943b957a bfc72dd2bf3d810a 3fa239054ca73020 400186f98887eaee
3ff990b4ab717a51 3fdae647f20c25b8 3ff3b5dda73ac9f7 3fd841263038369d
3ffcd3e71669f430 3ff8bc07e0c8a71f 3ff9ce747de75f47 3ff8dc22c62868aa
3ff406184b5d8be2 bfdcc2f3d9796b89 3fea05e86a3ff59d 3ff1d880cf0d4262
4003d70bc429a192 3fd7534b41eea428 40027b8260d91fce 3ffacd193fbe025a
3ff3625a664d7820 3ffa278df5d5bab2 40029a5111b42891 3fbda88ce2a8f313
400104a19180d47e 3ff2f1b8a7368e0e 3fe1544ac6ec2d5b 3fecd8abd25545cd
bfd4590f65e90396 bfc443c7ce7cd69f 3ff6916d54ed9c62 3ff4489ef5c5bf63
bfbd38c858921280 3fe6c7e30565ad0d 3fcfbd9846281be3 3ff822815dd5857f
3ffcdd061b428f48 3fe05d24628ffa1a 3ffb83c14d8b08dc bfceaa96ef9594c9
4002ea7cbb374421 bfa915b5d62b176d 3fed6e93c326eca1 3ff5d825f82e8c6d
bfd8a392c13766b8 3ff95dba0c6b51c7 bff0234f44bae28c 3fd258379b794401
3fb23a4f06524abb 3fe2255f88eee5b2 3fe98b540dbdffe1 bfdaa7a2c69d8c55
bf9bff48772a160b bfe2f975f37131b9 3ff7e9fa2015bc71 bfe2d5720bc3c605
3ff6b8c3b0b93970 bfeb0a0a9e560a0d 3f879bec4f2c9ebe 3ffe3deec0d854a2
3fe7c079b3d4f435 3ffbe1e678ce71f2 bfe4400349c83d49 3ff3ab6112f67e8e
3ff38bc75a40a8ad 3ff18d86331d292d 3ff8022af97bbe11 3ffaa3f8d77fbc16
3fe104a8dca542ec 400439db5bff7e96 3fab75a407f56010 3fec1287b1874b65
3ffce6b7e084deaf 3fe5b0b521bcd016 3ff34cf4c40c5c7f 4001ed5343dad88a
3fe4e92be764af06 3ff33fb44a45b9cd bfd8e09786e8c224 3ffd4f9b88458e9a
3fd25778786206c7 bfed1872b121860f 40063dfc6ad4679e 3fd6f8349c2c36f6
3fe9aac797bee874 3fd7b386860808aa bfd53ad3e3d87da0 40021ddf1981baed
3fe113331485fc62 bfbc5db74039848b 3ffb4e311d2ee095 bfd2fba0f445b0bd
3ff788950310e90d 4002c7b6dd8aaf33 3ff98547496b7ae2 bfc5578ce5bb7714
bfce09cd5f0b04c2 3fea5e1c2869cf06 bfd07b5a6ddafff7 3fd377e20ac08af7
bfec3f89e2be6fb5 bfe5dbdcc14698aa 3fe72a7726087105 3fff666d19617329
3ff62cd0540c53de 3fff8be39fefe4fc bfbc5db74039848b 3fe4a0c4b3f6ecf4
3fe80fa8f3bf95b7 3fcfbd9846281be3 3fee0d9bdc359858 3fd63869944f5c7d
bfd2efdc4e0a6910 3fd0e02d90328440 bfeadbeaaad7d13f 3fd7d2b21df1e732
40004f67aae9e139 3ffcd3e71669f430 4000c8c9e3c90eff 3fedae7f0410ec68
4002815c1dfd6052 3ffd43d8e92f633e 3fe5b14b5d430dc3 3fddff65ac89a6a3
3fe459dee9d6c2b6 3fbda88ce2a8f313 3fed773b99bb13f4 bfe65f4c2c48c295
bfcdc317e3ade285 3fff576c0c7fe7a0 bfe14cad79bd5c27 40041fd541635c4a
3ffd01ae2ac6d777 bfdcbe5b56ca03e0 3fe5a4a49be51891 3ffc9083ff749154
3ffb38b4672a3e4b 3f879bec4f2c9ebe 3ffb6a0e239c5a41 3ff4b2664d1370c0
3ffd5802fc6416c9 3ff2911a298a9395 3ffa9c116ac94aca 4000b6cde7c77d04
bfd69d9a36eec4cd 3fef48a9c3278aac 3fe9540236906e7e bfe1ab858d75a9d5
400022a562fd6e4c 3ff27301a5c72e7f 40074cf6c739df50 3fd6498abbafc84b
3ffb458f5b0ab3ce 3ff0bc4f21b1983a bfdb8d44df89d716 bfedadf23c68f7d9
bf99f4cd747f366c 3fdee928fa41caa5 3fe2e5375987e721 bfde5b3d1ffa080f
3ff15ff0c9da2259 bfc41ab3d989b051 3fd32108130a5342 bfcabe04e822262c
bfec79988ef1612e 3fe721232c6d30a1 3ff4489ef5c5bf63 3fe113331485fc62
3ff52b9cc252b0a9 400436f0fda2a342 3fc4d0f2d9941b8e 3ff73384517680fb
4001726f8402508d 4001efd8c816e1f8 3fe63ea5724f708a 3ffbc31d012aad60
bfd4dc470967a09d 4001dc3459cebf88 bfeb9919cb6e29e0 bfed3cd0068e69a4
4006378b538d73ee 3fefb78aa145e10b 3ff1bc215acbd87e 3fa028aff92f03df
3fffcdd9392e3b7d bfaaabb1a0198f7a 3fe77e10154fc597 3fca8dea7aa182a3
bff1ff5c87668821 3ff8f89d2a84b7d0 3fbd70b32cfabce9 3fd1cd0177ed34b1
3ff8709f43152875 3ff497113f324342 3fdcf2d951109ac2 3ff42affb0518376
bfe4e3107dade9fd 3fec1287b1874b65 3ffa196630d38c2c 3ffdb0deb0381d56
4005fdf9730212c8 3ffa5156bcc64834 3ff5a6265cd8da94 3fee21398a8bb569
3ff7c013ea448aff 3fea15fecd9fa9cc bfd69cc25ef82d7e bfd272fc11c4d8f9
bfb49b0fbfc11fdf 3ffc32374c3a4858 bfdab5c3c2d14910 40012867df2bd2db
3feb83b42b32d2d5 3ff56b47e9e1a4ed bfd2efdc4e0a6910 bfcfe9444cea982a
bfe4936e1eda4afc 3ffdca1db35145c1 3fd0bf2b58b7ab22 3fd9e4356037904f
3ff8cac84715f186 bfde4bbda4bd073f 3fe55c93c2cb8f18 3ff4b2ff743c2271
bfe3d0642c6ffcc4 3fe5a8580a747986 bf92675c7723302f 3ff2d44db25a41bd
bfe098d59b5f75bd bfe0f88e7c9f5235 3fe18a402da5a22b 4006378b538d73ee
bfae751d7ed0294e 3fdf4bd49532d4d2 3fd16079c7016c05 bff190cf40ade665
3ff6d2bf4bb3af2c 3fd382d51bcd9613 bfe9d9672227f8f3 3ff2cef496f87fea
3ffd1826f79711d0 3fd68427752dbc65 bfd78b1562513dcd 3fce29ad934ef906
3ff88cd527726ab6 3fdd49e4ccbc7385 3fffcaba5728b6f1 3fdcab5d33555bd9
3ff7346c6223d260 bfd6ec68d4e8abec 3fdfee34d0f0f4d7 bfd12a07c6f01904
400121c059bd9e61 3ff16683afc61bee 4002c3c67227776d 3ff5499d34a8ab68
3ff879c9f5a9af32 3ff69531be8ee5aa 3fdcd32ef74abde3 40037f9322fd553e
3ff2c61d9f4987fc 3fd8f7f6e02389ac 3fe539e36a0e32a4 3fc4d0f2d9941b8e
3fced67fc8692ae3 3fd43ff2999b21ef 3f66dcd9c48f9fc8 3ffbdfadb5b2a188
3ff454e16d2489b0 3fe5f11644d64d00 3ff8753d3e8800f9 3ff233d68671f661
3ff6158c860466cc 3fc566900a6450b6 3ffe7a9dbf5142a8 400093ee72fdea53
4006c99498852041 bfe0770a28ff11db 3fbc76cff7e8ff22 3fdd4baa1d33313a
3fef9cdde139d1af 3ff8b9b817100d6d 3fd3399950bcfad7 3ff5dac905a69805
3ff8b198d6c315fc 3fe19c7d348a1cd3 3fe8a655ea142193 4005bd17ae08fc87
3ffb38b4672a3e4b 3ff0221c67de9341 bfbf339d394d64d7 3fd5340bd5d4f130
3fecba0a85cb146e bfc9ab2d959576e4 3fd7b6908355c37b 3ff098638188c2ca
bfda54bda3056d96 3ff291b38f5dde51 bfd8a08ee8fdebea bfd8a392c13766b8
bfdb00bd2be3667c 400174c64de936c5 3fd24ed1ca3c322d 40037e52a7a402dc
3ffe2ee12b8287b1 bfa7391ddd16f094 bfd2a22111801f5b bff2658060425452
3ff5146e25404a6a 40010b40481f88cb 3fbab16cf3c791f7 4003d999dcdc93a9
3feae36a6abbb9c3 3fee130197ad52ff 3fe959b59bf7be7b 3ff7185840128b86
3ff3942a6b23a344 bff331b37761fd96 3ff4165bf4b911ac 3fe6f5417b350ea9
3fe7dc97c236105f 3ffe7bfd6f714c9d bfc0d7ebf928ffdd 3feaf25b48944b14
3fc54f80bb50623f 400003ca4edf349b 4000f2bf0857340a bff04ecf552a9f14
3ff993883da1fe4e bfba145bf20720e1 bfe0ac8f8eb859e0 bfb3e5ad33f1be52
3fee9b6f9170c727 3fe8c48ecd25bc5a bfdb8d44df89d716 3fef94a1748dab95
3ffbbdc4e9305f1c 3febb36407331acf 3fac9553dcfeb372 3fc98fd0be6fe625
3fe0bb31275166f3 3fffe5a8dd5824ef bf86fd380b4bb162 3ff69344e8fc6839
3fd968947137b022 4004f44c8fb5b7ad 3fe5b4b580a6d832 3ffbbaacd724eb50
bfc65920ad6acfcb 40041a1b3cc3badd bfd7cbe27cafad09 3ff4bcdf375a8eba
bfe36d63f292d08a 3fd4b44683de170d 4000dc3a4dae4d04 3fd5fea4fb40cfa4
3fea18727145351a 3ff05bc3dc3fb258 3fe8c258cd9ca19e 3ff7fe8ea33e43b1
bfe5a21d0bb269f4 3feb23cd2cbbb6a9 bfecfb924345daea 3ff4b2664d1370c0
3ff0163d637bf9f5 3ff59c6078a6c570 bfe62539ea69e3c5 3feffbb0ada09046
bfebd1ee8ce39d9e 3ff68a520ef7dd57 3ff4ea9a4b9cf808 4002d36dcaf34a0f
400193fe72ca7b33 3ff3ea608e9a8931 40015f630af5ed26 3fcb515aa51fafda
40047d7cf82725d1 bfe042ab910a4a18 3fc3d677cdc4a370 3fd42fdaeb9c04f2
bfd2a10e7d759839 bfeee6e848548eb3 bfeadb9553e5ec3c 3fe7dcf31c059513
3faff7fc960c5185 400041ccb8d3d737 3fd268abf7aa69e7 3ffbbffb3258fa92
3fe09dd58abd0b22 40022f30697f62b3 bff331b37761fd96 3fd2f94eb0f06a55
3fe17000aa0fad0c 3ff6574cd3e1d22e 3fb023869703ba2e bfbccae45bf1f721
bfc46bc68f23ac14 bfb11cc8185f11f9 bfd8382c800ad1ee 3ffd4d6ea1232825
3ffc60a00cb92d45 3fd1403ec0ef5e1a 3f9dbd5a123768c1 3ffcd3e71669f430
3fe0e1e61c2a6150 3ff06939041cd06b bfc1b2c079c198fc bfd2249c2b909ab4
3fd8a95ddc3e542e bfc1990f260118f9 bfe55b520d67de8d 3fee57eaaa3d57c5
3ffeab9e9d6e29ab bfdbe78406a4c491 3fffafc503875d3d bff404911357d68c
3fe38784e2abaa99 3ffddd1ec9bf6f6a 3ffc1c324ed58739 3fec885ea0c1638d
3ffda185aa8dae1c bfce09cd5f0b04c2 3ffd7416f5d8a3ee 3ff77e7d27e09c5b
bfec3370e863edec bfdacc0b73e81ce5 3ff7f71347840969 3ff2c5d257e94dec
bfe2e75a0cc101e1 3fd1217da13b937c 3fceb0380deaf505 4006412753db0635
40002c027ef945e5 bfcdb7ef074ecdc7 400282a58dcc43c7 3ff9472db2b8a300
3ff044a1cd6f50e5 bfdcbb780c6b832d 3ff80e9a816f7620 3fe14a422d1c3683
3fc341b92954e686 400174c64de936c5 bfd3e43ae06b7bb8 3fead3e375d84a65
3ffd6cebc7e3a284 3ff00508739b956a 3ff8eeb5d9a57bfd 3ff97f0f4af374eb
40075367de80d300 4003f14f1383cf26 3ff0cbf6415c7f65 3fc814437e518546
3ffc45bc2342e2cf 3fe631dbe9e4049e 3ff342a1be5a93fe 4005010831278566
3ff403c8d03b821f 3ff3fbbe7e921bd1 3ff4b2664d1370c0 bfdae5a29aa1be27
3fdaac62caa233d7 3fd63528bcfa6b4c 3ff6befdecdaddc0 3ffeab9e9d6e29ab
bfea01092bd7d83c 3fe553778cc9af04 bfb998a55bfb19b7 bff20a0f19c179aa
3fe8ad68e5c03c50 bfd1e6b77ee9f0a4 bff27dff994e8518 40075a6740e72286
3ffd7fde6e6d0369 3fd5baf2419d91b2 3fdd3574d21785bc 3ffd25e8a4331610
3ff7ea71aadc36af 3fe60f5ca9913d51 3fe608e87fac3354 3fda38d857c1632d
3fe8692f3b11b9b0 bfd67975cb331a32 bfcfd1d9fe744f46 3fe4cddcc99720b6
3fe92c24c7dae9fa 3fe919109ac202e4 bfc443c7ce7cd69f 3fdd00c91b8893d9
4005ee1900fd0cbd 3ff4b2ff743c2271 3ff627cc72109e22 3ff852ed7d11dd21
3fe51e493bdffa6c 3fff9b31d2133d25 3ffc3c30303c00b1 4002c0534c76c7de
3f919e046c33c600 3ffe8c691eea8bb7 3fc2438806c0cacb bff3746ee72c1186
3fb248dc8949e19d 3fc4c951c7c5f7b6 400179a0259d6e90 3fc41ee33f573551
3fdeeb31e970c4fb bff2d6a090bdd101 3ffce6b7e084deaf bfd47343e256e263
4001ba4fa3866447 4002611733ace0e5 3fbcd508999b269c 3fc8a97134519359
3fcbae6fcbc1840e 3fd772fa9b996d93 3ffcf7ffe3e7d5b9 bfba44a97e35bcb3
bff5b30482d548f9 3ffe7dd445e95ce6 3ffc6e3989241750 bfef1ceacfc61652
3f86e3a41c939380 bff0ee940189491c 40070f876b15af51 3ff0eaa802d51f15
3fd84996965697b9 bfd4362bd8d81bb0 3ff12f050bcb26f6 bfdb09511b242563
3ff7a0e14c1ad779 bfe99af3f9192608 3ff1e97f578ea73b 3fd238d8eddeb1fd
3ff2895f982d9657 3fdf065fde691f00 3ffe531054b9df6b 4001aa2160d9afb7
3fdd4baa1d33313a 3ffa0a264fd562a2 3fe894d778226a90 3fe6bdaecfa630cd
3ff5c59861502d1d 3ffe176744c1e8a8 3ffd7a587cd79a80 3ff1261918432b7c
bfa9e0de99987dd2 40015b8bd940b12e bfd3ec2fcf98abce 3feec617be541c36
3ff01e68b17c293d 4003d70bc429a192 bff0da52380a3b3f 3fe88ba4709d42b1
3ff0f96b7c71b359 400648ddc390ecdf bfdfdf51a9e92141 400135409ae0b14b
400205ec566a62c1 40068c253c60f042 3fef2cb3464d1314 3fe4d9fb2ea2cc05
3ff8e8275337bcaf bfd08d0076a41cdf 3ff0edc505895b77 3fc198d2419b8ae4
3fe9fb738028876b 3ff4a2270b5d81b5 bfe0120cd99e65b5 3ff6e15af2642dd7
3ffb3234bc5d2ae3 bfbfda6af7b85f0b 3ff6459f1c8972f1 3ff3a04e7fb8dedc
3fe8a03022f784d9 3ff4fcd919548b93 bfeb5f5df66a3872 3fb1cb16e6a429d4
3ff3c39182ae40a2 bfc6ecf57c0c1e8b 40017770b6269053 3fff210a6eb9fe6d
3ffd8fa2760bb99c 3feb23cd2cbbb6a9 3ff89d9d55d21c9a 3fd7bb5b0243c0aa
bfcc255ec3d3e7a3 3ff920a1676cda00 3ffad491ceff43aa 3fd39629c85364be
4002bc23284b0594 bfeb147f70d58396 bfe78b94a0032a73 3ff848277c477bd9
3ff1f5aabd010fcc bfe12ddf089a4850 3fd963e7e8e1c2ab 3fefd76b672fa9b4
3fff956029602683 4001b5ec1bee3d4a 3fe2dd3c5e54e9e5 3fdd1c9c064f4cb4
bff2789c607a1b67 40057c3bf052a314 3ff0eacd71d11c4e bfe6e7996341e489
4000c50eb5e29271 bff5b30482d548f9 bfe6654f007832d5 3fbc76cff7e8ff22
3ff05257f15fad4a bfe38aabaea70d90 3ffb62bd5c6ef8f3 400131c7a6a83b1c
3fec7fd0a573f2eb 3fe94ed6c8d1bbb8 3ff3ded27da0ad6e 4003e8d93e9fb570
3fab53a12f34ae26 400027585ccec48e 3ffade8502908423 bfa2b8116a573a95
3feff45839016af3 40009318aac1c98a bff3425e7a4abe19 3fd6f411aa1dc2e6
3ff0ab022d74fb71 bfdf8448df8d2851 3fd3566785a45be5 3fea76fcd2b43d04
3fef090ba41746f3 bfe579e09e7b272f 3ff006d79db4776a bfe912942e5e5d72
bfe30f7f08eff47b bfb883c0a76680e3 3ff737c173bf0bd6 bfe90b69f11b80f8
4003c1ccda4cae52 3ff1630d2a111bf8 400238d53fa2e5de 3fd7ae0f4e653374
4005d3c3b1358ac7 bf99fdd1366790b3 3fdf272a68f08043 3ffd7416f5d8a3ee
bf9a81a24031fc18 3ff78d0916d4923c bff268134a052f45 3ff18d546828041e
400824f2dec21ab3 3ff6916d54ed9c62 4003b7f3ea2f680d 3fbb1e7ca4e8dbf6
3fed8a588af37410 3fe9ca056724127c 3ff1d3f0f9cd3c1b 3ffa06b24315f567
400473673f0ae415 3fe9cabf2b2a49ba bfdf2ad7399628a8 400240c283bdb68b
bfe12ddf089a4850 3ffd276bdd214742 3ff76dd33f1adfd8 3fdf4ef4286e913e
3ff87acd01ca1bee 3fd63065e8e40275 3fffc01e809487a9 3fd2247d8b51bc8d
3fd63d0a5148b33e 3ffd534f8eaa0c64 bfed8c53cdfc7eb4 3ff068a15bc6ae17
3fb97c8b4389af1f bfe27ebd7e3950b1 3fe4180f7520437d 3fb2925895071a54
bfe2398c8b9e4a04 3f9dfb47b5b0d17a 3fef036bb3c775e9 bfe76f83311d0b3b
3fd1789a9f181ca6 3fea7db997acb7ed 4004139bf4132092 3ff2f16667f90d01
bfca7931eb51632c bfe98fa0e0fc8521 4000c9e1b4656b3b 3ffab3280dee046b
3fe4f6eeabd1f5cd 4000e395c44012e4 3ff722c7b27c4e7d 3fcadec9ae6019b5
3fe55c93c2cb8f18 3fde4dfa11232c38 3ff68a520ef7dd57 3fe6706957d810e6
3fc8f23a2f047814 3ff5b2dac534e233 3ff4096b3555c257 3fcfa65371fd6b33
3fd8590ec71b788d bfc39544ac1d9c83 3fe4146a72dc7544 3fcca592bb23d242
4006378b538d73ee 3fe37c13f0c36a12 3ffcded89f39b223 bfe13d314c006474
bfed0c1d474430ad 3fead10a833082dd 3fec154a7c6a618d 3fbaf8dcd1ddda28
3feaadfdf63d839d 3ff9078fcccbac9b bfe15c68e4c6d905 400250435e7c2756
bfe99ac49b5720ac bfdda514469d342c 3fee4a6cba2eb5f9 3fe6f5417b350ea9
3ff559d4e8cf7fea 3fcba3a217955a96 3febd92f86cc19d5 bfdfb58fe94ec87e
3ffe8742765ea4be 3fe2260c9b19e333 3ff2a03bf8ed9fb4 3fe3dde070a5f651
bfd6f05186e0c2a0 3fd837a657fb78cb 3ff102e274d5ae65 3ffff111f5e95799
3ff66d73056516d7 bfe0f1b0d7a7a36f 3fd44254186a2ecc 3ff4265e2e0bbc2c
3fe28cb2e3766ee6 3ffdaf4a0abffad7 3ff3528a6eeb6884 3ff259785e4dcb10
3ff83fa603caf8fe 3ff11f1d64921894 3fd963f57070d995 3ff1d8d60a947252
3ff8ce1307cb83aa bff1ff5c87668821 bfb48b49643fa8c6 3fbfc9fcd06784c2
3ff499c52beee4a7 3ff71db756b17544 40026323ee2f7d99 3feb3185231fa44c
3ff6236cc1c81b1d 4000b64651456839 3fefd46031ef49ea bfed8c53cdfc7eb4
3fdf8ae5284cf9a8 bfbb40d8eb47c4d3 3fbed7333c08bdbe 3fd1bb52c52b2420
3ff6eada19acc259 bfd87267fce251fe 3fd93303f5ee304c 400111935943d96f
3ff1c654f90d25fd 3f9965cca2cf5dfb 3ff06939041cd06b 3fd212baced53102
3ff2d3ebdd4e298f 3fdb96fa0d4695a2 3ffd25e8a4331610 3ff886d90f5dd8fd
3ffe07118b757fd2 3fd1b1a8a05defed 3fde1b2e669b74b0 3ff97f0f4af374eb
3fef2fd9a8e5e8b7 3fe59530548c7350 400130eed5f1d607 3ff3f86c9b3c6599
3ff9c89117945432 3ffcacae07929110 4005bd17ae08fc87 3ffd109bd5a6ebfb
bfcb16adddc21ab7 bfd76e292c67e946 bfe2b3c52596f4fd bf97c08f48202e90
bfc48d9383a6f5d2 3fff4b82b3ca8fdd 3fe4ad481cb3b391 bff522efe5b03ff2
4004452f09016d61 4001d8b3d2d3e210 3ffd71df5e25e6a4 4002f6f1f7884ee7
3fdbfafa4180cb62 3ff12c1d95bef51e 3ff119841e00a9a3 400089bdd1a84c80
40040bb5c31209e6 3fb3ef32cb4a8075 3ffefe31b98593ea 3fe986c702e695ae
400088cb92b6f0bf bfd932d9a3c9c9d4 3fd93303f5ee304c 3ff0a59ac7783181
3fd42fdaeb9c04f2 3fef4ea94ace5f37 3fe6b28428963add bf8180e3d588da06
3fdabc0e3d039c12 bfc213dcea50a751 3ff2e0c072990b7b bfe7ec6be66a9448
3ff9e0449f9b4ccf 3fff4019c908367b 3fa0373c812cf1b4 3fd6a5adb68de65a
3fda61292788b388 3fece65204e351dc 3fcd4368bd1443ad 3ffa0a77a4f5ae38
4003b3dc2ca9ca75 bfc443c7ce7cd69f 400238d53fa2e5de 3fff09129695ff29
3ff028ec71d77138 bfde768d69375dde bff43b07ed9a99e3 3fdecbcc4247c404
bfdab062aabe4964 bfa8170ad94913b0 3fe5ab089da66d88 3fec498bb7feb2a8
bfec5bfdc1703881 bfc28bbd61d10673 40033079a4b1aa9e 3ffd863abcec29ac
bfd07356a75ec3b1 bfec92200d6d153d 3ffaba94dcffda87 3ff403c8d03b821f
bfe852862ed71768 bfc8ca36e3c444e2 3fc7d2edda54dc5f 400003ca4edf349b
bfedadf23c68f7d9 bfeec1d4f04a61ea bfa55cd143b30c88 3f92b381260a4140
3ff7187b8679f7bc 3ff28fa30eb7982f 3ff173f676e90284 3ff3f33dd9bffd89
bfc1344d29bedbae 3fc058ce1b6bf638 bfd77abab57e043c 3fed168ff90ebbee
bfe648b48aa50dc2 3fe21f2360f37af7 3ff8b29ec98b7887 3ffb90a6748af9eb
3ff703a3e472786f 40044acb9363f2d3 bfe08ce869a7b4ee 4003c449b85699c9
bfd51693631ee6e5 bfeb08e955daf677 3fcad4a364e76f7f 3fe5184c78990f02
3ff019a6c32db3c4 bfd3b30432cd8dd3 40052f51b2a010d1 3fda4e5a2f2f63c6
3fb391573edfd4f0 bfe792135a7a5033 3ffe082ad3f06d92 3ffeb85b449e1d58
3ff41ae38fb49f0a 3fc76d2ff43666bd bfeb6636ce162056 400104a19180d47e
400022a562fd6e4c bfbf995b5d3a3d91 4007bb99dae30622 3ff4c6f66083bf42
3ff6eada19acc259 3ff28ee09403b1d1 bfe9d6065f6b5534 3ffac2139b98f7ab
3fedc3d694549838 3fe748f2d102c430 3feeb27e94d5f55c 3fe37dd9bbd062a5
bfc3cfa4f17edede 3ff60619991de521 3fe4d08ad5fe14c6 3ff08f78694db091
3feb8a5a17f5332a 3ff044a1cd6f50e5 3fe72a7726087105 4002815c1dfd6052
3fe8193aa65ff9d3 3fff2645ee17705a 3ffb4f623c5e09a0 bfeada3995057b86
bfce0037dec622ff 40016b8288b47676 bfca7931eb51632c bfb3e5ad33f1be52
3fee0d9bdc359858 3fee25cb72fa622b 3fbac5db9afc4bd9 40004092292b62e6
3fd9b9c983a239e9 3fd1cd0177ed34b1 bfc902caf090be73 3fddff65ac89a6a3
3fcf9a2c6a0dbdc5 3ff9033332916084 bfe5efa94b4560f2 bfd4aa587b7ac354
3ffd94764415e466 4004452f09016d61 400335b9162037ad 4001dc3459cebf88
40039dce7e8bfdfb 3ff0e696e6a93cb1 40070f876b15af51 3ffc9083ff749154
3fe2255f88eee5b2 40061f3e352b3a5f 3ff1db95a564fcc8 3ff323728dd17a0f
4001e6b196dabf44 3fe01369e95ef1a5 3ffb6c1a3d8b6eb1 3ff040552ff39092
3fe0bc27643fdead 3ff9472db2b8a300 bff5b30482d548f9 3ff5edf815cd5c49
3fd2af6281dbffb4 4005bd17ae08fc87 bfe36e490d3e868a bff123ab80201a0b
40007d2cb62045cb 3ff4cbbd1d1543ab bfda12df394214b3 3ffbbaacd724eb50
40022f8d0436ee12 3fe2250638feaded bfe03dd4d04e5340 bfdc8f1723c0f9f3
3ffb543533fad77f 3fd04e2ee6ae7196 bff10461989c2318 3ff7ffa1e2b85e74
3fea9959cc08587f bfb883c0a76680e3 bfeb9919cb6e29e0 bfd8004a173935b6
3ff8a3ec36427aa0 3fe6d88d8902f892 3fec2ae75f73ce43 bfdb1b38aff27291
3fc678467bbebfcc 3fdcff899738fb65 3fdbe4d6d58a3d92 3fdb3fb4f626fb6a
4003a56c05c03855 3fdcab5d33555bd9 3ff86056cf171db0 40084722737af626
3ff80e9a816f7620 3ff3f9b3a3d218a8 3fe61189ad0b08ce 3ffa8f4a33ffc9b3
3fb07d362fa7793e 3ff044796d67e358 bff14e204b7aaf9c 3f3d84217ac56168
bf7d27b803b959b5 3f94fd9774a178aa 3ffb14a539a732d1 3ff6e8c25c4b75e9
3fef368abb826017 3fecba0a85cb146e bfa59b2b13e262eb bfee981676c73358
3ff9907f3ed11ef6 3ff2aa9e995916fb bf86fd380b4bb162 3fdbe4d6d58a3d92
3ff1d7c7454ec3f7 40044c62064b74a9 4005d56de701dc76 bfd1df79dc72a9f7
3fdb0ea64944a1f2 3ff17b0241cae3ab 3fd5d4e0918d5ae6 3ff5f04aeaca4e88
3fea9a74250e3c02 3ffae5eece76af33 3ff28da2e45870d3 3fea1a6de1c91e55
3fdbb06ccf64aeba 3fecc6909b513a82 3ffa1156efd9f960 3fecae28b960655f
3fd84996965697b9 3fe96131e0378296 3fe72cb371c6c431 bfe829b822c58f0d
bff44babdc58f712 4007a190b00d5ba4 3ff50e55bc9a08ef 3fb6cb2eb6a78170
3fd31cee544b5c15 3feff93a21eca255 4002023140a69f8c bfa2b8116a573a95
3fed85da47ac2b67 bfb1701533246528 bfd48e706e6c02dc bff149876034b3cb
bfd02949da244abd 3fe2ac6efca8c6a8 3fe6c01e1b3bf991 3ffb0687c2bf11dd
bfe9bf2ba6a407f2 bf79449452f0f0ad 3fc34a58ab6ef3d0 bfd2a22111801f5b
bfdcdf4cdcde5b84 bfe24f9d6a9ae295 3fc447dd2c8616f4 3ff1e17fd5fb6cb6
3fec15f44c597910 3ff31810cce144ee bfd07356a75ec3b1 3ff6e4144430868c
3ffa4127759db0d1 3ff264f088b2256a bfed0c1d474430ad 3fdd4baa1d33313a
3febc268869cc801 bf79449452f0f0ad 3fe8ee46ff11cf9e 3fe470ff426e3d44
3fda8bd7e61b5b8d bfc55314be51993d 3fe8a655ea142193 3fd0a9f9625e9e04
3fdb4d8aa73608d7 40029ef64ecb2c44 bfe6b5b1b150b304 40009ff45602375a
bff5b30482d548f9 bfe30f7f08eff47b bfd3e43ae06b7bb8 4000d9295224410a
3f7d63e4abd69c34 3fe9bc337764cbd4 3ffaee1c17b9d509 3ff6da2918b04cf8
3fc33018fe5758a5 400498500f29f137 3fedace2e68d8100 bfe058fa0f9af858
400374fe8f2df9e4 3ffc7db12553c5dc 3fdfd682804b76cf 4000a4f903293788
bfd2f5ee92ac495e 3ff3bd807850afd7 bf9c61b2332d3cce 3fe882fabd764405
bff287cde3d87cc5 3fdd3375490468a9 3fb9296610948d4b 3ff0e26579c2ea92
3fe5130f377e7d63 3ff22699c8afd7a0 bfb30409cacbf956 3fedf1f5002f5add
40028a41aebc0978 40044e7fc100934e 3fd5e9d39b7b7596 3ffbe0fce6349fc1
bfe95b828ba7784d bfe4b53df9d17c17 3ff68c94db24986d 4007bb99dae30622
3ff5dac905a69805 3fe760a94db30bbc 3fe107dc3a0384c2 3fe6367cfbde25ad
3ff4b2ff743c2271 bff580f415f3f58c 3fe49423f7c56b2d 3fcfd62f629a8ecd
bff580f415f3f58c 3fbe84d9e1653c45 3fc2cc97ccac618c 4000ea08416488b9
3ffac91437ed44fb 3ff2cb66351dd1aa bfe280339a1c3388 3ffd8cd926cb2d15
3ff62aa4dbdbe88d bf983d3d81fec8f6 3fd50a4668dfbcfc 40043db2ba17d5d7
3ff194477cb1aad9 3fe1262d665ce166 3fe96e97c0259cfe 3ff52b9cc252b0a9
3fb3836cae2f9327 3feeb3e4ba1ba962 3fe2255f88eee5b2 3fe2b79be07d2cd5
3ff217c7bfc993f2 bfd555b236d68644 bfec3370e863edec 3ff48d49eb72b390
3febf1751a3654fb 3fd32c7870bc2e06 40053738a14a5201 bff44babdc58f712
bff2da543394762c 3fdf4651002ebff0 bfb4b6d16991b450 bfdb00bd2be3667c
bff09bfae3a58119 3ffb06366d9ec647 3fdd414ac8580cbd 3ff97fa8721c269c
3fc8a97134519359 3ff9247065c038ff 3fda29155ee9bc27 bfea23667a350bde
400131c7a6a83b1c 4002eb6aef01bb67 bfd2efdc4e0a6910 400016080f5256a7
bfd47d3e9bfd6c6f 3fb1cadc1e9d8f01 3ff6eada19acc259 3ff61023c0f92dec
3fdab22f927dd9a7 bfe91e0b43cb9cfb 40016c7a84eca126 3ffff1625fc427f5
3ff78d322309d280 4002815c1dfd6052 40059fdf16b58a55 3fb8e895b17b8659
3fe88ba4709d42b1 bfe0f95281183aec 3ff247519ed5ff6e 3febe50db52f3162
bfcf96e19e3cefa9 3fde0bb13ac2e017 3ff6e4a0e7aa8af8 3ff6b9e78c47cd9f
bff2658060425452 3fe7b8aee416a4ba 3fdfeb9563081ac6 3ffc15bb2038698c
3fe061923c6b8dc9 3ffac9f40628018e 3fd4cd5b4771d7fa 40013fe7137add54
3fd6a5adb68de65a 3ff82ec0e83c6502 bfd1340cf2a50433 3ff5dac905a69805
3fbc76cff7e8ff22 bfdab5c3c2d14910 bfedadf23c68f7d9 3fdff39fa9c2e6bd
4001d8c8a24e5d2e 3ffafb8dbf3f2624 3ff3112d89550087 bfed0d3e8fbf4443
3ff0d32fb0a8ee8f 3fef3fe560a530d9 3ff8393d7c80b06b 3ff4c53440e86235
bfd2a0ea963d0943 3fec856b0bc3385a 400067dceb453c8c 40015a767545de54
3ff7c3b1e2eb21c6 bfdd4d00eab293d0 3fc53214b08985b5 3ff6a14d219e483d
3fef260a62c04b54 bfc4eeca3fc08195 4002815c1dfd6052 3fe94ed6c8d1bbb8
3fe4b39cb78cc5fd bfe048b2ea982f93 3ff47b7b107bd1e9 3fe37dd9bbd062a5
bfe4b53df9d17c17 3fd7545e57d86742 3fe7673a038444e1 3fcfbd9846281be3
3fe425923dfd0568 bfdf63d568a49a44 bfb71d772879a470 3ffca67de8f4e3ed
bff268134a052f45 bfefed4d6a042dc2 3fd258379b794401 3fc609ab5b0e77c6
3fd39629c85364be bfc3fde9e195af00 4000c6a54b7cbe68 3fd258379b794401
3fe1262d665ce166 40031f934238aeac 3ffd73ab476d6a0c 3fe0e4b759006246
3fd5bc69b40ccf52 3ff470bb3821f04d bfc01a26209e29d8 3ffad01910ef1fe5
bfeaafb7290589af 3ff6c169858ad46e bfe815ec5b151742 bfd542c53cdb5514
bfc54fcdd162e558 bfdadb62cefc7c50 3ffdce0762252caf 3fdbb06ccf64aeba
3fe1aba7935b359d 3fce08729ca4bee7 3fe0d63dac2b5d45 3fc0bcf5dab79418
bff0d73c211e8ba4 3fe1f59d2f447fa4 3ff5aa8e1a6dd614 3fafe496cfda115d
bfd213ace0868419 4005a24db85cf181 40029ad9d7ae6c8f 3fd93303f5ee304c
3ffa859e66b49f4e 3ff5ae2e32412c1c 3fb53d0be165ad47 bfdf50afe434e9a3
3ff627317d808bd4 3ffd73ab476d6a0c 3fe9bc337764cbd4 3fdc055b11072eed
4002e29e9c524cc6 bfebd1ee8ce39d9e bfd7082bea50443f 40012327b8e831d0
40058b688959d3bc 3ffe8217d9577ab7 3ff853af63f001f6 3fc157996767d0ac
bfe62595127ac0cb 3fe2abd8c12288fb 3ff1c87bc1983edf 3fef803c5107d8be
3fe6ad50a2522647 3fe5184c78990f02 bfa7c876dcf25a69 3fd215aada664652
3ff3533dce73c437 bfb6b51bf63be02b 40065f2e5c026faa 3ff313ede9a0e483
bfe0120cd99e65b5 3ff822815dd5857f 3fe4146a72dc7544 3fdb0529ee377bbe
bff580f415f3f58c 3ff424c6fbec1f25 3fd0e40d78055e51 3fd571a88b87d188
bfd842d377df8a4d 3ff474ad518b6d46 3fc9f5439049585e 3ff4304aa57842ad
40015cbc3492efc9 3ff837f71c16e7d4 3ffacf6c3e3a4bee 3ff194477cb1aad9
bfe80dbbf58e517f 3ff18d546828041e 3ff1431ecc9a4b3d 4000136530774467
3ff225c613625626 40003bf92aea7a0f 3ffce6b7e084deaf 3ff2cb66351dd1aa
3fed2bdb1080126c 3ff07281c201a9d6 3fe7c1ed27ee35e9 3fe2954a18184c3f
4002e78effc8bf59 3ff2f1b8a7368e0e 3fc82f35e18064cb 400362d0999977d4
3ffca97898dd3ff7 bfaa5733b45aa40a bfbce7b954eb7e70 bfe13908e4fb80c4
3ff3ebb3cae8b9a1 3fd1df3dd9f18bfe 3ff088f517bf99c8 bff29d077106bb8a
3fe5184c78990f02 3fddd93b5d0b5976 bfcf5fe6880f8996 bfd55c5830d68dd8
4001071e6f8abff5 40008307ac4e1d1e 3fe2e415fe2178da 3ff3b3d88d122b3b
3fe6b4d062ae0879 3ff732b0ca73015d 3ff8091bf5d53e07 bfe0cd3f7cb34de4
3fe3e986e8a58faf 3ff2260473e1e461 3ff8a67376d3fc9b 400285936f5b5e9b
3fcaf6748137ccae bfcf0b49547a26ef 4002a347da1fda80 4004452f09016d61
3fe8a99d514e0f3c 400238d53fa2e5de 3ff6916d54ed9c62 3fe354ab27a71ace
bfe21f6b0a1034bc bfe09f92b8371ded 3fe8a03022f784d9 3ff323728dd17a0f
3fda9ac7e0d9b436 40044c62064b74a9 3fb8cae13cf3e8c0 3fd8f1a296a580d1
3ff069732c42bb29 bfd72ec99e63a437 3fe4881cd57f3ee2 bfe9bf0a8da197ac
3fea22eb0b522153 3ff044a1cd6f50e5 bfe05c132c02a725 bfe76440d69de24c
bfd0f3e090c2b891 3ffcd61837a243dc 3fedaeb93335c047 3fe08182ac405e36
3ff04502a9c06466 3ff8803d896c7c1d bfe3793e6627b62d 3fbf4aff00ffa9f9
3f66dcd9c48f9fc8 bfcef8cc366538f6 3ff31f87b2eee9eb 3febc268869cc801
40014312b03e0927 3fd5615e373235c7 3ff2f29d2be38a08 3ff6c927628d7146
3fe3b0820689fa64 40061f3e352b3a5f 3fe0a753869ba1e0 4000c9e1b4656b3b
3ff4fb4b848a30f9 4003e8d93e9fb570 3fe72a7726087105 bfeadbeaaad7d13f
3ff48d49eb72b390 3ffed75b6fc70de3 3ff953073093ac15 3feae201d1e7e8e7
3fcff57f5ff46031 3fd2a5e06137ca7e 4006be43a51ce432 bfe0f1b0d7a7a36f
3fd8a2cdc156e7c1 3ff34cb0e9cc572c 3fda74ea3276d577 3ffefe31b98593ea
bfc81d647172b575 bf9c61b2332d3cce bfa8f4df14f4d6d3 3ff454e16d2489b0
3fbd3aa8e608d7b5 3fd76205de27219d 3ff60f67315ef71d 3ffd2904e486eee1
40031f934238aeac bfeb08e955daf677 bfe65f4c2c48c295 bf7f51026242e604
40053738a14a5201 3feff93a21eca255 bfe2a728a0578251 bfec5bfdc1703881
bff3a12638532f25 bfc38ac12ddc0e2d 3ffd6cebc7e3a284 3ffc2de9591ca3c9
bfd6e9a74f45b37a 4003c1ccda4cae52 3ff429565c52d22a 3ff10a0defd7dae5
3ff9f9e36976f33d 3ff3625a664d7820 3fe09bd88af6af18 bfd8d4bbcd0bf98f
3ff8cb66f76856de 3fdf065fde691f00 bfe22c99223578f0 bfcffbdaa54efcb2
3fda3eaebccdfb8b 3fed9ee0cafd2e47 3ff22786018bc5df 4006be43a51ce432
3fc6d6a9a393ce7b 400259fa2ff5f362 bfdab5c3c2d14910 bfcf5a8e10793c56
3fdcb4c6e65254c3 3fff54899d4c99c7 bfc886183aec17b5 40016150f988105b
3ff65339d41b199e 3ff770a81b7de60d 3feae84a3f2563f4 3ffe8217d9577ab7
bfde768d69375dde 4006be43a51ce432 bfd932d9a3c9c9d4 bff18b352ffcc57f
bfa27bbc510fbe5c 3ff7a62740079c04 bf98f054ecd29ac9 4000bda0ac9ed9ac
3fd5d97754d3b6c2 bfe5d8d48774204f 3ff266bfb2cb076a bf983d3d81fec8f6
bfd175a499fde504 3fcd06279a8d90ae 3fdea3a3df0c27e0 bfd6b9687369cc42
3fe9c28b304077e8 3fe393d4dd23a0e1 3ff8535d2a6a397a 400183b6d2969f3e
bfdbce891622b03c bff3d280a676831f 3fdbfd57e3579164 3ff31ee9760b2381
3fc5962c2cb9e4e4 bfa90c9f31203ba4 bfbd07ebb44c527f 3fc98fd0be6fe625
4002a8551cffca04 3fb23a4f06524abb 3ff8e861d2b7c778 3fd80681d102652e
3f8b9f770238b16e 3ff92dc885afc18b bfeaf9745ad2f813 3fee7c83401fc371
3fd1055ba1b26808 3ffd82294ca28074 3fff4019c908367b 3fe2fe50baf627c4
bfc624f78fff0acf bfe24a97015f1a61 bfeb09bd2eb43f26 bff0a20760d2c1ff
3fb2a7920a16ddaf 3ff79aeb6e31c665 3fec5c16d350f2e0 3fef6eba086f7dae
3ff66cd96ccc93e8 bfad1bb8d5b1fc8a 4000d3940b970ccb 40033079a4b1aa9e
4000aeaa753d82e5 40047d7cf82725d1 3fe02acadf1afcf0 4005863a43155b98
3fe1bfde5344b13b 3fc83fbd4895bcbd bf8341b617bd42c6 3fd48acb452a8f82
3fd591fb8247fdbb 400209c2299fdeca 3ff04502a9c06466 3fe31681ce19a8b6
3fbe734ac8b8add1 bfdcbf60be6399e2 3fe2df9af0340fca bfe11c4506619fa8
3ffada703426f0b3 3fb9f31ed3f56e2d 3ff014210a65607b 3ffbfcf73b95c5f8
3ff2d44db25a41bd 3ff47ccf23d74e41 3fe0b15f6b8676b8 bfd72f705a9861ff
40043ebd5ddb1130 40010bbff5a689ad 3fe27b87af5c2911 3ff73d30fc5f67d3
40037cb4fee3e08e 3fe08182ac405e36 3ffb8c681d775537 bfceead3b2c9fbd3
4005d3c3b1358ac7 3fc5ad24a1ebc751 40021c761fd962b5 bf77e891a0c979f0
3ff13b2e538a980c 3fe38784e2abaa99 40005e299c698448 3fff9a597c57d775
3fdb2ffa6bab7a0d bfc443c7ce7cd69f 3fe239478331a819 3fffbb1f30fac7b6
bfed8c53cdfc7eb4 bfe374d0c7453651 3fef555e702fdb51 3fe0d14f5863d642
3fdb04c6f72cf6c2 bfe62539ea69e3c5 bfe048c4de34770e 3fc91fcf25fc94df
3ffd4d6a012e452b 3fc5ad24a1ebc751 40060e8702ffe27b 3fe143df5f2b44fc
3fe8d595803e5d69 3fe5331df1f934da 3fdbfafa4180cb62 bfe36d63f292d08a
bff18b352ffcc57f 3ff35416a0cb56fb 3fed51730e5b0f4d 3fbda88ce2a8f313
3ff6b066abb39a9d bfb030342b21d816 3ff6b8c3b0b93970 3fe42e7e9d164258
3ffa67f048786ac1 3fec3800e8600ec6 3fe5acbc0330bc2c bfe5d8f05c0770d7
3fea9d6a3b261821 3feb5c27e768d36e 3ff459cb38de169a 3ff61f6b838d65c1
bfdd4d00eab293d0 3fe944fecc3730e4 3fdf305b960acdfb 3fd6e6ae7270c6df
3feabbb1f6c9227c bfe08ce869a7b4ee bfd55c5830d68dd8 bfeec5d97b550439
bfed8c53cdfc7eb4 3fd5d4e0918d5ae6 3fdc3af5109fc2c9 3fe7673a038444e1
3fe9735a392a1383 3fed4a2bf0bc215d bfa8ed12a11fda1c 3ff4fe8a243f7a7a
3fa7dbfee30226dc 3ff440bbdec458a5 3ff1397dc8b5ce0f 4001b7175b1a04aa
400238f31ddddb0d 3fe4ecae4d1011d2 3fd8f1a296a580d1 3fea6ddeae629af1
3ff2d44db25a41bd 3fd72a0312f887a6 3feb9fb3d75aab66 3fd13d9f48d6f75a
bfdb5f3bf519173a 3fd91bda234cae8f 3fe894d778226a90 3fd09880c2f8b845
400287bbde78d05a 3fd80681d102652e 3ffa42e961caa97a 3ffa320b0414e36e
3ff8bcac22f94dd5 bfebaf504fc8d6c8 3ff9d800dff7db36 400498500f29f137
3ff2820faeb6c99e 3fa60842723428f9 3ff57859020f0992 3ffaedbcda073cd6
40031f934238aeac 4000de3670365f05 3ffea66ab7f190e0 3fdee5e574e7e5ac
bfef64ae5be1bf08 400502d814609c89 bfd09300f437afe9 3ffb4e311d2ee095
4003c1ccda4cae52 bfb6de35919b723d 3fd1b1a8a05defed bff2658060425452
bfbd07ebb44c527f 3fb23a4f06524abb 40050f1fa6b88313 bfdc211117c23f3c
3ffea764e3df2927 bfd30ca15bc39cee bfe21f6b0a1034bc 3fc02cb0d5d58671
3fba97781dd522f9 3fef3fe560a530d9 4002815c1dfd6052 3ffa1156efd9f960
3fedc6653332abae bfc6e46e8dabbdc3 3fe600bb1f756d8a bfcae14f06c1387f
3ffcd3e71669f430 3ff4e36ebc9b0f99 3ff37468ed18e0d5 bf1f1b12e51c1375
bfc5c5b345d350c6 3fc8719027de01cc bfa5d34385d55158 3ff03a2ddd865d04
3fe4811f47818ab2 3ff3581e6bb47a78 bfc65c44071a441e 3fe498950ae41f8f
3fe94ed6c8d1bbb8 bfeb15aa606bfff6 3fe9424e9eadade1 3feae36a6abbb9c3
40008ced0e7f0955 3fef0da86015d12c 3ff2183e96c91897 3fd9891c969733bd
3fc0ba5defe3bab4 400469a37b6d63cd 3ff45a6a71deed9c bfa947ebddde8c86
400022dc2b3270f9 3fdd1ee7a6362c83 bfeeaa1a55f46caa 3fec46a3a2da1036
3fc3736783da2014 4000ceb4d708b6bd bfe45749ce8bfeda 3fff41d6186b6fbe
3fe98b540dbdffe1 40022f8d0436ee12 3ff03668b9fa449d 3ff552238343a2f2
3fd5baf2419d91b2 3ffc2d7614dc9204 3fd873859c906958 3ff92f08d5b5f0cb
3ff474ad518b6d46 3fe1478b686bdea3 400343489ab622ed bf996ce081ddfbfa
3ff5aa8e1a6dd614 3ff52a94c14438b8 bfd1e81fca328632 3ff8091bf5d53e07
3fe8cc53b74f6fd6 bfeec1d4f04a61ea 3fed5146e0b641ed 3ff03dff1fb2eef2
3fec1e1fcfc5db08 3fd93303f5ee304c bfa075ae305b14b3 3ff214e39eaf95be
3fc92b583ed5e946 3fddafc3abfc0304 40025ebbe8605f22 40005b374e0755ac
3fdcc40aa4ab2258 bfad632d761cb5b2 4001c3690040e083 4004083a631b0b7b
bfb5cc80b85e2f1a 3fed158de1d1efed 3ffada703426f0b3 3ff5dac905a69805
3fd4005c3b255bf4 3fc875b8d6d43ffc bfd0fc3e256b196d 3f879bec4f2c9ebe
3ff3d9617908cbba 3fe68d15608fe53a 3ffed75b6fc70de3 3ff7a0476ddc6439
bfe4d15c105b7d93 3ffdd7cc7ee82ca3 4006e31bc740d00c 3ff06939041cd06b
3fbc28ef4b483d09 3ff7e6ea377625fe 3fef1426b86c8947 bfed8c53cdfc7eb4
3ff290b5a9613780 3ffc9cdd751af3a4 bfe755eb1f0706fe 3fddbd1b62b444c7
bfd25a394aa3765b 4004c2b97ac76ade 3fec1d1345bf522b 3fdebd33b3c48689
3fd6aef389f45828 4000b42dc5401d0b 3ff140a7d47066aa 4004eddae48f5b7c
40045da878d56c81 3ffb5bb9c8b1cbf0 3ffdae1521ae1fa1 bfc2558726e3ace3
3ff8cb66f76856de 3ffd020c0904cf76 40060e8702ffe27b 3ffe909952174607
40058108ef73afea 3ff73ea2eec9984a 3fed6f50d48f49db bf99f4cd747f366c
3ffdd2d531cc00f8 3fcbbdc9f7bad7ed 3fdc19955dd2f7a4 4000b428b8e3eca4
bfde768d69375dde 3ffecacc84616d44 3ff1060b175850bc 40024362c8e86500
3ff62ba9391bde44 3ff1aeab2edb148b bfcdc317e3ade285 3fc8719027de01cc
3ff9a47720f97ed3 3ff9c6b426a0a506 3fad509cfbfb0ff3 3ff5a34dfbb2505f
bfa8f4df14f4d6d3 3fda7300365f7e9d 3f879bec4f2c9ebe 3fbe6cbccae86554
3ff0c7fb112b92bb 3fe729cd4868d889 4002deaa8aeb385e 3fd0705413350baa
bfe4549d43294f01 3ff5c9758fd81f1b 3ff1247fd16f32af 3fffaf58de06ee47
4006a62e67369ac0 3ff52d85b5802d59 3fe8e6fff80f9bd3 3fc58ac8aad6a7b7
3ff6668909ccbad4 3fdb7c3370450a78 3ff37a7b713dca8f 3fddc00f82ca0183
3fed344e1a3a185e bfee5911ea8e3c56 4002bd71aeaec284 3fe27e9e84899fe9
400496fe22c7dfa1 3fe0cf5a8915c18b 40011543e34ee551 400533724fe4817d
bfc6e625da1b17a0 400496fe22c7dfa1 3fc2cd5b53fc5473 3fda61292788b388
3fcec1537b77f046 3ffcdd061b428f48 3ff6908f84bd59fb 4001d9f12aa05f6d
3ffda8e3d7a187ae bfbd92e2d523a481 3fed7a39c975820d 40019a2ab165ea6e
3ff1f88e2aa0e636 3ff2788af17df1f4 bfc81d647172b575 bfd05b9d5bde7054
3ffe31b26a7bded8 3ff224994b054288 3ff841c61c093d82 3fa6dedda75ec30e
3ff4db27f5f33975 3ff102e274d5ae65 bfe5b4acea89978b bfdef7051c5b5f18
4003ab92df170ba5 3ffb561e0fa79357 3fe6f5417b350ea9 3fe3ceda70d7a422
3ffeba332048ef42 3fd48acb452a8f82 3fe8a07c714e4137 3fddc362d1aa87b7
3ff2674cca045e18 3ff452c761fe6c51 3ffbd3fe9388ea9c 3fca9b5ba711c8eb
40068c253c60f042 40013a21f3337e27 3ff2f29d2be38a08 3ff044a1cd6f50e5
bff5b30482d548f9 3fd669cba3613d63 3fb5b8b1d089974f 3fdb48b98bb7c35e
3ffa1990694be66d 3fb3ef32cb4a8075 3ff979df5bb85930 3ff75196a47c4620
bfd2c77556648205 40021b4587fcabd7 bfb127531991125c bfd11a1f03aa982e
3fef5df13ce8da80 3ff9d150f9b13fc2 bfc990ef22d69891 bfe3ae907cd96507
bfec644d8990a756 3fffbb1f30fac7b6 bfe3bc9db248481f bfe402d7aeb72b48
3fe87dc42d073e46 3fe09dd58abd0b22 400104a19180d47e bfe9631455747f58
40017b2b8cb1129c 3ff454e16d2489b0 3ff799953635533f 3ff5514fc6fc454a
4006fb3d5cf284c7 40070f876b15af51 3fd1618b03fe6f62 3ffbce7baeb0fa2d
3fcf76ede4bf22a3 3ff040552ff39092 3fe0e2f974fc4d26 3ff5b95d3face0d6
3fe4bc318f7a9210 3fe37dd9bbd062a5 bf81d7687eaae840 3ff91bf28ec5d467
3ffd6faba6ca4e58 3ffcc50b8ba3980d 3fd4c6deacd8c686 400470ea6100f89e
bfeec5d97b550439 3ff96250d7e37c68 400496b62815228e bfdadf235cf9ee14
3fdeee5c699b2ecb 400357e9bfc707a2 3feb7d819c413988 4005713d13441a8b
3fe7c4afcf97fc2c bff580f415f3f58c 3ff141a042d372e2 3fd58ad9bfc4e89b
3fe9174101ea5ab1 3ff6edd22e6d33a7 3ff7afceb2f1600f bff1c5c367af72aa
3fd5e9d39b7b7596 3fd6bd33b898d5af 3ff454e16d2489b0 3ff65d2bb5763125
bfd746c90ae4b370 3fe104eb8fb7488a 3fc4d0f2d9941b8e 3fefc8cb95c8e82b
3ffe5a6b895ae3fe 3ff454e16d2489b0 3ffbab280cdf1935 bfe11bccd2bc7302
3ff3401ee883ec4f 4002e0b9877f408a 3fe95ade3d24f4fb 3ff63a903cbc6989
3ffb4f623c5e09a0 3ff8f89d2a84b7d0 3fe4a0c4b3f6ecf4 3fc9472729e0041d
bfc46bc68f23ac14 3fcfbd9846281be3 3ff7b7d6cbc0a3dd 3fe84d6c8d68c55a
bfdacb7b4f90095a bfebf95bdbddda32 3ffa8df8976ea4c3 3ffd77a0f7e9491b
3ff3ebb3cae8b9a1 3fe7e6e20ab16e0a 3ff215f895b0b1f2 3fe3970b5c24e1ce
3feee4b36ee8a458 3ff0302fffd776dd 3fe0997563490cef 3fc1ef7976d96eb9
3fd2754a68b760a9 3ff30f48421c62a1 bfed0c1d474430ad 3ffea22213fde824
bfed825aff249e9a 3ff49d27af81ca5c 3fd0d43532675ad4 3fd600732d4eb786
bfd25a394aa3765b 3feed6dd51d0f181 3ff7f02f40a5d159 3fe45e7d9cc6008e
3fd3782f1285dbd8 3ffe3ec414a6f59a 3fbd3aa8e608d7b5 3f9cd8b3a74a05c1
3fda37315a1b2b87 bfd6f05186e0c2a0 bf983d3d81fec8f6 3fee713c91090af6
40010ca1bad6c2d9 bff580f415f3f58c 3ff215f895b0b1f2 3fe894d778226a90
bfda1bf4e05cf699 3fe15f167d09fbf0 3fff08ffc8fef51e 3fbe5e644696f54c
bfb6566023ca03e9 3ffcaa51fe2a6e48 400470ea6100f89e bf983d3d81fec8f6
3fe709ae60397acf 3fe29b8b9af87f4d 3ff8d99607fee9ca 400228de902f2095
3fe1478b686bdea3 3fe44c1a967936de 3ff81712bba0b8d0 3fef1411a7db0df2
3fe5b4b976daa425 3ff0eacd71d11c4e 3fd8338dcc1df77b 3ff6ee95bd66ef10
3fcaf570d1123ed5 3ff4b2ff743c2271 bfec644d8990a756 3ffcf7ffe3e7d5b9
3fe6f5417b350ea9 3ffa389569d868c5 bfe8aba2bb9155e3 3f93f90be46c8711
3fe1bfde5344b13b 3feb5ecf310ec882 3ffa1156efd9f960 3ff4165bf4b911ac
bfb2635f2c6e7691 3fc898211c8e2f09 3fe5bcf61048c23a 4004f44c8fb5b7ad
3ffc377f6f2a9260 400152fabd3d8d3b bfe95f14bab05dfb 3ff01d6eecd1f064
3ffae5eece76af33 3fcbae6fcbc1840e bfd2efdc4e0a6910 3ff4b2ff743c2271
3fec29c7c041d191 3ff78141fc9a51ef bfcdb7ef074ecdc7 3feabbb1f6c9227c
3fedfc1eb38f4766 3fea4c41eaec7f3c 4001a35ac1fba4ed 3feb617a14f661b6
4005461ba97c29ed 3ffad8927eb6a809 3fdece0b6e2a94f9 bfc6e625da1b17a0
3fedc3d17ed0ecb0 3fefb7de8ea2aed2 3ffce18dbaf9718e 3ff81abac6577246
3fb09f2577339df4 3fece6207ca2a6cf 3fd571a88b87d188 bfd7d7f488440add
bff07b2548428ba3 3fea5d0da3099c60 3ff83e0f23834686 3fe1a57914fd6a46
3ff98311062882a3 400648ddc390ecdf 3feada5956a264c4 3fea15fecd9fa9cc
3fa8e39faa79a053 bfe9e6346790bd77 40076f265bf2bac3 bfa258da92d328d6
bfe68498783828b8 3fe75f3b258b57aa 3ff1e9ccbee7185e 40046eb525973731
3ff30a082d94283d 3ff06939041cd06b bfc4b1b0949270aa 3ff7c3b1e2eb21c6
bff03dc8922d9abb 3fbb8d651ce4435d 3ffada703426f0b3 3fd3f40f2bb44712
40021ddf1981baed bfa01c778db50182 bfe4ff0fd7ccbd3f 3fd9e7e8c1de4f6c
3fb248dc8949e19d 3ff008a22e04c772 3fa4d8255f09c7fb 40022f8d0436ee12
4002f949f278fc02 3ff3a64301c8095e 3fe6b4d062ae0879 4000b542435e74c0
3fe816dd03dc6f72 3ff27959345e43f2 3ffa320b0414e36e 3fc79429f7d48b79
bfe07ea453919f84 4005b22f247ed424 3fe3fe22c075c4bd 3fcc28fdaad759f6
4001d9f12aa05f6d 3ffd7fde6e6d0369 bfe4661d7808a40a bfdfdf51a9e92141
bfe1aa986025cc02 bfebcb3c36b994b7 3fd3f5e03356ad06 3ff3326bb971da7e
bfebf95bdbddda32 3ffe1841545e3cfc 3ffff9861b4a4624 3ff526d6bc925151
3ff2176f93ab0d79 bfdb09511b242563 3fe3616721454fa8 40059bdc06767b50
3ff4bc00dec1b4d5 4001f01b8046a0b5 3fee3099f55b1bb4 bfb4f7ab9eb3bc0c
3ff56019e27b9714 bfdb09511b242563 3fe1809d2303110b 4002611733ace0e5
3ff5d825f82e8c6d 3fe94c9f02d0a1bb 3faa9e02ea2565cb 3ff65f6b80043a1f
3ff62aa4dbdbe88d bfd9026769c03a3d 3ff3c06286461086 4000c9e1b4656b3b
3fe0249230de0f5b 40005059a503335b 3fe8692f3b11b9b0 3fd8f7d5a7294e75
3fb59b5e0b09e008 bfcb3502268390f1 3fcb238f7db9d34b bfa4a1dd2e3c5d98
3ff2d229ed2a89c5 3fcbae6fcbc1840e 3ff366377d856163 bfeabb3224c2a0f1
40007d9d5714e189 bfd8cddc6b3149a9 3ff5323548f6ea9d 3fe8692f3b11b9b0
3fe6485b557d450b 3fe3101d4b8c8243 bfce09cd5f0b04c2 bfecfb924345daea
3fec1287b1874b65 3fe2435f8c014c8a 3ff00d9498c176e4 3ffef7228811f44d
3ff58fcb41a6d70f 3fd3e2c0eb5f60c7 3fc70f45eabfee49 3fd3d971964d765f
bfcf0b49547a26ef 3fd93282cf12b4b8 3ff5dd970a053165 bfb9668ee38913fb
3ff247519ed5ff6e bfe852862ed71768 bfd72f705a9861ff bfe13908e4fb80c4
3ff658861f2f32fb 3fddd34de9ca5a2f 3fe8cc53b74f6fd6 bfb34ceceabfa9db
3ff08dcfec5ff8d9 3ffeab9e9d6e29ab 3ff115ca95bfbe8f 3fba117a35feaeb9
3fd63d0a5148b33e bfa4388365bb147d 3ff2f29d2be38a08 bff10461989c2318
3fe09c299fe4362c bff142c240e8e76d 3ff3f6f03802d250 3fc71f761390c882
bf79449452f0f0ad bfd7970ce605cc31 3ff1f7caaffc2ce4 3fca87d1b2c70d89
bfbc5db74039848b 3fdf4bd49532d4d2 3fa02086c009715c 3fe178a3f1038501
3fdeaf04e986c273 3ffec0fdb73856f3 bfeb9919cb6e29e0 bfe0101880a25a5b
3ff49598a897d443 bfc22ecf7770ab55 3fd3f40f2bb44712 4003c1ccda4cae52
3fbe2a01cc5458a4 bfbf37d799f14a93 3ffb0b5c0ff89d86 3fed6af56ef528a0
bfe5efa94b4560f2 3fd044a78e3196e7 3ff2cef496f87fea 3fe142f9d955b757
bfe0154c9b5f7b42 3fffdf98624501ae 4004feec4fd10dde 3ff45414a83ef069
3fbda88ce2a8f313 bff5b30482d548f9 3fecc8132443a80b bfea9261d4af406d
4001f1cf318a5654 4004df80c5cb1c57 3fffc747765b530d bfee5911ea8e3c56
bfe12d717afc45d6 bfb15f2d113fb680 3ff174bfbe01c12a 3ff3d16c6d7a8b79
3fe9e690f1502988 400374fe8f2df9e4 3ff2313c815fbd01 3fe894d778226a90
3fe6bee0b9c445be 3ff16a0282b44413 3ff9a484999acc9a 3fca7949e395f154
bfe65f4c2c48c295 3ff2d77887546927 3fe98d7668db7c05 bfc482dbf7cb96cc
3fdddc1a39589614 3ff655267408bf3f 3fece8f558f7816c 400131c7a6a83b1c
3ff99b3ef7d7fa9b bfaf1b35cd47b6a9 3ffe632a3cdc9ec2 3fec1e69dcc15ec9
3fcf837b58dfc840 bfea4ad3c92f4872 3fe0b2cbe33d3b5f 3fd0a9f9625e9e04
3ffd519212f98f9a bfd4e8006aed92da 3f7435ac295a8bd0 3fd3f5e03356ad06
3ff6d2bf4bb3af2c bfedadf23c68f7d9 3ffaacd5edf90d4c 3ff91112df2ecc14
40071364e6ae7e2a 3ff816791c0485c0 bfde768d69375dde 400196fa5f520748
bfd07356a75ec3b1 3fd5f7ee53984e11 3fe4180f7520437d 3ff527a0325db733
3fd39629c85364be 3fd93303f5ee304c 3fbc76cff7e8ff22 3fe5bcf61048c23a
3ff319918502700f 3fc4b8039c73d48d 4007ddc96f9be195 3ffd68a45680ef70
3faa84f61fb9c868 3fea8a120e3161fb 3fbda88ce2a8f313 3fef109ea6d04452
3fa028aff92f03df bfe158bfd81c27eb 3fe22b874705c0fa 3fe4146a72dc7544
3fd7d2b21df1e732 bfe78b94a0032a73 4001f1cf318a5654 3fb59b5e0b09e008
3ffc0e5e7d188212 3ffb2ced864ae2de 400439db5bff7e96 3ff55054592ac0f2
3ffb53a5f729c282 3fbfe7258c082a7a 3fec6cf3ac3870f1 bfa94466ffe578ab
bfd11a1f03aa982e bfb6fdfd2cbf4dc1 3ffc160ad77aac96 3fe1393d4259e218
3fff65943549e2f8 3fe1c7ef5188599f 3ffb4e311d2ee095 3ff6d01b6c8bae27
4002611733ace0e5 3ff2de1ed41db72b 3ff02d18a6236d0c bfcf0b49547a26ef
3faa84f61fb9c868 3fe7e6e20ab16e0a bf94730a2ecc0a6c 3ff896d2475c2d55
3ffb18faad3a6c62 bfd348be8b63d911 3fe28db686a1d449 3ff0f6dc7d02a91c
bfb75121795e892d 4004f6c96dbfa324 3ffbbffb3258fa92 4004ddb373949803
3fd48fc6a2442a79 bff2789c607a1b67 3fea1703baef3367 3fd4753eed89a870
3fdf272a68f08043 bfe321122a36bf6a bfe098d59b5f75bd 3fdd6599dbf621e3
400088a95389f4cb 3ff18f555d360b2d 4003f3cd2da3549a bfbd16cd1be9c3e3
3fd91f01ffd9acf3 3feb39f5a5dbe96e 3fdbaff064ec2c22 4003b7f3ea2f680d
3ff80fd7522d9330 3fec3ff039620456 bfc7388af6250aa7 3fe58e0320c5fe58
bfca7348c6c6235b bfd739c6f2e4d633 3fe22b9fa20cc94d 3ffa16463288af93
3fddaf94e4692c9f 3fe4a0c4b3f6ecf4 4000bb02c6bf6ac8 400111bb8b175345
3fe2533e2b85f422 3fe2abd8c12288fb 3fe8d414a48dbc7b 3fc0a116bb3f83c4
3ffe274a1b4209d7 3ffcb516dfa8e44b 3ffb2ced864ae2de 3fe0f0c0c18cd10d
3ffaacd5edf90d4c 3fead2a547a89941 bf82382b521ea934 bfa0c4150cd5dd8e
3ff5f2d9c5efe57c 3ff627317d808bd4 3fe74a74be347850 bfe721fce70f6708
3fd6bd33b898d5af 3fd6a08efcd34749 3fdef8a409edf064 3ff3b3d88d122b3b
bf9561f576bdda4e 3ffce6b7e084deaf bfe0f1b0d7a7a36f 3ffdccc180e7e3c4
3ff848277c477bd9 3fed9176c0d64d48 3fea653a11823f09 bfc531acef53ec2e
3ffd765393306996 3fc0a116bb3f83c4 3fe4562b0143ae59 4002186b406d7164
3ffde106e816029b 3ff21d6f9f9c3f69 400275e7c3c43cf3 3ff5f34fcd2e24ca
3ffc2034680f288c 4002195f36a006a5 3ffa66a4c217cbb1 bfb26f16fb43923c
3fe5750109055bb2 bff2d6a090bdd101 3fca052393b8d0d1 3fecac36914abfb7
bfc5a37531de0506 3fcc6ee90d954ab0 3ff8a0ab69491c4e bfd6f05186e0c2a0
3ffa1156efd9f960 3fc0a8a5fc3b82ab 4002299f4099df80 40009e75fcea868d
bfddb922fa08d99a 3ff75ce52ee5fb01 400498500f29f137 bff2789c607a1b67
3fcdc8596174e853 3fe4734323d655e7 3fc53214b08985b5 3fefd76b672fa9b4
3ffeda0eb4ad1404 3fee04cf6393a210 3fecb16ab67c41ca 3fbc76cff7e8ff22
3ffb4f623c5e09a0 4003bfcbe2375d27 3fe1ab0294118ea1 3ff91d594e37d809
3ff82ec0e83c6502 3fe690aa879da93d bfbd07ebb44c527f 3fd9971da7851266
bfcec58e12c176a6 bfa8ed12a11fda1c 3ffc1a490dd8ed0b 3fcadec9ae6019b5
3feb2313fb7986e5 bfec79988ef1612e 3ff7d03b9b056648 4004e320847f0e09
400062599f1d7a65 3fcaf945b088415f 3feeb78907a853fb bff4009a45ee3940
bfe21c1d7fe19fb2 bfad1bb8d5b1fc8a 3ffa6cb18ea876e5 bfed0d3e8fbf4443
3ff55e6edd7d6a1e bfe1aa986025cc02 3fddff65ac89a6a3 3fb324e944f0c1fd
bfe04029796a998a 3ffb15a59cc7d82b 3fdf305b960acdfb bfc2a4d65ee4d083
3fbe32370a51542d 3fdab0b7ca017d4a 3fe18b2716c2b06b 3ff0cb6bc68315b2
3fcbae6fcbc1840e 3ffafd0afadf82f4 bfbf1717aaa13b48 3fff000e7612ae1b
3ff7418a846a4e1b 3ff050af3beeb0c6 3fe0d0a59c272249 bfe1b69c235763b3
bfeaaa2e6d8eb522 3ff0ee2f6d5a2346 3ff2cb66351dd1aa bfc8c77199ea93fa
3fcec1537b77f046 4001de3cb007b9d2 3ff323728dd17a0f 3fe1b83832cc33fc
3fc878980b3bbe76 bfb9a8583f16a3aa bfe1402a4f00a73b 40054b36e742ff68
bfec6892e10b63a5 3ff95a712c4b80e7 40053738a14a5201 3fd39a532361d201
3ff38d9684598aad 3fd364fc742288cd 3fe09274525b0c78 40023b3ce4cb23ce
3ffcfa13d66bb3c6 bfe5abdcec8fc080 3ff47cc777683db3 3fe08d4d6275d6e6
3fe0abe4de1f8551 3f959b427960c545 3faf307b46060225 3ff7c4f11139fd63
bfdf58d2f3870324 40010961805c0077 bff23ea256a1998a 3ff59880cb73d9a9
3fbda88ce2a8f313 400041934c1ffd9b bfed8c53cdfc7eb4 3ff1f4af99241e67
3fcedc38b2a0a3cf 3ffaa9c4b58a8563 3ff6f434d4126a18 3ffff03c92ed69fb
3ff59880cb73d9a9 4005ba9acfff1110 4002683e18bf18c0 bfdcdd32c3b609c4
3ff215f895b0b1f2 bfeaf9745ad2f813 bfe13908e4fb80c4 3ff1df98f2d51a4e
3ff34d87f631415b bf9d2a78b01bafbb 4003763a76075ee0 bfcc1a02502f08c6
bfc28bbd61d10673 bfef4ff6901d8b95 3fde0bb13ac2e017 3fa955fb737acb11
bfd747e174409db6 3fea4c41eaec7f3c 3fb8e895b17b8659 3ffc7f0f5435f35c
3fc754c5cb2ce146 400111bb8b175345 3ff429565c52d22a 3febaeea46b257c8
40062341456a4964 bfd81919c9bf5b14 3ff1cec85a8ba08f bfde9c37c7edf2b7
bf4cca04a6a4b1d0 3fda914a442f17f1 3ff19d292fc5c59d 3ff8005dcd5725b1
bfe5b4acea89978b 4005466d6edf57c9 3fcd4368bd1443ad 3ff49c6ade38b75e
400151f944c696e0 40004353702e1105 3fedf78ac9a0d278 400088cb92b6f0bf
bfe42d4590e7e270 bfe07065c831c5ed 3ff8b286ad5051e4 bfe42d4590e7e270
3fcf838dc933dfcb 3ffd8cd926cb2d15 3fd0877afe4192f3 4003c1e574e8abfd
3fe178a3f1038501 bfdb8d44df89d716 bff5b30482d548f9 3ffe994e45ea7cc6
3ff90a4d9f27ce94 3ff01d6eecd1f064 3fadf89cd12860e1 bff408f780b94676
3ff8dc22c62868aa 3fd035b99a4c11fd 3fd1df3dd9f18bfe 3fddd93b5d0b5976
40016b8288b47676 3ff098638188c2ca 3fecc6909b513a82 3ff47d1a19c685f1
bfeaafb7290589af 3ff8d2062c31805d bfc958929d52eabe 3fe8b46a388b8690
bfe99af3f9192608 bfb9b96bf44f3de1 3fce0cc23b323ae2 3ff62b19d946369a
3fe894ada01c9f70 bfa2c9a546bba57f 4000904da14df77a 3ff075e827c2ef89
3fe94ed6c8d1bbb8 40000288635463d5 bff2d6a090bdd101 3ff4474aaf52c2c8
3f9a2f3b12336687 bff408f780b94676 3fed0a2822723c8f 3ff7d4461a8aec03
3ffe648efa76417d 3fe652d5a40ca09d 3fddd34de9ca5a2f 4004452f09016d61
4001ca9676051158 4002b94ef3671d29 3feb12f33cf95b17 bfdb00bd2be3667c
3ffec369a9a8e064 3ff08f78694db091 3f853ddbac29a108 3fbb77292688b60c
3fed291ac2d53489 3ffccb9bac28dbb4 3ff26f7ddb491fc0 bfef98ccb136cbdc
3fd47713b0c0b88b bfd418fb574bdfad 3fec74f62f990bbf 3ff9033332916084
3fbf3e17e0c41b0a 3fd0a9f9625e9e04 3ffca2430bc0433a 3ff454e16d2489b0
bfc2fe08195ff4b2 3fd03b848c30caf5 3fa4c1ae1cfa21e6 3fa97b21f06aed0e
3ffb4e311d2ee095 3fc90b0a417cb399 3ffcacde8d1caa38 3fff41d6186b6fbe
3fe43fa416f43db0 3fe37340304033bc 3ff788ea37241be6 bff0234f44bae28c
3fe5e99259d9aafd 3ff10b2f63283347 40029813367087bf 3fe7ad2314dbffc5
bfea1cde1cd35149 3fb23a4f06524abb 3fe6a22bba42f7ae 4000eada921278d1
3fb8e895b17b8659 3fd11df09c37b25b 3fe5fbf98facb2f0 3ffc43def5abafb3
3fef821ae29a0d9a bfe9b62cb35a9f3f bff2658060425452 3fe57052e9427f03
bff19b4bd9f591a1 3fe22c3262367fed 3faa84f61fb9c868 bfad6d0bc2f96057
bff580f415f3f58c 3febc268869cc801 3fe6f25fb11e6473 3ff9c6b426a0a506
3fe81dc87afd6408 3fdd40222c2007ef 3fe57c045cab6493 3fe835ee5c53d432
bff1c5c367af72aa 3fe4180f7520437d 3fdc2d9b97cc4dd3 3fe9a9e8b89d7c93
3ff088f517bf99c8 bfdd1de24b11eada 4002611733ace0e5 3fe583c56a907210
bff247bd4cc1da06 3ffac2a2a8a1179f bfd4e65f7f8578a9 4000c26b5a3e3e38
400003ca4edf349b 3fc443388d07e9f2 bfba5f68151c4d7c 4002f949f278fc02
bf86fd380b4bb162 40084722737af626 3ffbe0fce6349fc1 40009ff45602375a
3ff48ffa026e788a 3fdd94ba847a833d 3ff4d789e5d2756c 3ff025e7b563c10a
3ff3fbbe7e921bd1 3ff8753d3e8800f9 3fcca592bb23d242 3fd43afa1399a1ce
3ffb6a0e239c5a41 bfc225d26004a9a4 3fedb548a0efd0d1 3ff990dcb9024b95
3fed6e93c326eca1 3ff9e04d5055ea45 bfb127531991125c 3fe80fa8f3bf95b7
400240c283bdb68b bfa43b677b512265 bfd025666c381ba6 3fe1478b686bdea3
3fec44763d46800d 3ffbfa14e7fc521c 3ff6687a78eb68ab 3ffe994e45ea7cc6
3fe44514a9455341 3ff2bcc0158ea707 3ffe7d9d5d53f7eb bfe84f3cfbd94528
bfd0d4f1239e52d7 3fedeae9db3ec69a 3ff0564a87e7c130 4005520bb84d1d67
bfe53d3e3b973497 3ff541f6b57a4df2 3fb452a708b63dbe 4002dd5a25069ca4
3ff3ff66e7fef774 3fd55bc608273417 3ff96ee386204542 4000ef0d554d917e
bfe8023914cfba5f bfbd073aae097674 4001a2f527dcb893 3ff986bbfb8af2f5
3ff0e696e6a93cb1 400581822600beee 4002deaa8aeb385e 3fedc3d694549838
40084722737af626 3fff7148847efe17 3fa8950a4b43301c 3ff3d9617908cbba
3fdfeb9563081ac6 bfbd07ebb44c527f 3ff554ae2b5916cc 4002c7a699cfc729
3fe245b042a2430c 3ff7afceb2f1600f 400092fe5bc1e267 3fd545a097abf163
3ffd83468e4a86ef 3ff6b9e78c47cd9f bfdab5c3c2d14910 3ff4bad27bacb718
3fdd2eaa9f0dd389 40008ad44adbc55b 4007cefd04be2567 3ffb138dbe6b9d62
3ff21d5ddc79767d 3fc61a5b4bef32b0 bfbd07ebb44c527f bfa55a565b0f00d9
3ff2fb818b7f1804 3ff5b8abba0144a5 bfb93d91f8c8a3c8 40020215b2ea3a1e
bfa55cd143b30c88 3fdfa73b511e6737 3fc5501e3de6ef5e 3ff1c7fe1c8612fb
bfcb16adddc21ab7 3ff1f739b35f3f3b 40009ff45602375a bf8341b617bd42c6
3ff6a71bdbf4de52 3ff233d68671f661 3ff649e141174491 bfc3ed5f4c81101f
3ff499c52beee4a7 3ff31810cce144ee 3fe8ba6cf7681547 bfec6892e10b63a5
3ff8765be184a06f 3ff25c0e4deb16d3 3ff7dc41002399f5 3ffd8cc9e19c7d2b
3ff3571b0bfc9400 4005ba9acfff1110 3ff4096b3555c257 bfc7863672b65d66
3ff4956c6bbc1015 bff10c6895f99aa7 bff105bf861f5db7 3ff084ade477eac8
3ffa92172b88df72 3fc30b4bd34a068e 3fe843aebcc78265 3fc7e5414381e18b
bfe853297820c2d8 3ff3af0f67f13b62 3ff7aa07b4c5c56e 3ff11c4c58a9a49d
3fc2d9f21ecc5a1e 3fc80ecdb1f3e151 3ff3c39182ae40a2 bfc4c5ef59bc4376
3ff0d253fe60c2be 3fcd06279a8d90ae 4002deaa8aeb385e 3fda74ea3276d577
3ffb59eb8ff931fd bfdc3a0c8b5a60fc 3ff6631b84e0372c 40000f2a057acbf5
40030b1e0d2d8f69 3fd984d9b3b0d2c1 3ffbc31d012aad60 3fe94c9f02d0a1bb
3fc729431605141b 3ff1e9ccbee7185e 3fd65a7d1c963bb6 bfe54b1b368ff4e0
3fd51c26229b3078 3ff4fcd919548b93 3fcfbd9846281be3 3fded568547da88c
bfc5a37531de0506 3ff077020e6633f6 3fd254d859627ead bf983d3d81fec8f6
bfde9c37c7edf2b7 4002ee01201aaa67 3fd4ce265df70624 3fc8fb6c7d17ed48
bfe4339d77700bcc 3fe1709ded659419 3ff06939041cd06b 3fe8eef2897404af
bfc56161b632ef6d 3fe26d9e42739adc 3fdd35b4f1f77e23 3fe1db3403321f9f
3fd8d08df2daae72 3fe7a18f2b3d1927 3fdd4baa1d33313a 4001aead29a078b9
3ff2717aa433bd0a 3fcff57f5ff46031 3fdb48b98bb7c35e bfcf5a8e10793c56
bfb5128e612713c3 bfc443c7ce7cd69f 3fe894d778226a90 bfe0f25f7c11b61f
3ff73c0eeac8a14a 3ff438f11f02280a 3fffbafd8471c7ae 3fc6d6a9a393ce7b
3fff69684bbc4345 3ffabfef0d3754af 4004452f09016d61 400104a19180d47e
3fc15d26c90d8a75 3ff6601525f86520 3ff4e7c8e9cab217 bfd746c90ae4b370
bf9fe35340a2b047 3fdfc26a008ba7a1 40021ddf1981baed 3feb617a14f661b6
bfec902bb47109e3 3fad6a8901d0e7be 400099d7d64e9cd7 40000b30313c6113
3fea2f9647594504 bfd3e43ae06b7bb8 bfddc2ea92676ac0 3ff4bd5a1fdf5253
3fe1bfde5344b13b 3ff3118d2fef8d19 bfed8c53cdfc7eb4 3fdaa383034ad88a
40023778e5a5db20 bff44babdc58f712 bfd6b1d3ea159843 3ff8b6643a860a39
3feaedb561475081 3fd968947137b022 3fc67f603869b789 3ff221deb2e1e3f8
3ff61023c0f92dec 3feae201d1e7e8e7 bfba3dfacb48d343 3fe3f6586dac5a19
3febb469bca6c741 3ff6e15af2642dd7 4001e9519fb02571 400022a562fd6e4c
3ffb8bbeab463c54 3fb9af0a3393da43 bfeb776887dc82b7 bfdda18c2a1bc750
4002deaa8aeb385e 3ffa08e74da21cf6 3ff488d1f2da9408 3ff3571b0bfc9400
3fbc411ee9f435b1 3fee130197ad52ff 3fd57449f2b7db74 40006d17bdd625db
bfd2249c2b909ab4 3ffd2904e486eee1 3ff6eada19acc259 3ff0d8cc900c8b7f
3ff0089233b8bfdf bfb43ee803ec5090 3fcc5b3c3d161d1b 3ff54532807ea0b6
3ff9a47720f97ed3 3ff06a14a926a824 3ff2176f93ab0d79 3fec44763d46800d
bfec902bb47109e3 3fe944fecc3730e4 bfe46b848aee50d2 bfe6f9a3493e86a0
3ff93584ac2c73aa 3ffc15bb2038698c 3fdfa9dcd4c3ba6d 3ffee992fa661bde
bf820b8ff2a5394a 3fd25778786206c7 3fe5b4b580a6d832 3feb4d0b8398c329
40026257162a822c 40002b7aa913a5bb 3fd888e49a5eac53 3ff564029c950088
3ff0c629e24f3ecf 4000fb42994e6092 bfa4388365bb147d 3f914d43c48f67b5
40061f3e352b3a5f 3ff9078fcccbac9b 3ff4041ca3d2dc89 bfd07b5a6ddafff7
3fb73da1ba08a533 3fee5ccafa0e6148 3fc2757c7a9b817a 3fed6e93c326eca1
3ff0a575f00c57a2 3ff45c72c0dd106b bfe0f73a026905aa 3ff179ce3152f778
bff2da543394762c 3fd00684008f920e 3fec46a3a2da1036 3ffdb2d8bb63fad4
3ff53a094535f13c 400384d834664c6c bfd1e81fca328632 4000c6a54b7cbe68
3fe061923c6b8dc9 3fc21efafc67b0d4 400126517238b07c 3fda9497d3a628b2
bfd27766acc4e94d 3fdd2c50ad34cd1a bfe16744d36fd0ec 3fe4180f7520437d
3fddad9946ea29cc bfd21b7962f7ff2f 3ff96335813a87aa 3feca7ea8176580d
3fc0dc9e5631a54a 3ff7d65816192db2 3fbd3aa8e608d7b5 3ff8d2062c31805d
3ff10a0defd7dae5 3feee4b36ee8a458 3ff1fe453d4f5d02 3fe4146a72dc7544
3fd400cb28ed58c4 3ff2edbc01535eed bfd07b5a6ddafff7 3fe46e4cd192a8bc
3fb23a4f06524abb bfc84fc6a1901995 3fe54df1df0d2479 bfd99c7302aa5b1c
bfd746c90ae4b370 bfbd073aae097674 3f9a15960b8322e1 3fa0861c8d7ec0ac
3fd888e49a5eac53 400079aa0718d120 bfc8c77199ea93fa 3fefa0d846115563
3ff290184940ac6f bfd3a35afdfd2bda 3fffc2e2b2b917d9 bfdc417b1a8473c9
bff0899b37482863 bfd1f3f15604fc41 3ffd53ded74d34f0 3ff34d87f631415b
3ffd6edd59c40be4 3fdeeb31e970c4fb 400227b2dfd15104 3ffcd1fc011a9e61
3ffc6e27b3010636 bfdfb58fe94ec87e 3fef3fe560a530d9 3fc4d0f2d9941b8e
40053738a14a5201 3ff0e89dcd21dc00 3ff8b9b817100d6d bfc886183aec17b5
bfec2ad36b631372 3fe39c2726040dbb 3ff3f86c9b3c6599 3ff8bc07e0c8a71f
4000b428b8e3eca4 40034daadec77838 bfd2efdc4e0a6910 bfa5dfce2e04193e
3fd54cab543cf688 3fc3f9e56402c9c4 3fe7a4554e47b65b 3fed85da47ac2b67
400238d53fa2e5de 3ff2ca195019b83e 4001a1ab1c84f4c0 3fc71f761390c882
3ff300a02b6ac055 3fe98b540dbdffe1 3fe2c580272f3d8d 3ff9315f3da4144b
bfde940cd3e180ea bfe17acb9e943729 bfe5b4acea89978b 3ff3326bb971da7e
bf983d3d81fec8f6 40040b31bad6eab8 3fe80fa8f3bf95b7 bfbd07ebb44c527f
3fee87a287376417 3ff5870eb3b5d9d3 3fe92dd5e055d6bd 3ff50c309a1d681f
bfd8308f6ff0ed5f 3fb5c079f6a13d3f 3fc02cb0d5d58671 bfe1b69c235763b3
bfd2e0b01f991145 3ff2db430afd69d8 3fb5cb6d14ed4199 400031a678bf2c18
bfc54fcdd162e558 3ff3c90979de6375 bff1ff5c87668821 bfd6ec68d4e8abec
3fed6f50d48f49db 3fd9cf86c5cabf9e 3ff06939041cd06b bfd1372f20da1520
3fd200fead721276 3ffe3ec414a6f59a 400155b552441561 bfb2956fb0aaca0d
bfe0d4361ce78a27 3ffb8c681d775537 400227b2dfd15104 4004b3ee171d7707
4001ba4fa3866447 3ffd534f8eaa0c64 bfd2fc95edd4350d 3fdae647f20c25b8
3fc1874e2e46d3eb 40009e5d42af42fb 3ff722204ebacedb bfba3dfacb48d343
3fe95a27fdad0b38 3fd4172a2e1445dc 3ff0d58ac0c8e9ae 3fbb1e7ca4e8dbf6
3ffc001c2ec1f3f7 40059fdf16b58a55 3ff083ce44138021 3ff37bd191bec8a6
3fcb85da3157a066 3ff4b2664d1370c0 3fe1544ac6ec2d5b 3fddf6fc7120b69b
3fec154a7c6a618d 40018999a7c93849 bfdb6dca0156a10c 3fec2ae75f73ce43
bfb9c801493d3d8d 4001c5dcd1ccd399 bfea14d070456521 3fdd444ea5a7b3f6
3fe2c709c84b0c74 3ff121ece812bbf0 bfdb516e937eb351 bfa61f56f3ea195b
3ff6a71bdbf4de52 3ffb4e311d2ee095 3fef253f153e2665 3ff1aeab2edb148b
3fe2d80bdf92e088 bfe95f14bab05dfb 3fe19bd92c3082b4 3fd61d4e598ae9ad
3fd7d2b21df1e732 bfa8ed12a11fda1c 3fd230840f04b0a8 3fffa761e8ae709a
3fb07d362fa7793e 3fc9beff26a6a032 3fbc76cff7e8ff22 3ff6e2cb29e4e19b
bfdf82b8402fce21 40041fd32d4405e4 4001870ef5094707 3fdd19a3abb74285
3ff846947265cc1a bfe434aa840434ac 3feffbd54918e708 3feaf027eb424a45
3fd0705413350baa 3ff7f7505148b4b9 400622cc3881dbb1 4003a4f8cb3491c1
3fe754316e9b7a28 bfef47c27dd68587 3fcd58dd27f0af40 3fc53aca75ca8b60
3fd9fef14f0829c1 3ff6eada19acc259 400824f2dec21ab3 3fbb05a77bb85f20
3ffbbffb3258fa92 4005520bb84d1d67 400196fa5f520748 3ffa3a92fb56f492
3fcec6afa056be15 3fb46fd57a0e9b3b 40056252a3824660 3fe010ffbf523a2d
3fd963f57070d995 3fba9c7e5933111b bfc3fde9e195af00 3fe49fbc8064b27c
bfe8de4c05cec37d 3ffc1a490dd8ed0b 3feff81757e5b24b 3ffe7bfd6f714c9d
3ff160803d83596c 400796c715633d39 3ffb750b4aff26ec bfd6ec68d4e8abec
3fe49e8bd8234f36 3ff885a58fc391a3 bfd30e42472bb71f bfd4d1c720f21969
3ffb544f78c4eebf 3fe46e4cd192a8bc bfddaabc5bd0f56d 3ffed521250c8193
3ffffc4c6de85b93 4005ba9acfff1110 3ff3fbbe7e921bd1 4006046b1e286ef9
bfe99af3f9192608 3fd39776c75c5c34 40061f3e352b3a5f 3fe4c30e7e83436e
4004a24fbda6eeba 3ff452c761fe6c51 3ff32d680e839bc5 3fcb515aa51fafda
bfe3a2a34b21a437 3ff7cc4b6fe5e5dd 3fd748dc75818b2f 4004f44c8fb5b7ad
3ffd699ae8949c8b 3fda052cc88339b6 bff0724356dd6aeb 3ff73c0eeac8a14a
bfdb1f90d1b59eca 3ff8cfcca83af8f6 3ff3326bb971da7e bfbea73058d890f9
40022f8d0436ee12 3fe7b6311c352f1f bfc0610802ee2e6d 3ff02d18a6236d0c
3fcbae6fcbc1840e 3ff10d18b9827dcb 3fe39a370d15a5d3 3ff2b5d16d876a95
bfe19fe97dcae326 3ff1c70bc89c5ab4 bff18b352ffcc57f 3fd963f57070d995
bfdacc0b73e81ce5 3fe39da142fd24ac 3ff1397dc8b5ce0f 3fd39ee9e6a82ddd
bfc21751e3fa7ee3 3fdff059c60b37bd 3fd471973876abf1 bf79449452f0f0ad
3ff3275b135a9985 bfe42d4590e7e270 bfeada3995057b86 3ff7d354407660cd
3fe6d904a330c00a 3fbd3aa8e608d7b5 3fddff65ac89a6a3 3ff1eb60a094b442
3fee57eaaa3d57c5 3feefe927d37fb35 3ffb61946df4878d 3fe2435f8c014c8a
3fe3970b5c24e1ce 3fe1262d665ce166 3ff4d7599f30e19c bfa43b677b512265
3ff0a30d447d7f3e 3ff81712bba0b8d0 3ffd79c32bb35767 bfb6679e1d049fc3
3fe663b8ae3ac047 bfd6f05186e0c2a0 3ffcd3e71669f430 3fd9747cc897ad9e
3fdb6cb1547aef7d 3fc98fd0be6fe625 3febc268869cc801 400121d694d61fab
400496fe22c7dfa1 3ff40ccbc9c075b0 3fff2645ee17705a 4003d45ec601e3c0
3fbb2316167f3664 3feff07859f07e35 3ff98f0d8ee96995 3ffcde2cab0eee2e
3fe6494117c3ee11 3feb662b1bd5ee43 3fd03b848c30caf5 3feb23cd2cbbb6a9
bfe8589c088233f4 bfb5a887f3475743 3ffd7aec8f72bf84 40009318aac1c98a
bfbd07ebb44c527f 3fe048d90e4bd8f0 3fdd6aa6520a3210 3fc2ec30bf6ce52a
bfd72f705a9861ff 3ff30115884829f8 3fdb48b98bb7c35e 3ff36df408b2df49
3fd77ec8aa2abf63 bfad1bb8d5b1fc8a 4002bde4484898da bfeec5d97b550439
bfec5bfdc1703881 3fde8840e536de2f 4004452f09016d61 4001d7a19a1cc3d1
3fedd25bea6bda39 400196fa5f520748 3ff732b0ca73015d 3fd3e2c0eb5f60c7
bfea957bb0472937 3ff0340091b0ae50 bfbc5db74039848b 3ff0ab152fb562d3
3fea653a11823f09 3ff82c985745baea 3fcb12c6879a3d0a 3fd1707110973e3a
3fdda08a76d869ad bfec3f89e2be6fb5 3fd44bebb185f934 3fe58617497b22d4
3fd63d0a5148b33e 3fe9fc3e136d005f 3fccda411fe0860c 4003a4f8cb3491c1
3ff47cc777683db3 bfc72cb7652312fb 3fea33ea5f6465f0 bfe99af3f9192608
3fff95397175d88b 3fdf305b960acdfb 3ff79aeb6e31c665 bfdee142c91c2f61
3ff7185840128b86 3ff8efbfa194d981 40069c14106408bf 3ff1c7fe1c8612fb
bfcb591f8d4997dc 3fe583c56a907210 3ff0eacd71d11c4e bfcb539429953d45
3fc44d4231e3f97e 40049498b4666cc9 3fe8a1210f4700bf 3fef090ba41746f3
4003c6854d0d3e77 3ff0a45488b36b28 3fff2fb2da7fc1c9 3fd67b5f9b22ab1a
3fc46b3b64644bfd 3ff0b135e9ae3068 bfe87d2d71213812 3fee7c83401fc371
bfd3f5104db19f51 4003e9e52c0fb816 3fd74ca3449481c9 4005867b8a5f4784
bfa9ff6b54579df7 3fe600bb1f756d8a bfba145bf20720e1 40058108ef73afea
400147c25281e8b6 3fc5ad24a1ebc751 3ff9884a5046bbc4 3ffad18a8fbbc5de
3ff8710b4cad3faa bfcceb3f99e63e8b 3fe6d904a330c00a 3ffa0706a4705f98
3ff438f11f02280a 3ff9aac61f6faa50 400259fa2ff5f362 3fe50ece7640d517
3ffd8461d7259adc 3fc61a5b4bef32b0 bfe537569b336969 bfd3eb4c606674c0
3fe1262d665ce166 3ff8bad5561afb81 3ffc2d7614dc9204 bff2658060425452
3fd1463d55205fea 3feb7d819c413988 3ff2787bfd579863 3ffb7f8b7da4bb72
3fe6d88d8902f892 3feb5ecf310ec882 3fe36bfdede2e9e1 4005461ba97c29ed
bfd0fc3e256b196d 3fe299d5ee3c3aaf 3ffa6dfd932a0297 3fd7c02403c408d3
3fb0e306bb254c9d bfeaafb7290589af bfd1eb1e4656555a 3fea5d0da3099c60
3fed11b3c68a4d3e 3ffff6c653e5a8af 3fe2255f88eee5b2 3febc268869cc801
bfd70c81feb1703a bfed8c53cdfc7eb4 3ffd48d2fa65c0fb 400796c715633d39
3fd5baf2419d91b2 3ff319918502700f 4003f09b14cdf07b 3ff1c87bc1983edf
bfee251a746520c2 bfe0a0b400b23183 3ff2d44db25a41bd 3fecb1c70d922960
3f94c8ebe9113706 3fe6d88d8902f892 bfc635d68ecbbd78 3ffc43def5abafb3
3ff8753d3e8800f9 3ffdc9a55cdf841b 3ff05bc3dc3fb258 40022f30697f62b3
3fe2fe7a83eeb932 3ff85d02e1bcc17c 3ffbbab8f0bbb44e 4000d267417b6ad8
3fea1166434e7d84 3fd0e2c3cd8e646d 40021fe5ab6b1b71 3ff564c53dd02b05
3fdea5fa6a567818 3ffaacfff36559e3 3fe178a3f1038501 3fd76205de27219d
3ffeea8543dba88b bfbc5db74039848b 3fed56b3d39129ef bfdaa7a2c69d8c55
3fe4b2ada9c96451 3ff3fbbe7e921bd1 3fece6207ca2a6cf 3fe6a43bd6605f04
3ff4e66e3c13c526 3fe4e1496f7dd066 3ff937d1dffdc1e7 3fdb84aabfae1cd3
3fdeeb31e970c4fb 3ff53a094535f13c 4000d267417b6ad8 3fffcaba5728b6f1
3fe0249230de0f5b 400238d53fa2e5de 40059bdc06767b50 3fe8bf21b1edee49
3fc058ce1b6bf638 bfe18cb2c125bb83 bfaedc12701f85d2 40022ad24378ee0b
3fedcc517ff0b1d4 3ffdf03942ce6e0b 3fc1ced53c44e647 3ff28b9cb9480265
3f8a840e63a8f132 3fc88ef396017751 bfcb539429953d45 3ff6e4a0e7aa8af8
bf94730a2ecc0a6c 3ff0535aaedc0d99 3fe7c4afcf97fc2c bfd739c6f2e4d633
3ff3a64301c8095e 3ff87e84fe588822 3fe42ce80d30ef48 40005059a503335b
3ff00c76ec37d18b bfe46b848aee50d2 3febd2261a6679c0 bfb058ff7c52edca
bfd1e81fca328632 3ff2fdf33db33b73 3ff1d828692c2ddb 3fdb04c6f72cf6c2
3ff3c90979de6375 3fed49cb34f83ac7 3ff3942a6b23a344 3fe1b6d07f27c406
3fe84a3c9cb1ad50 3fefea8942067cf9 3fc585857adc4b5f 3fe36430a668bf96
3ff4acd7a7bb9dc2 3fd65cf7d899a6f4 3ffb2f74dd767f0f 40011a91c703b9be
3fd1cd1faaf928b9 bfe8d0143195424a 3ff006d79db4776a bfe1b69c235763b3
3fe7ad2314dbffc5 bfe7155a02eecc82 3ff5ae7987ec042d 3ffce094cbe68f45
3fc1c394278315cc 3fd2f6007a4a54f5 3ff1a83d8b0ab483 3ff696145e8ef4c8
3fd0ada9859657a7 3ff6a0ab7e134a43 bfd2efdc4e0a6910 400016b9fa7ca5ef
bfd39ac2a8ac0e18 3ffe5de6e701d255 40031816e7db5c6b 3fef24092fdd1384
3ff01d6eecd1f064 3fe539e36a0e32a4 3fddb206192ec61e 3ffc9c67150d68e8
bfd87267fce251fe bfe1aa986025cc02 3ff5c59861502d1d bfd19a6094cb5ea5
3ff73bf560b3e3d6 bfe0101880a25a5b 3ff2cb66351dd1aa 3ff6a71bdbf4de52
bfb7aeb8e19d53f7 3fe32a9e1846a42f 4002f949f278fc02 bfdd1e4976ef072b
bfd29ca67989f9b7 3fe817ef1d57769a 3fcb85da3157a066 3ffca67de8f4e3ed
40052bbbe02e9ad3 3ff8b29ec98b7887 40006ef681a8a003 3ff2bcc0158ea707
3fe88ba4709d42b1 3ff722204ebacedb 3ff8250342e385e8 3fdbbfecce0b559d
3ff82ec0e83c6502 3ffca95dcc3bcc0a 3fb86f203cedc5e1 3ff02b497c0a8b0c
bff04ecf552a9f14 bfd842d377df8a4d bfd932d9a3c9c9d4 4007e9062f93cfe5
3ff9c89117945432 3fe80f83fcb78176 bfecdf6c0be2d65c 3fd0c37cfbd08b08
3fe38171e08fbca8 bff04dc6d274c394 4000eada921278d1 3fdd3cf19a5b0625
bfbbb95c7c1e3826 bfdcdf4cdcde5b84 3fdd4987b58ecb97 3fe5184c78990f02
3fd63869944f5c7d 3ff4265e2e0bbc2c bfaadeeeb664c328 3fd7724f61b06551
3fc8f135867dc2eb 3ff55da10be173b3 3ffaedbcda073cd6 3ff253c7333f3e29
bfdcf192be5d748f 3ff694d9562e8cbe bff11ff3d12f4d15 400224228dc67eff
40007d9d5714e189 4002e6beca9d3528 3ff4875bf47f83d4 3fec45a217ee16fa
3fdaeca680e4a2e0 bfe609a160d3c4c7 3fd9bb61511583db 3ff81712bba0b8d0
3ff925f3536d6d85 3fec8c5235b489dd 40042b12015b4a9a 4000c9e1b4656b3b
bfe458edf5820df8 3fad509cfbfb0ff3 400101b7bd37a9e3 3ff2a03bf8ed9fb4
3fdfa73b511e6737 3fec1287b1874b65 3ffe1577bc7d113e 3fdd3cf19a5b0625
bfdfb58fe94ec87e bfdf8448df8d2851 3ffc001c2ec1f3f7 bfb5cc80b85e2f1a
3fd1bb52c52b2420 3fe68d032455070c 3ffd73ab476d6a0c 3ffbe7e98c4543ba
3ff7af8f28932297 bfd6dad143c3ea30 3ff1c654f90d25fd 3fc678467bbebfcc
bfe5b4acea89978b bfeec5d97b550439 3ff019a6c32db3c4 3fe52ad8c9324fc6
3fb8fe877dc2a9a7 3ffe96ba7841dd74 3fea9a74250e3c02 3fec0bd71614768a
3ff23d22fa01b32f 4000c9e1b4656b3b bfeb9b6dde1d3c21 3ff6044a6f050321
3ff4875bf47f83d4 bfeeaa1a55f46caa bfd8a392c13766b8 3ffe2eb28300130b
bfd348be8b63d911 3fd5440b01ee49d5 3fe4180f7520437d bfdd4d00eab293d0
3ff9b0bfdbaa2dc9 bfe464291f0c98b0 4003a4f8cb3491c1 3ffc324c46acb00f
3ff3d9617908cbba bfef29fa5517ab13 bfe38aabaea70d90 3ff8b9b817100d6d
4000a6e2cfc54298 3ff9e81bfe0be808 bf6206a65e5e7ee0 3fdddc1a39589614
bfcb3502268390f1 bfe71c2faf5002c1 3ff6459f1c8972f1 bfd927f5e4d887d7
3ff2a79c7ff4aef4 3fb377cf9fe870e7 3ff6b67463708a9f bfe30f7f08eff47b
3fc40c479f978e00 3ff85f6e992f866e bfbd073aae097674 3f8b9f770238b16e
3ff2f1b8a7368e0e 400539b57f543d78 3fc10c536d369070 3fe7cb5fa96f69bd
bf97c7bbf3e9d0e3 4003bfcbe2375d27 3ff0eacd71d11c4e 3fa7c37b5312069a
4002e0b9877f408a 3ff044796d67e358 3ff8ff17929a69eb 3fcbbdc9f7bad7ed
3ffde1303a0918a2 3ffaebd458c951fc 3ff46e531afe6013 3fc81b93ddd811ff
40009e5d42af42fb 3fc16eeed1fe7ff6 3fe2e042151107d8 4000928bb07d99bf
400496fe22c7dfa1 bfd27766acc4e94d 3ffe7a9dbf5142a8 3fe247e00408dbd4
3ff8d6cf0ba083bf 3ff1dea8bc0f4633 400502d814609c89 3ff608dc486e6ff6
3fe01326dcb65d82 3fe56309a395a977 bfde9c37c7edf2b7 bfdd4d00eab293d0
3fc3d677cdc4a370 3fff8cc60bf339bd 3ffe082ad3f06d92 3ff7d65816192db2
bfde0125885a295b bfaa84625ae4b72c 3fbc28ef4b483d09 3fe73ff024687ffc
3fdb3fb4f626fb6a 3fff184cc9476655 3ff4d7599f30e19c 3fe68efa28de4130
3fea653a11823f09 bfe12ddf089a4850 bfdd07085f7fe545 3f9651ef7c3c512a
400365448f1485a7 3ff04d71fedb92bd 3ff0f6dc7d02a91c 3ff16aae34119843
bff190cf40ade665 4000c51154b6f238 3ff2ca195019b83e bfe9ef0b98f6e362
3ff474b439a8f32b 3ffbeffb247d3de6 3fff51180ddf7198 40002359cfe4afce
3feb398c4d30fed0 3fe20bbb6056eef3 3ffd2d6d0bf96f35 3faa84f61fb9c868
bff3425e7a4abe19 bfeec5d97b550439 3ff53c3c960893d6 3fffbb1f30fac7b6
bfc911cef2f489f7 3fe8ee46ff11cf9e 3ff2cef496f87fea 3fe8e6fff80f9bd3
3fe022e962af296d bfdbad139b378f1f bfe6f414734de00b bfe426758b5ba26e
3ffc43def5abafb3 4002bc23284b0594 3feb4fed5d82dda4 3ff73384517680fb
3ffc43def5abafb3 3fc49fc4746e2c58 3ff5e45cd0437cb3 3feae201d1e7e8e7
3fddcf55547fec63 bfd81a62442d79a2 3ff653786a8db4a3 3ff91bf28ec5d467
3fdebee403408fe3 bfd932d9a3c9c9d4 3ff5014d8e19323d 3fd8f82a81ac08c0
3ff3dde0063dd366 3ff2fcfdf8ffe6c8 bfd11688b016d2b5 3fff16a4af0f84ad
3ffb4e311d2ee095 3ff0d7f3bd46458c 3fa93ea69129754f 3feb2313fb7986e5
bfe24f9d6a9ae295 3fc878980b3bbe76 3fa02086c009715c 3fe7539504745c84
3ff6ae91d32b8d8c 4005b42924d8b4df bff047a1aed1f16c 3fd3f5e03356ad06
3fa1c43fa9bde34d 3ffdfd3b84e7585c bfcc0e55ed8a986e 3fff8f910c71f523
3ff4894f31351902 3ff5edf815cd5c49 bff06603a05a9328 3ff557ededdd6388
3ffc1b04453dbecd 3fd65cf7d899a6f4 bfe86a4811dbd0ab 3fe9b9772c403844
bfe14cad79bd5c27 3fec93a433c45807 3fec93a5c9cacac7 bf983d3d81fec8f6
bfd88da909e6a8d0 3fda7300365f7e9d 3fe78d0b58799ae6 3fd888e49a5eac53
3fe98b540dbdffe1 40017b44ce7fe9ab 3ff86ba8dadef697 3fc3f9e56402c9c4
3fcffd30de279432 3fe7b6311c352f1f 400186f98887eaee 3ffed4f2d777a6f5
3ff5499d34a8ab68 bfd44eb373c1e787 3fdd0019025d5630 bfc9d87bd612a743
3fd66e6266a7993f 3fda97f9885299f7 3f7268892eebdb4c 3ff200432e0aa850
3ff9408e4ce5674b bf9003c20b1470bc 40029813367087bf 3fd9cf86c5cabf9e
3fc678467bbebfcc 3ff98549429c24bb 3fd41d874ff7773e 400282a58dcc43c7
40022c00176b7f8d 3ff52f5255660324 400275e7c3c43cf3 3ff0f41bdc5e6941
bfecccb3bace0a7f bff43b07ed9a99e3 4003c332476fdf8e 3ff1e9ccbee7185e
3fe9aac797bee874 3fe91cf83e33ff2d 3fc0a116bb3f83c4 400088cc75888b45
3ff65469f23bf4e5 bfb75121795e892d 3ff65214b8926468 3fee284247e73263
bfb6b51bf63be02b 3fdda74355e77078 3fc7ec9128c96636 3fff54899d4c99c7
3feef09dee42152e 400183b6d2969f3e 400470ea6100f89e 3ff4a579ae9a4da5
3fe47ab8aba02eb6 3fd5f7ee53984e11 3ffd6faba6ca4e58 3fe7c4afcf97fc2c
bfe86a4811dbd0ab 3fffec939bccee8e 3ff9078fcccbac9b 4005b0bf77f1baa1
3ff3fbbe7e921bd1 4001511b13a74b70 3fe894d778226a90 3ff44511284b1fcf
bfb20a023abcad23 bf86fd380b4bb162 bfd1eb1e4656555a 3fe6485b557d450b
bfe38dbff9b91750 bfa8ed12a11fda1c 3ffb369617e3a1a1 bfe99af3f9192608
3fd117001c13a715 bfd8b1d262086bd0 bfcf9042e8e267aa 3ff92f08d5b5f0cb
bfc24aac334f4bb5 3fe894d778226a90 3ff4f00d89ab1620 bfe36d63f292d08a
3ff2cb66351dd1aa 3fc33018fe5758a5 3ff121a6591a49fd 3ff2ca195019b83e
bff43b07ed9a99e3 bf83ce2e40ab39a0 3fc80d590941ed8b 3ff81afc8cfc4571
3ff9a535a6944693 400090f0903bfdba 3fe2c709c84b0c74 3fd65eecaf8d9180
40061f3e352b3a5f bfe2d6811d5b2080 4001d4db12546ed8 3ffbebc7718e3280
3fec5286f1bd08c5 3ff02192a765ed16 40020215b2ea3a1e 3ff15df4cba0f223
3fed6e93c326eca1 bff1486c5671c97e bfe62b228d59bf00 3fe8e9d6ca715795
3ffc70e964ce9978 40001e1d95c8e3fb 3fee92575386e120 bfe0d3f126ff5c48
3fd6e6ae7270c6df 3feb12f33cf95b17 3f9a2f3b12336687 3fd4cd5b4771d7fa
3ff92dc885afc18b bff4a9b00c9cacac 3fbc28ef4b483d09 bfe46b848aee50d2
3fd602035fdcde19 bff4a9b00c9cacac bfcad97da2d10b16 3ffb89f402c84f41
3fe5c3997869a376 bfdd86421c0528d4 bfe002674f9b87fb 3febcff0b24008ac
3fe7b6311c352f1f bf1f1b12e51c1375 bfd72ec99e63a437 3faa9e02ea2565cb
3ffdfd3b84e7585c bfd5fe8e57c80763 3fe6b131534d61ab bfe206ef32a12e56
3fdd35b4f1f77e23 3febc268869cc801 bfea1cde1cd35149 400383c66068d8b5
3fff000e7612ae1b 3fe709ae60397acf 3fe003ce578cde68 bfee300bbde52875
3fe1bd8b83ac0205 3ffc7151f1453d8b 40029813367087bf 3fb715a9d4432536
3fff71a97da93f65 40029e96334af666 3fe55160186fa576 40021d70f5320058
3fddc00f82ca0183 bfcc0c626defd8b3 40030b1e0d2d8f69 bf983d3d81fec8f6
bfb293e1a62881f5 3ff02ae39f4d6391 3fd5f138a1b95934 3fe0dd93809c2d35
3fe49423f7c56b2d 3fcdbeecf1826fba 3ffa06b24315f567 3ffb0c61623e0f63
4003047e5d199d4e bfc4c4707aad7fb0 3ffcfea56614e495 3ff3d4f936d2605f
3fe80fa8f3bf95b7 bfe7c85759bd4311 3fdf13eb88265a1d 3ff5e7dd50095aa4
3ff4d789e5d2756c 3ff5325125cfc6b5 bfe30f7f08eff47b 3fddb206192ec61e
bfec5bfdc1703881 40032196e4fde8f0 3ff21d5ddc79767d 3ff9c89117945432
3fe5bcf61048c23a 40061f3e352b3a5f bff1ff5c87668821 bfd272fc11c4d8f9
3ff9cda8f4bf564c 3ff5f6fb7c86edf5 3ffac9ba14885a41 3fe32a9e1846a42f
3fffcb05db2201e0 3ffbbf2fa930f197 3ff1431ecc9a4b3d 4000fb42994e6092
3ffaba88f97ab244 bfe14cad79bd5c27 3fd43c6eae605f7c 3ff25a32383c8ca8
3ff020b11ca2cd9a 3ff0b094c8936b0d 3ff3d43e466c93d4 bff522efe5b03ff2
3fd6ab5ab15c5a07 3fe8ca9742253560 4005520bb84d1d67 3ffdf185f2c318be
3fe39c2726040dbb 3ff98f0d8ee96995 3fde5db3c50e9daf 3fd81abebe61d5e3
3fe98b540dbdffe1 3ffc1a6be4ac8257 3fe2f62970367df9 bff4009a45ee3940
bfe0101880a25a5b 3ff990dcb9024b95 3ffc74b3ddfc67ea 3ffa6db61ecdf702
bfdb1f90d1b59eca 3ff06a14a926a824 3ff121ad9fda2f79 bfe5d8f05c0770d7
3fe877fb77590a2a 3ffd83468e4a86ef 3ff5f04aeaca4e88 3fe4a7e37aca54ea
bfbccae45bf1f721 4000c9e1b4656b3b bfe7ec6be66a9448 bfe573f7fb8b4bf4
3ff077aa4d6fd211 bfe426758b5ba26e 3fd766dc33e4b239 3fe20fea88a5af12
bfe98c1c921cad7f 3fbac5db9afc4bd9 3ff0ab022d74fb71 bfe16aa5953472f5
bfb7d2f94074c82a 3ff0b135e9ae3068 3fea5d0da3099c60 3fdfc8eab69562ee
3fe94ed6c8d1bbb8 bfd590884bc94cad 4001559043cfe796 3fe1bc2183c2fe24
3fe4e92be764af06 3fe9f8e2cb2b1725 3ffde757804dce27 bfe3c7a5ece9a8b2
3ffc1aa0139dc2be 3ff4b2664d1370c0 bfe9497c33475c1a bfbd073aae097674
3ff8ece18b48b821 400400926e085488 3ff7418a846a4e1b 3fd952ae5ad9c5d6
3ff732b0ca73015d 3fe7e74511a5cccd 3fec16c95995048f bff4a9b00c9cacac
3ff6d1febc8b94e1 bfcb59398fb40751 bfe42d4590e7e270 4001580e8bfc7907
3fe4f67496d36ce3 bfee893b4e27c79c 3fe9583482f07cb9 3ffb462734b9126c
3fcfb4708de25296 3ff1397dc8b5ce0f 3ff46b912f6c2c18 bf9ca1d76f3e0f3a
bfe8aba2bb9155e3 3fddb206192ec61e 3ff32ead274cce50 4000a0722d2dac63
3fee0aa43922d8c1 40019a2ab165ea6e 3ff5844820a86175 bfd927f5e4d887d7
3ff6b37d3115fcd5 3fd5baf2419d91b2 bfb8a5262672a27a 4001b7175b1a04aa
4008525f3372e476 3ff30f767caa8ae1 bfce63ab40054654 4002f949f278fc02
4001aa2160d9afb7 3fcf76ede4bf22a3 3ffb9698976cab82 4003f7dc8c970dda
3fdea5fa6a567818 3fe6c0f940b7385f 3ff06939041cd06b bfe537569b336969
3ff7a0476ddc6439 3ff145932b2f4aa0 3fd52c8e5cd718ef 3fe42e3ce54aff00
3fb68922480f0e39 3fd2754a68b760a9 3ff6fbc48bcafcb8 3fe87d289c92bf85
400160a09a4acd02 3ff459e95aa7d0fa bf9026a2a83b11d7 3ff4b2664d1370c0
3ffa5f42ab956f99 3fffae5fc18edc10 40029af20e83db06 400307b49671f6c3
3feb969330d6800c 3ffef494ae1a72bb 4000d001dd6e0394 3fe49fbc8064b27c
bfe48a086c4ee3ac 3ff014210a65607b 3fdd2eaa9f0dd389 3ffeef88c26d5bdc
bfcf9042e8e267aa 3feb5ecf310ec882 3ff22ea10b240a8b 4000ead3f96f059d
40014845a7373930 3fda592c14d4f62f 3ff2a4b87131f1ef 3ff6c12f07500162
400079755b5cc15e 3fd5de3cdb0c34af 3ff53a094535f13c 3ff2edbc01535eed
bfd8308f6ff0ed5f 3ffc9be05270f6e4 bfe7bfc7d431390a bfd4e824973d4bef
3fd9f3db10288da3 3fba117a35feaeb9 3ffbd3fe9388ea9c 3ff649e141174491
bff5b30482d548f9 3ff5ff4ea79a178e 3ff8bad5561afb81 3fcebd8acb1405e4
bfd769ec98d69e0a 3ffed901d565d62f bfa96ea3b8b3c970 40005084d00456eb
3fe49dab6996b557 3fe539e36a0e32a4 bfc9dafcb4721758 3fffcdd9392e3b7d
3ff70b41214877d3 bfe38f575305ed44 bfed8c53cdfc7eb4 3fe572273858ef3a
bfd29ca67989f9b7 3fe04661651c70ec 3fe6c7e30565ad0d bfcd660b4dcbbeca
3fc2438806c0cacb 3fe2260c9b19e333 bf6e647cc561e250 3ff92f08d5b5f0cb
3ffd188ecc90e9e4 3f905a56735bcb2e 3ffff889dd94dbe9 3fbdb946399a6323
3fd3f5e03356ad06 3f8f9f3b1b2e9b84 40023b20b97697b1 3ff4e21b804cdf29
3ff59880cb73d9a9 3ffeb13d0f35890d bfe9899f647c0302 3ffbaf578e34c062
3ffae5eece76af33 3ff1e2440bc114b9 bfbd07ebb44c527f 3ff5b3bbd0740acd
3f879bec4f2c9ebe 3fea301673a9d488 3ff497113f324342 3ff3118d2fef8d19
bfd4e824973d4bef 3ffca6b8ada9e881 3fdf4ef4286e913e 40006376759b26af
3fea5d0da3099c60 3ff98f0d8ee96995 3fffbb1f30fac7b6 400172c0f2637d1b
3fdfb7914c10b2e5 3fe2afc7aadb3324 bfd3f5104db19f51 bfd369e74ee492ec
3ff0d166d88482ee 3fe6c7e30565ad0d bff2524ecd44ea56 3ff3326bb971da7e
3fdab22f927dd9a7 bff0c407d09e48c2 3ff7c63984293edb 3fd3399950bcfad7
3fd76a5a5b637031 3fce18d8b41e4e90 3fe72a7726087105 3fe1db3403321f9f
3fe2586b0c1336f3 bfd69d9a36eec4cd 3ff0be334f578c36 3ff22786018bc5df
3ff6b9e78c47cd9f 3ff7fc96d1f3e9d9 3ff5eac2917ddd84 bfd2f5ee92ac495e
bfd7d51f6a0b5201 3fe7bce50d19e276 3ff2bcc0158ea707 bfe27ebd7e3950b1
3ffd8cd926cb2d15 bfe2b3c52596f4fd 3fcd4182131ea4a8 bfeeaa1a55f46caa
3fb9af0a3393da43 3fda592c14d4f62f 3fe8d414a48dbc7b 3ffff1625fc427f5
bff3746ee72c1186 3ff7cb45e2ac6ba3 400278bbd910cad5 3fd6f3e2e28aec81
4001dfcef2708197 3fc9472729e0041d 3fd93303f5ee304c 3ff3326bb971da7e
3ff55fbeb8547d2e 3fef30f97ea50ff6 3fff8c4294874b7a bfe90944e94d3294
bf92a744d5ac7cc2 bfae045eda0fbc06 4005c7207ddcba6a 3fe1f52baa811795
3ff724d8f57d7948 40047c89a931782b 3fd9747cc897ad9e 3ff1d3f0f9cd3c1b
bff20d1640afbf9e 3fe80389eed03c5d 3f641cd2ac942e70 400502d814609c89
3fff58f1bc8daebe bfd25a394aa3765b 3fec160deb999ab7 3fe9fa4670fed15a
400231cee79ef63c 3feb39f5a5dbe96e bfd9eb20d4a3b3f2 3feda649edd91ea4
3ff9218a6b048ddb bff5b30482d548f9 bff5b30482d548f9 3fdb7b3df27aee14
3ff6a571f56c2983 3ff34a4e1cf53f3f 3ff0f4a5c0bd5df6 bfe2c95b331dbc44
3ff47f3ff8da67a0 40044e7fc100934e 3fec673126b4fb5a 3fcdfb1b450c2999
3ff5883114e485fd 3fc0a2fd9c7dc9cc 3ff121ad9fda2f79 3fee130197ad52ff
3fe2d80bdf92e088 3ff119841e00a9a3 bfd7f06bbfc53a90 3ff33bee8a8bc026
3ffb2ced864ae2de bfe08dec8cb3494c 3fc2ec30bf6ce52a 4005ba9acfff1110
3fda61292788b388 bfdfb58fe94ec87e 3fff305a6c06602a 4002ae4ce3fa9e4d
3fef5786e2a46996 4002b17fbba426eb 3fccda411fe0860c bfd4a00c7644fae3
3ff325bc82ad05ca 3ff290b68e05f53f bff287cde3d87cc5 3ff36040a952c3b4
3ffd79c32bb35767 3fcbae6fcbc1840e bff2d6a090bdd101 400824f2dec21ab3
3ff2f29d2be38a08 bfd8a08ee8fdebea 3ff0f2d696a47bf6 3ff454e16d2489b0
3fef3fe560a530d9 3fef5a1f88fa404d bfe235063c1596d2 3fef86410f82d807
3ffd534f8eaa0c64 bfe9704cb6cf055e 3fee79d31296406d bfe491663a0bd777
3fd8a4205bcd6614 4003f09b14cdf07b 4000978d6770275d 3fe1d0fe69048b42
3ffc43def5abafb3 bfa5d34385d55158 3fd3015162d95d65 bfe03dd4d04e5340
3fe49e8bd8234f36 bfec703edac40ff5 3ff604f03713e11b bfe4cc5704d152fd
bfdd3ea1f85acda7 bfe525446555941c 3ff3af0f67f13b62 bfe4f228ec3efb6d
3fa6e659312df969 4000914aa6e3e639 400318cae6913b16 bf9d381c27cc725f
bfb98b3c61bf4d3e 3fb90f0b63ab6aa9 3fdd6aa6520a3210 3fe3b6f4c6fef29a
3ff4db3325d07e5c 3feba22ac601f559 3ff1e9ccbee7185e 3fd3566785a45be5
3fe72cb371c6c431 3fc7760f0fd66d05 bfdd86421c0528d4 bfcec58e12c176a6
4006a62e67369ac0 3fd31cee544b5c15 3fc7d2edda54dc5f 3ffce7906583b493
3ff175464603c829 3ff9552dec3ade61 3fee284247e73263 bfed0c1d474430ad
400196fa5f520748 3feb523a75295378 bfd300fe849831c4 bfa27bbc510fbe5c
bfd255d3d806be7e 3ffa0a264fd562a2 40001599c7e7ca18 3fec0d42d93b64be
3fd475c3e5017be6 bfc1f62bc7631081 3fe959b59bf7be7b 3fdd9c18ae499bc3
3fe7c079b3d4f435 3fd3015162d95d65 3fe2435f8c014c8a 3ff191e2f5447ca3
bfd5958dae723070 3ffa3abdfa1d64c8 bfd2486439774b58 4005ba9acfff1110
3ff7320b0ec13094 3ff4ffc9528e5617 3fee39eacd1f15f1 3ff47ccf23d74e41
bfe0f25f7c11b61f 3fffa70f40812a75 3ff7f536834ebb25 3fc35bc9a33c36d4
3fe022e962af296d 3ffd8cd926cb2d15 bff179d3d9e395d7 3ffc0e5e7d188212
3fe15dce1a804642 3ff9ea289497aabb 3ff2176f93ab0d79 3feec617be541c36
3fec877078426fd1 bfe6a7090aa7bf6e 3ffddb151d4bcfb4 4001efd8c816e1f8
3fea7db997acb7ed 400439ae49a728bd 3ff529a751a221f6 bff0234f44bae28c
3fd1df3dd9f18bfe 3fd1cd01880915e1 3ff4fe8a243f7a7a 3ff4ebb6d8444d0f
bf946d671252ee48 bfd170e4d1718291 3ff3b3d88d122b3b 4004083a631b0b7b
bfc9d5d2d55462e3 3fe178a3f1038501 3fe5bf7a70f602a4 3ff10178ebcdf64b
3fe9f8e2cb2b1725 40033079a4b1aa9e 3feef09dee42152e bfda3487b3d4dd52
3ff1d7c7454ec3f7 3fdfc26a008ba7a1 4004452f09016d61 bfe2319c387eade7
400482ae089f82e2 3fea2f9647594504 bfebd1ee8ce39d9e 400272aede587e7e
3ff8155606a86d13 bfdfe5ecdbafa7f7 3fd00849f311945f 4005bd17ae08fc87
3fe27afcc93ce3fe bff20d1640afbf9e 3ffada703426f0b3 400404a289e15f92
3ffe723aeb4b290e 4001de3cb007b9d2 400824f2dec21ab3 3fe600bb1f756d8a
3ff076214ce8a631 3fe94c9f02d0a1bb bfd1dbf3b790811c bff5b30482d548f9
bfe26be811f1e019 400000acd3867eab bfe7c1c4a420739e 3fc9472729e0041d
bfebaf504fc8d6c8 3fd8d1de8f3a512b 3ff8e8275337bcaf 3f879bec4f2c9ebe
3fcbae6fcbc1840e 4006a62e67369ac0 bfe27ebd7e3950b1 3fdfeb9563081ac6
3ff0702e5cbff886 bfd72ec99e63a437 bf94730a2ecc0a6c 3ff2cef496f87fea
bfe80dbbf58e517f bfd216053a9dd6de 3ffa92172b88df72 3fed6af56ef528a0
bfc1a72bbefad255 3fdb3fb4f626fb6a 4005d56de701dc76 3ffb5826d53b76af
bfc886183aec17b5 3fdb2e7119d69ad4 3ff937d1dffdc1e7 3ffa31975023bb0e
bfedadf23c68f7d9 3fe94dc4adf7edae 3f919e046c33c600 3fc754c5cb2ce146
3ff24b624eaa58af 3feaec39f48f6abf 3fe9da1eef12655e 3ff65339d41b199e
3fe395e2e824f74c bff3425e7a4abe19 bfee02393a08ba8e 3ffddd4e25ffcdcb
3fe904cec9385e95 bfe5b4acea89978b 3ff9efeb655c6a8c 3fdcf283677d76c3
3fd63d0a5148b33e 400147454c67ded4 3ff26c44f0f3704a 3fec39f04b3eea0f
3fe6c0f940b7385f 3f92d34c0e6e8420 bff18b352ffcc57f bfd6a06d589f21eb
4001832e0c9c5b41 bfd77abab57e043c 3fead2a547a89941 3ffed4f2d777a6f5
3ff4a3a2cf8a5eaf bfc3706c5435ddf9 bfb4af10288fc7f1 3fe7caeee8acf588
3fe539e36a0e32a4 3fe18851e53156b4 3feb62b878a3c629 3fcffd30de279432
3fd382d51bcd9613 3ffc1e81768cab11 3ffdb46bb630014a 3ff3cf539119dd45
3ff10a0defd7dae5 3ff2a03bf8ed9fb4 bfd4e198c2a5c174 4003a77ec55a713d
3fe437439646814e 3fe2c52936406dbc 3ffc39d72d8c4039 3fdc913c09b3114d
bfe5009806d6f414 40029813367087bf 3fd708ff4552d90d 400030e76c394584
3ffbc31d012aad60 3fec39f04b3eea0f 3ff370efd387e979 3fef604af2f479ab
3fe35b03470c7624 3fe0856004e75ecc 3fe4cddcc99720b6 3ff65d2bb5763125
3ffd71df5e25e6a4 bff268134a052f45 40041237a01f1553 3fd015605cbfdcc8
3fe27b87af5c2911 bfdcbf60be6399e2 4003c8a9caaa21aa 3fe8a03022f784d9
3fef0d175672b60c 3fecce89050dc26a 3ff98f0d8ee96995 3fe2fe7a83eeb932
3fe9540236906e7e 3ffb4257d97db891 bfe0f1b0d7a7a36f 4000c34697a14cd9
3fcffd4c1829fd5c 3ff5a049697d8e21 3fe66fe2bc9f70d7 3ff47cc777683db3
3fb68922480f0e39 3fe709ae60397acf 3fffb755d2a24245 bfccd79785801462
3fe48d0f1219493e 3ff71db756b17544 3ffe7bfd6f714c9d 3fdf305b960acdfb
3fe9511386519c11 bfd2efdc4e0a6910 3fdd73213c9c1c89 bfe5dc33f9abca83
3fde4da407469cfc 3ff8a3ec36427aa0 bfced2b2412cf608 bfd5cae5bf7b6458
bfd7f06bbfc53a90 bfe743007261d1bb 40010a06f718bd55 bfe2a85edafe2513
3ff4db3325d07e5c 4002f2d847529fd1 bfd55c5830d68dd8 3fe6c14d23d7f4ce
3fb2b8870a7caeb6 4005aa042812d82b bf9dbc32888d4643 3fdebe8703ab4d27
3fcadec9ae6019b5 3ffe909952174607 3ff7f71347840969 40047d7cf82725d1
bfe27ebd7e3950b1 3fe14ba3f5515f3f bfdbef974eaf38e0 3ff9f929b06f1e21
4005520bb84d1d67 3fc1ced53c44e647 bfef29fa5517ab13 3ffa9edda5ba558a
3fe608e87fac3354 3f879bec4f2c9ebe 3fd5615e373235c7 3ff5abdc6589c9a9
3feec874fc94d2d9 bfe8023914cfba5f bfdc4d5f1a2882ad bf9bff48772a160b
3ffd4d6ea1232825 3fd5340bd5d4f130 bfda440588a94b18 3ff217c7bfc993f2
40054b36e742ff68 3fd65eecaf8d9180 3ff6a71bdbf4de52 3fe709ae60397acf
4004f44c8fb5b7ad 3ff42affb0518376 3ff9247065c038ff 400121b23a8b188d
3fdd00c91b8893d9 3ffae5a210ba26d3 3ff4cd0c818e1822 3ff12c1d95bef51e
3fc2ec30bf6ce52a bfeec6fac3d017cf 3fef351a9e87f2f7 3fcf86d325f8154a
3ffadc7a790c5918 40017770b6269053 4001d86029a7295f 3fc1ea59455f72a6
bfa8ed12a11fda1c 3ffb0b1c5ffff287 3ff4332031106e6a bff331b37761fd96
3ff348c18aa2de8e bfc28bbd61d10673 bf1f1b12e51c1375 3fc92b3fdc0538ef
3feae201d1e7e8e7 3ff51754b3ce8867 3fe5cffa6dfee8e6 3fddee860d49309d
3ff1291c9a5f3571 3fe95c9e6720a953 bf983d3d81fec8f6 400282a58dcc43c7
3fd5fda015f45876 3fd77ec8aa2abf63 bfec88c1a7b40d46 3fffc4e6ae9661bd
bfe01dfa0b562685 3fe037eccd6b3516 3fecc48fcc622d73 4002186b406d7164
3ffef7228811f44d 3fef260a62c04b54 bfd11260059c216f bfe04029796a998a
3fef5df13ce8da80 3febd460b0efec48 3ff46b912f6c2c18 3ff56f3654046477
bfe773d1656eb9af 40036566a419aada 3fef85a443740da3 3ff48074326ac8b3
3fb5bb50f188411f bf983d3d81fec8f6 3ffe7a9dbf5142a8 bff18b352ffcc57f
3fe8fc3769774f72 bff081e0b9c42932 bfdefeadebd76d22 40056682b333cd18
3fde7fc38867b5ee bfe5efa94b4560f2 3ffc160ad77aac96 bfc28bbd61d10673
3ff070f22dbf423b 3ffeffbff8581608 40022f8d0436ee12 3ff640e49d059566
3ffb4cdae5326d6f 3ff9f51602639493 3ff5870eb3b5d9d3 3ffde757804dce27
4001aa2160d9afb7 3fedbc3a8139af9c 3feab7c9a81c9c9f 3ff4ee5d48e24150
3fe59ae58664966e 3ffcc269c1c59991 bfd977b06fbcd796 3fef41153d4ed19a
3fc344e53fd8a741 3ffe987fdf010b5c bfaaa88167b0b7ed 400485f36b43cd76
bfe180ac5644b252 3ff1d3f0f9cd3c1b 40068c253c60f042 3fe7cb5fa96f69bd
bfce0037dec622ff 400131c7a6a83b1c 40006b84556cfeb3 3ff527a0325db733
3fe01b3cfe558637 bfd7f55b3186755e 3ff5499d34a8ab68 3ff3fbbe7e921bd1
bfe8d7ad8a1092f0 3ff4cdf00f894a7e bfc326144ef7bc4d bfd7d469bc1a4245
3fd01c4adfe69267 40011543e34ee551 3feeb5fb06a26abb 3ff945bebd11cc7d
4001bc1c084104d8 3ffb6c2a0367328a 3ffc5a32a9377190 3ff76f61237edc06
3ff9d6f7f419683e 3ffd25e8a4331610 4002611733ace0e5 bfed075e8731b8a1
3fe0f83255f72c1d 3ffb14a539a732d1 3ff5a26add264808 3ff9a47720f97ed3
3fb90f0b63ab6aa9 3fdfa73b511e6737 3fb53d0be165ad47 bfe9704cb6cf055e
3fe27e9e84899fe9 4003bd0fc8c5c220 3fe894d778226a90 3fec78e3a2954933
40000e26e6551eb3 40003bf92aea7a0f 3fd4856c64d84416 3fcd0d06b943d9a8
3fecaac64bc20281 3ff6cc38b7dd9ef9 3fdf4569887e465f 3fe2435f8c014c8a
3fd475c3e5017be6 400049ca2e43810a 3fd3826b51514267 bfe426758b5ba26e
3fe76b5423f87c3b 3ff910ba32c1bd20 3fb300805bbab12e 3ffc420ca256d990
3fe58e0320c5fe58 3ffc420ca256d990 3ffe7a9dbf5142a8 bfd008115a080e40
3ff434c95bc3b691 3fd87435ded69d7a 40015b8bd940b12e 3fdd7027b1ec0004
3fed6af56ef528a0 3fe7c3be00c8fdba bfe9291efa752964 3feca701841269f3
3fd748dc75818b2f 3ff083ce44138021 3ff5c8e48ec66f73 3ff088f517bf99c8
3fecdbf1476db1e0 3fde67791322796f 3feae201d1e7e8e7 3fe7e74511a5cccd
3fec856b0bc3385a bfd4a00c7644fae3 3ffa1156efd9f960 3fe9511386519c11
3fda052cc88339b6 3ff4fc8292069e25 bfd7970ce605cc31 3ff7a0e14c1ad779
40074cf6c739df50 3fed85da47ac2b67 bfe153a436108e25 40004a82bda9e19a
3ff0361a4e8136df 3ff459e95aa7d0fa 40003a9217da1a9d bfa61f56f3ea195b
3ffd7df9cb4cd947 bf92a744d5ac7cc2 bfdbb94801fe7bda 40062f059b077f68
3fd2ae0967c4c3a1 40008307ac4e1d1e bfb482f3a298c541 3ffb2ced864ae2de
3fe27e9e84899fe9 3ffe88a301d971ef 3ff7fa3ae2bc6feb bfd8e64161dc15ed
3fc53214b08985b5 bfef1ceacfc61652 3ff25f5b196ee81e 3fd117001c13a715
bfcb5e433a489814 bff46d77f3c241bc 3fcad4a364e76f7f 3ff18c44dac91319
bfdab062aabe4964 400796c715633d39 bfd29083e20b5e55 bfd927f5e4d887d7
3ffa988876c82370 3fda61292788b388 bfc31a11152744ed 3ff643557ad8c813
3ff7524b8f552c22 bfe95f14bab05dfb 3ff959f0ffc196ef 3fe894d778226a90
40040bb5c31209e6 3ff7be04d19c8298 3ff7dd9360beb4d1 3ff4a9bf3ac66bb9
3fec29c7c041d191 3ff0c24b9280d3ef 3ff8eeb5d9a57bfd bff3ed15c61f0bf8
3ff0e46147d595b2 3ff0e666e8f652a0 bff3d280a676831f 3ffc2d7614dc9204
bfcd41431f7c72f6 3ffe5c4dc889455d bfcdfa09e1fdf035 bfc911cef2f489f7
bfec319e8ac01e7f 3fefc8cb95c8e82b bfe5d8f05c0770d7 3ff8155606a86d13
3ff81ed07eb17c1f 3ffeac792cb57aaa 3fe7b9938403bf12 3fe6a6b746700838
3fe35ff84fe43344 3ff8525df98a4941 3ffe051e051a7021 3ff0d41073a50f4c
3fe6f5417b350ea9 3ff6fbc48bcafcb8 3fe1f59d2f447fa4 bfe36d63f292d08a
bfd2efdc4e0a6910 3ff8b29ec98b7887 bfc55256da0bc5da 3fe354ab27a71ace
3fe44c1a967936de 3ffa859e66b49f4e bfa55cd143b30c88 3fea4c41eaec7f3c
3fe88ba4709d42b1 3ff04d8a433336be 3ff3e084d16d5775 bff5b30482d548f9
3fd13c257ca39e00 3fee284247e73263 bfe1c9065899f357 bfb7aeb8e19d53f7
4003a77ec55a713d 3fe009d772b7c527 3ff3d3df61a6d4b0 4005a24db85cf181
3ffc9a4b3542909f 3fe875ff9e0cae11 3ffb08c6ff63963e 3ff77c2401dc42cb
bfd81f6718f8dad6 bfe24f45fb3d22aa 3ffb15a087154a12 3ffe07118b757fd2
3fc678467bbebfcc 3ff07deaea2453c2 bff2d6a090bdd101 3ff3cf539119dd45
3ff83e0f23834686 3ff7c4f11139fd63 3fd3566785a45be5 3ff02b497c0a8b0c
3ff33cffcdc6c2ac 3fd47920d69f0aa2 3fdeb03629e088e9 3ff6c7e7aee52e4d
3fcc6ee90d954ab0 3ff1d4dcadd5ef24 3ff75365ce952820 3fb481538d306b14
3fe768cb3d03ce71 3fd873859c906958 3fdd35b4f1f77e23 3fe0249230de0f5b
3fbe3e305141d0d1 bfca400efd964ebf 3fdadd9b0130242c 3ff529a751a221f6
3ff2671ed0bbfddd 3fe2f2e020bd6f73 3fd9c2cfc565a226 3fff5be688836a6c
bfe4bd433e50eaa1 3fceddc27fa01d56 bff5b30482d548f9 3fd60cbebb87f0fa
3fdd2e34c3863dd9 bfeb776887dc82b7 bfd746c90ae4b370 3ff3c3a5647c85c1
3fed85da47ac2b67 3fd39776c75c5c34 bfe25191c396edef bfd17f36e40a7d1e
3ff2d44db25a41bd 3fe178a3f1038501 bfed175168a67279 3feb617a14f661b6
4001ed5343dad88a 3fe25611b66b47c4 3ff73680ca41d466 4002e64ab064c3bd
3ff8a3ec36427aa0 3ff1ac62ce6d536c 400043720ff277c3 3fd4361c0d5eaeb6
3ff0089233b8bfdf bfc6245793c40ec3 bfba98f5966fa95e 3ff044a1cd6f50e5
3fea5d0da3099c60 bfe07ea453919f84 3ff8b286ad5051e4 3ff9935546874ff4
4001a35ac1fba4ed 3feb3a1388ec2da2 3fd7534b41eea428 bfbb6a1f8012334d
3ff2591040a15907 400160a09a4acd02 40009e5d42af42fb 3fff8cc60bf339bd
3fe66f472085e0ce 3fd6f411aa1dc2e6 3fcc6ee90d954ab0 3fd117001c13a715
3ffd5b5677e34e3c 3ff97fa8721c269c bfd2efdc4e0a6910 bff05e101867863a
3fece6207ca2a6cf bfc07fbf1f8451d7 bff07b2548428ba3 bfbccae45bf1f721
3fe59ae58664966e bfec644d8990a756 3ff8709f43152875 bfeec5d97b550439
3fef604af2f479ab 3ff00273b769a795 3fe10856d245e00e 3ff30fc129b6f98d
bfbf5b1c71cc2386 4003ae0c3e6daff1 bfe15c68e4c6d905 bfce6a289e42cc5d
bfe9ef0b98f6e362 3ffa1189c9348bc2 3ff1efa0c981e1fb bfcad97da2d10b16
3ff1bae009e9b26b 3ff32fa48c4bdec0 3fea7db997acb7ed 3ff1b7ca7b202e5d
3fdd19a3abb74285 3ff3cdbe187fcea5 3fe652d5a40ca09d 3ff3326bb971da7e
3feb969330d6800c 3fbf0c635a78b54d 4005ba9acfff1110 bfe12ddf089a4850
3fbadf5e735a4eb9 3fd145e81080ce27 3fd49ebf72a3e1d0 3fa1a58d4b196c55
3ffa301214939700 3ffb2f74dd767f0f 3fd600732d4eb786 bfb048ea817a87a3
3fdf065fde691f00 3fd7fef05d787ef3 3fd33d022c0c2da7 bff2d6a090bdd101
3fce0637c5e19214 400033c42baf96bb 3ff9455e889fc100 3feea27236343e9e
40056682b333cd18 3ff6c46ffdf017d4 bfe0f1b0d7a7a36f 3ffb15a087154a12
bfedadf23c68f7d9 bfc8c62ccb5d1b6a bfb7d2f94074c82a 3fee4a6cba2eb5f9
3ff069732c42bb29 3fb09c528d8fac46 3fdeffc3995d8829 bfe30f7f08eff47b
3ff84ac9bf84d721 3ffd7a587cd79a80 bfe87cf2fabbdaf2 3ff2875f71d33d03
bfd07b5a6ddafff7 3fee57eaaa3d57c5 3fe1ab0294118ea1 3ff4474aaf52c2c8
3ffee0f3796e604d 3fff68999acae97f 40020a720c17a26d bfd4ffb0f9abd65a
3feefda4aeb1d500 3ffc39d72d8c4039 3ff4bd5a1fdf5253 3fc12f992afc2f55
3ff6427a25f652a0 40060e8702ffe27b 3fa5bd421846c0a4 3fd6f21c037ef5fd
bfc482dbf7cb96cc 3ffed4f2d777a6f5 3fe684e5b77d17d5 3ffae01f70735204
3ff3165d2a9803dc 3ffd59d5cccbccd6 3ff4826ac648d992 3fce26073e426d89
bfed8c53cdfc7eb4 3fded87ab2cccb0c 3feb3673c7d9faa8 3fdf065fde691f00
400405fb2a4c776b 3fe6c7e30565ad0d 3fed6ec4bfdf7b8a 3ff235d2c7265733
3ff79d86da77ce4a 3ff2788af17df1f4 3ff65214b8926468 bfd3b11fe7bf73ab
3ffcded89f39b223 3fe894ada01c9f70 3ff413998d47afe1 3ff688033501edbc
4005025aeb8ceb48 400473673f0ae415 3fc82f35e18064cb 3fd51e0debec51d6
3ff29b91e75ee398 bff35f11ed2fc5a5 3fe94ed6c8d1bbb8 3ffed75b6fc70de3
3ff6f10c20a2bc75 bfc8706cf0dec0d2 3ff2d44db25a41bd 3fddb206192ec61e
3ff97fa8721c269c 3ffac4793e06d0a0 4000629ea12316d7 3ff20bf29cdcb8fa
400244e3a8562ee5 3ff0bc4f21b1983a 3ffae5eece76af33 3ff8f17885f0ba27
bfbfda6af7b85f0b 3ffbea605491796f 3ff879c9f5a9af32 3fe61aae1239208b
3fe55657bfee459f 3ff2d862a635ab61 3ff47a35d0f98bed bfe098d59b5f75bd
bfeec5d97b550439 3fef105497de2fda 40004ede419ad80e 3ff3e7751101af55
bfe6ee8cb35cccc1 3fdcecb8f9d42763 3ff0221c67de9341 3fe3c2d74c4f59d1
3ff0523824cf6bdd 3fdd7027b1ec0004 bfe05c132c02a725 3fe1be81da9c1dae
3fe033d45551c04f 3ff52a94c14438b8 3fe5fbf98facb2f0 bfb21423bc8eef57
3fde8840e536de2f 3fa6bf2217d108a8 3fec45a217ee16fa 3fdc61c84332ad0d
bfd8d6a486fc6da9 bfdb8d44df89d716 3fdeeb31e970c4fb bfed0c1d474430ad
bfe95f14bab05dfb bfd5e6d78c798158 3fff4ffc43ea2a1f 3fe3885303e8dc05
4003cae14a7568b2 3fd8a95ddc3e542e 3ff306979a48bacc 3fff9b31d2133d25
3ff028540eb38970 3ffd109bd5a6ebfb 3ff7ccef6cdd3826 bfd8a392c13766b8
40002b7aa913a5bb 3fbac5db9afc4bd9 3fb1cadc1e9d8f01 3ff9147a5295c226
4000afd2414f2697 bfc886183aec17b5 3fd6c51ffd96b834 bfba98f5966fa95e
3fc6fe942408f6de 3fd0ce716ecf65b4 4003a4f8cb3491c1 3ffed190fc5e6dfa
3fec29c7c041d191 3fb09f2577339df4 3fe1393d4259e218 bfed0c1d474430ad
3ffb89f402c84f41 bfdc4eefb985dcdd bff1a1585722d287 3ff40e5d06066e0f
bfc01a26209e29d8 3ff1f523795857c9 3fef351a9e87f2f7 bfe11c4506619fa8
3fe640a533214d25 3fe51e493bdffa6c 3fc22dd6883eca2d 3fb25b379663510e
3feef93cbfcda682 3ff6eada19acc259 3ffefe31b98593ea bfe57f6663c5059c
3fe8b126739c7e5b 4004bfec02cc0235 3ff91112df2ecc14 3ffcaa51fe2a6e48
3fdbfafa4180cb62 3fd7bb5b0243c0aa 3fea5d0da3099c60 40004c0b37d037bf
3fbba19754786b63 3fe1b6efbb56cd0d 3fe2a8acf697032d bfe852862ed71768
bfe9704cb6cf055e 3ff53a094535f13c 3fe944fecc3730e4 40028978a90b888d
3ff62ed4c639ef3c bfa59b2b13e262eb 3ff56b47e9e1a4ed 3fd033ba709e8321
3fe3b0164c112a99 bfbed681e9ecb055 3fe58e0320c5fe58 bfdfdf51a9e92141
4001d86029a7295f 3ffa4f76285b70f6 3fc8043a35e32c63 bfd932d9a3c9c9d4
4002a347da1fda80 3ffc95734fbac4ca 3ffb4e311d2ee095 3ff6f9756cfdf763
3fd4b44683de170d 3fc2cc97ccac618c 3ffa513e0a044206 3fd39ee9e6a82ddd
bfe5a21d0bb269f4 bfb127531991125c bfefe06f89bee586 3fe80389eed03c5d
3fe1bfde5344b13b bfb7d2f94074c82a 3facbf0f9d87daa8 3fdd4baa1d33313a
bfe434aa840434ac 3ff5addeeac3648d 4003477f1bc0a9a3 400008253912e0cf
3fd3afdb6f4175fe 3fb852cdd6b4b5b4 40018eb0dcb6164a bfe07c0dbb34dbfe
4000864f977e9fce 3fd39776c75c5c34 3fe94ed6c8d1bbb8 4001c62095b01cc5
bfe65f4c2c48c295 3ff5c8e48ec66f73 400147454c67ded4 3ff34d630ee15a93
400581822600beee 4002a8551cffca04 3fefc2139143bdb1 3ff84bb335d313cc
bfe69fa7b68726f5 3ff7b385078de89c 3ff898270880ee21 3febbc5676d2ffb9
3ff2c8a062497fcc 3fc051721dd266be bfbea73058d890f9 4002205bf78ba664
3fe817ef1d57769a 3fc5ad24a1ebc751 bfe08ce869a7b4ee bf9d381c27cc725f
3ff3571b0bfc9400 3fef02f48b39e0b2 3fbddfbf82991e39 3fd5f04ac31de9ac
400592e86bd39133 bfeca8d386c16db0 bfdfb68bdb3ebdfc 3fe70ed346414175
400041934c1ffd9b 4002c7b6dd8aaf33 3ffeab9e9d6e29ab bf9f85ccd7024ce9
3ff65c62ac58e68f 3ff860fc73e8247e 3ff5ed61f78ab998 3f9df3049c42985b
3ff1eb60a094b442 3ff38cecf9e6bd1c 4005010831278566 3fbda88ce2a8f313
bfafdc0cbb13ea5f bfb33d273fb993c5 3fecf912496fa6e9 3fea259fb9367e80
3ff67501e6c72b14 3fe959b59bf7be7b 3ff5d8f9db8db605 bfe87f21ca1d436c
3fda7110e896aad0 400278bbd910cad5 3ff057e8ae7ccfe3 3fe8193aa65ff9d3
3ff30b921c8c7561 3fe92baa1e3f72d3 bfd3ec2fcf98abce 40002828af253072
40031f934238aeac 3ff35aed9de4a79b bfa8ed12a11fda1c 3fe4f67496d36ce3
3fe1393d4259e218 40037292935b1da7 bfe42d4590e7e270 3fee532f17bd294a
3ffa2771dbb849f0 3ff824e67700247a 40037292935b1da7 3ff656b6f51650fb
bf98e6c431efcbde 3fddb206192ec61e bfbd38c858921280 3fecff0f32709edd
3ff61b23a66587b3 3fe33ee9f54b7091 3ff65214b8926468 3ff476782175e644
bfe434aa840434ac bfeb776e77afb979 3ffa31975023bb0e bfe29b07db46c273
3fe0a89d363a171e 3ff6ee95bd66ef10 3ff3a58b94752ccc 3fe29130def40ca0
3ff6735498a06c75 bfd536ac8ce7ab34 3fe32a9e1846a42f 3ff88a9c013a46e7
bfe84f3cfbd94528 3fd380e05b2895c8 bfc9891dbee66b29 3ff438f11f02280a
3fd31cee544b5c15 40016150f988105b 3ff24cea7360a4e8 bfeb3296f83d5317
3ff6b5aee4ae29d6 bfe0770a28ff11db bfd71f05fb8fb596 3fe90602c78d2f4c
3ffbbffb3258fa92 4003c1e574e8abfd 3fff2645ee17705a 3ff098638188c2ca
3fed9ee0cafd2e47 3fe4180f7520437d 3ff06939041cd06b 3fd8b174470ef24b
3ff47cc777683db3 3ffeb488549c3f97 3ff51875a78f68b0 bfebc6b291a06c98
bfe9704cb6cf055e bf9cf8c4c1c3083b 3fe9366d5f7aaf10 3ff89c1a81e11c8e
3f9ce6e9eef3cc51 400296e93955e2a1 3ff90a28b7e55415 bfd750310dc6b6f6
bfe2398c8b9e4a04 3ffd8fa2760bb99c 3fdd3cf19a5b0625 3fff000e7612ae1b
bf983d3d81fec8f6 400824f2dec21ab3 3fdaf382b2435c5e bfae7362cf2a2177
4000eada921278d1 bfe4d6e859545e09 3fd1789a9f181ca6 3fdf61c0391d5d5e
3fe652d5a40ca09d bfc0f0b452c90065 3fe32a9e1846a42f bfd833e41bbf1a49
3fbac5db9afc4bd9 3fe6b29786da6abb 400238d53fa2e5de 3ff1fe453d4f5d02
3ffd7a587cd79a80 bfd76ebd7e01a42a 3ffef98e0abf7566 3fd5d711bac29f0d
3ff70232efd3444a 4003058c4cff0973 bfd8d2dcb8575716 bfa7391ddd16f094
3fe9da1eef12655e 3fb3e3bc6e4d6253 bfeaaed6acfa9019 40031f934238aeac
3ffce6b7e084deaf 3ffa05e2f01f988b 3ff1feeb428da81e 3ffcf9bfbbd595f6
3ff9bdb388c9aede 3fdcd32ef74abde3 bff56714f5096c26 3fe600bb1f756d8a
3ff2a03bf8ed9fb4 bf99f4cd747f366c 3ff49c6ade38b75e 3feb200bd02d823d
400088a95389f4cb bfeca8d386c16db0 3ffd188ecc90e9e4 3ff61b11d26f2c83
3ffad01910ef1fe5 3ff61015bb0f80c3 3ffd51f1ff3e5c13 bff268134a052f45
bff0724356dd6aeb 3fecd8cea2b8867a 400631c20751a5e7 bfd255d3d806be7e
3ff60f67315ef71d 3fd6f21c037ef5fd 3fe724d5a7898aec 3fd047e9c0b508c0
3fed9176c0d64d48 3fe86f4ce8507f95 bfda4ae2047f7c53 bff43b07ed9a99e3
4002500282556f12 3fe409ae7769b765 bfde9c37c7edf2b7 3ff6a0ab7e134a43
bfe0f1b0d7a7a36f bfec6892e10b63a5 3fe63e0c6e4a957a bfd2e9e4b1b7e6fb
3fc2438806c0cacb 3fb1bd25b94844b5 3fdd2e34c3863dd9 bfd6185ce4babde2
bfba81a15521d3f9 3ff5cf393f30aa87 3fe2c41ad40244e8 3ffd7bec6139d063
400030d56e6d49a5 3feb43f4e28e2f0b bff43b07ed9a99e3 bfe15c68e4c6d905
400475a312f1c2e5 4000a6e2cfc54298 3fc16eeed1fe7ff6 bfd6b3f9694137d4
3ffec0fdb73856f3 3fe887a65b2532e0 bff18b352ffcc57f 3ff11f426c795322
bfc742b1d881a3cf 3fde6dc9acbf32d7 3ff052be770198ce bfd55c5830d68dd8
bfe36e490d3e868a bfe9704cb6cf055e 3ff454e16d2489b0 bfeec5d97b550439
3ff598c938652919 bfe7a946a89c1295 40047c89a931782b 40024851177102bf
4002b51a60156abb 3ff6eada19acc259 4002f949f278fc02 3ffd25e8a4331610
3fea05e86a3ff59d 3ff6163d1d9302c2 bfc0ee93c4bb4208 bfd9a6b9f8c0836f
3fdbb6fea0f44931 3fdccae51ec51eca 3fe1262d665ce166 bfe7343fcb489697
bfe55d6457cf6eeb 3ff5d8f9db8db605 bfdab062aabe4964 3fd32a3a04f7ae55
4004f44c8fb5b7ad 4005fdf9730212c8 bfe4cc5704d152fd 3ff95b5430c0d224
4002bc554c929a1c 40058108ef73afea 3ff454e16d2489b0 3fe17000aa0fad0c
3fe25dd2c4505541 bfed0c1d474430ad 3ffba50a53a8eae0 400796c715633d39
bfc4fb2979b23845 3ff25a32383c8ca8 3ff4a73e3d98c8c0 bfbfe212e0fb4dea
3fef4b1817be7acd bfc4d06443fc8209 400224228dc67eff 3fe3ed8566e99e1d
bfec89e2f02f20dc 3fdbc4f832871ab9 3ff4bd5a1fdf5253 3fd94479c5605085
4004139bf4132092 3ff1eb60a094b442 bfdd4830fa00d771 3fee47ab9536c550
3ff4459318194816 3ff4a73e3d98c8c0 3fdef2ec487226e1 3ffb59d79af69769
3ff22ac2d6d8c91a 3ff3c4af4d470f0c bfdb237983adb57f 3ff1de97b8ed11fc
3fcfbd9846281be3 40022f8d0436ee12 bff247bd4cc1da06 3fe944fecc3730e4
3fefdc7a22a7be73 bfd55c5830d68dd8 3fc0a116bb3f83c4 3fe7fa00d94768d9
bfd09300f437afe9 3fe49ef464ab7f40 3fb689a3d3d96cbd 4004e320847f0e09
40009bac5062cce5 bfcb0f1cdfb35e7a 3feb5ecf310ec882 400306aceea0956e
3fbe32370a51542d bfe852862ed71768 bff268134a052f45 4003c1ccda4cae52
3ff95b5430c0d224 3fa0373c812cf1b4 bfbd07ebb44c527f 3ff3fbbe7e921bd1
bff2d3af471e47fc 40084722737af626 3feef03e612c90b3 3fc330cbcd753748
4000f106631d2371 bfd72ec99e63a437 3ffb3e2db587eacb bf7040e2f69049c8
3feaf25b48944b14 bfcaeaaa775cab50 bfd927f5e4d887d7 3ffb4abb296dca06
bfeb6636ce162056 3ff2d6cd5a29f3fd 3ffafa7a13dd2bd1 3ff61fd6a30401a8
3ff79317ce34f1ca bfe5b4acea89978b 3fb300805bbab12e 3ffcacde8d1caa38
3fb1cb16e6a429d4 3fee6df4879bb5d4 4003c1e574e8abfd bfea2b7b0d6a6cb2
bf93b0d3c4619767 3fe37dd9bbd062a5 bfeebbf4e7bcd648 bfd8a392c13766b8
bfe87f21ca1d436c 3fe88ba4709d42b1 40015f630af5ed26 3ffae5eece76af33
3fd076377467bb0a 3f810429dd892828 bfda539c99818771 bfd94b4033f56b30
400470ea6100f89e 3ffb79b07d05e570 3fe5bcf61048c23a 3fd4ce265df70624
3fecaac64bc20281 3ff05f7a0bd99ac2 bfdcaf2f2858ac15 3fcf1f60a9d12d02
bfcb18d291952359 3fe013f7e7fc8302 4003c1ccda4cae52 3fd3c40a92ab7e6d
4002815c1dfd6052 3ff3bf84b03fe505 3fe13c1df1df044e 3ff73c0eeac8a14a
3ffd25e8a4331610 3ff9ab4a00174a9d bfde69b83fca562e 3ff57594cbe03f94
bfe2b5c43940b65a 3fa97b21f06aed0e 3fe5eb8f8790f458 4000a33441a2a751
bff5b30482d548f9 bfd2efdc4e0a6910 bf6f1a2432f80b5f 3ffb2ced864ae2de
3ff241757719e6f6 3fec44763d46800d 3ffb5a2f7ff13c8f 3fe1478b686bdea3
3feae36a6abbb9c3 bff07b2548428ba3 3ffb01d40104ed2b 3feb5ecf310ec882
3fecfe4b1f036a78 4001ae6cb8ea372d 3ffcb08d6946ef24 3fbc28ef4b483d09
3ffb475d924aa002 40015f630af5ed26 3ff03348ccf3e804 3feaa89fac991cd3
4003262ba9f9824a 3fd86f253647984a 3fd748dc75818b2f 3fe49ef464ab7f40
3ff479710abc329c 3fffa17b751ac9fc bf7d6e599807043c 3feb28d1a6ffffe1
3ffd49ad89ad11fa 3feae2a73dcc397a 3fe418acdb5af9fe 3fee9888acceeae9
3ff0e46147d595b2 3ff696145e8ef4c8 3fb481538d306b14 3ff57c959eb5eebe
3fe709ae60397acf 3fd258379b794401 3ffb3478e61b8dc8 3fe1640d4ce4f946
3fe86f4ce8507f95 40012867df2bd2db 4000f2f701e09c1e bfa056d547da1553
bfd1a883e877e0cf 3ff875c136d0faf3 3ff80e9a816f7620 bfec3370e863edec
bfe18cb2c125bb83 3fea5d34e129efaa bff149876034b3cb 3ff7d7b804d01482
40022657569177a2 bfec3370e863edec 3fed4613da4324c7 bfc15e1df717750d
3fd560bc1627b621 3fe74c09a6303b08 3ff9884a5046bbc4 bfe8d7ad8a1092f0
bfc3cfa4f17edede 3ff29ca6fe32a487 3fe5c35eaaeb125c 3ffe723f0654a081
3fe68d15608fe53a 3fe197543cd39e50 3feb200bd02d823d 3fed180cf2ba1a44
3ffd758b9db16cd6 3fed60109fd01271 3ff1c462d312f133 3fd8ae1f8d17b40a
3ff6496395f2709a bfd5cae5bf7b6458 3fe5b4b580a6d832 bfe68498783828b8
bfa4388365bb147d 3ffb51f13d02e59b 3fffa5ae6657888e 400307b49671f6c3
3fe627ec2c06d47e 3ff18ce382d60b93 3ff7550dc0017716 3fd189747124f7f7
4001a99d63901957 3fdea5fa6a567818 3fec594e3df77e93 3fedef8244044e11
3fe58617497b22d4 bfcbb4e5be797bae 3ff73bf560b3e3d6 bf97c0e17a8df1ac
3ff7389f8eb7eb5e 3fe2283e488c02c5 3ff56ec1569a166a 3ff4015946c749c0
40074cf6c739df50 3fd6a5df0b0f7467 bfc83dcf6374c27e bfeb15aa606bfff6
3fe2abd8c12288fb 3fea4edb19e9c454 3ff3fbbe7e921bd1 40061f3e352b3a5f
4005c5d6fc1796df 3ff7090d8bf82b40 3fd5d58a07d4d130 3fe467d85bf52157
3fb1c6142c188695 3fd63869944f5c7d 3fd11c07ab904d45 3fddf527ade9a681
3fd800badc3966b3 400275e7c3c43cf3 bfdb09511b242563 4000495af1983de8
3fe0257eb2deb47b bfc70a88a09e6eb7 bfe98ecb68a7dd3d 3ff0ab022d74fb71
3fd4493b21dcc1ee 40030df2421177d4 3fe47ab8aba02eb6 bfd7970ce605cc31
3ff081f1ab2278c9 3fea4376aad86630 bfd555b236d68644 3fe859f2d0748167
bfd8308f6ff0ed5f bfd1e81fca328632 4001fef5a022f4bc 3ff98c8d8f98c42a
3ff816791c0485c0 bff1000b91167799 40065fb9988c10fd 4006d55f615c2b2a
3fec39f04b3eea0f 4001ae6cb8ea372d 3fe71003edb9684e 4003687921475b87
400092fe5bc1e267 3fe31681ce19a8b6 4002f949f278fc02 3fea7db997acb7ed
3ff0fbf86493af86 3fe73b272768863b 4000e53f878737bb 3ff4b2ff743c2271
3fda8bd7e61b5b8d 3ff1c87bc1983edf 3fdf4ef4286e913e 3ff0f6ddd15bec83
4002c7b6dd8aaf33 3ff3165d2a9803dc 3ff5014d8e19323d bfd7517db34e6ff4
3f9ecd34fdc28369 bfa3075e6fc711f3 bff3aaf3507590dc 400284ddbb69e504
400041ccb8d3d737 3fdc19955dd2f7a4 3ffa96702bfcd0bb 3fe4734323d655e7
3ff429565c52d22a 3fdb7c3370450a78 3fcea6ae95cd5ade 3ff4c894f3e97d15
40033a66593b7a12 3fcd4368bd1443ad 3ffbc83962f62df6 3fee39eacd1f15f1
bfea4ad3c92f4872 4006046b1e286ef9 3ff9218a6b048ddb 4000e5ce3e7dceb5
3ff0b52d7d346b74 3ffc420ca256d990 bfefd9b0d4f03e69 3fe5167f686cef3e
3ffdb46bb630014a 3fea653a11823f09 bff132cb106a30cb bfdab5c3c2d14910
3f923fa5b4960095 3fbb9b1b82398da9 3ffa3a41a636a8fc 4000e7b61821eb1d
400003ca4edf349b 3ff55e6edd7d6a1e 3feacdb49fd99e4a 3ff1c654f90d25fd
3ffec7c522ff9b8a 3fe536f4b9cf6037 3fe47ab8aba02eb6 3fd252f0a65feadc
bf683b09820f3d03 3ff0b094c8936b0d bfd4ffb0f9abd65a bfe4c1406eef991f
bff522efe5b03ff2 3ffe515edd4355b3 3fb74152f298b843 3ff9131c38058fb6
bff11c74adb3b65b 3ff753e0731df965 3ffb78f20aa45020 3fcd68e04a58faf2
bfd4e824973d4bef 3ff2d64b56ff6ab7 4001aa2160d9afb7 4001401011116d78
40025ebbe8605f22 3fcffe0ffa018d8a 3ff2df42060cbcc7 3fd883922873510d
bff43b07ed9a99e3 bff1a1585722d287 3fe5590d5b242046 3fb23a4f06524abb
3fe9c28b304077e8 3feb59965b7b04f3 3ffddd1ec9bf6f6a 3fed6e93c326eca1
3fd883922873510d bfcc0e06ce17e682 3fedb5e2823b395b 3fe3b18a5c7043e3
3fbeb79ed26c3137 3fe32c25f2a8df03 bfe30f7f08eff47b 3feb40e4b18db036
3ffd8cd926cb2d15 3f8b9f770238b16e 3ff0340091b0ae50 bfd707ca5c989204
4005dfe026ed415d 3ffb0d92bf874ab0 3fe4180f7520437d 3fd8f1a296a580d1
3ff54532807ea0b6 bfe5894282afd178 bfab9cc0fa0d16e3 bf7319b0c958cc80
3fef821ae29a0d9a 3fee6ece904dc587 bfdd03ca1e1da6ee 3ff4287550e6dbf9
3ff2cb66351dd1aa 4007a190b00d5ba4 3feee4b36ee8a458 3fe0b5d88069a3c0
3fd7bb5b0243c0aa bfd67b1244448c14 3ff27959345e43f2 3ff5b7e8451164e9
3fa6ade62717bdf1 3ffb475d924aa002 bfe27ebd7e3950b1 3fdc6ac06fb079f4
3fcff76ba01528fd 3fe0afd9a6ba8ca2 bfe5c135465780b6 3ff0db308616a92d
3fe94ed6c8d1bbb8 bfcad97da2d10b16 3ff5dc4901024237 3fd867d09a1d6c01
3fc2276bce9b01a8 3fd5baf2419d91b2 3ff853af63f001f6 3ff969a7adeba16a
bfd679ca9a06d69e 3fef07396fc3793a 400235ecc4b25e1a 3ffa5d2f8f5091e2
bfd06950350f234c 3ff44fd95d66d070 bfebaabbd291a0b0 3ffb01d40104ed2b
3fe539e36a0e32a4 400003ca4edf349b 3f9ecd34fdc28369 3ff1ccefe428b6de
3ff6b9e78c47cd9f 3fd3afdb6f4175fe 3ff977308f1a2d82 bfe0f1b0d7a7a36f
3ff3b98bfc3586cb 3fdb206635cf34ee 4003ff4bd9eac3d3 3fe01b3cfe558637
3fed9a7ed74df17c 4003327e5ce88772 bfde5700373d9c43 3fd86bf8f46e8d2c
4002611733ace0e5 bfeeaa1a55f46caa 3fe851f6426243e5 bf9f85ccd7024ce9
3ffb15a59cc7d82b 3fc0e6f0943b957a 3feb0c325c5577d2 400238f31ddddb0d
bfcdc317e3ade285 40001a287e532094 3ffd68c0594d4b8c 3ff5bc198acb37ac
3ff80ee4c1d41f63 400104a19180d47e bfe4d6e859545e09 3fe5b8706a0f22a4
bfe0f25f7c11b61f bfd3a35afdfd2bda 3feb6a1dd9c20f66 bfe2c95b331dbc44
3fe72cb371c6c431 3fe94dc4adf7edae 3ff1c9039b0b642b 3ff87accf4c31800
3ff81712bba0b8d0 3fd65260006dbb2d 3ff6ee95bd66ef10 4005ba9acfff1110
bfcb6c9ce7022677 bfc326144ef7bc4d 3ff7afceb2f1600f bfc81d647172b575
3fc31597403ca2d9 3ff5ae2e32412c1c 3ffa859e66b49f4e bfc8b02e21d808fe
3ffc77350dffdc50 bfe7c85759bd4311 400103acc65487f8 3ff72e59e249db77
3fcadec9ae6019b5 3ff816e8709353fc 3ff7511e73dd0847 3fe734e0e05d66e4
3fe5ba3e226b7136 400000a16faf886c bff43b07ed9a99e3 bfed0c1d474430ad
3fd1bbf205e44b73 4002ee01201aaa67 3fec1e1fcfc5db08 bff331b37761fd96
bfcdc317e3ade285 40059bdc06767b50 3fe1dba9935bc96c 3fb74152f298b843
bfe0101880a25a5b bfeadb9553e5ec3c 3ffa3788bb97417c 3fe84660ca67a069
3fe476e1780819e8 4002611733ace0e5 3ff7d03b9b056648 3fb74152f298b843
3ff8aa84e71da5d3 3fecd8abd25545cd bfe4661d7808a40a bfb752bee88fcffb
3fe3a64b4ac00e6d 3fd1bb52c52b2420 3fda61292788b388 bfcfbd2272cbf390
bfb8a5262672a27a 3ffed75b6fc70de3 4002815c1dfd6052 3ff2d44db25a41bd
3ffa278df5d5bab2 40002985a0ef5a6e 3fe4ff65b36c2cf2 4006fb3d5cf284c7
3fec74f62f990bbf 3fc3f9e56402c9c4 3ff65c62ac58e68f bff1ff5c87668821
bfa35a3d0ac2936f 3ff38d9684598aad bfa0fb31a242481b bff5b30482d548f9
3fda698927fc3ae4 3ffe2944266de755 bfdacb7b4f90095a bfe581efa70486a6
3fe72aeb802b8357 3ff3528a6eeb6884 3f9a6496c7f9dfb2 3ff717414401c456
bfe9704cb6cf055e 3fe5130f377e7d63 400824f2dec21ab3 4001d8c8a24e5d2e
bfceaa96ef9594c9 3fed9ee0cafd2e47 3ff18d0b399ae85a 3fb97c8b4389af1f
3ff06939041cd06b bfdd28d87a0da81c 3fe5184c78990f02 3fd6ae6da0accda9
3ff599e31cd723c1 3fe24bf58eeae9cd bfd977b06fbcd796 bfc31a11152744ed
bfdcd9343c7c3d5b bfed8c53cdfc7eb4 bfea2b7b0d6a6cb2 3ffe29de15385178
3ff2da77f275a614 3ff05f510060c279 bfd5ffbe2c4f8aa5 400383c66068d8b5
3ff91112df2ecc14 3fe6a6b746700838 bfebd1ee8ce39d9e 3ff6c12f07500162
3fea7dd77abcfc21 40009e75fcea868d 3ff45e41eaf5f26b 4005ba9acfff1110
bfc9a7e32fca810f 3fa46fd70e46c648 40031251143832b9 3fdf3f2f46215850
3ffbe0fce6349fc1 3fd5c2e01b39bf75 3ffced0a58a7530e 3ffdca43d3654148
40041a1b3cc3badd 40053738a14a5201 3fe335a6849cb7c6 4000c51154b6f238
bfd255d3d806be7e 3fe7c079b3d4f435 40036b40aeff4fe0 3ff290b5a9613780
3ff723419b49ef79 4004eddae48f5b7c bfe316fde75dcdca 3ff25400c2531dfd
bff408f780b94676 3ff6e39a81cdec08 3fc35bc9a33c36d4 3ff89d9d55d21c9a
3fd32108130a5342 3fed7db8ab107c63 40017547251688b0 3fbe7b2252b14cf2
3ff9875789ac63bb 400205ec566a62c1 3fe061923c6b8dc9 bfd4b5254f1c7c97
3ff15df4cba0f223 3ff6e9a803481cf6 3ff6c9d73f3b0243 bfeb147f70d58396
3fdbaff064ec2c22 bfbd073aae097674 3ffd765393306996 3fe5772966d9e1d6
4000dbdb45a15525 40053738a14a5201 bfe8b89f921291d6 4000b84f096a3cd5
bfd998b7c109817f 3fe63cef4d557b1f 40002985a0ef5a6e 3fd1ef94363410e7
3fd3e76061f4ffea bfed0767c1981e74 3ff783657f9e73e5 3fdf4ef4286e913e
bfb51aa2d09dd15e 3ffb0ce53730903c bfeb5f5df66a3872 3fe03d88d9707816
3ff73384517680fb 3ff2de88866b3a16 bfbd07ebb44c527f 3fee72efec271738
4005ba9acfff1110 40019ce1bd015a8e 4003763a76075ee0 3feb3a121c0bbea9
4005ba9acfff1110 3ff751400fc0ec79 bfdb68df9d8af975 4003a4f8cb3491c1
3fd883922873510d 4001e57a20624036 3fe78f8e1855efbb 3ffcf7ffe3e7d5b9
4000f0022079de03 40042b12015b4a9a bfc48fd87507fc99 bfe36d63f292d08a
bfafdc0cbb13ea5f bff27c5e2dd6d22e 3fffc378d66bd22e 3fb23a4f06524abb
3fef2a89194c2f75 3fd377e20ac08af7 3feb471af8747f18 3ff10a0defd7dae5
3ffef27ddd0e8e15 bfc3cfa4f17edede 3ff65316583aa25e 3feef60706c1c9a8
3fd1bb52c52b2420 3ff91112df2ecc14 3ff8b198d6c315fc 3ff008a22e04c772
3fd32108130a5342 3ffabfef0d3754af bfc5dd72db52e11c 3fde4da407469cfc
3ff331ed1e46957e bfde9c37c7edf2b7 3ffe11b52fc7d2a9 3ffaf025d3df91e5
3fedbc3a8139af9c 3ff28de038c7df39 3fd77ec8aa2abf63 3fcad4a364e76f7f
3fff2645ee17705a bfdc211117c23f3c 3fdf205fe798019d bfb3e5ad33f1be52
bfbd604773505a18 4001e9519fb02571 3fd602035fdcde19 3febf8d11439c0af
3fef1c39a0c96142 3ff1d828692c2ddb bfe5ce8aa1921ecb 3ff0e6d35405d9a2
3fd377e20ac08af7 3ff119841e00a9a3 3fe71003edb9684e 3ffa2b62169f4d18
3fd3d971964d765f 4005b42924d8b4df 3ff60f67315ef71d bfe755eb1f0706fe
3ffbfa14e7fc521c 3fda61292788b388 3fe6d88d8902f892 3fd42fac24092e8d
3fba27dd98c43ed1 3fbc28ef4b483d09 3ff26067f5570b20 3fe6b28428963add
3ff4897c41df9ecf 3fda38d857c1632d 3fc9b56c56ccca06 3fe09dd58abd0b22
3ffd7fde6e6d0369 bfe9704cb6cf055e 3ffedb515125209e 3ff3aa389606e38a
3fe69e80f0d68c1d 3fe47ab8aba02eb6 3fe64c51a54879b9 3ff35e4631ef47f0
3fec92ed69035020 bfe426758b5ba26e bfda13356b562b4b bfca01ad543eb988
bfed0c1d474430ad 3ff3a57117343ff5 bfde9c37c7edf2b7 bfcadd328fa14465
40043ebd5ddb1130 3fed7c3c4a8e094c 3fecc933bf2bdbd1 3f68b2d514950731
3ff73d8dabc48157 bfdb448b532d2b70 3ff6dab5a80bacb2 3febf8d11439c0af
3ffa2136bc8aa53b 3ff2cb66351dd1aa bfd72ec99e63a437 3ff822815dd5857f
400151f944c696e0 3fcb9c2cf8fbd7c0 3fd76205de27219d 3fd5298b436392e6
40009e5d42af42fb 3ff977308f1a2d82 3ff8fbbb5ce9b55b 4001b7175b1a04aa
3fe7b08714ca9382 4000b9dbdf3dc8e3 3fa975e2e0d5eb59 3fe2d8a29e58f64d
3ff3d16c6d7a8b79 bfe80dbbf58e517f 40036566a419aada bff4a9b00c9cacac
3fd1bb52c52b2420 3fcfbd9846281be3 3ffbf3249ad46f46 3feeb78907a853fb
3fd81579eaf24992 3ff718211f359254 bfe95f14bab05dfb 3ff79ecaff0d1c91
3ff0f8a0d09bb0f0 3ff083ce44138021 3ff55da10be173b3 3fc98fd0be6fe625
3fe4ecae4d1011d2 3fd22f3b179ee360 3ff29fc6c9a42062 bff0724356dd6aeb
4006fb3d5cf284c7 3ffb08c6ff63963e bfd6f05186e0c2a0 3fe87d289c92bf85
bf94730a2ecc0a6c 400306515aff3652 40040bb5c31209e6 3fe652d5a40ca09d
3ff9d1692f0e2767 3fd392a82d5123a8 bfcea970ac6b5205 3ff2e34692413980
4004cea9f7235770 bfdd86421c0528d4 3fe9bc337764cbd4 3ffc97e7845963a4
bfb0879900bf9aec 3ff050af3beeb0c6 3ff2263e7afd4f66 bfcfbd2272cbf390
3ff1c7240b99407e bfbb134f07851f42 400498500f29f137 3ff84c87ef5ceeb3
4003c1ccda4cae52 4005863a43155b98 3fd4700665ab51c5 4005aa042812d82b
3ff53a094535f13c 3fe7c079b3d4f435 3ff08f78694db091 3feef60706c1c9a8
400470ea6100f89e bfb3bdc9b450cb2a 3ff7b385078de89c bff3aaf3507590dc
3f879bec4f2c9ebe 4005520bb84d1d67 3ff454e16d2489b0 bf918b9adbc9316f
3ffe994e45ea7cc6 3ff7b7d6cbc0a3dd 3fffae5fc18edc10 3ff732b0ca73015d
3ff53a094535f13c 40015a767545de54 bfd4b4b46da1df64 400103acc65487f8
3fd47920d69f0aa2 3fe6f5417b350ea9 bfe4bd433e50eaa1 40052bbbe02e9ad3
3ff3c90979de6375 bfd7970ce605cc31 3ff7a0476ddc6439 bfeeaa1a55f46caa
bfe8505e445458be 3fdfa8719ac7f088 3fea7fa11beb61b7 3fe02acadf1afcf0
3fe9c62984723be9 40002828af253072 3fe04661651c70ec 3feb34b02919db32
3ff1424dac5ea7af 3feffbb0ada09046 400275e7c3c43cf3 40034b190c3f6e42
40017639ee43fc06 3ff03812ac0e2f59 3fb46fd57a0e9b3b 3ff822815dd5857f
3fdab10f14b50a06 3fe5fb1c33272321 400062599f1d7a65 3fa93ea69129754f
3ff06939041cd06b 3ff65469f23bf4e5 3ff1d2a06bcfff61 3fe1afe0b2fe5480
3ffd73ab476d6a0c 3fd9b056ae59b27f 3fc6fd74ba3d4e19 bfd5f4f6a09ee7f9
bfe0081410a8d1f3 3fcc6ee90d954ab0 3ff3219f7879633c bfd1f3f15604fc41
3ff1fb75084639b2 3fdddc1a39589614 3fe56b2abed8e71a 4001c62095b01cc5
bfe12ddf089a4850 bff1ff5c87668821 bfc886183aec17b5 3ffb543533fad77f
bfe2b1c77bd2c7ff 3ff0ab152fb562d3 3ffaba88f97ab244 40029ef64ecb2c44
400475a312f1c2e5 bff3d280a676831f 3ff94f1709b81e7c 3fe6eeb6f9f9ee26
3fe7dc97c236105f 3fe418acdb5af9fe 3ff90b816bfe1bfa 3ff649e141174491
3ff2717aa433bd0a 3fda45b7183068e7 bff19b4bd9f591a1 400041ccb8d3d737
3fd8f7d5a7294e75 3fd4361c0d5eaeb6 3ff55686863aeea7 4002de07b73b43c3
bfbfe212e0fb4dea 3ff789153a9bf833 bfdfdf51a9e92141 3fe4bf07a1897445
3ff4897e99af2c94 3ff3bf84b03fe505 3ff7027ff10a19f3 bfd9b22cb3f4f989
4000864f977e9fce bff5b30482d548f9 3fb23a4f06524abb bfecc88a164b2b11
3fe22619c1092aaa 3ff19a3e4d4334b6 3fe919109ac202e4 3ff3563aeb512f46
3fe88ba4709d42b1 400340c245b1acd0 3ffc567b8efb00ca 3ff274fc66e357d9
bfd4ca24dbfbe288 3fd0ada9859657a7 3fdb48b98bb7c35e 3fef171568ab5bb3
bfd213ace0868419 3fd5d711bac29f0d bfe6ee8cb35cccc1 bfc6b671133c1461
3fe1c7e9bcdb2e67 3ff2cef496f87fea bfc1990f260118f9 3fe8cc53b74f6fd6
3ff6916d54ed9c62 3fd14709967e7a29 bf98e6c431efcbde bfbd073aae097674
3fe01369e95ef1a5 400130eed5f1d607 bfe5d0fbc3376636 40016f0149d8a3da
3ffa96702bfcd0bb 3fec89efe98c85a7 3ff87f759d67d78a 3fea05e86a3ff59d
3fddf6fc7120b69b 4002b94ef3671d29 3ff333762b53202f 40060e8702ffe27b
bfd422819a057958 4001a2f527dcb893 3fa228cb8dd56357 3fea9d6a3b261821
bfd2414b07ebeeec bff580f415f3f58c 3fe5c7af904a13ec bfd07356a75ec3b1
40013b4fe3bc965c bfe36d63f292d08a bfc2b35ffcc8adf5 400152fabd3d8d3b
3fc5f5c05c8c7ebb bff247bd4cc1da06 bfa4388365bb147d 3fff716522de9eb4
bfd70c81feb1703a bfdb45c42082d494 bfe0f88e7c9f5235 3fcca592bb23d242
3fa8e39faa79a053 3fe80fa8f3bf95b7 bfee14076f27aa0d 3ff73756f9779403
3ff56b8f7705c3ff 3ff1bfd8745d0da3 3ff0e696e6a93cb1 bfcec58e12c176a6
3ff4cd8b5c1faef3 3fc4d0f2d9941b8e 3ff4db3325d07e5c 40000f0348a3d2a2
3ffab64b8f01c019 3fbc28ef4b483d09 40048bd21653d7d4 3fd0a9f9625e9e04
4002c7b6dd8aaf33 3ff6c9e323a1a123 400183b6d2969f3e 3ff7b583be2ffe32
3fde4dfa11232c38 3ff1c532738aee2e 3fe6a43bd6605f04 3ff993883da1fe4e
3fea8e9a28805ffe 400246ab2ee3191f 4000e5ce3e7dceb5 3ff7a0476ddc6439
3ff8005dcd5725b1 3ff1cf883d9992fd 3ffbdc1959622936 3fad509cfbfb0ff3
bfdcd9343c7c3d5b 3ff92dc885afc18b 3ff2a03bf8ed9fb4 bfe3c08c9b54db7c
3fee842376b93448 3fd28d29d81f98fa bfb45a745cecbcf8 400218fc8d232315
4006046b1e286ef9 3fd354dc8f7baf46 3fbdabe0f074ef36 3ff06939041cd06b
3fb9f31ed3f56e2d 3feea46128afea5b 3ff7914d554bf35b 3fdba588d18228f8
bfe08ec63129e967 3ffc34e5580a6d5f bfa056d547da1553 3fd1cd0177ed34b1
4005ced6ed92a114 3ff76d077beb31ac 3fe236339f8fbfc3 40021f7de0966db0
bfd4a00c7644fae3 3ff1eb60a094b442 bfed9f4a332e84ab 3f905a56735bcb2e
bfd4fe39e2f4fa48 3fd0d1d24c5048f2 bff27c5e2dd6d22e bfe07c0dbb34dbfe
3fe53ea9c6fdbfb4 4003e9e52c0fb816 3ff30f767caa8ae1 3fee284247e73263
3ffdd0cc0f1fa7e7 3ff63ced7230f6ae bfc4eeca3fc08195 3fe32ba69d4199b6
400041ccb8d3d737 bf918b9adbc9316f 3fe760553ce6a085 3fdb9b5af1e19e51
bf996ce081ddfbfa 3fe42e7e9d164258 bfa8ed12a11fda1c 3fd8204135197939
3fe4f76fe1f0ebed bfd71f05fb8fb596 bf9a947b946f2a0a 3ff55054592ac0f2
bfa8ed12a11fda1c 3ffdb46bb630014a 3fdabc0e3d039c12 bfd927f5e4d887d7
4001c62095b01cc5 3ff65dff8bfef922 4002deaa8aeb385e 3fe51d1b4e753c8f
4001aa2160d9afb7 3ffa3abdfa1d64c8 40009ff45602375a 4000b42dc5401d0b
3f5f1aa202d5768f 3ff79aeb6e31c665 3fdb0a845749345c bff18b352ffcc57f
4003f7dc8c970dda 3fe754316e9b7a28 3ff70649da7fe0c6 3fdf36329014ed56
3fc38fd348bbdb65 400008d52f1b2240 40033840f30f9c1a 3ffda85f7e4b3421
3fe041f81b0f5559 3fe27f1edc01fc04 4000b64651456839 3ff655267408bf3f
bfe7e64ea69414ec bff04ecf552a9f14 3ffe96ba7841dd74 3ff45bd6c5c7b1cb
bfc4883afe9cda7a 3ff5e0d4ac422dcd 40039bd633926a6c 3fddb206192ec61e
3fd600732d4eb786 3ffdea33a996a115 400261713668865a 4007a190b00d5ba4
3fd91f01ffd9acf3 bfeec1d4f04a61ea 4003b8e480bb4699 4000027bf3d1255b
3fe0f83255f72c1d bfe30f7f08eff47b 3ff3a61ac895deff 3fe3ea467acaee4e
bfdd07085f7fe545 bfe9aa4f6bdeb7a9 3ff40e46837c6824 3ffa0793b3a0ca6f
3ff6e15af2642dd7 3fff2bfbbae3f775 3ffc612c53a67711 4005b0bf77f1baa1
3fc2ec30bf6ce52a bff5b30482d548f9 bfd07b5a6ddafff7 3fc82f35e18064cb
3ff6601525f86520 bfe3e258a687144b 3ffd51f1ff3e5c13 bfe7018cc8cc7ef9
bff3f44c988b6fa8 bff29a23b6e682b2 3fe6c4fc3a4042cb 3fda80ce5337a861
bff580f415f3f58c 3ffc567b8efb00ca 3fea05bc73f16723 3ff5cd2c8f7974d4
3fda0b1d969f1bcf 3feccae4b78117d1 400088a2bbf946f9 3fed56b3d39129ef
3feeb78907a853fb 3fe26d750d15b209 bfdc3a0c8b5a60fc 3ffd42fd6c42d7fb
3fc1a6e41024056b 3fb42ebc9177dde7 3ffc2441e701b223 3ff235d2c7265733
3ff1ad2b381042e6 40018540bd183cae bfde1e8f9a20468d 3ff96004ca988185
3ff5addeeac3648d bfba98f5966fa95e bfe55d6457cf6eeb 3ff102e274d5ae65
3ff7abb639c37ff4 3fe1bfde5344b13b 3ff9874d999e0c86 3ffb42a1a54ec845
3feab11496dbf4bf 3feec1df6643e19b 3ff30115884829f8 40031f934238aeac
3fdf4ef4286e913e bfe51e1ffac39026 400089bdd1a84c80 3fed1279062a4ba5
3fee03b3d3a3f8b4 3ff7d5a384f1ab48 3fe5167f686cef3e 400104a19180d47e
bfd959af95781982 3ffb4e311d2ee095 4000c6a54b7cbe68 3ff4260bf07d9365
3fc79429f7d48b79 bfd4444eca7fb259 3fbe2a01cc5458a4 3fd86f208ab8ef11
3ff696145e8ef4c8 3ff2260473e1e461 3fe934fbb176ba8e 3ff7b965fa426df2
3fd1d32e994ff0cf 3ff0e696e6a93cb1 bfd2f5ee92ac495e 3ffe1bf4d035197c
bfc08a3ff67c820c bfeadb9553e5ec3c 3ff4acd7a7bb9dc2 3ff7d4461a8aec03
4006046b1e286ef9 3ff2717aa433bd0a 3ff990dcb9024b95 3ffebc76998550ba
3ff81712bba0b8d0 3fe2fe7a83eeb932 3fd52d87b542b2a5 3fefe885cf03273f
3fec707a4d756902 3fae0f7017771ced 4005466d6edf57c9 3fecbaefb2d4b804
3ff84ac9bf84d721 3fff305a6c06602a 3fe6b4d062ae0879 3fef7b57bb387f83
3fdd9c18ae499bc3 3fe4a8ade646cbac bfd255d3d806be7e 3ff4ba8eb4e3b0eb
3fc2cd5b53fc5473 3fe3012c2aa52abd 3ff24269840104c2 3fd44bebb185f934
3ff49c6ade38b75e 3fd25778786206c7 3fec154a7c6a618d bfe72b72efcd0398
3fd6e9df4bac790d 3ff0535aaedc0d99 3fee39eacd1f15f1 3ff4474aaf52c2c8
3ffb2f74dd767f0f 3ff4b07a96206904 3ff6b84b1ebf5afc 3fd842b51553267c
3fe8fdf23d61f882 3ff6574bd4456840 40068c253c60f042 3ff57bcba937d16f
3fe655db834908cd 3fd39776c75c5c34 3ff35b673a81f82e 400404902edd2a32
4006be43a51ce432 3ff65c62ac58e68f 3fed2bdb1080126c 4004f84fc1006a0a
3fd65eecaf8d9180 3ff40b6549d9e43a bfe506d0b4ed9f75 bff580f415f3f58c
3ff22c854983cea1 3fe80fa8f3bf95b7 3fefaf529f71f7ee 3fdf8e95429fda7f
4003eb4181d6b450 bfeeaa1a55f46caa 3ff49afd1a56023c bfec6892e10b63a5
bfd5e6d78c798158 bfdd30316a8c8761 3ff89d51a5881828 3ff3deb5a7547575
bfe49d0405ed616b 3ff9eaf32ac29f0a 3ff3a8de84240547 bfc6245793c40ec3
3febf4236693276e 3ff1ef361d4e3148 3ffd68c0594d4b8c 3fca8dea7aa182a3
3ffb39d5056852c1 3ffa1156efd9f960 3ff044a1cd6f50e5 3fc8719027de01cc
3feaadfdf63d839d bfd4e65f7f8578a9 bfc8c77199ea93fa 4001d86029a7295f
4004ddb373949803 3ff3fbbe7e921bd1 bfbd8791b96d3ca9 bfee598a148e9f78
4005a24db85cf181 3fd0add720c05b51 bfced579a2ed0809 3ffd9cbe9d1f3da5
3fdf79229e3bc1f4 3fd61342135de4c5 bfe0f88e7c9f5235 3ff629f3e22fba00
3ff2cb66351dd1aa 3fd1bb52c52b2420 3fef3fe560a530d9 3fff58f1bc8daebe
3ff4b2664d1370c0 bfbd073aae097674 3fe2ed0e0ef9d0ea 3fe31ce9e88b7663
bfeec6fac3d017cf 3fc5ad24a1ebc751 3ff5708efdcf634c 3fe80fa8f3bf95b7
3ff4165bf4b911ac 3fdf74bc9ab0dba7 bfbfc33cacd02cbe bf9d2a78b01bafbb
3ffa103aed9d565e 3ff9828447ebd71d 3fbc76cff7e8ff22 4007c3c044c63717
3ff06cfc30aead33 3ff8ff17929a69eb 3ff078785f733076 3fefc19ca51c9ef7
3ffc7c6bb76b1875 3fd4ce265df70624 3fd7d2b21df1e732 3fb25b379663510e
bfa35a3d0ac2936f 3fe359ca3964991e 40015a767545de54 3ffafb8dbf3f2624
40058108ef73afea 3ffb62302f89770b bfd6b1d3ea159843 bfc990ef22d69891
3fc8fb6c7d17ed48 3ff5c6c1133688bd 3ff02b497c0a8b0c 3fb23a4f06524abb
3ffe3ec414a6f59a bfe506d0b4ed9f75 3ffe3ed83bb18cbc 4003ab92df170ba5
3ffaedbcda073cd6 3fe259a241b6cb06 3fe709ae60397acf 3fda74ea3276d577
3fd1ef94363410e7 3fed9176c0d64d48 bfed0c1d474430ad 3fd1bb52c52b2420
3ff5fa52b83e89b4 3ff88451f5c23f1c bff5b30482d548f9 bfec88c1a7b40d46
4005c3dc03ca7234 bfb0879900bf9aec 3fd6f0e7633f49eb 4001e51969fcf4f6
40029813367087bf 4000154f1e7ce64c bff268134a052f45 400365448f1485a7
bfe4924cd65f3766 3fd38c18f12326d7 3ff7514654b4941b 3feb7f2797648182
3fe104eb8fb7488a 3fb9f5101c03804d 3ffbc31d012aad60 3ff737c173bf0bd6
bfdef7051c5b5f18 3fc223e87c9aebe4 3fdef2ec487226e1 bfd739c6f2e4d633
bfe4dcd73c03f467 3ff822815dd5857f 3faff7fc960c5185 bfe12ddf089a4850
bfe3d0642c6ffcc4 3fe438e68be1d1e3 3fd1ef94363410e7 3ffd87ee40408f25
3fe600bb1f756d8a 3ff1543771025702 3faa71782c8aa763 3fef59f2de04fe73
3fff8f910c71f523 3fd644049ebe1d6b 3fde0144e44b4d97 3ffcf7ffe3e7d5b9
4003763a76075ee0 bfec3370e863edec 40059bdc06767b50 3ff4b2ff743c2271
bf94730a2ecc0a6c bfe8aba2bb9155e3 3fee8f4c644d75d2 3ff2911a298a9395
3fe8a03022f784d9 400176f6fce49a63 3ff66a454d2bf3b4 4001b29366bf6f47
3fd9ffd3e6296464 3ffe449dcc67092b 3ff4b2ff743c2271 3fb2b8870a7caeb6
3ff49c6ade38b75e 3f9fe44a405a24e9 3ff8de587869c71c 3fdeaf04e986c273
3fe57c7e0872a866 3ffd8461d7259adc 3fd50a4668dfbcfc 3fd5b3a2850cca6e
4005b42924d8b4df 3fd883922873510d 3ffbe5ef8119663c 400440648c836d87
40054b36e742ff68 3fedfc1eb38f4766 3fc886f512580560 4002bea00654f10b
3ffa8df8976ea4c3 3ff88f3935bfc29f bfe73d5a125e87f4 bfe5dbdcc14698aa
3feb23cd2cbbb6a9 3ff18d86331d292d 400092fe5bc1e267 3ffc001c2ec1f3f7
3fe9f50926ad7317 3ff9cd39635f74ff 400282a58dcc43c7 3ff47ea50851e6b3
bfd69cc25ef82d7e 3ff19ebc4252023f 4005d56de701dc76 3fe66828ec9ee478
3fd74bfe2da4f258 3fdafc8757647861 3ffec7c522ff9b8a bfc03287b2c55432
bfcd660b4dcbbeca 3fd3bfa2a7e34710 3fdbdf9d81ae5629 bfdd4d00eab293d0
bfc3987e28e2da1b 3ff9552dec3ade61 3fef48b592aa146b 3ffa737192aae2bc
bfd2c77556648205 3fe89fea7c2e591f 3ff174a1623d3380 bff580f415f3f58c
3ff2d44db25a41bd bfdde0c232782cba bfc0610802ee2e6d 40010bbff5a689ad
3ff5abb70ec5674c 3ff132fd3cfe6f46 bf9c4bf1202d0d87 bf79449452f0f0ad
bff03dc8922d9abb 3fe747bddebdac95 40020215b2ea3a1e 3ff06939041cd06b
3ffd58a7054f0f04 3ffa0a77a4f5ae38 bfe7155a02eecc82 3fe212839a45e592
4001ff458f3983c4 3ff3c06286461086 bfc1990f260118f9 3fa8968087d9e345
bfdfdf51a9e92141 3fe009d772b7c527 3ffba3e2c97c3b2c 3fb19978d75d8f0b
3fe14ee4119cb290 3ffc74b3ddfc67ea bfe929e5b760829e 3fa6bf2217d108a8
400088cb92b6f0bf 3fe44c1a967936de bfd2efdc4e0a6910 3ff57dd0abf63303
3ff7b57e8d47a6cc 40000d27bd3dca94 bfde4cafe346b505 bfab05c88e3ad535
bfe56b10bf73a27e 40044c62064b74a9 3ffcb08d6946ef24 3fe21e9eddaba2fc
4003c1e574e8abfd 3ffb42a1a54ec845 3ff747fe93cb24d7 bfc1920a77bb0c0e
3feb5a7893261fc5 3ffe2622c6caa07f 3ff8f85413643cd8 bff2d6a090bdd101
3ff38bc75a40a8ad 3fe7c4afcf97fc2c 3feb0d056395b6a9 3fd3473809fc42e2
4000cb6e98c3b110 bfe42d4590e7e270 3fd5d711bac29f0d 3fd963f57070d995
3ff44b77be9c6a5a 3ff732b0ca73015d bfd55c5830d68dd8 3fffdcf1fd8473b2
3fe5b8706a0f22a4 3ff1e9ccbee7185e 3fe9a9e8b89d7c93 3fecb17871014713
3fe7a18f2b3d1927 3f5a09f973d04467 40009fb22b88360b bfe1b69c235763b3
3ff3a04e7fb8dedc 3ff781ea6618902d 3fe197543cd39e50 3fefad09f3fa7d90
3fe29b8b9af87f4d 3ff076214ce8a631 3ff81ed07eb17c1f 3fea7db997acb7ed
3ffe02111e1e34e3 3fe89fea7c2e591f bfdfb58fe94ec87e bfcc0c626defd8b3
3ff306979a48bacc 3fcff76ba01528fd 3ff4b2ff743c2271 3fef21912dfde0ba
4005461ba97c29ed 3fd76205de27219d bfe86a4811dbd0ab 3ffb6a0e239c5a41
bfd886585cadf81c 3feccae3f1723644 bfdba62e14383180 3ffcc91454fd3f83
bfeec1d4f04a61ea 400496fe22c7dfa1 3fe25dd2c4505541 bfe5d8f05c0770d7
3fe46e4cd192a8bc 3fe32a9e1846a42f 3fc1b595c43fd0b5 bfb3760fac011be2
3ff7fa56fffefa89 3fe53ea9c6fdbfb4 40009fb22b88360b 400062599f1d7a65
bfe25191c396edef 3fdb3fb4f626fb6a 3ff34a4e1cf53f3f bff2d6a090bdd101
bfc33097160e3ec3 bff1ff5c87668821 3ffac00a7d7e4b8b bfb717f394ddf3cf
3ffd7a587cd79a80 3ff724d8f57d7948 3fe2fe7a83eeb932 3ff5e9c0ef74a8b4
40058108ef73afea 3fda8bd7e61b5b8d 3fdd1e322b1e23c1 3fd4361c0d5eaeb6
3fff2645ee17705a 4000f82b2d7b0c40 3fe183ca4cdb87cc 40024fc53ef36e86
4000d6aac7a894f7 40010fe641198059 3fdcab5d33555bd9 40058778b04a9164
3ff19fb95f4f0d4e bff2f5a28c6e1958 bfb45a745cecbcf8 3ff73d8dabc48157
3fde2afe78202215 bfd8d9e2c85eac00 bfdbad139b378f1f 3ff2e41a9617da3f
3fd77ec8aa2abf63 3fc058ce1b6bf638 3ff048883c38ae88 3fef0da86015d12c
3ff1caeaa25b4487 4001870fdd62e35e 3ff2d3590cc8325d bf9164838dda7ba2
bfde5700373d9c43 3fff58f1bc8daebe 3fef368abb826017 bf72cc056869db79
bfdf2cc4289197bc 4000c6beab2bae7b 3fffbb1f30fac7b6 3fd5340bd5d4f130
3ff0ab152fb562d3 bfa58b2f7309393e 3ff6801ad73316e6 3fba076908988a5b
3fe6d88d8902f892 3fb9af0a3393da43 400195698e64217b 3fe74a74be347850
3fd9be328dd36fcd 3fd5186aef5abd75 40009e75fcea868d bfccead351192d5e
3fd99510acb2f1ae bfe431ffe7db8ef2 3fe9366d5f7aaf10 3ffa09974c19fcaa
3ff2e5c58a33528b 3ff7e6ea377625fe 3fcce48209086f33 3ff2d876b765fb1c
3ffbbffb3258fa92 3fc5dec2ffd34d82 bfeb9919cb6e29e0 3ff0b52d7d346b74
3fdebe8703ab4d27 3fe6fdd787940a07 3fef7d5a909a7711 3ff25adf6aba3b07
bf9d2a78b01bafbb 3fffc01e809487a9 3ff849ae9f4868d8 bfdbad139b378f1f
bfbaf37b1bcfddf7 bfe6d1d4f81a5fc7 3ff50e55bc9a08ef 3fe616de60520a7d
3ff1feeb428da81e bfc35050d993ddaf 3fa8e39faa79a053 bfa27bbc510fbe5c
3fddf527ade9a681 3fef24092fdd1384 3fdaea9d395aba63 3fdd4baa1d33313a
3fe3ceda70d7a422 bfcaf5f464e79a90 3ff0cf4c6ce44525 3ffee3a93538525d
3fddafc3abfc0304 3feeb0f543cdf568 3fdd853f6b947fd0 bfc83dcf6374c27e
bfd2c3d237d912a3 40046eb525973731 3fe894d778226a90 3fbd55932327d41d
3fe527050269149f bfdf2cc4289197bc 3fec44763d46800d 3fdd3cb85a444d6e
bfe0770a28ff11db 3fede631c91d9546 3ff194477cb1aad9 3fd600732d4eb786
3fdbaff064ec2c22 3ff4cdf00f894a7e bfce8151f61dae66 3fffbb1f30fac7b6
40037cb4fee3e08e 3fbda88ce2a8f313 bfe25191c396edef 3ff459e95aa7d0fa
3ffd06a3ca84ea2b 3fc4611a13464199 3ff53a094535f13c 3ff07bacff8b1eb9
400287bbde78d05a 3ff92dc885afc18b 3fddda44dcc13868 bfd34f24034400e1
3fe7e6e20ab16e0a 3ff824e67700247a 3fb1c9128a0d1af1 3fb74152f298b843
3fd3e2c0eb5f60c7 3fd9971da7851266 40002353385401fc bfe7db12ab50e3e5
3ffe449dcc67092b bfdda18c2a1bc750 3fbab16cf3c791f7 3fb90f0b63ab6aa9
bfe09f92b8371ded 3ffe0bf2991e1da2 bfd07b5a6ddafff7 3fe2255f88eee5b2
3fc3b7c8355e25bc 3fdd414ac8580cbd bff5b30482d548f9 3fc90b0a417cb399
3fc9036ff27d1572 3fbe84d9e1653c45 3fc86986bbd63189 bfeec6fac3d017cf
3fe4d9fb2ea2cc05 bfdf0d5e93f539f2 3ff3cf539119dd45 3fd258379b794401
3ff9247065c038ff 4007c3c044c63717 3fd5298b436392e6 400147454c67ded4
bfe0b64f5d552dda bfe4e3107dade9fd 3fe139bc6ead80c8 3feaf7b8467e364c
3ffa103aed9d565e bfad632d761cb5b2 3ffd94764415e466 3fffab9280b8ea47
3ff9d46948ac5151 4001cc3daa5afa3f 3ff44575c5d350a4 3ffa06b24315f567
3fdfd7675c2fe7eb bfbb5540ddea9b86 3ff751b605bcdb96 3ff0d90066966b91
3fec44763d46800d bfd977b06fbcd796 3ff08f735e0fb251 bfe9704cb6cf055e
4004f44c8fb5b7ad 40059bdc06767b50 3fedcc517ff0b1d4 3ffc937ed0dbbeda
3ffd7c95eda7d9d7 4005ba9acfff1110 3fec62dcf5fea00b bfd5b1573e8f6212
3fbdc7d196982bcd 40019bba7a21dbba 3ff9b3e35929db9b bfc2637fbf022efb
3fe37340304033bc 3fdf1e30a428cfa8 bfea183258d2ca12 3ff3581e6bb47a78
3fe5a4a49be51891 4001efd8c816e1f8 bfbb134f07851f42 3fdecbcc4247c404
3fee5080b65f5a34 3ffbd270abbe8664 4000904da14df77a 3fedfc1eb38f4766
3ff1632dd888feb9 3ff247519ed5ff6e 3fdbaff064ec2c22 3fdebb05d73928af
bfec40e9e75dd22d 3feb39f5a5dbe96e 3fccd89656b5caf6 3ff9247065c038ff
bfe4c1406eef991f bfc22ecf7770ab55 3fddbe51a767ff92 bfeec5d97b550439
4000904da14df77a 3feefe927d37fb35 bfc36818bded6e8c 3ff11b5348198ba3
3ffe1e48848a4c64 3fffe607f63b4f1d 3fd0ee896c42fa87 3fe80389eed03c5d
3fca707498630c7d 40010aed1acc7d89 4004139bf4132092 3fdfeb9563081ac6
bfdd4d00eab293d0 3fe8c258cd9ca19e 3ff08f78694db091 3ff0565baa3bcd47
bfe136f21209181a bf9c61b2332d3cce 4000c50eb5e29271 40022f8d0436ee12
3fd09f28b172bd03 bfd47d3e9bfd6c6f bfe31e58b8b70a98 400161c360e8456d
3ffed190fc5e6dfa bff1ff5c87668821 3fc4d0f2d9941b8e 3ffa1990694be66d
3fd51c26229b3078 3ff2cc75655d884f bfa647dc386242ed 3fe898aaf23426ce
3fd615256f956699 3feb19f8f96f4af6 3fcf4844612f4b39 3ff33153c71d92b9
3ff459074323b6e4 3ffae5a210ba26d3 3ff7c57b974834a2 3fea3b43b97d124b
3ff33138c0952970 3fdd21538847319c 3ff6a0ab7e134a43 3fe88ba4709d42b1
bfe2c0d96079ef92 3febf8d11439c0af 3fe8bd7cf902757a bff2978558c9eea8
4002b51a60156abb 3ffba3e2c97c3b2c 3ff0564a87e7c130 3ff9552dec3ade61
400282a58dcc43c7 bfe20bd89a885b52 3feb7382851d6cbd bfae501f12d00e45
3fc3c4677d29a201 bff2d6a090bdd101 400498500f29f137 3fe5ba3e226b7136
3ff36c673f1c026d 3fe9e710f17cb665 3fddad9946ea29cc 3ff545653836afd5
bfeabb3224c2a0f1 4006815ba1b6d1d7 bfc68463d41a97a0 400212a620261237
3fda7ff9139bd628 3ffb4e311d2ee095 3ff83fa603caf8fe bfe46b848aee50d2
3ffb0aa474ff99b3 4000ef0d554d917e 3ff82ec0e83c6502 bf97c7bbf3e9d0e3
3fb53514e3c25b11 bfdcbe5b56ca03e0 3fd79a41a952c63a bff2889db0b38a2a
4001e0ea739b5907 3ff37f6d6e2d1e17 3ff24884978fdac1 bfe3bb85932c4bfd
3ff727cfb7e0e11d 3fd84996965697b9 3fe4a7e37aca54ea 3ff719b1e53f5133
3fe85b4cd35f8b8e 3fefc8cb95c8e82b 3fc0a116bb3f83c4 bfd932d9a3c9c9d4
bf9e05d3b8b9cb65 4004b3ee171d7707 bfe36e490d3e868a 3fee6cbf61c9c46d
3ff910ba32c1bd20 3fecb16ab67c41ca bfdab062aabe4964 bfa9ff6b54579df7
4004452f09016d61 3ffe389c9f359be0 40040bb5c31209e6 4000278dd2075513
4007cefd04be2567 3fbe32370a51542d 3fe5c50d23b9d9b9 40047d7cf82725d1
3ff62aa4dbdbe88d 3fea9f05e8d640e0 3fe104eb8fb7488a 3ff6ee95bd66ef10
400135485f25e53a bfcf2b909992b3fd 3fd9ffd3e6296464 400622cc3881dbb1
bfe21f6b0a1034bc 400622cc3881dbb1 3fd9b1bc8af5662d 3ff7459ea0d33f90
3fbb8562791a437f 3fe6c7e30565ad0d 3ff2260473e1e461 3fe1439fc8e92466
3fc4cb2c2e8d2ff8 3fd1423b21e5bb3c 3ff078785f733076 bfc8c77199ea93fa
3ffd25e8a4331610 bfe0792aa158cba9 3ff81712bba0b8d0 3fddb206192ec61e
3ff2025ba2015241 40053738a14a5201 3fd7534b41eea428 3ff47e69423155cb
3ffe3af6928140ac 3ff0e59cd1c21c53 bfdcdf4cdcde5b84 bfc2fe08195ff4b2
3ffe4f223780df80 bfdd7e340ca21b96 bff07fdcdd51043a 3fc8ebe640254a2f
3ff3a57117343ff5 40009152029e8ae2 bfeadbeaaad7d13f 3fe266a255cb1058
3febf8d11439c0af 4006b32fcca34fb7 3ff4459318194816 3ff2d876b765fb1c
3fe663b8ae3ac047 3ff10a0defd7dae5 3fcff57f5ff46031 40044e7fc100934e
3ff14a95f4372be8 3fe572273858ef3a 3fdb3fb4f626fb6a 3ff126087ee2122b
3ff2cb66351dd1aa 3ffca67de8f4e3ed bff1ff5c87668821 400371625bff20ec
3ff32c436a2ea82b 3ff906a5021605cf 40008307ac4e1d1e 3fe3c2324fcbbefa
40000054c85feaed bff1486c5671c97e 3ffc6e27b3010636 bf9642e91128149d
3ff1cd48a2861364 4002c7b6dd8aaf33 bfe4d30daed02e6d bfbd4546195c90c5
4006c99498852041 3ff73593a0be98ae 4001eec2f972ba97 bfd5958dae723070
3ffa47c4e3dba304 bfed175168a67279 bfe87d2d71213812 3ff8753d3e8800f9
bfd99133160bf15b 3fdc38e0ae68fa36 3ff1895b1d0c5725 3fd254d859627ead
bfb26f16fb43923c 3ffe519afe3ab048 40020e131998bddd 3fcd7aa263e472fc
3ff8eeb5d9a57bfd bff4a9b00c9cacac 3ff92dc885afc18b 3fecfbf5c0d33dfc
3ff74f33793529c6 3fe87417b5e1464a 3fe80fa8f3bf95b7 3ff2d44db25a41bd
3ffd9b6609de218e 3fc32f9db9461bdc bfdf63d568a49a44 3ff2723578e17379
3ff724d8f57d7948 bfc28bbd61d10673 3fd7724f61b06551 3feefe927d37fb35
3fe19bf4ca1fc39c 3fffa92b204c02be 3fdae647f20c25b8 400198cd93d9872c
3ff3f5f5dff8ef17 3ff5c2edcf117a1b 3fb994ad4194b2c3 3fca28e234327360
3ff1150d7ca16c1b 3fc41ee33f573551 3fbab16cf3c791f7 bfd72ec99e63a437
4004c2b97ac76ade 3feb1d2fd0d75210 3fe9c884534f6fc9 3fddafc3abfc0304
3ffae5eece76af33 3fe6b7763c66c87e bfd07027af7ff75e bfe431ffe7db8ef2
bfe84ccfc496563c 3ff66d73056516d7 3fe99b61d78f5266 3fe79c47fd0fb256
3fee3099f55b1bb4 3f94c8ebe9113706 3ffcf1e9600228f2 3ff4b2664d1370c0
bfc61f17d9439c42 bff5b30482d548f9 3ffeae5eda61188d 3ffceffdffc6c89f
bf985b8fe3ded5d2 3f9fe44a405a24e9 bfa8170ad94913b0 3fef5df13ce8da80
3ff7576fb2a6659f 3ffd82294ca28074 3fdc0bc1f6595639 4003a89f6cb1dc26
bfa61f56f3ea195b 3fc898211c8e2f09 3ffd109bd5a6ebfb 3fdf98209254be2b
3fefc2139143bdb1 bfe6d78c9568a836 3fe09bf2242e31ef bff0973266b8a395
bfbe9103f4cf6816 3ffa16463288af93 4002a1e5fdffba7b 3fe7dcf31c059513
bfe58b36dbabdcd2 4001bf8deda889fc 3fb74152f298b843 3ff6c12f07500162
400371625bff20ec 400631c20751a5e7 3fd1fa168c918463 3fe19bd92c3082b4
3ff31caf8aff054f 3febcb401ed99181 3feaef299843478a 3ff5bf4ea14b1a8b
3fe4a8ade646cbac 3ffc555f2410bc19 3fff0a17d060ae77 40017770b6269053
3fe4ad481cb3b391 3ff697c1d884e737 40037cb4fee3e08e bfeec5d97b550439
bfd008115a080e40 3ffc37b2f201c013 bfe3a291a8938edd bfe1aa986025cc02
3fd4b44683de170d 3fe7e6e20ab16e0a 3fd117001c13a715 3ffed4f2d777a6f5
3fc46b3b64644bfd 40008d686799a786 3ffd9b6609de218e 3fc41ee33f573551
bfe1b69c235763b3 3fd206f5dc032fcb 3ff08f78694db091 3fe7b8cf6d0aced3
bfdbc4edfea710be bfe80044bbd3af05 3ff657ed59fb298f bfe1aa986025cc02
bfe7155a02eecc82 bfed0c1d474430ad bfcf5e61781beb85 3ff81a026940ef2a
3fee21398a8bb569 3ff5499d34a8ab68 40013c4256b4e1b9 40029af20e83db06
400244e3a8562ee5 3ff5ae2e32412c1c 3ffb51f13d02e59b 3ffba3e2c97c3b2c
3ff47e27c54f045f bfe076dacb3d0c7f bfbb8be3dbc860a3 3fe279976bb045fd
bfd395aa31a85f80 3fae0f7017771ced 3feaddf85bf479ec bfd9b22cb3f4f989
3ff5426d09ef99d7 bfef8a9628ee59e3 3fddd93b5d0b5976 3fd80681d102652e
3febfc7a8bd526e4 4003b78107219a93 bf9a81a24031fc18 3ff8a3ec36427aa0
3fef86410f82d807 bfed79c449682677 3ff24269840104c2 3fc341b92954e686
3fd13d9f48d6f75a 3fe0b4dcf652daa6 3ff05c6746d0d415 4000d08aea5f7e16
400498500f29f137 3fe5c6bf61e46965 3ffd46afdd6e436c bfa01c778db50182
3ff549acee2887b0 3fe0dd93809c2d35 3fdabc0e3d039c12 40033079a4b1aa9e
3ff15e1fcd70ad14 3facbf0f9d87daa8 bfee251a746520c2 bfcd1f9844f02bc1
3ffcf7ffe3e7d5b9 3fe6f5417b350ea9 3fe975f1f0888cd2 3fd87435ded69d7a
3ffcc50b8ba3980d 400824f2dec21ab3 bfe974a7f299fd0b 3fffd2482ab2036d
3ff272fef8994233 3febd9487f3d3eb2 bfe457e0e8ede518 3ffd53ded74d34f0
3fe44c1a967936de 3fcf59e08ae2e0ae bff0899b37482863 3ff28f22db60e6c5
bfeba540fb8c2760 3ff78fd9960853dc bfc44340ddaadbd4 3fb2b8870a7caeb6
3ff7c10804b442c4 3ff681c8858d8a36 3fe7774155b03d0b 3ff9552dec3ade61
4002642cd114004a 3fd21a2de8cf4acb 3fd13c257ca39e00 3ff60f67315ef71d
3ff45bd6c5c7b1cb 4006c99498852041 3ff45c72c0dd106b 3ffab83609d95c1f
3fe0c2e26a218f1b 4000fc584ac86bf1 bff0234f44bae28c bfd932d9a3c9c9d4
4002e8ae0ebf2989 3feef09dee42152e 3fe3b57c41691ddd 3fe4f76fe1f0ebed
3ff1c692b6f3d35b 3ff454e16d2489b0 4002a141483679fc 3ffc9601753b028e
3f9e385b0990e6d1 bfdee35ce2448121 3fee130197ad52ff 3fefc8cb95c8e82b
3ff2a4b87131f1ef 3fec10b574873750 3ff845a71eca122a 3fd033ba709e8321
bfe0101880a25a5b bfc24897b854b0d4 3fe0a6c2193e410f 40040bb5c31209e6
bfddd8b9696929f7 3fe6a031a623e7f8 3ff39275bcaebe7a 3ffe542d175789c4
3fe8cc53b74f6fd6 40029ef64ecb2c44 3fef1411a7db0df2 3fd6e6ae7270c6df
bff43b07ed9a99e3 40014e6595430d6e bfcf5a8e10793c56 bfe09c0eeeb088a2
bf725d3a8b229f02 bfe7ec6be66a9448 3ff25400c2531dfd 40033079a4b1aa9e
3fffcaba5728b6f1 3fdf065fde691f00 bfdc9ccf45758223 3ff10ed9c1f2c100
bfe11c4506619fa8 3fe2435f8c014c8a 3ffee3720256dc37 3fe7c4afcf97fc2c
bfe91e0b43cb9cfb bfedadf23c68f7d9 3fee39eacd1f15f1 bfe9e55453dfb9c3
bfd19a6094cb5ea5 4000cbb02bb8ac0e bfb6b51bf63be02b 3ffe7a9dbf5142a8
3ff0cfebc79609c8 3ffb4e311d2ee095 3fcffd4c1829fd5c 3fedfbc002074086
3ffd71df5e25e6a4 3ff98c8d8f98c42a 3fe2fb16563b96c3 3fc2c9ce0694050d
4004f44c8fb5b7ad 4006378b538d73ee 3ff898270880ee21 bfe0770a28ff11db
3ff306e9389ec50e bff2658060425452 bfbfe212e0fb4dea bfc069c0727d53cf
3fbda88ce2a8f313 bfd746c90ae4b370 3fe131784a2aac49 bfe7df12542afb1c
3fdd00c91b8893d9 bfe91e0b43cb9cfb 3fe587e9158d0a4f 3fff2645ee17705a
3ff5b7e8451164e9 3ff6a0ea24839511 bfcb3502268390f1 3feb43a1177e4676
3ff44b77be9c6a5a 3ff3a64301c8095e 3fd5d97754d3b6c2 3fe3a4d20d6971a1
3ff5c823855b5fbb 3fe60a92c271949b 4004d17ffb20441b 3fbd3aa8e608d7b5
3fff346436e3f6dd 3fe8692f3b11b9b0 3ff23d22fa01b32f 4000ffb9e0185a69
3feecf48c9c1d610 3ff4578a56d64c23 3fe010ffbf523a2d 3fceb3040f15deeb
3ffaba88f97ab244 3fe495c4b991ee9d 3fa6f45a38c6ca1f bfe402d7aeb72b48
3ff5627eb1771b06 3fe65ff845b283a7 bfa8ed12a11fda1c 40014d5b068c47eb
3ffe1841545e3cfc 3ff6601525f86520 3ff748b2b8b2de04 3fdd414ac8580cbd
3ffe7a9dbf5142a8 3feb8a5a17f5332a 3fdccae51ec51eca 40058108ef73afea
40001847c0757363 3fe6f5417b350ea9 bfdd07085f7fe545 3ff6b64a5271e1ca
3fee5080b65f5a34 3ff89c1a81e11c8e 3ffb62e0758b8a29 3fe27b87af5c2911
bfe1aa986025cc02 3ff8a67376d3fc9b 4000fb42994e6092 3ff32be48a3d71bb
bfe81b5214cca7b6 3fcff57f5ff46031 3fcadec9ae6019b5 3ff1bae009e9b26b
bfe0101880a25a5b 3ff35c7271fe8d90 bfebcb3c36b994b7 3ff4bdb01425aba5
3ff4d26846eadbb1 3ff60f67315ef71d 3fcaf945b088415f 3feffd17d164307f
3febe50db52f3162 3fee03b3d3a3f8b4 3ff6916d54ed9c62 3ff45a6a71deed9c
3ff8e3ec37815d86 3ff323728dd17a0f 3fd5d711bac29f0d 3fe9511386519c11
3ff007ba28d0d190 400383c66068d8b5 3fe4381f4be56f65 3fee5080b65f5a34
3fd1217da13b937c 400631c20751a5e7 4005713d13441a8b 4003ae0c3e6daff1
3fe0840810436778 bfc580dbeb71d386 3ffcdabe4238ca97 bfb96c8855f338b3
3fddd8f8a60f5dfb 3fdb752ce598bb09 3fdbfafa4180cb62 bfd959af95781982
3ffa0a264fd562a2 40022e8999316464 3ffb267521a69c6e bfc49bdc52071f3c
3fddb206192ec61e 400357e9bfc707a2 3ff7d6bf79efc1d9 3fd7534b41eea428
3fe1809d2303110b 4001ed5343dad88a 3ffd83a918df92a2 3ffa7ee213f776aa
3fe5772966d9e1d6 3ff53a094535f13c 3ff0df267021d20f bfa20e5eeaf56bc5
3ff2653135a678e8 3ff642b82e680623 4007e9062f93cfe5 3ff17a8f466d4377
3ffa93886e751cae 3fe5b0b521bcd016 3fe0b2a9feae60ef bff1ff5c87668821
3fd2ea03e788e7f0 3ff1c532738aee2e 3fbafb9d186e7fd9 40020215b2ea3a1e
3ff594febe76aaf2 3fd68e64967666bf 3ff2c9ed1b952a6a 4002104a869961f0
3ffae3802cea50bf 3fe4604c4fa72cfb 3ffea764e3df2927 bfe2e75a0cc101e1
3ff3a452832185cb 3fd963f57070d995 40020ae807994add 3fd4361c0d5eaeb6
3fd0b743f5979019 3ff7a0476ddc6439 bff0735ee70542f0 3ffa859e66b49f4e
3ffa16463288af93 bfebaf504fc8d6c8 3ffb0a8b8a88370d 3ff9218a6b048ddb
3feb51868344e80d 3feea46f9761aee7 3fe80fa8f3bf95b7 bfe03dd4d04e5340
bfec3370e863edec bfd3b8d3a9ab7e78 3ff448d9f24a03d2 3ff649e141174491
3ff476e72c1cf9fe 3ffb4e311d2ee095 3fedbc3a8139af9c 3ff7b385078de89c
bfe36d63f292d08a 3ff3ba44422d19e2 3fe8ba6cf7681547 3fe2ca435165e650
3ff5ed61f78ab998 3ff2cb66351dd1aa 400365448f1485a7 3ffad39cf12e90d3
bfd338d335800b1d 3ffae7459be11f3b 3f7d63e4abd69c34 400054a5aec128a4
3feacfa9479d7956 3fdb7b3df27aee14 3fe0840810436778 3ff454e16d2489b0
3ff7cb7f02f0a456 bfd2719983d2a86b bfedadf23c68f7d9 3fd382d51bcd9613
3ff74a0b2a56e78a 3fc0a116bb3f83c4 bfcbb4e5be797bae bfd338d335800b1d
3ff80897fd8c440d 40000d27bd3dca94 bfeb55085fae6c0e 3ff098638188c2ca
3fb4a29c7f89993e 3fe4d9fb2ea2cc05 bfe36d4b8c8dde06 4003c1e574e8abfd
3fd93303f5ee304c bfd1528749a81b9d 3ff7eb64833d0785 3fe2435f8c014c8a
3fc1970998105b91 bf7e8424b7498634 3ff04d8a433336be 3ff3a8de84240547
3fdff808793d77b4 3ff02b497c0a8b0c 3ff4cdb911e6ddc4 3ff5325125cfc6b5
bfdaa36ae939c92a 3ffec52bb3394b86 bfaf65684844882b bfa947ebddde8c86
3fc65d1bce486123 3fe247e00408dbd4 bfdda18c2a1bc750 3fe02acadf1afcf0
3ff47139e5a06721 400033c42baf96bb 3fe3dde070a5f651 3ff1c8d09d4fcc83
bfe9d6065f6b5534 3fb324e944f0c1fd bfe0d3f126ff5c48 3fb248dc8949e19d
3ff0be334f578c36 3ff48074326ac8b3 bf9dc233312710c7 bfd445f3febb1c39
400329c9253ddf88 3fd9382e14a68af8 bfe47899ddafbc11 3fff9b31d2133d25
3f8a840e63a8f132 3fd5baf2419d91b2 3fd7d2b21df1e732 bfe99ac49b5720ac
3ff7d345d25ddbf2 3fef8b84d7f10791 bfe73d5a125e87f4 3ff59644acdd6998
40010760b40380bc 3fc08e3604f7aaeb 3febdf663dab5b3c 3fd4005c3b255bf4
3fffc01e809487a9 bfbc2ca7b2e5a295 3fd15d219ff015cf 3ff82bfdaa5c4646
3fedcc517ff0b1d4 bfdcdf4cdcde5b84 3ff9078fcccbac9b bfdb6dca0156a10c
bff26b1a70f37539 3fc08f3a845d3b7c 3fe9553023b2b19e bfd3ec2fcf98abce
3fd8a2cdc156e7c1 3f9995e712fe9bf8 bff43b07ed9a99e3 bfa5b3d3f905b15a
3fc5ad24a1ebc751 3ff9a47720f97ed3 3fd76205de27219d 3ff65214b8926468
3fe9540236906e7e 3fdba588d18228f8 3ff173f676e90284 3fdbbfecce0b559d
3ff7185840128b86 4003bfcbe2375d27 bfe648b48aa50dc2 bff1e925bc76adb6
4000890447ee8955 bfdcdf4cdcde5b84 3fe781ae69818998 3fddafc3abfc0304
3ff4ab9309fffa67 3ffb543533fad77f 3ff339808afd87d9 3fe43bac3933997a
bfe79242b83c558f bfc9891dbee66b29 3fefd3cd12fde5b3 40018540bd183cae
3fcd7aa263e472fc bfd51693631ee6e5 3fc315975e3cfca6 3feb523a75295378
3fb4de5103b0b343 3ff59ab6079859a9 3ff73225be115fa5 40052f51b2a010d1
3f823ba165d7f96a bfe6a7090aa7bf6e 3ff3275b135a9985 3fc2438806c0cacb
bfc326144ef7bc4d 3ff00508739b956a 400135409ae0b14b 4004c21f8c8fd03f
bfde768d69375dde 40029ad9d7ae6c8f 3ff80ee4c1d41f63 3ff8bec1fea780b2
3fc425b0db26a5cd 4004659b7d3fda46 3ff9ccccc7f7ec80 3ff3ba44422d19e2
bfb15f2d113fb680 400246ab2ee3191f 3ff02075566bf4ab 3fccd89656b5caf6
3fec692abe3a6a9e 3fe64829d43b6da2 3ffd8cd926cb2d15 4004f44c8fb5b7ad
bfcb591f8d4997dc 3fa93ea69129754f 3fedeae9db3ec69a 40030e248e259764
3ffe176744c1e8a8 3ffb4cdae5326d6f 3ff8b7cb24c47c13 3fdfde31ddb4f49a
bf983d3d81fec8f6 400275e7c3c43cf3 4002edc456c44bc6 3ffca95dcc3bcc0a
3fed49cb34f83ac7 3ff6801ad73316e6 4000aa9c3e93a0e0 bfe0f1b0d7a7a36f
3fc7d2edda54dc5f 4001a99d63901957 bfe0ff07fceafb12 3ff9247065c038ff
3ff06939041cd06b 3fd0b62576fb028e 3fb950700d943511 3ff723419b49ef79
3fd883922873510d 3ff53a094535f13c 3fd6f21c037ef5fd 3fe6e6eb6eca6d81
3ff7fa3ae2bc6feb 3fdcdd4d7e35812d 3ffabe97a3a8f992 3ffdb32c0e17a59f
3ff30fc129b6f98d bfe68dc1f96edbeb 3fb391573edfd4f0 bfd5cae5bf7b6458
3feecca3507b60ef 3fe79ebf42d93027 bfa55cd143b30c88 3ffd763b64c44636
3ffb0fbec7509f42 bfe2ea7517d4c513 3fc98fd0be6fe625 bfefb88340c0893e
400383c66068d8b5 3feaf25b48944b14 3ff2ca195019b83e 40006d17bdd625db
3fe94725d6a5a191 3ff77220e413661c bfeba540fb8c2760 3fe7754e4367a09e
3fd72a0312f887a6 3ffbc31d012aad60 bfebff6a29ba6619 4006046b1e286ef9
3ffe52fbeabdbe8a 3ff6163d1d9302c2 3fe6494117c3ee11 bff14e204b7aaf9c
40009e5d42af42fb bfd4e798ff7e8425 3ff1a90e981f8590 bfc4c4707aad7fb0
4002b340502ef89a 3ffc43def5abafb3 3f9a6496c7f9dfb2 bfe98c1c921cad7f
4002bde4484898da 3ffea2ca993746d0 400259fa2ff5f362 3fe09dd58abd0b22
3fc5015d714a6d8a 3ff3aa389606e38a 40041237a01f1553 3fdea54cf61c5823
3ff4cdb911e6ddc4 3fe54ed67acf8df5 3ff4e81532a857d9 bfeb0a0a9e560a0d
3ff406184b5d8be2 3fe74fb3c7f437e9 3fef1411a7db0df2 3ff724d8f57d7948
3fd117001c13a715 3ff910ba32c1bd20 3ff1eb60a094b442 3fd39629c85364be
3fe1700a90b3328a 3ff7d65816192db2 3ff1291c9a5f3571 3ffd83468e4a86ef
bfe63e93ddec15bb bfbeeed9645e3fca bfa3075e6fc711f3 4001e7fea3b144ea
40022f8d0436ee12 4006046b1e286ef9 3ff47ea50851e6b3 400002b63df36bca
3ffa5d2f8f5091e2 3fd2247d8b51bc8d 3ff1529cd3482211 3febb586fe3429c8
3ff19c53f87e13f0 3ffd58a7054f0f04 3fb248dc8949e19d 3fff35bd2886e93d
bfd1f6b634040f70 3ff1c654f90d25fd 400275e7c3c43cf3 bfb9fdd97f6a70c8
bfdcdf4cdcde5b84 3fff54899d4c99c7 3fd1ef94363410e7 400121d694d61fab
3fd5b72464bdf0e8 40015f630af5ed26 bfe374d0c7453651 3fd5d711bac29f0d
bfdc39e8a421d206 3ff8a67376d3fc9b 3ff881fb5e27bcca 3fee26d22aecc543
bfa5242f5f58f120 3fe3101d4b8c8243 3fe8fdf23d61f882 3ff87d50afa351b5
3fdb3fb4f626fb6a 3ff7c2a17916b5ff 3fd4956d0953cd5d 3ffee992fa661bde
3ff08f78694db091 3ff0528a82016b0e 3ffb15a087154a12 4002e4d3651d4569
3fdddc1a39589614 bfce0037dec622ff 3fd5f37d250a1bb7 3ff9428026d93c93
bfdd7ef45b4c88af 3ff9218a6b048ddb 3ff8f17885f0ba27 bfc16d2f9cac403c
3fe3929e27f05c3a 3fe94ced4ad2e272 3fe39a5324010383 3fd8b8b262862fd1
bfed0c1d474430ad 3ff2e7be8ccd8800 3fe5167f686cef3e 3fbb9b1b82398da9
3ffa2ea896df3f9d 3ff0f1255fd54a88 3ff436fd8e8ea1c3 4000278dd2075513
3ff7324f68551686 3ffa1990694be66d 3ffd49ec603bad8d 3fe6f5417b350ea9
bff4d70057e4631f bfdb3fae02ec4596 bfdf0420fedd16f5 bfb49b0fbfc11fdf
3ffe04d5705a4a88 3fd72a0312f887a6 3ffeee060059aead bfc31014165f76a5
40074cf6c739df50 bfdff8209aee7486 4003c6854d0d3e77 40002ff96da1e7bd
400261713668865a 3fc754c5cb2ce146 bfd28ebf272c714d 3ff5870eb3b5d9d3
3febabb46c502da0 bfda440588a94b18 3ff0ab152fb562d3 40022f8d0436ee12
3ff2010c61d0fedf bfd09300f437afe9 3ff6a71bdbf4de52 bfe5b4acea89978b
3ff76163403f2464 400470ea6100f89e 40023615fb5c807f 3fe37dd9bbd062a5
3ffbb513ed24730c 3ff4fb65fd04e007 3ff088f517bf99c8 3ffa17e839a398a4
3fe47cc776f5a6f5 3ffb79442c35d822 3fed2a7c19d0c59b 3fd6a5adb68de65a
40015ab1168e60f3 3fffbafd8471c7ae 3ff9e6849cb4161c 3fc83c389259094d
bfbf39f1cf7c3715 400496fe22c7dfa1 3faa84f61fb9c868 3ffb6c1a3d8b6eb1
3ff2e5c58a33528b bfa4a1dd2e3c5d98 bfb75121795e892d 3fd883922873510d
3fc49fc4746e2c58 3fe08e0c2b06634d 3ff6044a6f050321 3ff2347b8f722f23
3ff6a71bdbf4de52 bfec6892e10b63a5 bfceaa96ef9594c9 4004a24fbda6eeba
3febbc2600f4832e bfe852862ed71768 3fe0f0c0c18cd10d 3ffdc9a55cdf841b
3fe97ccf29bc8059 3fff441ab4f20da0 4003262ba9f9824a bfd32934828bcd9d
bfeb15aa606bfff6 3fe5bcf61048c23a 3feb5a7893261fc5 3ff4c42f97d2090f
3fd5c2e01b39bf75 3ff7459ea0d33f90 3fedace2e68d8100 3fe3e986e8a58faf
3fe9c28b304077e8 3ff4c839cf765aa6 bfec7fa24ec1c27e 3fd133402b4ccb30
3ff343724b919d69 3ff627317d808bd4 3ffbcd99956ef8c9 3ff993883da1fe4e
3fd4005c3b255bf4 3ff2b217d52b1168 400071bd6c8019fb 3ff7de6b36cedf83
bfc56161b632ef6d 3ff8a1d2a2f127a1 3ff3a04e7fb8dedc 3ff65214b8926468
3fd118e6cf0da220 3ffd87ee40408f25 bfdfb68bdb3ebdfc 3febf21668a40934
bfd8e09786e8c224 3fc7ec9128c96636 3ff2e1119d1eaa77 3ffa392b8bf70046
bfe6f9a3493e86a0 bfe2e75a0cc101e1 3fe67744a700972c bfdb1f90d1b59eca
3fe495c4b991ee9d bfd6cdbd0f9ab200 3fac9553dcfeb372 bfe431ffe7db8ef2
3ff291a931130e66 3ffd5c6e9382a77c 400275e7c3c43cf3 3fc14760b0841705
4002815c1dfd6052 3fcd03261c131df6 3ff8fdf80f5063e3 3ff788914252fe39
3fd5d711bac29f0d bfc65c44071a441e 3fe15f167d09fbf0 3ff40f2c330084b4
3fe3dde070a5f651 3ffb78b8598e2550 3fee9b6f9170c727 40084722737af626
3ffca9bcbe4af76a 3fedbc3a8139af9c 3ffe7a9dbf5142a8 3fdeb7ef193c8fb3
3fb3b1139599a943 400022a562fd6e4c 3f879bec4f2c9ebe 3fa6140340455291
3ff9552dec3ade61 3fd38a014adf6aca 40047c89a931782b 3fe8a03022f784d9
3fec74f62f990bbf bff580f415f3f58c 3fd273b463bf44f9 3ff7f9a289e79611
3fe0997563490cef bfbd073aae097674 40008326ff6b6fa5 3fead2a547a89941
400677db2e3dc5b8 3fef07396fc3793a 3ffaf1941fc08e76 3fcad4a364e76f7f
3fdc2c62c13b2aa0 3ffed75b6fc70de3 3fd238d8eddeb1fd 3ff06054c7b9a1d8
3fe32ba69d4199b6 3ffd71df5e25e6a4 3ff459e95aa7d0fa 3feb04716f172b2d
3fd7101128661c69 3ffcacde8d1caa38 3ff323728dd17a0f bff11f88c38f3f92
3fc3840da0267541 3fe5cba8130c200d bff247bd4cc1da06 3fe1dba9935bc96c
4001739f2a85f681 3fde7fc38867b5ee bfe15c68e4c6d905 3fe6a621276ffcf4
3fee57eaaa3d57c5 3fd7d78328d3ed1a 3ff66341a28ee57d 3fd5f0e958fb508e
bfd5f8570bcf3e5e 40051c5693a53a85 3fec707a4d756902 3fd254d859627ead
40021b4587fcabd7 3fdd19a3abb74285 3fdf065fde691f00 3fed7db8ab107c63
3ffd82294ca28074 3ffafd0afadf82f4 3fb637cf4f6154e1 bff5b30482d548f9
3ffe771a0d0fa5f9 3fbf0c635a78b54d bff05c91e2d7f535 3fd32014be0f83c0
3fc6a4dcd680de99 3fce29ad934ef906 bfe852862ed71768 bfde6687854d270a
3fb09f2577339df4 bfe76470345fe7a8 3fe178a3f1038501 3ffe5c063d15d1e1
3fb253eadaa0aae7 3fd77ec8aa2abf63 3fe42e7e9d164258 3f87b0ed73ff5dc4
3fea6953d228c2d6 3fe42e7e9d164258 bf983d3d81fec8f6 3fdc2c62c13b2aa0
3ff8753d3e8800f9 3fdcc236070df0dd 3fcd4182131ea4a8 40034daadec77838
40035b76d71bd6ae bfe21f6b0a1034bc 3ffc4f23d6a887e4 3fee29a5b76009d4
bfe9526cc6c45802 bfc567ded152512b 3fdfa73b511e6737 bfeada3995057b86
4001eab2feb02a15 bf9e05d3b8b9cb65 3ffcd61837a243dc bfd1eb1e4656555a
bfbccae45bf1f721 40001e1d95c8e3fb 3fea4c41eaec7f3c 4000ead3f96f059d
3fc56020af73b8b4 bfebd1ee8ce39d9e 3ffd8cdbbc3f1143 3ff4c839cf765aa6
bfe426758b5ba26e 400430888891d2cf 3ffdbb47076cfd9a 3fd8f2ac110b6598
3fd382d51bcd9613 bfec88c1a7b40d46 3fd6d0514708aa4e 3fe5b4b976daa425
3fd33d022c0c2da7 3fd39776c75c5c34 bfde1e8f9a20468d 3fad8580569331a3
3fdd93442abcb179 3ff06939041cd06b 3fd5baf2419d91b2 3fd65ba71ce63cea
3fe6b131534d61ab 3fe8a07c714e4137 3fc058ce1b6bf638 400470ea6100f89e
3ffb12641904587a bfef47c27dd68587 3ffbdcdd40a9f394 3fe0bb31275166f3
bfdacb7b4f90095a bfef64ae5be1bf08 3ff02075566bf4ab 3ff68a1bce8a4f96
3fc7a414046e75b9 3fb3836cae2f9327 3fe6ea0474cde949 bfd2b25b9d27ede3
3ffeab9e9d6e29ab 3fecc8132443a80b bfd72f705a9861ff 3ffc15bb2038698c
3fd9c2cfc565a226 3fcff76ba01528fd 3fedb1b4dc47f695 3fcfb9ad1c0f60d8
400318cae6913b16 400088cb92b6f0bf 40001a287e532094 bfd3e30e64734484
3fe2260c9b19e333 3fdec0c77397fa27 3ff7c4f11139fd63 3ff841c61c093d82
3ffba95ba417de92 3f979971179fa5b0 3ff4a3a2cf8a5eaf bfd16b7c667237cf
bfd55c5830d68dd8 bfe81b5214cca7b6 3ffdca464aaf1684 3fc86986bbd63189
3ff214e39eaf95be 4001ef7c2d5f5699 3ff440bbdec458a5 3fe5bc011462cd5b
3fe236339f8fbfc3 3fefe885cf03273f bfe1b69c235763b3 3ffd62375fa6dafe
3ffa919bdec34512 400033beb079a20c 3fda61292788b388 3fcbae6fcbc1840e
bff0899b37482863 bff308b0fd9f246e 3fee4a6cba2eb5f9 bfbd07ebb44c527f
bfe12ddf089a4850 3f9fe44a405a24e9 3ff27301a5c72e7f bfe4bbee1afc6a80
3ff066cac73978db 3ff554ae2b5916cc 3fe5cf8bc07fe1f7 3fe08182ac405e36
4001cc3daa5afa3f 3fc46a165b46d590 40019a57542d22ed 3fd43c8e82e83c51
40013908b8e94215 3ff6e9a803481cf6 bfd833e41bbf1a49 bfca7931eb51632c
3ffc0bb127219459 bfab9cc0fa0d16e3 3fedef8244044e11 bfafdc0cbb13ea5f
3ff9455e889fc100 3ffc2d7614dc9204 3fe47ab8aba02eb6 bfee5911ea8e3c56
3ff8a1d2a2f127a1 3fee9b6f9170c727 3fe9f96ebe69235d 3feef47a3cb8c2fa
3ff5851559e9be3b bfe6a9dfd0ec783e 40006376759b26af bff0ef88bb2f834c
40008d686799a786 3fdf272a68f08043 bfa8ed12a11fda1c 3ffe2cc83da5e48a
3ff13fe216c8f837 3ff68a520ef7dd57 3ff80a89b96c0abf bfef64ae5be1bf08
3ffe7a9dbf5142a8 3ff96e55e44bbd38 3ffcb8337096d1cb bfe9704cb6cf055e
3ffe9beace58d72e 3fd815f65f594cc7 bfd72ec99e63a437 3feb5ecf310ec882
3fd76205de27219d 3ffb8c681d775537 bfeadb9553e5ec3c bff408f780b94676
3fcff76ba01528fd 3fe4bd3d2895f6ad 3ff3590c78014419 3ffc5a3735b26018
3fddb206192ec61e 3ffae2d2ce0d5ce1 3ffc2cf4c6bd8b3c bfe12ddf089a4850
3ff3dc945bc55d57 3ff476782175e644 3ff9f6be2fe0f271 bfe7df12542afb1c
3fd81579eaf24992 3ff81712bba0b8d0 3ffdca464aaf1684 3fdb48b98bb7c35e
3fe5bc011462cd5b 40008d686799a786 3faff7fc960c5185 3fb09c528d8fac46
3fea5d0da3099c60 3fe04179ea807693 3fc76bd50bc226d1 3fda7bf032cf99c6
3fdd82fb00bf7b3d 3ff7d7b804d01482 bfec6892e10b63a5 3fe8e6fff80f9bd3
bfeb6636ce162056 bf6e647cc561e250 3ff190c7c9ce29c5 3ff92590381ab09b
bf820b8ff2a5394a 3ff61ab0fe87cf68 bfbd8791b96d3ca9 3fe3970b5c24e1ce
3ff068a15bc6ae17 3fd142cfbbcb0d57 3fd4f4bacd866e9c bfd4a339f1dc1237
bfdcbe5b56ca03e0 3fcc6ee90d954ab0 3ff8f018b2807827 3ff6801ad73316e6
3fff71a97da93f65 400581822600beee 3ff03348ccf3e804 bfdb1f90d1b59eca
3ffcd1fc011a9e61 3ffa1156efd9f960 3fdf065fde691f00 3fe32a9e1846a42f
3ffee6de2675d6bc 4005ba9acfff1110 bfe7c85759bd4311 3fec44763d46800d
400003ca4edf349b 3fccd89656b5caf6 3fead3e375d84a65 40005963f3794968
3fffc0b28a0a3b05 bff4009a45ee3940 4003f7dc8c970dda bfe1163acb24f60d
40033840f30f9c1a 3fd7d78328d3ed1a 3fe1262d665ce166 40021ddf1981baed
40022f8d0436ee12 bf996ce081ddfbfa 3fdd19a3abb74285 3ff28f22db60e6c5
bfeecd0911aca3b6 3fdc053c2af528dc bfd5f8570bcf3e5e bfccb0445ec918a3
3ff61ab0fe87cf68 3fe4b2ada9c96451 bfeb9919cb6e29e0 bfb0b846c3f2b10d
bff1ff5c87668821 3fda2d5d88038ed6 400796c715633d39 40029ef64ecb2c44
bfd2efdc4e0a6910 3ffae3802cea50bf 3ffe073bbb255eea 3ff30afd96150ab7
bfeadd89ace1f796 3ff44b77be9c6a5a bfc9d87bd612a743 400370a36411992a
3ff0a493269f873e 3ff30afd96150ab7 3fc85cec130d2151 3ff20f1d22f55c9a
40058108ef73afea 3faa84f61fb9c868 3fff2bfbbae3f775 3fe5b0b521bcd016
3ff8b29ec98b7887 3ffbfe8557f2f65a bfe56b10bf73a27e 3fe1478b686bdea3
3fd3fa9b4ecee266 3ff25ed8d243d334 3fe2d8a29e58f64d 3fdcfba124d952bb
4000c26b5a3e3e38 bfe19fe97dcae326 4000fb42994e6092 3fe17000aa0fad0c
3fc1b2b8c2cf5261 3fec6f8b4e349164 4004452f09016d61 3fb7f34e3d3cdaaa
3ff454e16d2489b0 3fe0997563490cef 400088cb92b6f0bf 4001a2f527dcb893
3fbdd2cd7c934db1 3ffc612c53a67711 3fe6c7e30565ad0d 3fd31cee544b5c15
3ff306979a48bacc 3ff1efa0c981e1fb 3ff78e061c0d7b34 3fe7e7a9c723151d
3ff52f5255660324 3fe9ca056724127c 3fec29c7c041d191 3fddee860d49309d
3ff0cf96dccebc96 3fe41fa6b021ba5e 3fbd3aa8e608d7b5 3ff2717aa433bd0a
3fe2ca435165e650 bfe5b4acea89978b bfbf02b8712ca9d1 3fbf4aff00ffa9f9
3ffaaf7edf52c15d 3f95327152c038cd 3fcbae6fcbc1840e 3fff9587b6f01f67
40040bb5c31209e6 bfe1ac758212ce79 3ff73225be115fa5 bfe2a728a0578251
3ff36df408b2df49 3fdc2f9406252836 3fe690aa879da93d 3ffc092dac528a51
3ffdd1375c1271f5 3ff168a1ca7ca20a 3ffc34e5580a6d5f bfd30ca15bc39cee
3ffb08c6ff63963e 3ff81e9e65a3726f 3ffafa9bcac60588 3fd7433d7d6ffb53
40029a5111b42891 bfd3eb4c606674c0 bfe41990133b7b73 bfdfdf51a9e92141
3ff24cea7360a4e8 bfb53e383d9a7ac6 3fc4d6ed1017b036 bfb1e31e69230c8f
3fda74ea3276d577 3fb23a4f06524abb 3ffd7f0e745cded0 3fff2bfbbae3f775
3fb9f425b85980ce 3fe1bfde5344b13b 3ff9b57aaa50315c 3ff3581e6bb47a78
4005d56de701dc76 4002683e18bf18c0 bfdb09511b242563 bfeeaa1a55f46caa
4006e31bc740d00c 3ffe9beace58d72e 3ffc6789c3a90a03 3fb97c8b4389af1f
3ff750ca4b48d0c7 3ffee4f9c9181da0 3fea7da3a63d9e41 3ff40e46837c6824
40029ad9d7ae6c8f 3ff78d322309d280 3ff69344e8fc6839 3ffa9e3274e1b043
4002815c1dfd6052 bfe7c1c4a420739e 3fe2e5375987e721 bfe6ee8cb35cccc1
3febd460b0efec48 4002f6f1f7884ee7 3ff0c7570b34692c 40060dbb34acd56c
40000f2a057acbf5 bfe42d4590e7e270 3fe7e6e20ab16e0a 3ffd570036f07bc6
3fea9a2676370fec bfee09d11acd9904 3fd863bd7d4d51e0 3ff258d4d9d9e214
3fcf4844612f4b39 bfdd4d00eab293d0 bf983d3d81fec8f6 bfb4af10288fc7f1
40029ef64ecb2c44 3ff7afceb2f1600f 400084e755e45ee8 3ff81712bba0b8d0
40058108ef73afea 3ffa06b24315f567 3ffd0632adfc1fcc 40006d17bdd625db
40004a82bda9e19a bfdd4d00eab293d0 3fcf838dc933dfcb bfc03287b2c55432
3fd963e7e8e1c2ab 3ff01d6eecd1f064 3fdfa73b511e6737 bfd8a392c13766b8
3fc4d6ed1017b036 bfe211b6bc02b795 4002f949f278fc02 3fe60c86d3ddf755
bfd75dd7717335cb 400498500f29f137 4004e320847f0e09 3ff01d6eecd1f064
3ffe0bf2991e1da2 bfa55cd143b30c88 bfe929e5b760829e 3fbd86dc1366b2a6
bfee598a148e9f78 3fe4f73a11e7c333 3fe5167f686cef3e 3f9eaa52ca978e36
3feae201d1e7e8e7 3fd873859c906958 3ff89608fba1b63d bfe4d30daed02e6d
bfdb5f3bf519173a bfc8c62ccb5d1b6a 3ff2b5d16d876a95 3fd84996965697b9
bfeeb5efd446b034 3ffa859e66b49f4e 3fe749f109df9a5c 3ffb4cdae5326d6f
4001aead29a078b9 3fddbe51a767ff92 3fc9b56c56ccca06 4000c6a54b7cbe68
3fbcd1ebafe94896 3fe76d34cdc2ed0e 3fe0af03177ba469 3fec856b0bc3385a
3fc7d2edda54dc5f bfabd95f6c412120 bfdff8209aee7486 3ff860fc73e8247e
400089bdd1a84c80 3fbd70b32cfabce9 3fe600bb1f756d8a 3ffbf58043f704d6
bfba2830cc7d340a bfe48f70ba747cba 3ff2cef496f87fea 3ffd25e8a4331610
3fe80389eed03c5d 4004650d2046a4f0 bfdd1de24b11eada 3fe1478b686bdea3
3fe72458951b114c 3ff985a616bd89eb 3fc3c4677d29a201 3fcfbd9846281be3
3ffa26c2cfeca38c bfece44250fb23a1 3fffcaba5728b6f1 400240c283bdb68b
40055aaa2027be45 3fd61d4e598ae9ad bfd967b3e4fb9ff3 3fff71a97da93f65
3fb74152f298b843 4001d8b3d2d3e210 bfc8b628ae3ed18d 3fe894d778226a90
bff408f780b94676 3fe7e7a9c723151d 4001c5037bda1edf 3fee0d9bdc359858
3fe63cef4d557b1f 3fe164808b657275 3fb74152f298b843 bfea65ab9cf08645
3ffaa51845a02a57 bfdef7051c5b5f18 400362d0999977d4 3ffbaa048dcb2edf
3fdf65f2d6da60db bfedd159561a5fe5 3fb23a4f06524abb bfe62539ea69e3c5
bfe5a21d0bb269f4 3ff215e53103f6e0 3ffd7bec6139d063 3fd48acb452a8f82
3fd6f411aa1dc2e6 3f6c91af93678fc6 bfceaa96ef9594c9 bfeebbf4e7bcd648
bfd5525c109ae430 3fd6ae6da0accda9 bfb38f951811e0b4 3ff73bf560b3e3d6
40018ac2f8e7ed2a 3ff18b3adaf15b01 4000c34697a14cd9 3fd19d18eec8611f
3ff696145e8ef4c8 3fc344e53fd8a741 3fe4f67496d36ce3 bfde940cd3e180ea
3fe134537fb742c4 bfe0120cd99e65b5 3fd5fb6cdeb4c18a 3ff247519ed5ff6e
3ff5e015f20f0403 3fdf205fe798019d 3ff58fcb41a6d70f bff2889db0b38a2a
4004c2b97ac76ade 3ffa1156efd9f960 3fdfa8719ac7f088 3ff606e17f3a4a4b
3fdeeb31e970c4fb 3ff31ac669522ed2 3feb3ce4561abbdb 3ffa25285dcb446b
bfe6fd116e7c5e51 40047d7cf82725d1 40019a2ab165ea6e 3ff9ea289497aabb
4006378b538d73ee 3fed6af56ef528a0 3fcd0300345f4f93 bfd30e42472bb71f
3fd3e2c0eb5f60c7 3ff81712bba0b8d0 3ffa06b24315f567 bfe506d0b4ed9f75
3fe1bd8b83ac0205 4000eada921278d1 3ffb41ddba359159 3ff2d44db25a41bd
3ff29cbf55e13e36 3fda8bd7e61b5b8d 3ff0da7c5ee8d739 3fc3ed175dec4979
3ff130117bcc0d8a 3fff000e7612ae1b 3ff7e6ea377625fe 3ff3ba44422d19e2
3feeb3e4ba1ba962 3ff483259af69001 bfd7d469bc1a4245 3fef9048ab8abced
3fd13b8a9ff92cca 4004c21f8c8fd03f 3fec29c7c041d191 3ff163b0e9ae275f
bfd65e104f066c8d bfbccae45bf1f721 3fed6af56ef528a0 bff268134a052f45
3fe638f77d31f57b bfa43b677b512265 3fe5f411e2d021a3 400172c0f2637d1b
bfe948ec44580b30 3ff30fc129b6f98d 3ffabe97a3a8f992 3ff4b2664d1370c0
bff5b30482d548f9 bfddaabc5bd0f56d 40061f3e352b3a5f 3fd3f5e03356ad06
400037fa7ee4f929 bf983d3d81fec8f6 bf94730a2ecc0a6c 40014845a7373930
3ffcdd061b428f48 3fd51893a3c9d8da 3ffa5d29d9d4d060 bfb8a5262672a27a
3fddaf94e4692c9f 3fc22dd6883eca2d 3fea20937748eb69 bff4a9b00c9cacac
3ff6636d3acfd911 3fcfa10714e5ee60 3fecf38bcfccd8c1 3ffae871b9c5f29b
bfe17f1ca1e7654d 40044c62064b74a9 3feeeb294fb2beb7 3ff3f66e5b1ba089
3ff08f78694db091 bff1ff5c87668821 3feb20a981bf18ec 3ff37bd191bec8a6
bfe72fc3f32031bd 3ff068a15bc6ae17 3ffe7a9dbf5142a8 3fd54b2de75a6db9
bfdee35ce2448121 3ffa19b0e08e0ad7 3fc678467bbebfcc 3ff89e816bb13a14
400796c715633d39 3ffc9083ff749154 3ffa0a264fd562a2 bfdf0d5e93f539f2
bfd60e3e29c75e79 40024851177102bf 3ff506ef74717bff 3fce30a342f2fa43
3ff945bebd11cc7d 400088cb92b6f0bf 4006412753db0635 3ffb38b4672a3e4b
bfd2efdc4e0a6910 bfdb186b9c9a67fc 3fe3da37b9818004 3ff1c532738aee2e
3fbf48effbe3694c 3f68b2d514950731 400293ded55330b2 bfe84f3cfbd94528
3fed9abc88cbcdab 3ff263a53525c452 3ff2e41a9617da3f 3fab53a12f34ae26
3fd7b6908355c37b bfdf24f074b68454 bfeca8d386c16db0 3ff0093fd43eb3a9
bff29d077106bb8a bfc28bbd61d10673 3ffee3a93538525d 3ffa0e2a16566e31
3ff26ff4e8cc5303 3fc3f9e56402c9c4 bfcba5a47ad676b3 3fe7754e4367a09e
3fb9c47f4d5a6958 3ff25400c2531dfd 3f94c8ebe9113706 40013e23b720f55b
3ffd109bd5a6ebfb bfe53d3e3b973497 bfcf5fe6880f8996 3fcf2fb7d3e7353a
3ff0b094c8936b0d 3fed85f57fded7a3 3fef95914d819328 3fb411f648c4d473
3fd6580f81fe1ed7 3ffbe4335fe8f454 3fcbb8b188b72524 3fe8a03022f784d9
bfbd38092d0d5e9a bfbccae45bf1f721 3ffdda66c760b13f bf96e6f253e38c08
bfe36d63f292d08a 4001c2e05e027a85 bff43b07ed9a99e3 3fe6f5417b350ea9
3ff4a1ed13ff578e 3f9ba9b1b399b20b 3ffaa51845a02a57 3fd5baf2419d91b2
bfe62f59403cfe3e bfcbe383551f4227 3ff6574cd3e1d22e bfeab3fe61d974f6
3ff6beaef25737a1 4005ba9acfff1110 3fd93303f5ee304c 400631c20751a5e7
3ff08d0167080aa0 bfd55c5830d68dd8 3fb9b674757c8ba1 3fdd4baa1d33313a
3fb9c0d61480aba1 3ffca95dcc3bcc0a 3fd611f68faee686 3fdbfd57f3819a65
3ff824e67700247a 3ffce094cbe68f45 3ff1d5a73086e753 4003d45ec601e3c0
3fc1ea59455f72a6 3fedaeb93335c047 3ff1c654f90d25fd bfd144331c0e45d9
3fe074fb7ff3145e 4004e8eb3c3723c6 3fea63a6522fd4b9 4000b51aa5432cef
3ff4b2664d1370c0 3fe5b4b976daa425 3ffb6a0e239c5a41 3fdf1e30a428cfa8
3fd020a3279021e9 3fc341b92954e686 3fe9eac08f0dae01 3feb65fc6f8f2838
3fdf4ef4286e913e 3fe5987cb90559fd bfd927f5e4d887d7 3fe4e2df884ca47a
3fd2ae0967c4c3a1 bfa5b3d3f905b15a 4003ff4bd9eac3d3 bfe6fd116e7c5e51
bfd0eed59a023038 bfdb448b532d2b70 bfe07ea453919f84 3ff02cfb786c7a63
400172c0f2637d1b 3fcca592bb23d242 3ffc80581f298161 3fe99b61d78f5266
3fcc70f64630dd61 3ffe1e48848a4c64 40024851177102bf bfc81d647172b575
3ffe3ec414a6f59a 3feac570be44cbbc 40017770b6269053 bff247bd4cc1da06
4002a61697a65ade 3feeeb294fb2beb7 bfe58153be8a57ab 4000a33441a2a751
3fe88ba4709d42b1 3ffed9557af2eb61 bf493959b19b553c 3ff6fc8dcfdbb627
3ff2d44db25a41bd 3fb8e895b17b8659 bfcfcd42a6ae936b 3ff0528a82016b0e
bfe21f6b0a1034bc 3fec29c7c041d191 bfd927f5e4d887d7 400200e7a9436293
3ff48d49eb72b390 3fbcd567578002ba 3fe49aa038a094f9 bff580f415f3f58c
3fe9e25d1828ea2f 3ff1d6d7a88581b6 3ff3726fc6a62fb1 3ff95dba0c6b51c7
3fe8cc8bf0e45b07 bfdd4d00eab293d0 3ff01d6eecd1f064 3fe46e4cd192a8bc
3ffe39424d2f92d5 4001b5ec1bee3d4a 3ffa3abdfa1d64c8 3fd41d874ff7773e
4003c332476fdf8e 3ff2a10b7521fc81 3fcf76ede4bf22a3 bfeec5d97b550439
3fe5dbf68cf88ee4 3ff1e9ccbee7185e 3ff8091bf5d53e07 3fed6df62517b078
3ff8005dcd5725b1 3ff624859d435467 3feec874fc94d2d9 4003c8a2c0bbf43c
3fce0cc23b323ae2 bfd0ecf0fa809c1a bff5b30482d548f9 3ff91112df2ecc14
bfc457c462b640e9 4002f949f278fc02 bfe49e44b2025c46 3fec8bb5e2840cb5
3ffb4e311d2ee095 3ffd6cebc7e3a284 bfecdf6c0be2d65c 3fd5baf2419d91b2
4001559043cfe796 3fe87e5e0e52a6ce 3ffbb513ed24730c 3fd0ada9859657a7
3ff90ce4ecbd3e66 3fbb35ff86559ff4 bf7e4341d54329c3 3fffec939bccee8e
3ff21b6bb67f41b3 3ff637dd7cd94219 3fea653a11823f09 bfdfdf51a9e92141
bfbe1e47e80286ac 3fef3fe560a530d9 3fe1f0fe80f26dbf 40015f630af5ed26
bfd72ec99e63a437 bff3d280a676831f 3fec1287b1874b65 4002c7b6dd8aaf33
3fd1463d55205fea 3fe944fecc3730e4 3fe6955d3f74d1d0 3ff8c0de713a8a2b
3ffc937ed0dbbeda 3fe1544ac6ec2d5b 3ff25a8d80f0ce3a 3fd3e76061f4ffea
3ff5c2edcf117a1b 4004e29e24976d2c 3fc064ac1e235c40 3fff2645ee17705a
bfcb591f8d4997dc 3fb90f0b63ab6aa9 bfdef1b634e2505d bfeabb3224c2a0f1
3fe9540236906e7e bfd5f8570bcf3e5e 3feba256ed038a8b 3fb97c8b4389af1f
bfecfb924345daea 3fb74152f298b843 3fe1c7e9bcdb2e67 3fdddc1a39589614
4005ba9acfff1110 3ff233d68671f661 3ffd699ae8949c8b bfb33d273fb993c5
4005520bb84d1d67 3fee87a287376417 3ff24cea7360a4e8 3fe1bc2183c2fe24
3ff470bb3821f04d 3fd5e9d39b7b7596 3fa97b21f06aed0e 3ff2bcc0158ea707
3fe70ed346414175 4002c7b6dd8aaf33 3f96bfb31386abda 3feea46128afea5b
3ffafa9bcac60588 3fe6a43bd6605f04 3ff25400c2531dfd bfd7299bcb2af7f5
4000904da14df77a bfeadbeaaad7d13f 4001d8b3d2d3e210 bfef47c27dd68587
3fe56309a395a977 3fd600732d4eb786 3fea5d0da3099c60 3fc443388d07e9f2
4005fdf9730212c8 bfdefeadebd76d22 3fe652d5a40ca09d 3fff65943549e2f8
4004c9c98b6a5d71 bfdaa7a2c69d8c55 3ffe987fdf010b5c 3ff47a35d0f98bed
3fe2255f88eee5b2 3feaca7beafff988 3fff1a3fa112ad97 3fe2f3c6b4e0cf51
3fe005fdb2feac97 3fd0a9f9625e9e04 4003c1ccda4cae52 400131c7a6a83b1c
3fff5c61f76677ad 40047b5f3d72072c 3fe1a2b5f816bacf bfe30f7f08eff47b
3ff3c39182ae40a2 3ff5c41c39a60f64 3fa7dbfee30226dc 3ff040552ff39092
3fd6f411aa1dc2e6 3fe9c8e86ecf08a5 bfd2a10e7d759839 bfdd07085f7fe545
3ff38cecf9e6bd1c 3fff716522de9eb4 3fb481538d306b14 3ff4b53fb50dfb34
3ff64467d210fb18 3fed55e08a44abb6 3ff5db76843a0314 40009ff45602375a
3fdba588d18228f8 400482ae089f82e2 3ff65214b8926468 3fef01d6f4c3a1a6
3fc566900a6450b6 3ff30f767caa8ae1 bfe99ac49b5720ac 3fe28cb2e3766ee6
3fe9174101ea5ab1 400135409ae0b14b bfc1990f260118f9 3ff381581b20a83a
3fdf4ef4286e913e 3fd7da61b0b16e54 bfd2fba0f445b0bd 3fea3c55d7a960a0
3ffbe30a601237f1 bff2da543394762c 3febb586fe3429c8 3fc757719c578fc7
3ff4096b3555c257 3fff4ffc43ea2a1f 3fe1c7e9bcdb2e67 3fe09c299fe4362c
bfe0f1b0d7a7a36f 40004a5245e9d6d4 3ff173f676e90284 3ff81712bba0b8d0
3feee4b36ee8a458 3fd6e9df4bac790d bfce114c183cd3dc bfde9c37c7edf2b7
4002b51139bc13fa 3feacdb49fd99e4a 400131c7a6a83b1c bfcb539429953d45
3ff3da99f0c390ca bfde573d351de23f 4001b05e9df476a4 400089bdd1a84c80
bfe153a436108e25 400022dc2b3270f9 3ffc7e344d382714 400325b5fa45cd51
3ff448d9f24a03d2 3ff47f3ff8da67a0 3fe6f5417b350ea9 4003bd0fc8c5c220
bfeb55085fae6c0e bfdb448b532d2b70 3ffcbaf27843df5f 3fc31597403ca2d9
3fd86f208ab8ef11 3ff8155606a86d13 bfe46b848aee50d2 3faacfcdb01f0750
3fa5cd348f7cd26f 3fe8a07c714e4137 3ff30907cad4cd73 bfa2097f0b6d8f82
3ffd8cd926cb2d15 bfd6575b4d875b82 3fdfa73b511e6737 bfd0449a06778755
bfe9d09a35432f8c 3ff4d93399263379 bfe2398c8b9e4a04 3fe7c3be00c8fdba
3fb23a4f06524abb bfe581efa70486a6 bfed8c53cdfc7eb4 bfdd911a8d77ce7b
bfd379b49f640aa0 bfd4de6ace6796eb 3fe9e2a2eca2dcc7 3fff58f1bc8daebe
bfedcbc41c4e4805 bf9259dce2e95d6f 3fe19bc257f07cbf bfc6fdc6fd6cdbca
3fed11b3c68a4d3e 3fdba606533cb1df 3fe24eee5bd29068 bff1a1585722d287
3fdfeb9563081ac6 bfbf1717aaa13b48 3ffbbdc4e9305f1c 3ffd94af5d825115
bfca01ad543eb988 400631c20751a5e7 3feb23cd2cbbb6a9 3feb1df3043540d0
3fe88ba4709d42b1 3ffe4fd7627d2bf2 4000e9c4a42b9543 3ffa737192aae2bc
4000b42dc5401d0b 3ff8eb48cf8d77e4 3feae201d1e7e8e7 3fdf065fde691f00
3ff08f78694db091 3fe5a3d0213c68d3 3ff19c53f87e13f0 4000629ea12316d7
3fd5b3a2850cca6e 3ff7ea2123149fde bf98e36515320447 3feeff2a96d68fe1
bfe76f83311d0b3b 3faa84f61fb9c868 3ff540d33b176c0d 3ff6ee95bd66ef10
3fd5440b01ee49d5 bfe648b48aa50dc2 3fe851fc6bf75b93 3feb12f33cf95b17
3f9a22e86aeb0afb 400272aede587e7e 3fffc0b28a0a3b05 3ff070818d0c063a
3ffacdcef03d36f7 4007c3c044c63717 4003b78107219a93 400165e5ba3db3b5
bfd1ad704d41af5d 3fcf9a2c6a0dbdc5 3fca8dea7aa182a3 3fd1bb52c52b2420
bfd18c4359ba7944 bfdb9bfd6061bf15 3ffcdd061b428f48 3fe019a2e8ce09d1
3ff524ccd51f5169 bfd3ec2fcf98abce 3fea9a3ec5723947 3fd27e784491edac
400549181daa334a bfe22502ae8a1290 3ff1feeb428da81e 3ffe53695d442997
bf98e6c431efcbde bfe2e75a0cc101e1 3fe2954a18184c3f 3fe59ae58664966e
bfc0345513ba78b9 3ffda2b9db6aba9e bfd51693631ee6e5 3ff58072b2d4cf83
3fef036bb3c775e9 bfd5fe8e57c80763 3ffeda0eb4ad1404 3fe4274f46dac450
3ff077aa4d6fd211 3fdfd7675c2fe7eb 4000629ea12316d7 4000904da14df77a
bff5b30482d548f9 3fffa17b751ac9fc bfd10244a03b168f 4003a37d5cd3b174
3ffa097bc772c65e 3fff58f1bc8daebe 3f94c8ebe9113706 4004a24fbda6eeba
3ff9078fcccbac9b 3ff34cf4c40c5c7f 3fe2b632395ba3e7 bfe6f9a3493e86a0
bfd81ff4fc127084 bfd6b3f9694137d4 bfe97a979214b6eb bfd274f550000bcc
3fd963f57070d995 3ff454e16d2489b0 3fdf4ef4286e913e 3fc886f512580560
bfe2dcfca7439c55 40047d7cf82725d1 3ffd48d2fa65c0fb bfd3ce5ee1a9e6b4
3fcf0d31321de47a bfd0923a16ef96d5 bfe0f1b0d7a7a36f 3ff8b198d6c315fc
3fd252f0a65feadc 3ff4320a2948c62e 3fdc19955dd2f7a4 bf9e324709f602dd
3fffae5fc18edc10 3fffb755d2a24245 3ff8155606a86d13 bfce510cf8f275a9
3ffb40ee975f5ca7 3fddbd1b62b444c7 bfbe7654f9eb4b23 40033079a4b1aa9e
3ff9fa352697f186 3ff621a6cb5868e7 3ff1c872d43f9fdc 3fc364dbdb026106
3fc83c389259094d 3ffca2ae91716e82 bfd2e9e4b1b7e6fb 3fbd8dd25b960c25
3fb5cb6d14ed4199 3fe0f0c0c18cd10d 3ffb2f74dd767f0f bfe815ec5b151742
3fd8193d1437384c 3fd888e49a5eac53 40027b8260d91fce bfc5eb4dd25a1dd4
3ff47cc777683db3 3ff696145e8ef4c8 3ff4099d43158ec5 3ff1dda7fa30d8ad
3ff81ed07eb17c1f bfecfb924345daea 3ffc1aa0139dc2be 3fc5dec2ffd34d82
3ff0cf4c6ce44525 3ff3d189c9ed9cb3 400269f470d1eb43 3ff1312e12e58d46
bfd679ca9a06d69e 3fb0201088e05b38 bfe4524ef2f74633 bfe458fc359b0980
bfed8c53cdfc7eb4 3ff2416e49a29074 3ffdbd356865a5fb 3fb1038e29cbdc9b
bfd55b7fd615f046 40009318aac1c98a 3ff00508739b956a 3fd9b9c983a239e9
bfc096dd6f5ce8a9 3fddc4da4a240c57 3ff0340091b0ae50 40047fe74cc8d3b5
40044e7fc100934e 3fefa7b0ef421ce5 bfa8ed12a11fda1c 3ffba7c7a54652c7
3ff5400c7a8411dc 3fe849cef5632b6b 3fe5ba3e226b7136 3fec856b0bc3385a
3ff90669384387e1 bfd72ec99e63a437 3fe91ec97d6f554a 3ff6bf8f7df22f5e
3fdcb2c06a43bba5 3ff4096b3555c257 3ff75196a47c4620 3fff9b31d2133d25
bf77e891a0c979f0 bfe951e5c6d03e10 bfd833e41bbf1a49 bff132cb106a30cb
3fd54cab543cf688 3ffe39dd508eab6a 3fe0bc27643fdead 3fca7633f7449816
3ff71af5700e08b3 3fef18963c52021f bfd8219ce424cdf1 3ff5a6265cd8da94
bfe36d63f292d08a 3fcb2603d468fd9b 3fdf1e30a428cfa8 3ffb4abb296dca06
3ffd77a0f7e9491b 400043720ff277c3 3ff03f5bc10197be 3ff8b6643a860a39
3fd07b1a35a0a2f2 bfa45e9646039162 bfd833e41bbf1a49 bff4a9b00c9cacac
3ff48d49eb72b390 3ff5d1d06cafab12 3fa46fd70e46c648 3fe98b540dbdffe1
bfe1ac758212ce79 3ff01e68b17c293d 3ff361926a411178 bfd6ec349ca3718c
3ffe519afe3ab048 40001206deebbb21 bfde0ebf5758031e 3ff3aa9563042280
bfcdacdf74aaa9a3 3fd0d8693bce6f3b bfe87cf2fabbdaf2 3fd44f6c941fee52
bfd4e824973d4bef 3fed11b3c68a4d3e 3ff345ecd4c93bd0 3fdfdd5ffff60ec3
3fc8719027de01cc 3ffd4d6ea1232825 3fe1478b686bdea3 3ff73d8dabc48157
3ff1d29f872b41a2 3ff2d26d4c91d318 bfdcdf4cdcde5b84 3fc2438806c0cacb
4002bc554c929a1c 4003c1e574e8abfd bfd09300f437afe9 3ff554ae2b5916cc
bfe2b3bbd4ced359 400354be17f8228c 3fe9553023b2b19e 3fe94dc4adf7edae
bfeec5d97b550439 bfeadbeaaad7d13f 3fe6f25fb11e6473 400354be17f8228c
bfe08dec8cb3494c 3fe92d4eaff96973 bf983d3d81fec8f6 3fc27f6d22d10e1b
3fd9575f19e052c9 3fe5eb8f8790f458 3ff1aa08fe982f16 bfd5e5d443c3bcf3
3ffa6aa1e551d1f9 bfe0f73a026905aa 3fc820dda00b0ebc 3fe53910c1a49bc8
3ffd6f97affd3a64 3ff0f8a0d09bb0f0 3fc6a4dcd680de99 3fdc3a4c0cacc0e1
bff4a9b00c9cacac 3fd8d3a5ef9d2634 bf9529c3fcb0d3e0 3fe1f3fefd64eab1
3ff2d89c8442d5fd bfa35b17320a33a2 4002bd71aeaec284 3ff2be6e18d39e77
3ff649e141174491 3f9f5ab0312d343d 4004f44c8fb5b7ad 4003f14f1383cf26
4001efd8c816e1f8 3fbddfbf82991e39 3feefe3b94dbeb6c 40034b190c3f6e42
3ff8155606a86d13 4004847692702f33 4007a190b00d5ba4 3ff68c94db24986d
3ffb440cd58e47f4 3fe6c01e1b3bf991 3ffcf1e9600228f2 3ff1ef7f6e0443a5
3fe5cba8130c200d bff18b352ffcc57f 4001b0674426949b 3fd968947137b022
4003c1ccda4cae52 bfce8333a62311b1 3fe4f67496d36ce3 3fe79ebf42d93027
3ffc9da0f8b3d51c 3ffc420ca256d990 bfcb6e9922617bae 3fc2438806c0cacb
40008d686799a786 bfecccb3bace0a7f bfe17acb9e943729 3fe7c4afcf97fc2c
bfd48e706e6c02dc 3ffe632a3cdc9ec2 3fbfba0d59e333e7 3fe495c4b991ee9d
3ff247519ed5ff6e 3ff965562f280a82 bfec88c1a7b40d46 bfe68498783828b8
3fd84a8e082eb0d6 bff2d6a090bdd101 3ffdca1db35145c1 bff190cf40ade665
4003c1ccda4cae52 4001fed91d39b86d 3fe0f83255f72c1d bf82382b521ea934
3ff3e7751101af55 3fedd42605667bba bfb15f2d113fb680 3ff0a47b4a0cbec8
4003e915e40e1db2 3fe49f0f065a75ba 3fdf4651002ebff0 bfc302f4cbb85cec
3fe775b2e046766d 3ff4db3325d07e5c 3fe1ed814bf36fde bfdff9faf58e302a
bfe45749ce8bfeda 3ff83cc1f5a265d9 3fddd3134a3d6bc4 3fefaf1c52687930
3fdfaf9b075b2f6f 3fae40d63d6af0b3 3ff06939041cd06b 4002023140a69f8c
bf9bff48772a160b 3ff35819f73650c9 40006a88cb6ac933 4004ae7f72d70c21
3ff00969c0632f9f 3ff1c929736c0455 3fe7860da5dc0b3f 3fdd2eaa9f0dd389
3ff4d93399263379 3fe381cc5feca3d6 bfe1627da133fd12 3fecba0a85cb146e
4003e9e52c0fb816 3ffdf185f2c318be 3fe5b2692ea8e187 3ff9428026d93c93
3fd1450f51c497ac bfe6f9a3493e86a0 bff3d280a676831f 3fe437439646814e
3fe56309a395a977 bfe7c85759bd4311 3ff142ba8e924f26 bff2789c607a1b67
3fed09d985a87425 3fdfa73b511e6737 bfd39ac2a8ac0e18 bfe95e4a0e403f98
3fdb48b98bb7c35e 3ff025e74586bdcc 3fe8193aa65ff9d3 3feaca94a266687e
"""

# --- GOLDEN END ---


# --------------------------------------------------------------------------
# Bits
# --------------------------------------------------------------------------


def _bits(v: Float64) -> UInt64:
    """The IEEE-754 bit pattern, which is what is compared. `==` on Float64
    would call +0.0 and -0.0 equal and would say nothing about a NaN
    payload; the claim under test is that the same double arrives, so the
    bits are the comparison."""
    return v.to_bits().cast[DType.uint64]()


def _f64(u: UInt64) -> Float64:
    return bitcast[DType.float64, 1](SIMD[DType.uint64, 1](u))


def _order_key_bits(u: UInt64) -> UInt64:
    """`binning.order_key` applied to a bit pattern rather than a value, so
    a mismatch can be reported in ulps without routing a possible NaN
    through a Float64."""
    if (u >> 63) != 0:
        return u ^ 0xFFFF_FFFF_FFFF_FFFF
    return u | 0x8000_0000_0000_0000


def _ulps(a: UInt64, b: UInt64) -> UInt64:
    """Representable doubles between two patterns. 1 is the smallest
    non-zero distance a moved multiply can produce, and it is the distance
    both of this round's FMA findings produced."""
    var ka = _order_key_bits(a)
    var kb = _order_key_bits(b)
    return ka - kb if ka >= kb else kb - ka


def _hex16(u: UInt64) -> String:
    var out = String("")
    for i in range(16):
        var d = Int((u >> UInt64((15 - i) * 4)) & 0xF)
        out += chr(48 + d) if d < 10 else chr(87 + d)
    return out^


# --------------------------------------------------------------------------
# Parsing the checked-in literals
# --------------------------------------------------------------------------


def _parse_hex(text: String) -> List[UInt64]:
    """Whitespace-separated 16-digit hex words. Anything that is not a hex
    digit is a separator, so newlines and indentation carry no meaning."""
    var out = List[UInt64]()
    var b = text.as_bytes()
    var acc = UInt64(0)
    var have = False
    for i in range(len(b)):
        var c = Int(b[i])
        var d = -1
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        if d >= 0:
            acc = acc * 16 + UInt64(d)
            have = True
        else:
            if have:
                out.append(acc)
            acc = 0
            have = False
    if have:
        out.append(acc)
    return out^


def _parse_ints(text: String) -> List[Int]:
    """Whitespace-separated signed decimals."""
    var out = List[Int]()
    var b = text.as_bytes()
    var acc = 0
    var have = False
    var neg = False
    for i in range(len(b)):
        var c = Int(b[i])
        if c >= 48 and c <= 57:
            acc = acc * 10 + (c - 48)
            have = True
        elif c == 45 and not have:
            neg = True
        else:
            if have:
                out.append(-acc if neg else acc)
            acc = 0
            have = False
            neg = False
    if have:
        out.append(-acc if neg else acc)
    return out^


# --------------------------------------------------------------------------
# One fixture's recorded state
# --------------------------------------------------------------------------


@fieldwise_init
struct _Record(Movable):
    """Everything a fixture pins.

    `ints` is a flat stream: the tree count, then per tree the node count
    followed by (feature, threshold_bin, left, right) per node. Flat because
    a golden file is read as a diff, and one stream of integers diffs
    legibly while four parallel arrays per tree do not.
    """

    var name: String
    var ints: List[Int]
    var vals: List[UInt64]
    var raw: List[UInt64]
    var pred: List[UInt64]


def _record_trees(
    trees: List[Tree], mut ints: List[Int], mut vals: List[UInt64]
):
    ints.append(len(trees))
    for t in range(len(trees)):
        var n_nodes = len(trees[t].feature)
        ints.append(n_nodes)
        for i in range(n_nodes):
            ints.append(trees[t].feature[i])
            ints.append(trees[t].threshold_bin[i])
            ints.append(trees[t].left[i])
            ints.append(trees[t].right[i])
            vals.append(_bits(trees[t].value[i]))


def _record(var name: String, b: Booster, data: BinnedMatrix) -> _Record:
    var ints = List[Int]()
    var vals = List[UInt64]()
    _record_trees(b.trees, ints, vals)
    var raw = List[UInt64](capacity=data.n_rows)
    var pred = List[UInt64](capacity=data.n_rows)
    for r in range(data.n_rows):
        raw.append(_bits(b.predict_raw_row(data, r)))
        pred.append(_bits(b.predict_row(data, r)))
    return _Record(name^, ints^, vals^, raw^, pred^)


def _record_multiclass(
    var name: String, b: MulticlassBooster, data: BinnedMatrix
) -> _Record:
    var ints = List[Int]()
    var vals = List[UInt64]()
    _record_trees(b.trees, ints, vals)
    var k = b.n_classes
    var raw = List[UInt64](capacity=data.n_rows * k)
    var pred = List[UInt64](capacity=data.n_rows * k)
    var bins = List[Int](capacity=data.n_features)
    for r in range(data.n_rows):
        bins.clear()
        for f in range(data.n_features):
            bins.append(Int(data.bins[f * data.n_rows + r]))
        var row_raw = b.predict_raw_bins(bins)
        var row_pred = b.predict_proba_bins(bins)
        for c in range(k):
            raw.append(_bits(row_raw[c]))
            pred.append(_bits(row_pred[c]))
    return _Record(name^, ints^, vals^, raw^, pred^)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _check_length(
    fixture: String, array: String, got: Int, want: Int
) raises:
    if got != want:
        raise Error(
            "golden shape changed in fixture '",
            fixture,
            "', array '",
            array,
            "': the run produced ",
            got,
            " entries, the checked-in golden holds ",
            want,
            ". A length change is a structural change (a different number of",
            " trees, nodes, or rows), not a rounding change.",
        )


def _check_ints(fixture: String, got: List[Int], want_text: String) raises:
    var want = _parse_ints(want_text)
    _check_length(fixture, "tree_structure", len(got), len(want))
    for i in range(len(got)):
        if got[i] != want[i]:
            raise Error(
                "golden mismatch in fixture '",
                fixture,
                "', array 'tree_structure' (flat stream of tree count, then",
                " per tree the node count and (feature, threshold_bin,",
                " left, right) per node), index ",
                i,
                ": expected ",
                want[i],
                ", got ",
                got[i],
                ". Tree structure moved, so this is not a rounding",
                " difference: a split was chosen differently.",
            )


def _check_bits(
    fixture: String, array: String, got: List[UInt64], want_text: String
) raises:
    var want = _parse_hex(want_text)
    _check_length(fixture, array, len(got), len(want))
    for i in range(len(got)):
        if got[i] != want[i]:
            raise Error(
                "golden mismatch in fixture '",
                fixture,
                "', array '",
                array,
                "', index ",
                i,
                ": expected bits 0x",
                _hex16(want[i]),
                " (",
                _f64(want[i]),
                "), got bits 0x",
                _hex16(got[i]),
                " (",
                _f64(got[i]),
                "), ulp distance ",
                _ulps(want[i], got[i]),
                ". Do NOT regenerate the golden values to clear this; see",
                " this file's docstring.",
            )


def _verify(
    rec: _Record,
    ints: String,
    vals: String,
    raw: String,
    pred: String,
) raises:
    _check_ints(rec.name, rec.ints, ints)
    _check_bits(rec.name, "tree_values", rec.vals, vals)
    _check_bits(rec.name, "row_raw_score", rec.raw, raw)
    _check_bits(rec.name, "row_prediction", rec.pred, pred)


# --------------------------------------------------------------------------
# The fixtures
# --------------------------------------------------------------------------


def _features(n_rows: Int, n_features: Int, seed: UInt64) -> List[Float64]:
    """Column-major `_uniform(k + seed)` over `k = f * n_rows + r`."""
    var out = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        out.append(_uniform(UInt64(k) + seed))
    return out^


def _regression_target(x: List[Float64], n_rows: Int) -> List[Float64]:
    """`3*x0 - 2*x1 + x2*x3 + 0.5*(x2 - 0.5)^2`: linear in two features,
    an interaction, and a curve, so a shallow tree has something to split
    on at every depth."""
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = x[0 * n_rows + r]
        var x1 = x[1 * n_rows + r]
        var x2 = x[2 * n_rows + r]
        var x3 = x[3 * n_rows + r]
        y.append(3.0 * x0 - 2.0 * x1 + x2 * x3 + 0.5 * (x2 - 0.5) * (x2 - 0.5))
    return y^


def _fixture_l2() raises -> _Record:
    comptime n_rows = 200
    comptime n_features = 8
    var x = _features(n_rows, n_features, 0)
    var y = _regression_target(x, n_rows)
    var data = bin_equal_width(x, n_rows, n_features, 32)
    var params = BoosterParams(8, 0.3, TreeParams(8, 5, 1.0, 1e-3))
    var booster = train(data, y, SQUARED_ERROR, params)
    return _record("l2", booster, data)


def _fixture_logit() raises -> _Record:
    comptime n_rows = 200
    comptime n_features = 8
    var x = _features(n_rows, n_features, 10_000)
    var labels = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var s = (
            2.0 * x[0 * n_rows + r]
            - x[1 * n_rows + r]
            + x[2 * n_rows + r] * x[3 * n_rows + r]
        )
        labels.append(1.0 if s > 1.0 else 0.0)
    var data = bin_equal_width(x, n_rows, n_features, 32)
    var params = BoosterParams(8, 0.3, TreeParams(8, 5, 1.0, 1e-3))
    var booster = train(data, labels, BINARY_LOGISTIC, params)
    return _record("logit", booster, data)


def _fixture_multi() raises -> _Record:
    comptime n_rows = 150
    comptime n_features = 6
    comptime n_classes = 3
    var x = _features(n_rows, n_features, 20_000)
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        var s = (
            0.5 * x[0 * n_rows + r]
            + 0.3 * x[1 * n_rows + r]
            + 0.2 * x[2 * n_rows + r]
        )
        var k = Int(s * Float64(n_classes))
        labels.append(n_classes - 1 if k >= n_classes else k)
    var data = bin_equal_width(x, n_rows, n_features, 32)
    var params = BoosterParams(4, 0.3, TreeParams(6, 5, 1.0, 1e-3))
    var booster = train_multiclass(data, labels, n_classes, params)
    return _record_multiclass("multi", booster, data)


def _fixture_bagged() raises -> _Record:
    comptime n_rows = 200
    comptime n_features = 8
    var x = _features(n_rows, n_features, 0)
    var y = _regression_target(x, n_rows)
    var data = bin_equal_width(x, n_rows, n_features, 32)
    var params = BoosterParams(8, 0.3, TreeParams(8, 5, 1.0, 1e-3))
    var booster = train(
        data, y, SQUARED_ERROR, params, bagging=BaggingParams(0.6, 1, 7)
    )
    return _record("bagged", booster, data)


def _fixture_ffrac() raises -> _Record:
    comptime n_rows = 200
    comptime n_features = 12
    var x = _features(n_rows, n_features, 30_000)
    var y = _regression_target(x, n_rows)
    var data = bin_equal_width(x, n_rows, n_features, 32)
    var tree = TreeParams(8, 5, 1.0, 1e-3, 0.0, feature_fraction=0.5)
    var params = BoosterParams(8, 0.3, tree^)
    var booster = train(data, y, SQUARED_ERROR, params)
    return _record("ffrac", booster, data)


def _cat_weight(code: Int) -> Float64:
    var table: List[Float64] = [0.0, 1.5, -1.0, 0.7, -2.0, 2.5]
    return table[code]


def _fixture_large() raises -> _Record:
    """The only fixture above the thresholds the production code now has.

    The other six are 150 to 200 rows at 32 bins, five of them on
    `bin_equal_width`, and between them they cover **neither** of this round's
    two deliberate bit-moving changes: row-block private histograms need 8,160
    rows at 255 bins to engage, and the binning defaults need `fit_bins` and a
    column with levels of one or two rows before `min_data_in_bin = 3` merges
    anything. A fixture that cannot see a change is not a contract, and this
    file was silently green through both.

    So: 12,000 rows, which clears the block threshold with room; 255 bins,
    which is the production bin count rather than a toy one; `fit_bins`, so
    the binner is exercised rather than bypassed; four continuous columns and
    two deliberately low-cardinality ones, the second of which carries rare
    levels so the `min_data_in_bin` merge has something to merge.

    It is slower than the other six put together and that is the price of
    covering the code that actually runs.
    """
    comptime n_rows = 12_000
    comptime n_features = 6
    var x = _features(n_rows, n_features, 90_000)
    # Column 4: sixteen levels, so it packs at four bits if the row-major view
    # is ever built, and every level is populous enough to survive the merge.
    for r in range(n_rows):
        x[4 * n_rows + r] = Float64(r % 16)
    # Column 5: a long tail. Levels 0..7 hold hundreds of rows each and levels
    # 8..39 hold one or two, which is exactly what `min_data_in_bin = 3`
    # collapses and what nothing else in this file contains.
    for r in range(n_rows):
        if r % 97 == 0:
            x[5 * n_rows + r] = Float64(8 + (r // 97) % 32)
        else:
            x[5 * n_rows + r] = Float64(r % 8)
    var y = _regression_target(x, n_rows)
    var mapper = fit_bins(x, n_rows, n_features, max_bins=255)
    var data = mapper.transform(x, n_rows)
    var params = BoosterParams(8, 0.3, TreeParams(31, 20, 1.0, 1e-3))
    var booster = train(data, y, SQUARED_ERROR, params)
    return _record("large", booster, data)


def _fixture_misscat() raises -> _Record:
    """Feature 1 is missing on every seventh row, feature 2 is an
    integer-coded categorical with six categories, and the target depends on
    both, so a split search that mishandles either produces a different
    tree."""
    comptime n_rows = 200
    comptime n_features = 5
    comptime seed = UInt64(40_000)
    var x = List[Float64](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            var u = _uniform(UInt64(f * n_rows + r) + seed)
            if f == 1 and r % 7 == 0:
                x.append(NAN)
            elif f == 2:
                x.append(Float64(Int(u * 6.0)))
            else:
                x.append(u)

    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x1 = x[1 * n_rows + r]
        var v = 2.0 * x[0 * n_rows + r]
        v += 0.5 if isnan(x1) else x1
        v += _cat_weight(Int(x[2 * n_rows + r]))
        y.append(v)

    var mapper = fit_bins(
        x,
        n_rows,
        n_features,
        max_bins=32,
        categorical_features=[2],
        use_missing=True,
    )
    var data = mapper.transform(x, n_rows)
    var tree = TreeParams(
        8, 5, 1.0, 1e-3, 0.0, cat=CategoricalParams(4, 32, 2.0, 10.0, 5)
    )
    var params = BoosterParams(8, 0.3, tree^)
    var booster = train(data, y, SQUARED_ERROR, params)
    return _record("misscat", booster, data)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_golden_l2_regression() raises:
    _verify(_fixture_l2(), _L2_INTS, _L2_VALS, _L2_RAW, _L2_PRED)


def test_golden_binary_logistic() raises:
    _verify(_fixture_logit(), _LOGIT_INTS, _LOGIT_VALS, _LOGIT_RAW, _LOGIT_PRED)


def test_golden_multiclass() raises:
    _verify(_fixture_multi(), _MULTI_INTS, _MULTI_VALS, _MULTI_RAW, _MULTI_PRED)


def test_golden_bagged() raises:
    _verify(_fixture_bagged(), _BAG_INTS, _BAG_VALS, _BAG_RAW, _BAG_PRED)


def test_golden_feature_fraction() raises:
    _verify(_fixture_ffrac(), _FFRAC_INTS, _FFRAC_VALS, _FFRAC_RAW, _FFRAC_PRED)


def test_golden_large_shape() raises:
    _verify(_fixture_large(), _LARGE_INTS, _LARGE_VALS, _LARGE_RAW, _LARGE_PRED)


def test_golden_missing_and_categorical() raises:
    _verify(
        _fixture_misscat(),
        _MISSCAT_INTS,
        _MISSCAT_VALS,
        _MISSCAT_RAW,
        _MISSCAT_PRED,
    )


# --------------------------------------------------------------------------
# Generate mode
# --------------------------------------------------------------------------


def _emit_ints(name: String, values: List[Int]):
    print("comptime " + name + ' = """')
    var line = String("")
    for i in range(len(values)):
        if line.byte_length() > 0:
            line += " "
        line += String(values[i])
        if (i + 1) % 12 == 0:
            print(line)
            line = String("")
    if line.byte_length() > 0:
        print(line)
    print('"""')
    print("")


def _emit_bits(name: String, values: List[UInt64]):
    print("comptime " + name + ' = """')
    var line = String("")
    for i in range(len(values)):
        if line.byte_length() > 0:
            line += " "
        line += _hex16(values[i])
        if (i + 1) % 4 == 0:
            print(line)
            line = String("")
    if line.byte_length() > 0:
        print(line)
    print('"""')
    print("")


def _emit(prefix: String, rec: _Record):
    _emit_ints(prefix + "_INTS", rec.ints)
    _emit_bits(prefix + "_VALS", rec.vals)
    _emit_bits(prefix + "_RAW", rec.raw)
    _emit_bits(prefix + "_PRED", rec.pred)


def _generate() raises:
    print("# --- GOLDEN BEGIN ---")
    print(
        "# Generated by `mojo run ... tests/test_golden_bits.mojo"
        " --generate`."
    )
    print("# Do not hand-edit; see this file's docstring before changing.")
    print("")
    _emit("_L2", _fixture_l2())
    _emit("_LOGIT", _fixture_logit())
    _emit("_MULTI", _fixture_multi())
    _emit("_BAG", _fixture_bagged())
    _emit("_FFRAC", _fixture_ffrac())
    _emit("_MISSCAT", _fixture_misscat())
    _emit("_LARGE", _fixture_large())
    print("# --- GOLDEN END ---")


def main() raises:
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--generate":
            _generate()
            return
    TestSuite.discover_tests[__functions_in_module()]().run()

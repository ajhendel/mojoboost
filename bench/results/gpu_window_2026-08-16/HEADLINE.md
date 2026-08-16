# The GPU end-to-end headline, 1,000,000 x 50, against LightGBM stock+det

**Taken on the merged tree at `901e31d`, so `lambda_l2 = 0` and LightGBM's stock
binning defaults, on both sides. One pair, one process, five repeats.**

**Canary: `stable`.** CPU drift 3.8 percent, GPU drift 1.1 percent, both under
the 5 percent threshold. This is the first headline this project has taken with
the regime instrument agreeing it was quiet.

| | binning | training | **end to end** |
|---|---|---|---|
| **mojotrees GPU** | 0.115 | 2.965 | **3.080 s** |
| **LightGBM stock+det, 10 threads** | 0.325 | 3.311 | **3.636 s** |
| ratio | **2.84x faster** | 1.12x faster | **1.18x faster** |

Our arm's samples: 2.951 2.962 2.965 2.968 2.974 -- a spread of **0.8 percent**.
LightGBM's: 3.115 3.129 3.311 3.369 3.381.

**The GPU is ahead end to end.** The CPU campaign's arm on the same shape and the
same comparator is **1.75x behind** (6.573 against 3.757). So the two backends
sit on opposite sides of the comparator, and which number is "mojotrees against
LightGBM" depends entirely on which backend a user gets -- which is why
`device='auto'` reaching the GPU above the crossover was worth a lane.

## Both engines throttle, ours less, and that inflates our margin

The first attempt at this ran **twelve** repeats, on the CPU campaign's advice
that a five-repeat median measures a transient. The canary called it
`SHIFTED-DURING-SESSION` at **102.9 percent** CPU drift, so it is discarded --
but the shape of the discard is the finding:

| | first-5 median | last-5 median | rise |
|---|---|---|---|
| mojotrees GPU | 2.983 | 3.822 | **28%** |
| LightGBM | 3.529 | 5.709 | **62%** |
| ratio, LightGBM over ours | **1.18x** | **1.49x** | |

**The margin grows with heat, from 1.18x to 1.49x, because the ten-core CPU
comparator throttles more than twice as hard as our GPU does.** That is exactly
the mechanism behind the 1.50x claim this project withdrew a day ago, and it
would have been reproduced here by anyone quoting a long run.

So the honest reading is narrow and it is the cool one: **1.18x, and the 1.49x is
us measuring LightGBM's thermal envelope rather than our own speed.** A user
running one fit on a cool machine gets roughly the first number; a user running
fits back to back gets something between, and so does LightGBM.

## What it does not say

- **Accuracy is not in this table** and the rule is that speed and accuracy are
  reported together. The real-data gate is re-run on this tree before this
  number goes anywhere near a summary, because `lambda_l2` moved and the
  `GAIN_FORM_CROSS` arm's advantage lives exactly where `lambda_l2` moves things.
- **No baseline was recorded for the canary**, which refused itself again at 7.9
  percent CPU spread against its own 3 percent bar. So `stable` here is a
  start-to-end drift verdict, not a ratio against a known-good window.
- **The canary's regime verdict ORs its two probes.** The CPU campaign found a
  run marked SHIFTED on 6.9 percent GPU drift whose CPU drift was 0.4, which
  cannot affect a CPU-only pair. For a GPU window the OR is conservative in our
  favour, but the verdict should be per-engine and is not.

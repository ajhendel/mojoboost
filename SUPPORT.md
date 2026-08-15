# Getting help with mojotrees

mojotrees is an experimental public alpha maintained by volunteers. Questions
are welcome, including ones that turn out to be about your own code. If
something in the documentation was misleading enough to send you here, saying
so is a useful contribution by itself.

## Where to go

| You want to | Go here |
| --- | --- |
| Ask how to do something, or why mojotrees behaves a certain way | [Discussions, Q&A](https://github.com/mojotrees/mojotrees/discussions/categories/q-a) |
| Report something broken or wrong | [Bug report issue](https://github.com/mojotrees/mojotrees/issues/new?template=bug_report.yml) |
| Share a benchmark or a result from real hardware | [docs/HARDWARE_CONTRIBUTORS.md](docs/HARDWARE_CONTRIBUTORS.md) first, then the [hardware result form](https://github.com/mojotrees/mojotrees/issues/new?template=hardware_result.yml) |
| Propose a feature or an API change | [Discussions, Ideas](https://github.com/mojotrees/mojotrees/discussions/categories/ideas) |
| Show what you built with it | [Discussions, Show and tell](https://github.com/mojotrees/mojotrees/discussions/categories/show-and-tell) |
| Report a security vulnerability | Privately, per [SECURITY.md](SECURITY.md). Never in a public issue. |
| Report a Code of Conduct problem | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |

If you are unsure whether something is a bug or your own mistake, file the bug
report anyway. Sorting that out is the maintainers' job, not yours, and a
question that turned out to be a real bug is worth far more than the cost of
reading a question that did not.

## Before asking

Two documents answer most questions faster than a thread will.

- [docs/LIGHTGBM_PARITY.md](docs/LIGHTGBM_PARITY.md) is the behavior contract.
  If mojotrees differs from LightGBM, this file says whether the difference is
  deliberate. It is authoritative where it and the README disagree.
- [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) covers accelerator status.
  Before reporting that the GPU path is slow, absent, or refusing to run, check
  whether your hardware and workload are in the validated set. The set is
  currently one Apple M4, so the answer is usually no, and that is a finding
  rather than a support question. [docs/HARDWARE_CONTRIBUTORS.md](docs/HARDWARE_CONTRIBUTORS.md)
  is the protocol for turning what your machine did into a record the project
  can use.

A search of open and closed issues is worth the thirty seconds, and adding
detail to an existing issue is more useful than opening a second one.

## What to include

The single most useful thing is a snippet somebody else can run. After that,
the environment, because an alpha fails differently on different machines.

```sh
pixi run mojo --version
python3 -c "import platform, sys; print(platform.platform(), platform.processor(), sys.version)"
```

Include that output, your accelerator if a GPU is involved, whether you passed
`device="cpu"`, `"gpu"`, or `"auto"`, the full error text rather than a
paraphrase, and what you expected instead. Data shapes and parameter values
matter more than the data itself; if a dataset is required to reproduce the
problem, a small synthetic generator beats an attachment.

The bug report template asks for these fields, so filling it in covers it.

## What to expect

Nobody here is on call. Issues and Discussions are read in batches, and a
question can sit for a few days before anyone answers, particularly one that
needs hardware the maintainers do not own. No response after a week is not a
rejection; a comment on the thread is a reasonable nudge.

Answers are honest about uncertainty. "That path is untested on your hardware
and I cannot tell you whether it works" is a real answer that you should expect
to receive, and it is more useful than a confident guess.

Things that get answered quickly are reproducible failures with a snippet,
anything involving a wrong numerical result, installation failures on a
supported platform, and results from NVIDIA or AMD hardware, which the project
actively needs and currently has no way to produce.

Things that will take longer are questions about unvalidated hardware, requests
to match a LightGBM behavior not yet listed in the parity contract, and
performance questions without a reproducible benchmark.

## Commercial support

There is none. mojotrees has no company behind it, no support contract, no
service level agreement, and no paid tier. It is Apache-2.0 software provided
as-is. See [GOVERNANCE.md](GOVERNANCE.md) if you want the longer version of
what the project is and is not.

If you depend on mojotrees for something that matters, the practical answers
are to pin a commit, keep your own tests against your own data, and contribute
the fixes you need. Write access is available to anyone contributing regularly,
and [GOVERNANCE.md](GOVERNANCE.md) says how.

## Security

Do not report vulnerabilities through issues, Discussions, or any public
channel.

[SECURITY.md](SECURITY.md) is the whole policy. It has the private reporting
channel, what to include, what is in and out of scope, the response targets,
and the disclosure terms. Read the scope section before deciding which door to
use; several things that feel like security problems, including any behavior
difference from LightGBM and any crash caused by parameters you chose
yourself, are ordinary public issues.

If you are genuinely unsure, report privately. A private report that turns out
to be a normal bug costs nothing to move into the open.

## Contributing a fix

If you found the problem and know the fix, [CONTRIBUTING.md](CONTRIBUTING.md)
covers how to make the change and which tests to run. During the alpha no
contribution waits on a human approval, so a pull request is frequently the
fastest way to resolve your own issue.

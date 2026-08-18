PURPOSE: Index of copy-and-paste launch messages.

Every file in this directory is plain UTF-8 text with Unix LF line endings. The first lines of each file identify exactly where it should be posted. Delete the DESTINATION and TITLE lines before posting if the destination has separate title and body fields.

The Modular Forum and Discord announcements went up on 2026-08-14 under the project's former name, MojoBoost, and drew no readers before the 2026-08-15 rename to MojoTrees. 01_MODULAR_DISCORD.txt and 02_MODULAR_FORUM.txt carry the new name, URLs, and install command, and are written to replace those posts in place rather than to announce a rename.

BLOCKER BEFORE POSTING ANY FILE HERE: the install line must actually work. Publish 0.1.0a2 to PyPI first, confirm the install from a clean environment, then post. The old `mojoboost` package stays on PyPI and must not be deleted.

WHY EVERY FILE SAYS `pip install --pre mojotrees` AND NOT `pip install mojotrees`: 0.1.0a2 is a pre-release, and pip ignores pre-releases unless asked. Plain `pip install mojotrees` reports "No matching distribution found" even on a supported Mac. Do not simplify the install line in any of these drafts before a final (non-alpha) version is published, or the announcement ships an install command that fails for everyone who tries it. When 0.1.0 ships, drop the `--pre` everywhere and the line becomes true.

Remaining optional outreach:

1. Post 03_X_LINKEDIN_BLUESKY.txt from personal social accounts.
2. Post 04_REDDIT_MACHINELEARNING.txt only after checking the current subreddit self-promotion and project-post rules.
3. Post 05_SHOW_HN.txt after the Modular community has had a chance to catch obvious installation or documentation problems.
4. Post 06_CLAUDE_BUILD_STORY.txt separately if you want to discuss the parallel AI-assisted development process.

CREATOR_BIO_AND_TALK_ABSTRACTS.txt contains short and long creator bios plus
conference, meetup, podcast, and project-page descriptions. Use these when a
site asks who built the project or when proposing a technical talk. They make
the creator and maintainer role explicit without claiming unvalidated results.

The public alpha is mojotrees 0.1.0a2 on PyPI. Its only published wheel is for CPython 3.14 on Apple Silicon with macOS 26 or newer. Do not claim near feature parity, production readiness, NVIDIA or AMD validation, or performance leadership unless the linked repository evidence has established it. The strongest current description is broad LightGBM-shaped feature surface with experimental-alpha maturity and a working, narrowly targeted binary install.

IF A POST EVER CARRIES A SPEED NUMBER, IT CARRIES THIS WITH IT, IN THE SAME POST AND NOT AS A LINK. Every performance figure this project has ever produced was measured on one Apple M4 laptop, one machine and not a chip family, and no NVIDIA or AMD device has ever executed this code, so anything said about a datacenter card is a prediction rather than a measurement. That machine's CPU and GPU share one memory bus, so a mojotrees GPU arm and a CPU comparator were contending for the same bandwidth, which handicaps every arm identically on that machine and does not represent a machine where the accelerator owns its memory and the CPU comparator keeps the whole host bus. State both halves of that; do not post the half that flatters us. It is a laptop, so its timings drift by a factor of two to three between thermal windows, and only arms interleaved inside one run are comparable. Name the run id, the shape, and the arm set, per rule 10 of bench/results/LANE_RULES.md. A number without those is not publishable here, however carefully it was measured.

Do not write that mojotrees is the only or the first gradient-boosted tree library to reach an Apple GPU. What the repository supports is narrower and is stated in the README's "Why this exists" section, which is about three named libraries and cites their own documentation. Apple ships MLBoostedTreeRegressor in Create ML and this project has never measured it. XGBoost and CatBoost both train on CUDA, so "competitors cannot use a GPU" is false and must not be implied.

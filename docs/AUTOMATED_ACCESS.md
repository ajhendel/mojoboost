# Automated contributor access

MojoBoost can promote contributors from public pull-request history without a
maintainer manually tracking activity. The automation is intentionally
transparent: the thresholds live in
`.github/governance-access-policy.json`, the evaluator lives in
`tools/governance_access.py`, and every workflow run writes its decision table
to the GitHub Actions job summary.

The scheduled workflow is safe when unconfigured. It evaluates the policy
daily in dry-run mode but cannot change access. Applying promotions requires a
repository variable, a GitHub App ID, and that app's private key.

## Policy

The initial ladder is:

| Permission | Merged PRs | Contribution days | Span | Reviews | Reliability contribution |
|---|---:|---:|---:|---:|---:|
| Triage | 2 | 2 | 14 days | 0 | no |
| Write (`push`) | 4 | 3 | 30 days | 0 | no |
| Maintain | 8 | 5 | 60 days | 2 | yes |

Every level also requires a GitHub account at least 90 days old. The evaluator
does not count bots, unmerged pull requests, excluded/reverted changes, tiny
changes, or documentation-only changes. Maintainers may label a valuable small
or documentation contribution `governance-credit`; the labels
`governance-no-credit` and `reverted` exclude a change.

A reliability contribution touches tests, validation, packaging, workflows,
CI tooling, or Python tests. This is only an abuse-resistant approximation of
judgment, so automatic promotion stops at Maintain. The report makes sustained
contributors visible, but an existing Admin must grant Admin explicitly.

Promotions are monotonic. The automation never grants Admin, demotes, or
removes somebody. Access can be changed or revoked manually at any time under
repository **Settings → Collaborators and teams**.

## One-time GitHub App setup

The repository's normal `GITHUB_TOKEN` can read pull requests but cannot grant
repository access. Create a GitHub App so the automation is not tied to a
personal access token.

1. Open the `mojoboost-ml` organization and go to **Settings → Developer
   settings → GitHub Apps → New GitHub App**. If Developer settings appears
   only under the personal account, create the app there and install it on the
   organization.
2. Name it `mojoboost-contributor-access`. The homepage may be the repository
   URL. Webhooks are not required; clear **Active** under Webhook.
3. Give it these repository permissions:
   - **Administration: Read and write** — required to invite collaborators and
     change their repository role.
   - **Metadata: Read-only** — GitHub includes this automatically.
   - **Pull requests: Read-only** — used to evaluate contribution history.
4. Give it **no organization permissions**. This automation grants access only
   to `mojoboost`; it does not invite organization members.
5. Install the app on `mojoboost-ml`, choosing **Only select repositories** and
   selecting only `mojoboost`.
6. Generate a private key on the app settings page and download the `.pem`
   file.
7. In the repository, open **Settings → Secrets and variables → Actions**:
   - Add variable `GOVERNANCE_APP_ID` with the app's numeric ID.
   - Add secret `GOVERNANCE_APP_PRIVATE_KEY` containing the complete `.pem`
     file, including its BEGIN and END lines.
   - Add variable `GOVERNANCE_AUTOMATION_ENABLED` with value `true`.
8. Open **Actions → Contributor access → Run workflow**, select `apply`, and
   inspect the first applied report.

The scheduled run executes daily at 08:17 UTC. Removing or setting
`GOVERNANCE_AUTOMATION_ENABLED=false` immediately returns it to dry-run mode.
Deleting the app installation revokes its authority.

## Two-factor authentication

The automation cannot determine whether an outside collaborator has enabled
2FA. GitHub enforces that at the organization boundary. First enable 2FA on
the owner's account under **Personal Settings → Password and authentication**,
save recovery codes outside the development laptop, and register a second
recovery method. Then enable the organization's 2FA requirement under
**Organization Settings → Authentication security**.

Enabling organization-wide 2FA can remove existing members and outside
collaborators who have not enabled it. Review the warning GitHub displays
before confirming. Do this before the project has many collaborators, not
after.

## Local dry run

Evaluation is read-only and uses only the standard library:

```sh
GH_TOKEN="$(gh auth token)" python3 tools/governance_access.py \
  --repository mojoboost-ml/mojoboost
```

Never pass `--apply` with a personal owner token merely to test the script.
Use the narrowly installed GitHub App for changes.

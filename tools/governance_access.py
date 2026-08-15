#!/usr/bin/env python3
"""Evaluate and optionally apply MojoBoost contributor access policy.

This deliberately counts merged pull requests rather than commits or lines.
It uses only Python's standard library and GitHub's REST API. The normal
GITHUB_TOKEN is sufficient for evaluation. Applying permissions requires a
GitHub App installation token with repository Administration: write.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


API = "https://api.github.com"
PERMISSION_RANK = {"none": 0, "read": 1, "triage": 2, "push": 3, "maintain": 4, "admin": 5}
CURRENT_PERMISSION_ALIASES = {"write": "push", "pull": "read"}


class GitHub:
    def __init__(self, token: str):
        self.token = token

    def request(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            API + path,
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "mojoboost-governance-access",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"GitHub API {method} {path}: {error.code}: {detail}") from error

    def pages(self, path: str) -> list[Any]:
        separator = "&" if "?" in path else "?"
        page = 1
        values: list[Any] = []
        while True:
            batch = self.request("GET", f"{path}{separator}per_page=100&page={page}")
            if not batch:
                return values
            values.extend(batch)
            if len(batch) < 100:
                return values
            page += 1


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def is_documentation_only(files: list[dict[str, Any]], extensions: set[str]) -> bool:
    return bool(files) and all(Path(item["filename"]).suffix.lower() in extensions for item in files)


def current_permission(api: GitHub, repository: str, login: str) -> str:
    try:
        result = api.request("GET", f"/repos/{repository}/collaborators/{login}/permission")
        permission = result.get("permission", "read")
        return CURRENT_PERMISSION_ALIASES.get(permission, permission)
    except RuntimeError as error:
        if ": 404:" in str(error):
            return "read"
        raise


def qualifies(stats: dict[str, Any], level: dict[str, Any], account_age: int) -> bool:
    return (
        account_age >= stats["minimum_account_age_days"]
        and stats["merged_pull_requests"] >= level["minimum_merged_pull_requests"]
        and stats["distinct_days"] >= level["minimum_distinct_contribution_days"]
        and stats["span_days"] >= level["minimum_span_days"]
        and stats["reviews"] >= level["minimum_reviews"]
        and (not level["require_reliability_contribution"] or stats["reliability_contributions"] > 0)
    )


def evaluate(api: GitHub, repository: str, policy: dict[str, Any]) -> list[dict[str, Any]]:
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=max(item["minimum_span_days"] for item in policy["levels"]) + 30)
    excluded = set(policy["exclude_labels"])
    credited = set(policy["credit_labels"])
    doc_extensions = set(policy["documentation_extensions"])
    reliability_paths = tuple(policy["reliability_paths"])
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)

    pulls = api.pages(f"/repos/{repository}/pulls?state=closed&sort=updated&direction=desc")
    for pull in pulls:
        merged_at = pull.get("merged_at")
        user = pull.get("user") or {}
        login = user.get("login", "")
        if not merged_at or not login or user.get("type") == "Bot" or login.endswith("[bot]"):
            continue
        if parse_time(merged_at) < cutoff:
            continue
        labels = {item["name"] for item in pull.get("labels", [])}
        if labels & excluded:
            continue
        files = api.pages(f"/repos/{repository}/pulls/{pull['number']}/files")
        changed = sum(item.get("additions", 0) + item.get("deletions", 0) for item in files)
        forced_credit = bool(labels & credited)
        if changed < policy["minimum_changed_lines"] and not forced_credit:
            continue
        if is_documentation_only(files, doc_extensions) and not forced_credit:
            continue
        by_author[login].append(
            {
                "number": pull["number"],
                "merged_at": merged_at,
                "reliability": any(item["filename"].startswith(reliability_paths) for item in files),
            }
        )

    reviews_by_author: dict[str, set[int]] = defaultdict(set)
    for pull in pulls:
        for review in api.pages(f"/repos/{repository}/pulls/{pull['number']}/reviews"):
            user = review.get("user") or {}
            login = user.get("login", "")
            if login and user.get("type") != "Bot" and review.get("state") in {"APPROVED", "CHANGES_REQUESTED"}:
                reviews_by_author[login].add(pull["number"])

    decisions: list[dict[str, Any]] = []
    for login, contributions in sorted(by_author.items()):
        user = api.request("GET", f"/users/{urllib.parse.quote(login)}")
        account_age = (now - parse_time(user["created_at"])).days
        dates = sorted(parse_time(item["merged_at"]).date() for item in contributions)
        stats = {
            "minimum_account_age_days": policy["minimum_account_age_days"],
            "merged_pull_requests": len(contributions),
            "distinct_days": len(set(dates)),
            "span_days": (dates[-1] - dates[0]).days + 1,
            "reviews": len(reviews_by_author[login]),
            "reliability_contributions": sum(item["reliability"] for item in contributions),
        }
        desired = "read"
        for level in policy["levels"]:
            if qualifies(stats, level, account_age):
                desired = level["permission"]
        current = current_permission(api, repository, login)
        decisions.append(
            {
                "login": login,
                "account_age_days": account_age,
                "current": current,
                "desired": desired,
                "promote": PERMISSION_RANK[desired] > PERMISSION_RANK.get(current, 0),
                **stats,
            }
        )
    return decisions


def markdown(repository: str, decisions: list[dict[str, Any]], applied: bool) -> str:
    lines = [
        "# Automated contributor access report",
        "",
        f"Repository: `{repository}`",
        f"Mode: **{'applied' if applied else 'dry run'}**",
        "",
        "| Contributor | Current | Eligible | PRs | Days | Span | Reviews | Reliability | Action |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in decisions:
        action = "promoted" if applied and item["promote"] else ("would promote" if item["promote"] else "none")
        lines.append(
            f"| @{item['login']} | {item['current']} | {item['desired']} | "
            f"{item['merged_pull_requests']} | {item['distinct_days']} | {item['span_days']} | "
            f"{item['reviews']} | {item['reliability_contributions']} | {action} |"
        )
    if not decisions:
        lines.append("| _No eligible contribution history yet_ | | | | | | | | none |")
    lines.extend(["", "Generated from `.github/governance-access-policy.json`.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, help="OWNER/REPO")
    parser.add_argument("--policy", default=".github/governance-access-policy.json")
    parser.add_argument("--output", default="governance-access-report.md")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("GH_TOKEN or GITHUB_TOKEN is required")
    policy = json.loads(Path(args.policy).read_text())
    api = GitHub(token)
    decisions = evaluate(api, args.repository, policy)
    if args.apply:
        for item in decisions:
            if item["promote"]:
                api.request(
                    "PUT",
                    f"/repos/{args.repository}/collaborators/{item['login']}",
                    {"permission": item["desired"]},
                )
    report = markdown(args.repository, decisions, args.apply)
    Path(args.output).write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

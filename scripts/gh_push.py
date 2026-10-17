#!/usr/bin/env python
"""SemiForge 三级降级推送脚本 (作者: 晨星, CJX0712)。

用法:
  python scripts/gh_push.py [--repo CJX0712/semiforge] [--branch main] [--tag v0.1.0]

三级降级:
  1. 若已配置 remote origin 且可推送 -> 直接 push (+ tag + release)。
  2. 若 remote 不存在 -> 用 `gh repo create` 建仓库并推。
  3. 若 gh 不可用 / 网络失败 -> 打印手动推送命令, 退出非零。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO_DEFAULT = "CJX0712/semiforge"
BRANCH_DEFAULT = "main"


def _run(cmd, check=True, capture=True):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=check,
                          capture_output=capture, text=True)


def _have_gh() -> bool:
    try:
        _run(["gh", "--version"], check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _existing_remote() -> str | None:
    try:
        r = _run(["git", "remote", "get-url", "origin"], check=True)
        return r.stdout.strip() or None
    except subprocess.CalledProcessError:
        return None


def _git_push(branch: str) -> bool:
    try:
        _run(["git", "push", "-u", "origin", branch], check=True)
        return True
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False


def _create_and_push(repo: str, branch: str) -> bool:
    if not _have_gh():
        return False
    try:
        _run(["gh", "repo", "create", repo, "--public", "--source", ".",
              "--remote", "origin", "--push", "--branch", branch], check=True)
        return True
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False


def _tag_and_release(tag: str, repo: str):
    if not tag:
        return
    try:
        _run(["git", "tag", "-f", tag], check=True)
        _run(["git", "push", "-f", "origin", tag], check=True)
        print(f"pushed tag {tag}")
    except subprocess.CalledProcessError as e:
        print("tag push failed:", e.stderr, file=sys.stderr)
        return
    if _have_gh():
        try:
            _run(["gh", "release", "create", tag, "--repo", repo,
                  "--title", f"SemiForge {tag}",
                  "--notes", "世界顶级半监督学习系统 (作者: 晨星). 见 README。",
                  "--latest"], check=True)
            print(f"release {tag} created")
        except subprocess.CalledProcessError as e:
            print("release create failed:", e.stderr, file=sys.stderr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SemiForge 三级降级推送")
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--branch", default=BRANCH_DEFAULT)
    p.add_argument("--tag", default="v0.1.0")
    args = p.parse_args(argv)

    # 确保是 git 仓库
    try:
        _run(["git", "rev-parse", "--is-inside-work-tree"], check=True)
    except subprocess.CalledProcessError:
        print("不是 git 仓库, 初始化中...")
        _run(["git", "init", "-b", args.branch], check=True)

    # 提交（若有待提交）
    status = _run(["git", "status", "--porcelain"], check=True).stdout.strip()
    if status:
        _run(["git", "add", "-A"], check=True)
        _run(["git", "commit", "-m", "SemiForge v0.1.0 (作者: 晨星)"], check=True)
    else:
        print("无待提交改动, 跳过 commit。")

    # 1) 已存在可用 remote -> 直接推
    remote = _existing_remote()
    if remote:
        print(f"已有 remote: {remote}, 直接推送...")
        if _git_push(args.branch):
            _tag_and_release(args.tag, args.repo)
            return 0

    # 2) 用 gh 建仓库并推
    print("尝试 gh repo create ...")
    if _create_and_push(args.repo, args.branch):
        _tag_and_release(args.tag, args.repo)
        return 0

    # 3) 失败 -> 打印手动命令
    print("\n[降级] 自动推送失败。请手动执行:")
    print(f"  git remote add origin https://github.com/{args.repo}.git")
    print(f"  git push -u origin {args.branch}")
    if args.tag:
        print(f"  git tag -f {args.tag} && git push -f origin {args.tag}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

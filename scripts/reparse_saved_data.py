"""Update from an adjacent local bundle and reparse existing data without HTTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def update_code(repo: Path, bundle: Path) -> None:
    if not (repo / ".git").exists():
        raise RuntimeError(f"Gitの保存先が見つかりません: {repo}")
    if not (repo / "data" / "keiba.sqlite3").is_file():
        raise RuntimeError(f"既存のデータベースが見つかりません: {repo / 'data'}")
    if not bundle.is_file():
        raise RuntimeError("同梱のkeiba-recovery.bundleがありません。ZIPをすべて展開してください。")
    if shutil.which("git") is None:
        raise RuntimeError("Gitが見つかりません。Gitのインストール後に実行してください。")
    # Absolute paths also work when the extracted package is in Downloads.
    for arguments in (("bundle", "verify", str(bundle.resolve())),
                      ("fetch", str(bundle.resolve()), "HEAD"),
                      ("merge", "--ff-only", "FETCH_HEAD")):
        subprocess.run(["git", "-C", str(repo), *arguments], check=True)


def reparse(repo: Path, job_id: str) -> dict:
    source = repo / "src"
    if not (source / "keiba_scraper" / "__main__.py").is_file():
        raise RuntimeError(f"更新後のプログラムが見つかりません: {source}")
    # Use the code just updated above even if the virtual environment contains
    # an old non-editable installation or points to another checkout.
    bootstrap = ("import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                 "runpy.run_module('keiba_scraper',run_name='__main__',alter_sys=True)")
    command = [sys.executable, "-X", "utf8", "-c", bootstrap, str(source),
               "--offline", "reparse", "--job-id", job_id]
    summary = None
    log_path = repo / "data" / "reparse_console.log"
    with log_path.open("a", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    try:
                        message = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(message, dict) and message.get("event") == "reparse_summary":
                        summary = message
                returncode = process.wait()
            except KeyboardInterrupt:
                # The console sends Ctrl+C to the child as well. Give it time
                # to close SQLite before the launcher exits.
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
                raise
    if summary is not None:
        report_path = repo / "data" / "reparse_report.json"
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    if returncode or summary is None or summary.get("complete") is not True:
        raise RuntimeError(f"補完は未完了です。保存済みデータは残ります。確認用ログ: {log_path}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "keiba")
    parser.add_argument("--bundle", type=Path,
                        default=Path(__file__).resolve().with_name("keiba-recovery.bundle"))
    parser.add_argument("--job-id", default="64bca8f8ad301623")
    args = parser.parse_args(argv)
    try:
        print("保存済みHTMLから項目を補完します。収集・別の補完処理は停止しておいてください。", flush=True)
        update_code(args.repo.resolve(), args.bundle.resolve())
        summary = reparse(args.repo.resolve(), args.job_id)
        print(f"補完が完了しました: {summary['done']}レース。SQLiteを更新しました。", flush=True)
        print("既に出力したCSVを使う場合は、補完後にexportで出力し直してください。", flush=True)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"停止しました: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("中断しました。同じファイルを実行すると残りから再開できます。", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

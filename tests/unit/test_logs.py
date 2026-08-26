import os
from datetime import datetime, timedelta

from local_llm.logs import current_log, new_run_log, prune_logs, tail_lines

T0 = datetime(2026, 8, 26, 10, 30, 5)


def test_new_run_log_creates_private_file_and_symlink(tmp_path):
    path = new_run_log(tmp_path / "logs", now=T0)
    assert path == tmp_path / "logs" / "llm-router.2026-08-26T10-30-05.log"
    assert path.is_file()
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    link = tmp_path / "logs" / "llm-router.log"
    assert link.is_symlink() and link.resolve() == path.resolve()
    assert current_log(tmp_path / "logs") == path.resolve()


def test_second_run_repoints_the_symlink(tmp_path):
    new_run_log(tmp_path, now=T0)
    second = new_run_log(tmp_path, now=T0 + timedelta(minutes=1))
    assert current_log(tmp_path) == second.resolve()


def test_current_log_is_none_without_a_run(tmp_path):
    assert current_log(tmp_path) is None


def test_prune_removes_old_files_but_never_the_current_one(tmp_path):
    old = new_run_log(tmp_path, now=T0 - timedelta(days=40))
    stamp = (T0 - timedelta(days=40)).timestamp()
    os.utime(old, (stamp, stamp))
    current = new_run_log(tmp_path, now=T0)
    os.utime(current, (stamp, stamp))  # even an "old" current file survives
    assert prune_logs(tmp_path, days=30, now=T0) == 1
    assert not old.exists() and current.exists()


def test_tail_lines(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("\n".join(f"line {i}" for i in range(50)) + "\n")
    assert tail_lines(f, 3) == ["line 47", "line 48", "line 49"]
    assert tail_lines(f, 100)[0] == "line 0"

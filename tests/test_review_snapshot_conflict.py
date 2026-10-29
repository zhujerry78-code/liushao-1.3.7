"""Append-only review writes reject stale editors without changing the schema."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading

import pytest

from liushao import reviews


@pytest.mark.parametrize("expected", [-1, True, False, 1.0, "0", [], {}])
def test_invalid_expected_revision_does_not_create_database(tmp_path, expected):
    database = tmp_path / "new" / "reviews.sqlite"
    with pytest.raises(ValueError, match="复核修订"):
        reviews.save_review(database, "record", "v1", "待补证", "draft",
                            expected_review_id=expected)
    assert not database.parent.exists()


def test_legacy_writes_and_scoped_revision_comparison_keep_return_contract(tmp_path):
    database = tmp_path / "reviews.sqlite"
    assert reviews.save_review(database, "record", "v1", "待补证", "legacy") is None
    first_id = reviews.history(database)[-1]["id"]
    reviews.save_review(database, "record", "v2", "待补证", "other model",
                        expected_review_id=0)
    reviews.save_review(database, "other", "v1", "待补证", "other record",
                        expected_review_id=0)
    assert reviews.save_review(database, "record", "v1", "确认正常", "new basis",
                               expected_review_id=first_id) is None
    before = reviews.history(database)
    with pytest.raises(RuntimeError, match="复核记录已被其他窗口修改"):
        reviews.save_review(database, "record", "v1", "待补证", "stale",
                            expected_review_id=first_id)
    assert reviews.history(database) == before


@pytest.mark.parametrize("existing", [False, True])
def test_two_writers_with_same_revision_append_exactly_once(tmp_path, existing):
    database = tmp_path / "reviews.sqlite"
    if existing:
        reviews.save_review(database, "record", "v1", "待补证", "initial")
    before = reviews.history(database)
    expected = before[-1]["id"] if before else 0
    ready = threading.Barrier(2)

    def writer(note):
        ready.wait(timeout=10)
        try:
            result = reviews.save_review(database, "record", "v1", "待补证", note,
                                         expected_review_id=expected)
        except RuntimeError as error:
            assert "复核记录已被其他窗口修改" in str(error)
            return "conflict"
        assert result is None
        return "saved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(writer, ["first writer", "second writer"]))
    assert sorted(results) == ["conflict", "saved"]
    after = reviews.history(database)
    assert after[:-1] == before
    assert len(after) == len(before) + 1


def test_write_checks_revision_under_write_lock_and_rolls_back_and_closes_on_failure(
    tmp_path, monkeypatch,
):
    database = tmp_path / "reviews.sqlite"
    reviews.save_review(database, "record", "v1", "待补证", "initial")
    before = reviews.history(database)
    connection = reviews.connect(database)
    statements = []
    connection.set_trace_callback(statements.append)

    class FailedCommit:
        rolled_back = False
        closed = False

        def execute(self, *args):
            return connection.execute(*args)

        def commit(self):
            raise sqlite3.OperationalError("injected commit failure")

        def rollback(self):
            self.rolled_back = True
            connection.rollback()

        def close(self):
            self.closed = True
            connection.close()

    proxy = FailedCommit()
    original_connect = reviews.connect
    monkeypatch.setattr(reviews, "connect", lambda path: proxy)
    with pytest.raises(sqlite3.OperationalError, match="injected commit failure"):
        reviews.save_review(database, "record", "v1", "待补证", "must roll back",
                            expected_review_id=before[-1]["id"])
    assert proxy.rolled_back and proxy.closed
    assert statements[0] == "BEGIN IMMEDIATE"
    assert statements[1].startswith("SELECT MAX(id)")
    assert any(statement.startswith("INSERT INTO reviews") for statement in statements)
    assert statements[-1] == "ROLLBACK"
    monkeypatch.setattr(reviews, "connect", original_connect)
    assert reviews.history(database) == before


def test_conflict_rolls_back_and_closes_connection(tmp_path, monkeypatch):
    database = tmp_path / "reviews.sqlite"
    reviews.save_review(database, "record", "v1", "待补证", "initial")
    original_connect = reviews.connect
    connections = []

    def observe_connect(path):
        connection = original_connect(path)
        connections.append(connection)
        return connection

    monkeypatch.setattr(reviews, "connect", observe_connect)
    with pytest.raises(RuntimeError, match="复核记录已被其他窗口修改"):
        reviews.save_review(database, "record", "v1", "待补证", "stale",
                            expected_review_id=0)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
    assert len(reviews.history(database)) == 1

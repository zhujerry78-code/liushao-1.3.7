"""Confirmation requires evidence without changing stored notes or review identity."""

import pytest

from liushao.reviews import current_reviews, history, save_review


BLANK_NOTES = ["", "   ", "\t\r\n", "\u3000\u00a0"]
CONFIRMED_STATUSES = ["确认异常", "确认正常"]


@pytest.mark.parametrize("status", CONFIRMED_STATUSES)
@pytest.mark.parametrize("note", BLANK_NOTES)
def test_confirmation_without_evidence_does_not_create_database(tmp_path, status, note):
    database = tmp_path / "not-created" / "reviews.sqlite"

    with pytest.raises(ValueError, match="必须填写判断依据"):
        save_review(database, "source:2", "v1", status, note)

    assert not database.parent.exists()


@pytest.mark.parametrize("status", CONFIRMED_STATUSES)
@pytest.mark.parametrize("note", BLANK_NOTES)
def test_rejected_confirmation_preserves_all_existing_history(tmp_path, status, note):
    database = tmp_path / "reviews.sqlite"
    save_review(database, "source:2", "v1", "待补证", "保留原结论")
    save_review(database, "source:3", "v1", "确认正常", "另一记录的判断依据")
    save_review(database, "source:2", "v2", "确认异常", "另一模型的判断依据")
    original_history = history(database)
    original_bytes = database.read_bytes()

    with pytest.raises(ValueError, match="必须填写判断依据"):
        save_review(database, "source:2", "v1", status, note)

    assert database.read_bytes() == original_bytes
    assert history(database) == original_history
    assert current_reviews(database, "v1")["source:2"]["status"] == "待补证"


@pytest.mark.parametrize("status", CONFIRMED_STATUSES)
def test_confirmation_preserves_evidence_and_appends_scoped_history(tmp_path, status):
    database = tmp_path / "reviews.sqlite"
    save_review(database, "source:2", "v1", "待补证", "原始记录")
    save_review(database, "source:3", "v1", "待复核", "另一记录")
    save_review(database, "source:2", "v2", "待补证", "另一模型")
    original_history = history(database)
    note = '  依据：核对特征原值与任务上下文。\n保留 "引号" 与换行。\t'

    save_review(database, "source:2", "v1", status, note)

    updated_history = history(database)
    assert updated_history[:-1] == original_history
    assert len(history(database, "source:2", "v1")) == 2
    assert current_reviews(database, "v1")["source:2"]["status"] == status
    assert current_reviews(database, "v1")["source:2"]["note"] == note
    assert current_reviews(database, "v1")["source:3"]["note"] == "另一记录"
    assert current_reviews(database, "v2")["source:2"]["note"] == "另一模型"


@pytest.mark.parametrize("status", ["待复核", "待补证"])
@pytest.mark.parametrize("note", BLANK_NOTES)
def test_unconfirmed_review_still_allows_blank_note(tmp_path, status, note):
    database = tmp_path / "reviews.sqlite"

    save_review(database, "source:2", "v1", status, note)

    rows = history(database)
    assert len(rows) == 1
    assert rows[0]["status"] == status
    assert rows[0]["note"] == note

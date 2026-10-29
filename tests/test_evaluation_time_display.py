"""Display-only polishing must preserve recorded evaluation time semantics."""
import pytest

from app_evaluation import evaluation_time


@pytest.mark.parametrize(('recorded', 'display'), [
    ('2026-10-16T09:40:10.905670+00:00', '2026年10月16日 09:40:10（UTC）'),
    ('2026-10-16T09:40:10Z', '2026年10月16日 09:40:10（UTC）'),
    ('2026-10-17T01:40:10+08:00', '2026年10月16日 17:40:10（UTC）'),
    ('2026-01-02', '2026年01月02日'),
    ('2026-01-02T03:04:05', '2026年01月02日 03:04:05（原记录未注明时区）'),
    ('2026-02-30', '时间格式待核对'),
    ('unreadable', '时间格式待核对'),
    (None, '时间格式待核对'),
])
def test_evaluation_time_is_readable_without_inventing_timezone(recorded, display):
    assert evaluation_time(recorded) == display

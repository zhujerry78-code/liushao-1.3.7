"""Session-owned structural preflight; never predicts or adapts feature values."""
from dataclasses import dataclass, field
import hashlib
import io
import csv
import pandas as pd
from liushao.data import read_csv_checked

MAX_BYTES = 50 * 1024 * 1024
MAX_ROWS = 50000
PREFLIGHT_VERSION = 1

@dataclass
class Preflight:
    digest: str
    model: str
    rows: int
    columns: int
    required: int
    _raw: object = field(repr=False)
    version: int = PREFLIGHT_VERSION

    def summary(self):
        return dict(rows=self.rows, columns=self.columns, required=self.required, extra=self.columns-self.required)

    def preview_copy(self):
        if self._raw is None: raise ValueError('预检输入已消费。')
        frame = self._raw.copy(deep=True)
        frame.attrs = {'source_rows': list(self._raw.attrs['source_rows'])}
        return frame

    def take(self, content, model):
        if self.version != PREFLIGHT_VERSION or self.model != model or self.digest != hashlib.sha256(content).hexdigest():
            raise ValueError('预检输入或模型身份已改变，请重新选择文件。')
        frame, self._raw = self._raw, None
        return frame

def preflight_input(content, bundle, model):
    if len(content) > MAX_BYTES: raise ValueError('文件超过 50 MiB，请分成较小的文件后再上传。')
    raw = read_csv_checked(io.BytesIO(content), max_rows=MAX_ROWS)
    if raw.empty: raise ValueError('文件需要包含 1 至 50,000 条记录。')
    features = bundle['config']['features']
    missing = [name for name in features if name not in raw]
    if missing: raise ValueError('缺少必需特征: ' + ', '.join(missing))
    return Preflight(hashlib.sha256(content).hexdigest(), model, len(raw), len(raw.columns), len(features), raw)

def problem_sample(features, kind):
    features = list(features)
    if not features: raise ValueError('缺少模型字段清单。')
    if kind == '缺少必需列':
        frame = pd.DataFrame([[1.] * (len(features)-1)] * 2, columns=features[1:])
    elif kind == '缺值与无效记录':
        frame = pd.DataFrame([[1.] * len(features) for _ in range(4)], columns=features).astype(object)
        frame.iloc[1, 0] = ''
        frame.iloc[2, :] = ''
        frame.iloc[3, 0] = 'bad'
    else: raise ValueError('未知格式练习。')
    return frame.to_csv(index=False, lineterminator='\n').encode('utf-8-sig')

class UserError(str):
    """Keep the legacy string error contract while providing structured UI text."""
    def __new__(cls, reason, action, detail):
        result = super().__new__(cls, detail)
        result.reason, result.action, result.detail = reason, action, detail
        return result

def present_error(error, stage='文件检查'):
    message = str(error)
    if isinstance(error, UnicodeDecodeError):
        reason, action = '文件编码暂不支持。', '请另存为 UTF-8 CSV 后重新选择。'
    elif '缺少必需特征:' in message:
        reason, action = '你的文件缺少必需数据列。', '请按格式说明补齐这些列：' + message.split(':', 1)[1].strip()
    elif '50 MiB' in message or '50,000' in message:
        reason, action = message, '请确认文件包含记录；较大文件按上限分批导入。'
    elif isinstance(error, csv.Error) or any(word in message for word in ('CSV 表头', '多行 CSV', '字段数不一致', '包含多行字段', '重复字段名')):
        reason, action = message, '请修正表头、分隔符或引号；每条记录占一行后重新选择。'
    elif isinstance(error, FileNotFoundError):
        reason, action = '所需文件不存在。', '请恢复完整配套包，或重新选择输入文件。'
    elif isinstance(error, PermissionError):
        reason, action = '本机目录暂时无法读写。', '请检查目录权限和占用情况，或将软件解压到自己的可写目录后重试。'
    elif isinstance(error, TimeoutError):
        reason, action = '本机操作等待超时。', '请先完成另一窗口中的操作，再稍后重试。'
    elif any(word in message for word in ('恢复清单', '保留文件', '恢复缓存', '清除未完成')):
        reason, action = message, '可重新上传原文件继续分析；如需删除旧副本，请使用“清除已保留文件”，并确认出现清除成功提示。'
    elif any(word in message for word in ('模型', '冻结身份', '兼容', '源码', '依赖')):
        reason, action = '分析程序或配套资源检查未通过。', '请使用配套启动器并恢复匹配文件，详细原因见技术详情。'
    else:
        reason, action = '本次操作未完成，暂不能确定原因。', '请保留原文件和技术详情，修正后重试。'
    return UserError(reason, action, stage + ' · ' + type(error).__name__ + ': ' + message)

"""Literal, Unicode-normalized match promotion within a filtered record table."""
import re
import unicodedata
import pandas as pd
from app_ui_state import STATUS_NAMES


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC',str(value)).casefold().split())


def promote_matches(table, keyword):
    query=normalized(keyword)
    mask=pd.Series(False,index=table.index)
    if not query: return table,mask
    match=re.fullmatch(r'(?:第\s*)?(\d+)(?:\s*行)?',query)
    if match:
        mask=table.source_row.eq(int(match.group(1)))
    else:
        for index,row in table.iterrows():
            visible=[f'第 {row.source_row} 行','提示异常' if row.rf_anomaly else '未提示异常',
                     STATUS_NAMES.get(row.review_status,row.review_status)]
            mask.at[index]=any(query in normalized(value) for value in visible)
    return pd.concat([table.loc[mask],table.loc[~mask]]),mask

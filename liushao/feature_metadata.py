"""CICFlowMeter field units and meanings, independent of presentation modules."""


def feature_definition(field, name=None):
    """Units follow the imported CICFlowMeter schema, never infer a new unit."""
    name = field if name is None else name
    if field == 'Protocol':
        return name, '协议编号', '例如 6=TCP、17=UDP、1=ICMP；编号大小不表示风险。'
    if 'Port' in field:
        return name, '端口号', '通信端口编号，需结合资产提供的服务核对。'
    if ('Duration' in field or 'IAT' in field or field.startswith(('Active ', 'Idle '))):
        return name, '微秒 (μs)', '1,000,000 微秒 = 1 秒；IAT 指相邻数据包的到达间隔。'
    if field == 'Down/Up Ratio':
        return name, '无量纲比值', '返回方向与发出方向的包数比；按数据源导出的值显示。'
    if field.endswith('Pkts/s'):
        return name, '包/秒', '该方向或整条流平均每秒的数据包数。'
    if field == 'Flow Byts/s' or 'Blk Rate' in field:
        return name, '字节/秒 (B/s)', '传输速率；数据块速率只统计采集器定义的批量传输。'
    if field == 'Pkt Len Var':
        return '数据包长度方差', '字节² (B²)', '包长的方差，不能按普通字节长度解读。'
    if field.startswith('Init ') and field.endswith('Win Byts'):
        return name, '字节 (B)', '连接起始窗口大小；负值需核对采集器的未采集/无效编码，不能理解为实际负字节。'
    if 'Flag' in field:
        return name, '次（标记计数）', '采集器统计的 TCP 标记计数；需核对协议与采集器版本。'
    if 'Pkts/b' in field:
        return name, '包/块', '采集器定义的批量传输中，每块的平均数据包数。'
    # TotLen Fwd/Bwd Pkts names describe total packet length, not packet count.
    if any(word in field for word in ('Len', 'Size', 'Byts')):
        return name, '字节 (B)', '1 字节 = 8 比特；发出/返回相对于流的起始方向。'
    if 'Pkts' in field:
        return name, '包', '发出/返回相对于流的起始方向，不自动等同于内网/外网。'
    return name, '数据源原始量纲（未确认）', '当前未提供可靠单位定义，请先核对数据字典。'

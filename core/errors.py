"""SemiForge 错误码定义。

作者: 晨星 (CJX0712)

错误体系: E1xx 配置 / E2xx 数据 / E3xx 模型 / E4xx 评测 / E5xx 系统。
所有自定义异常统一继承自 SemiForgeError。
"""

from __future__ import annotations


class SemiForgeError(Exception):
    """所有 SemiForge 异常的基类。"""

    code = "E000"
    message = "未分类错误"

    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(f"[{self.code}] {self.message}{(': ' + detail) if detail else ''}")


class ConfigError(SemiForgeError):
    code = "E100"
    message = "配置错误"


class DataError(SemiForgeError):
    code = "E200"
    message = "数据错误"


class ModelError(SemiForgeError):
    code = "E300"
    message = "模型错误"


class EvalError(SemiForgeError):
    code = "E400"
    message = "评测错误"


class SystemError(SemiForgeError):
    code = "E500"
    message = "系统错误"


class ConvergenceError(ModelError):
    code = "E310"
    message = "迭代未收敛"


class LeakageError(DataError):
    """检测伪标签泄漏到测试集时抛出（硬性不变量）。"""

    code = "E210"
    message = "数据泄漏: 测试集参与了伪标签生成"

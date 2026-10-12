"""会话记忆后台写队列：把 SQLite 落库从请求关键路径摘出，交给单线程 FIFO worker 执行

动机（首 Token 耗时优化）：prepare 的 user 消息写入、act 的 tool 结果写入、finalize 的助手回答
与安全事件写入都是**同步阻塞的磁盘 I/O**，串在「请求 → 首 token」的墙上路径上。本模块把这些
写入提交到**单一 worker 线程**的 FIFO 队列异步执行：
- 提交顺序即落库顺序 → 同一会话内 seq 单调、跨 user/tool/assistant 的时序不乱（正确性等价于同步串行）；
- 调用方拿到结果所需的信息（如 prepare 预先分配的 seq、finalize 预生成的 message_id）在提交前即确定，
  不依赖写入返回值，故无需阻塞等待；
- 单 worker 保证 SQLite 串行写（避免多线程抢写锁），读路径仍走各线程的独立连接。

关键路径需要「前面轮次的写入已落库」时（如下一轮 prepare 读 next_seq / 历史前必须含上一轮
助手回答），调 `drain()` 等待队列排空——稳态下队列为空、瞬时返回。
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MemoryWriteQueue:
    """单线程 FIFO 后台写队列（fire-and-forget，异常仅告警不落主链路）。"""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem-write")
        self._lock = threading.Lock()
        self._pending: deque = deque()
        self._not_empty = threading.Condition(self._lock)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """提交一次落库：立即返回，由 worker 线程按 FIFO 顺序执行。"""
        task = self._executor.submit(self._safe_run, fn, args, kwargs)
        with self._lock:
            self._pending.append(task)

    @staticmethod
    def _safe_run(fn: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as e:  # 后台写失败不得影响主链路，仅告警留痕
            logger.warning("会话记忆后台写入失败：%s", e)

    def drain(self, timeout: float | None = 10.0) -> bool:
        """等待已提交的全部写入完成；返回是否已在超时前排空。

        注意：仅排空「调用时刻之前」已提交的写入；worker 顺序执行，FIFO 保证时序。
        """
        from concurrent.futures import wait

        with self._lock:
            # 清理已完成任务，取出仍需等待的句柄
            while self._pending and self._pending[0].done():
                self._pending.popleft()
            tasks = list(self._pending)
        done, not_done = wait(tasks, timeout=timeout)
        with self._lock:
            for t in tasks:
                if t.done() and t in self._pending:
                    self._pending.remove(t)
        return not not_done

    def qsize(self) -> int:
        with self._lock:
            return sum(1 for t in self._pending if not t.done())


# 全局单例：所有会话记忆写入共用同一 worker，保证跨节点提交顺序
memory_writer = MemoryWriteQueue()

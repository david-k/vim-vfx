import vim # type: ignore
import time
from typing import Optional
from enum import Enum


class LogScope:
    log: "Logger"
    header_line_no: int
    start_time: float

    def __init__(self, log: "Logger"):
        self.log = log
        self.header_line_no = self.log.cur_line_no()

    def __enter__(self):
        self.log.increase_indent()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log.decrease_indent()
        duration = time.perf_counter() - self.start_time
        self.info_add(f" [{duration*1000:.4f}ms]")

    def info_add(self, msg: str):
        self.log.info_add(msg, self.header_line_no)


class LogLevel(Enum):
    NOTHING = 0
    INFO = 1
    DEBUG = 2

class Logger:
    buf_no: Optional[int] = None
    indent_level: int = 0
    log_level: LogLevel = LogLevel.NOTHING
    counter: int = 0
    first_line: bool = True

    def init_buf(self, buf_no: int, log_level: LogLevel):
        self.indent_level = 0
        self.buf_no = buf_no
        self.log_level = log_level
        self.counter = 1
        self.first_line = True

    def cur_line_no(self) -> int:
        if self.buf_no is not None:
            return len(vim.buffers[self.buf_no])

        return 0

    def info(self, msg: str):
        if self.log_level.value >= LogLevel.INFO.value:
            self._append_line(msg)

    def info_add(self, msg: str, line_no: int):
        if self.log_level.value >= LogLevel.INFO.value:
            self._append_to_line(msg, line_no)

    def debug(self, msg: str):
        if self.log_level.value >= LogLevel.DEBUG.value:
            self._append_line(msg)

    def scope(self, name: str) -> LogScope:
        self.info(name)
        return LogScope(self)

    def increase_indent(self):
        self.indent_level += 1

    def decrease_indent(self):
        assert self.indent_level > 0
        self.indent_level -= 1

    def _append_line(self, line: str):
        if self.buf_no is not None:
            counter_str = ""
            if self.indent_level == 0:
                counter_str = f"#{self.counter} "
                self.counter += 1

            log_str = (" "*self.indent_level*4) + counter_str + line
            buffer = vim.buffers[self.buf_no]
            if self.first_line:
                vim.buffers[self.buf_no][0] = log_str
                self.first_line = False
            else:
                buffer.append(log_str)

    def _append_to_line(self, line: str, line_no: int):
        if self.buf_no is not None:
            vim.buffers[self.buf_no][line_no-1] += line


LOG = Logger()

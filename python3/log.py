from abc import ABC, abstractmethod
import time
from typing import Optional
from enum import Enum


class Level(Enum):
    INFO = 1
    DEBUG = 2


class LogScope(ABC):
    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def info_add(self, msg: str):
        pass


class Logger:
    @abstractmethod
    def cur_line_no(self) -> int:
        pass

    @abstractmethod
    def info(self, msg: str):
        pass

    @abstractmethod
    def info_add(self, msg: str, line_no: int):
        pass

    @abstractmethod
    def debug(self, msg: str):
        pass

    @abstractmethod
    def scope(self, name: str) -> LogScope:
        pass

    @abstractmethod
    def increase_indent(self):
        pass

    @abstractmethod
    def decrease_indent(self):
        pass


class DefaultLogScope(LogScope):
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


class NullLogScope(LogScope):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def info_add(self, msg: str):
        pass




LOG: Optional[Logger] = None

def info(msg: str) -> None:
    if LOG:
        LOG.info(msg)


def debug(msg: str) -> None:
    if LOG:
        LOG.debug(msg)


def scope(name: str) -> LogScope:
    if LOG:
        return LOG.scope(name)

    return NullLogScope()

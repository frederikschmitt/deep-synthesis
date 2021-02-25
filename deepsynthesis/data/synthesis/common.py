import enum

class Semantics(enum.Enum):
    MEALY='mealy'
    MOORE='moore'

class Status(enum.Enum):
    REALIZABLE = 'realizable'
    UNREALIZABLE = 'unrealizable'
    TIMEOUT = 'timeout'
    ERROR = 'error'

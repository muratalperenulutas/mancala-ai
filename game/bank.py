from .base import Base

class Bank(Base):
    def __init__(self, index):
        super().__init__(index=index)
from base import Base

class Pit(Base):
    def __init__(self, index):
        super().__init__(index=index, stone_count=4)
        self.next=None


class A():
    LEN = 0

    def __init__(self):
        self.cls_me()

    @classmethod
    def cls_me(cls):
        cls.LEN = 100


a = A()
print(a.LEN)

class A():
    LEN = 0

    def __init__(self):
        self.cls_me()

    @classmethod
    def cls_me(cls):
        cls.LEN = 100


s = [1, 2, 3]
m = ["a", 'b', 'c']

s_z = zip(s, m)

print(dict(s_z))
s_zi = iter(s_z)
# print(1)
# next(s_zi)
# s_i = iter(s)
# print(s_i)
for _ in range(len(s)):
    print(next(s_zi))
# s_l = list(s_i)
# print(s_l)


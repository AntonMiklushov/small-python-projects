from QSystem import *


def main():
    deutsch()
    # print(*map(lambda x: str(x) + "\n", (o := s.measure())[0]), o[1])

def test():
    s = QSystem(4)
    s.apply(H, 0)
    s.apply(CX, 0, 1)
    s.apply(CX, 0, 2)
    s.apply(CX, 0, 3)
    print(s.get_state())

def deutsch():
    s = QSystem(2)
    s.apply(X, 1)
    s.apply(H, 0)
    s.apply(H, 1)
    s.apply(CX, 0, 1)
    s.apply(H, 0)
    print(s)

if __name__ == '__main__':
    main()

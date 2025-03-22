class _Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class _Segment:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def is_between(self, c):
        # Check if slope of a to c is the same as a to b ;
        # that is, when moving from a.x to c.x, c.y must be proportionally
        # increased than it takes to get from a.x to b.x .

        # Then, c.x must be between a.x and b.x, and c.y must be between a.y and b.y.
        # => c is after a and before b, or the opposite
        # that is, the absolute value of cmp(a, b) + cmp(b, c) is either 0 ( 1 + -1 )
        #    or 1 ( c == a or c == b)

        a, b = self.a, self.b             

        return ((b.x - a.x) * (c.y - a.y) == (c.x - a.x) * (b.y - a.y) and 
                abs(_cmp(a.x, c.x) + _cmp(b.x, c.x)) <= 1 and
                abs(_cmp(a.y, c.y) + _cmp(b.y, c.y)) <= 1)

def _cmp(x,y):
    return int((x>y)-(x<y))

def is_between(a,b,c):
    pa = _Point(a[0],a[1])
    pb = _Point(b[0],b[1])
    pc = _Point(c[0],c[1])

    return _Segment(pa,pb).is_between(pc)
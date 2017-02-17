
def in_rect(p,rect):
    '''check if point lies within rectangle '''
    x1,y1,x2,y2 = rect
    return (p[0] < x2 and
            p[0] > x1 and
            p[1] < y2 and
            p[1] > y1)

def intersects(point,obj):
    '''check if point is within (bounding box of) object '''
    bb = (obj.min_x-.05,obj.min_y-.05,obj.max_x+.05,obj.max_y+.05)# bounding box
    return in_rect(point, bb)


def rect2points(rect):
    '''give rect as list of corner points'''
    x1,y1,x2,y2 = rect
    return [(x1,y1),(x1,y2),(x2,y2),(x2,y1)]

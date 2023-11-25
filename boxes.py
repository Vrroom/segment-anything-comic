import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
from copy import deepcopy
from functools import reduce
from more_itertools import flatten
import matplotlib.pyplot as plt

def points_to_box (points_np) : 
    """ 
    Converts numpy points of shape [N, 2] to our box
    """ 
    x = points_np[:, 0].min()
    X = points_np[:, 0].max()
    y = points_np[:, 1].min()
    Y = points_np[:, 1].max()
    return BBox(x, y, X, Y, X - x, Y - y)

def pathBBox (path) : 
    """
    Get path bounding box
    
    This is to replace the path.bbox
    method from svgpathtools with better error handling. It
    also converts the bbox into a nice standard representation.
    """
    try : 
        x, X, y, Y = path.bbox()
    except Exception : 
        # An exception is likely when bbox is called on a degenerate path.
        x, X, y, Y = 0, 0, 0, 0
    return BBox(x, y, X, Y, X - x, Y - y)

def polylineAABB(doc, lines) :
    box = pathBBox(svg.Path(*lines))
    dbox = getDocBBox(doc)
    box = box / dbox
    y = 1 - box.Y
    Y = 1 - box.y
    newbox = BBox(box.x, y, box.X, Y, box.w, box.h)
    return newbox.translated(-0.5, -0.5).scaled(2)

def shape2obb (shape) : 
    rect = shape.minimum_rotated_rectangle.exterior.coords
    return corners2canonical(rect)

def polylineOBB (doc, lines) : 
    from vectorrvnn.geometry import normalizePts2Doc, equiDistantPointsOnPolyline
    pts = np.array(equiDistantPointsOnPolyline(doc, lines, normalize=True))
    pts += 0.002 * np.random.RandomState(0).randn(*pts.shape)
    ls = sg.LineString(list(zip(*pts)))
    return shape2obb(ls)

def getDocBBox (doc) :
    """
    Get the view box of the document

    Wrapper for the svgpathtools function for the same purpose.
    Gives a standard representation for bounding boxes.
    """
    x, y, w, h = doc.get_viewbox()
    return BBox(x, y, x + w, y + h, w, h)

def setDocBBox (doc, box) : 
    """ Set the view box of the document """
    string = ' '.join(map(str, box.tolist()))
    doc.set_viewbox(string)

def pathBBoxTooSmall (pathbox) : 
    """ Return true if the path occupies a very small portion of the graphic """
    return pathbox.normalized().area() <= 5e-4

def union(boxes) : 
    """ Compute the union of bounding boxes """
    return reduce(lambda x, y: x | y, boxes)

def intersection (boxes) : 
    """ Compute the intersection of bounding boxes """
    return reduce(lambda x, y: x & y, boxes)

def pathsetBox (t, ps) : 
    """ Compute the bounding box for a set of paths """
    return union([t.bbox[i] for i in ps])

def isclose(x, y, atol=1e-8, rtol=1e-5):
    """ 
    np.isclose replacement.

    Based on profiling evidence, it was found that the
    numpy equivalent is very slow. This is because np.isclose
    converts the numbers into internal representation and is 
    general enough to work on vectors. We need this function 
    to only work on numbers. Hence this faster alternative.
    """
    return abs(x - y) <= atol + rtol * abs(y)

class BBox : 
    """
    Standard representation for Axis Aligned Bounding Boxes.

    Different modules have different but equivalent representations
    for bounding boxes. Some represent them as the top left corner
    along with height and width while others represent them as the
    top left corner and the bottom left corner. This class unifies
    both representations so that we can write bounding box methods
    in a single consistent way.
    """
    def __init__ (self, x, y, X, Y, w, h) : 
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y
        self.w = w
        self.h = h
        self.assertConsistent()

    def assertConsistent (self) : 
        assert(self.X >= self.x)
        assert(self.Y >= self.y)
        assert(isclose(self.X - self.x, self.w))
        assert(isclose(self.Y - self.y, self.h))

    def iou (self, that) : 
        if (self | that).isDegenerate() \
                or (self & that).isDegenerate() :
            return 0
        intersection = (self & that).area()
        union = (self | that).area()
        return intersection / (union + 1e-5)

    def isDegenerate (self) : 
        return isclose(self.w, 0) and isclose(self.h, 0)

    def area (self) : 
        if self.isDegenerate() : 
            return 0
        else : 
            return self.w * self.h

    def center (self) : 
        return complex(
            (self.x + self.X) / 2, 
            (self.y + self.Y) / 2
        )

    def __eq__ (self, that) : 
        return isclose(self.x, that.x) \
                and isclose(self.X, that.X) \
                and isclose(self.y, that.y) \
                and isclose(self.Y, that.Y)

    def __mul__ (self, s) : 
        """ 
        Scale bounding box by post multiplying by a constant 

        A new box is made, scaled with respect to its origin.
        """
        return self.scaled(s, origin='center')

    def __or__ (self, that) : 
        """ Union of two boxes """
        x = min(self.x, that.x)
        y = min(self.y, that.y)
        X = max(self.X, that.X)
        Y = max(self.Y, that.Y)
        return BBox(x, y, X, Y, X - x, Y - y)

    def __and__ (self, that) : 
        """ Intersection of two boxes """
        x = max(self.x, that.x)
        y = max(self.y, that.y)
        X = min(self.X, that.X)
        Y = min(self.Y, that.Y)
        if y > Y: y = Y
        if x > X : x = X
        return BBox(x, y, X, Y, X - x, Y - y)

    def __contains__ (self, that) :
        return (self.x <= that.x <= that.X <= self.X \
                and self.y <= that.y <= that.Y <= self.Y) \
                and not self == that

    def __truediv__ (self, that): 
        """ View of the box normalized to the coordinates of that box """
        nx = (self.x - that.x) / that.w
        ny = (self.y - that.y) / that.h
        nX = (self.X - that.x) / that.w
        nY = (self.Y - that.y) / that.h
        nw = nX - nx
        nh = nY - ny
        return BBox(nx, ny, nX, nY, nw, nh)

    def normalized (self) : 
        """ Convert this box into the closest fitting square box """
        d = max(self.w, self.h)
        nx = self.x - (d - self.w) / 2
        ny = self.y - (d - self.h) / 2
        nX = nx + d
        nY = ny + d
        return BBox(nx, ny, nX, nY, d, d)

    def tolist (self, alternate=False) : 
        if not alternate : 
            return [self.x, self.y, self.w, self.h]
        else : 
            return [self.x, self.y, self.X, self.Y]

    def __repr__ (self) : 
        x = self.x
        y = self.y
        X = self.X
        Y = self.Y
        w = self.w
        h = self.h
        return f'BBox(x={x}, y={y}, X={X}, Y={Y}, w={w}, h={h})'

    def __xor__ (self, that) : 
        """ check whether boxes are disjoint """
        b1 = sg.box(self.x, self.y, self.X, self.Y)
        b2 = sg.box(that.x, that.y, that.X, that.Y)
        return b1.disjoint(b2)

    def rotated (self, degree, pt=None) : 
        if pt is None:  
            pt = sg.Point(0, 0)
        else : 
            pt = sg.Point(pt.real, pt.imag)
        x, y, X, Y = sa.rotate(self.toShapely(), degree, origin=pt).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def translated (self, tx, ty=0) : 
        x, y, X, Y = sa.translate(self.toShapely(), tx, ty).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def scaled (self, sx, sy=None, origin=sg.Point(0, 0)) : 
        if sy is None : 
            sy = sx
        x, y, X, Y = sa.scale(self.toShapely(), sx, sy, origin=origin).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def skewX (self, xs) : 
        x, y, X, Y = sa.skew(self.toShapely(), xs=xs).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def skewY (self, ys) : 
        x, y, X, Y = sa.skew(self.toShapely(), ys=ys).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def toShapely (self) : 
        return sg.Polygon([
            (self.x, self.y),
            (self.x, self.Y),
            (self.X, self.Y),
            (self.X, self.y)
        ])

    def draw_box (self, ax=None) :
        if ax is None:
            ax = plt
        ax.plot(
            [self.x, self.X, self.X, self.x, self.x],
            [self.y, self.y, self.Y, self.Y, self.y]
        )

def corners2canonical (corners) :
    """ 
    Canonical representation of an oriented 
    bounding box given as rectangles is a
    BBox centered at origin, a translation 
    vector and a rotation vector.
    """
    if not isinstance(corners, np.ndarray):
        corners = np.array(corners)
    a, b, c = corners[-2], corners[0], corners[1]
    center = (a + c) / 2
    u1 = ((c - b) / (1e-5 + np.linalg.norm(c - b))).reshape(2, 1)
    u2 = ((a - b) / (1e-5 + np.linalg.norm(a - b))).reshape(2, 1)
    corners = corners - center
    rot = np.hstack([u1, u2])
    corners = (corners @ rot)
    m, M = corners.min(0), corners.max(0)
    box = BBox(m[0], m[1], M[0], M[1], M[0] - m[0], M[1] - m[1])
    return OBB(box, center, rot)

def canonical2corners (obb) :
    from vectorrvnn.utils import argmin
    corners = np.array([
        [obb.x, obb.y],
        [obb.X, obb.y],
        [obb.X, obb.Y],
        [obb.x, obb.Y],
    ])
    corners = obb.C + (corners @ obb.rot.T)
    mi = argmin(corners.tolist())
    corners = np.vstack((corners[mi:], corners[:(mi + 1)]))
    return corners

class OBB (BBox) :

    def __init__ (self, box, center, rot) :
        super(OBB, self).__init__(
            box.x, box.y, box.X, 
            box.Y, box.w, box.h
        )
        self.C = center
        self.rot = rot

    def center (self) : 
        return complex(self.C[0], self.C[1])

    def __eq__ (self, that) : 
        corner1 = canonical2corners(self)
        corner2 = canonical2corners(that)
        return isclose(0, np.linalg.norm(corner1 - corner2), 1e-4)

    def __or__ (self, that) : 
        """ Union of two boxes """
        return self.toShapely().union(that.toShapely())

    def __and__ (self, that) : 
        return self.toShapely().intersection(that.toShapely())

    def __contains__ (self, that) :
        return (self.x <= that.x <= that.X <= self.X \
                and self.y <= that.y <= that.Y <= self.Y) \
                and not self == that

    def __truediv__ (self, that): 
        raise NotImplementedError

    def normalized (self) : 
        raise NotImplementedError

    def tolist (self, alternate=False) : 
        boxlist = super(OBB, self).tolist(alternate)
        rotlist = np.ravel(self.rot).tolist()
        cenlist = np.ravel(self.C).tolist()
        return [*boxlist, *rotlist, *cenlist]

    def __xor__ (self, that) : 
        """ check whether boxes are disjoint """
        p1 = sg.Polygon(self.corners)
        p2 = sg.Polygon(that.corners)
        return p1.disjoint(p2)
    
    def rotated (self, degree, pt=None) : 
        if pt is None:  
            pt = sg.Point(0, 0)
        else : 
            pt = sg.Point(pt.real, pt.imag)
        shape = sa.rotate(self.toShapely(), degree, origin=pt)
        return shape2obb(shape)

    def translated (self, tx, ty=0) : 
        shape = sa.translate(self.toShapely(), tx, ty)
        return shape2obb(shape)

    def scaled (self, sx, sy=None, origin=sg.Point(0, 0)) : 
        if sy is None : 
            sy = sx
        shape = sa.scale(self.toShapely(), sx, sy, origin=origin)
        return shape2obb(shape)

    def skewX (self, xs) : 
        shape = sa.skew(self.toShapely(), xs=xs)
        return shape2obb(shape)

    def skewY (self, ys) : 
        shape = sa.skew(self.toShapely(), ys=ys)
        return shape2obb(shape)

    def toShapely (self) : 
        return sg.Polygon(canonical2corners(self))


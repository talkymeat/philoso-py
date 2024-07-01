import numpy as np
# from icecream import ic
# import cmath
# from scipy.special import cbrt

class Cuboid:
    def __init__(self, min_x, min_y, min_z, V):
        """This class allows a set of constraints on the dimensions of a cuboid
        to be specified: the minimum lengths of each side $min_{x}$, $min_{y}$,
        $min_{z}$, and the maximum volume. Given a set of weights, $w_{x}$, $w_{y}$,
        $w_{z}$, and the further constrint that the cuboid must be of integer-
        valued side lengths, `Cuboid` will compute the side-lengths
            $$x = \\lfloor min_{x} + kw_{x} \\rfloor$$
            $$x = \\lfloor min_{y} + kw_{y} \\rfloor$$
            $$x = \\lfloor min_{z} + kw_{z} \\rfloor$$
        """
        self.mins = np.array([min_x, min_y, min_z]) 
        self.V = V

    def a_b_c_d(self, x, y, z):
        a = x * y * z
        b = (
            (self.mins[0] * y * z) +
            (x * self.mins[1] * z) +
            (x * y * self.mins[2])
        )
        c = (
            (self.mins[0] * self.mins[1] * z) +
            (x * self.mins[1] * self.mins[2]) +
            (self.mins[0] * y * self.mins[2])
        )
        d = np.prod(self.mins) - self.V
        return a, b, c, d
    
    def np_solve_cubic(self, a, b, c, d):
        return np.polynomial.Polynomial([d, c, b, a]).roots()

    #####################################################
    # XXX TODO It's worth trying again at a hand-rolled #
    # cubic solver, as this can speed up runtime: but   #
    # for now, let's go with numpy                      #
    #####################################################
    # def p(self, a, b, c):
    #     return ((3*a*c) - (b**2))/(3*(a**2))

    # def q(self, a, b, c, d):
    #     return ((2*(b**3)) - (9*a*b*c) + (27*d*(a**2)))/(27*(a**3))

    # def shift(self, a, b):
    #     return -b/(3*a)

    # def solve_sad_cubic(self, p, q):
    #     half_q = q/2.0
    #     det = cmath.sqrt(half_q**2 + (p/3.0)**3)
    #     S_cubed = -half_q-det
    #     S = cbrt(S_cubed.real) if S_cubed.imag == 0.0 else S_cubed**(1/3)
    #     T_cubed = -half_q+det
    #     T = cbrt(T_cubed.real) if S_cubed.imag == 0.0 else T_cubed**(1/3)
    #     return S + T, (-S-T)/2.0 + (1j * np.sqrt(3) * (S-T))/2, (-S-T)/2.0 - (1j * np.sqrt(3) * (S-T))/2 

    # def solve_cubic(self, a, b, c, d):
    #     _p = self.p(a, b, c)
    #     _q = self.q(a, b, c, d)
    #     shf = self.shift(a, b)
    #     return tuple([sol+shf for sol in self.solve_sad_cubic(_p, _q)])

    def ks_cubic(self, x, y, z):
        # return self.solve_cubic(*self.a_b_c_d(x, y, z))
        return self.np_solve_cubic(*self.a_b_c_d(x, y, z))
    
    def real_k(self, ws):
        ws = np.array(ws)
        if np.sum(ws!=0.0)==2:
            ks = self.ks_quadratic(*ws)
        elif np.sum(ws!=0.0)==3:
            ks = self.ks_cubic(*ws)
            # ks = self.np_cubic(*ws)
        else:
            ValueError(
                f'`ws` should have len of 2 or 3 nonzero values: {ws}'
            )
        ks = np.round(ks, 6)
        return np.max(ks[~np.iscomplex(ks)].real).item()

    
    def ks_quadratic(self, *ws):
        ws = np.array(ws)
        nonzero_mask = ws!=0.0
        if np.sum(nonzero_mask)!=2:
            raise ValueError (
                '`ks_quadratic` can only handle exactly two nonzero ' +
                f'weights, not: {ws}'
            )
        ws2 = ws[nonzero_mask]
        mins2 = self.mins[nonzero_mask]
        V2 = self.V/np.max(self.mins[~nonzero_mask])
        return self.solve_quadratic(*self.a_b_c(ws2, mins2, V2))

    def a_b_c(self, ws, mins, V):
        a = np.prod(ws)
        b = ws[0]*mins[1] + ws[1]*mins[0]
        c = np.prod(mins)-V
        return a, b, c
    
    def solve_quadratic(self, a, b, c):
        """Solves a quadratic, given input values a, b, c, using the quadratic formula
        
        >>> c = Cuboid(5,3,2,1000000)
        >>> c.solve_quadratic(2,6,4)
        (-1.0, -2.0)
        """
        det = np.sqrt(b**2 - 4*a*c)
        return (-b + det)/(2*a), (-b - det)/(2*a)
    
    def dims_from_weights(self, x, y, z):
        ws = np.array([x, y, z])
        zero_mask = ws==0.0
        zeroes = np.sum(zero_mask)
        if zeroes == 3:
            ws = np.ones(3)*.1
            zeroes = 0
        if zeroes == 2:
            min_dims = self.mins*zero_mask
            big_dims = np.array([self.V/np.prod(self.mins[zero_mask])]*3)*(~zero_mask)
            return min_dims+big_dims
        else:
            k = self.real_k(ws)
        return tuple([
            int(_min + dim*k) 
            for _min, dim 
            in zip(self.mins, ws)
        ])
    
    def __call__(self, wx, wy, wz):
        """Returns values for dimensions x, y, & z of a cuboid, such
        that all dimensions: 
        
        * are int-valued 
        * are greater than or equal to min_x, min_y, and min_z
        * the product of the sides is not greater than the maximum volume $V$
        * the ratios of the differences between the dimensions and their
          minimum values are approximately equal to the ratios of the
          weights, so:

          $$\\frac{x-min_x}{y-min_y} \\approx \\frac{w_x}{w_y}$$
          $$\\frac{y-min_y}{z-min_z} \\approx \\frac{w_y}{w_z}$$
          $$\\frac{z-min_z}{x-min_x} \\approx \\frac{w_z}{w_x}$$

          (approximate because of the `int` constraint)

        The values returned are the largest values compatible with 
        these constraints.

        >>> test_size = 1_000#_000 # <- uncomment the extra zeroes for a more rigorous test
        >>> cube_1_mil = Cuboid(5, 3, 2, 1_000_000)
        >>> cube_1_mil(0,0,0)
        (101, 99, 98)
        >>> cube_1_mil(1,1,1)
        (101, 99, 98)
        >>> cube_1_mil(0.95, 0.97, 0.98)
        (100, 100, 100)
        >>> for i in range(test_size):
        ...     ws = np.random.uniform(0, 1, 3)
        ...     if i%7 == 0:
        ...         ws[0] = 0.0
        ...     if i%7 == 3:
        ...         ws[1] = 0.0
        ...     if i%9 == 0:
        ...         ws[2] = 0.0
        ...     wx, wy, wz = ws
        ...     x, y, z = cube_1_mil(*ws)
        ...     assert x*y*z <= 1_000_000, f"({_}): x*y*z = {x*y*z}"
        ...     assert (x+1)*(y+1)*(z+1) > 1_000_000, f"({i}): (x+1)*(y+1)*(z+1) = {(x+1)*(y+1)*(z+1)}"
        ...     if y>3:
        ...         assert abs((x-5)/(y-3) - wx/wy)/max(x-5, y-3) < max((0 if x<6 else 1/(x-5)), 1/(y-3)), f"{i}, {ws}"
        ...         assert abs((z-2)/(y-3) - wz/wy)/max(z-2, y-3) < max((0 if z<3 else 1/(z-2)), 1/(y-3)), f"{i}, {ws}"
        ...     if z>2:
        ...         assert abs((x-5)/(z-2) - wx/wz)/max(x-5, z-2) < max((0 if x<6 else 1/(x-5)), 1/(z-2)), f"{i}, {ws}"
        """
        return self.dims_from_weights(wx, wy, wz)
    



def main():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    main()
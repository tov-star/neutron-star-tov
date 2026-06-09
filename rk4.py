"""
rk4.py  --  one classical 4th-order Runge-Kutta step (built from scratch).

"4th order" means the slope is sampled four times per step and matched to
the Taylor expansion through 4th order, so the *global* error scales as
h^4: halve the step size and the error drops by 2^4 = 16. (See convergence.py.)

We advance the coupled system (P, m) together using the same four stages.
`rhs(r, P, m, eos)` must return (dP/dr, dm/dr).
"""


def rk4_step(r, P, m, h, eos, rhs):
    k1P, k1m = rhs(r,           P,             m,             eos)
    k2P, k2m = rhs(r + 0.5*h,   P + 0.5*h*k1P, m + 0.5*h*k1m, eos)
    k3P, k3m = rhs(r + 0.5*h,   P + 0.5*h*k2P, m + 0.5*h*k2m, eos)
    k4P, k4m = rhs(r + h,       P + h*k3P,     m + h*k3m,     eos)

    P_new = P + (h / 6.0) * (k1P + 2.0*k2P + 2.0*k3P + k4P)
    m_new = m + (h / 6.0) * (k1m + 2.0*k2m + 2.0*k3m + k4m)
    return P_new, m_new

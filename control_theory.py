import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import IPython.display as display
import hashlib

class Plant:
    """Base class for a plant"""
    PLANT_DIM = 0
    CONTROL_DIM = 0
    OTYPE = 'Empty object'

    def __init__(self):
        pass

    def __str__(self):
        return self.OTYPE

    def __call__(self, x, u=np.zeros(CONTROL_DIM), t=0.):
        return np.empty(0, dtype='double')

class PendulumTorque(Plant):
    """Plant representation of a pendulum controlled by applied torque"""
    PLANT_DIM = 2
    CONTROL_DIM = 1
    OTYPE = 'Pendulum controlled by applied torque'

    def __init__(self, omega, nu):
        self.omega = omega
        self.nu = nu

    def __str__(self):
        return self.OTYPE + (": omega = {0:.2g}, nu = {1:.2g}").format(
            self.omega, self.nu)

    def __call__(self, x, u=np.zeros(CONTROL_DIM), t=0.):
        f = np.empty(2, dtype='double')
        self.omega * self.omega
        f[0] = x[1]
        f[1] = self.omega * self.omega * np.sin(x[0]) - self.nu * x[1] - u[0]
        return f

class PendulumHorizontalForce(Plant):
    """Plant representation of a pendulum controlled by applied horizontal
    force"""
    PLANT_DIM = 2
    CONTROL_DIM = 1
    OTYPE = 'Pendulum controlled by applied horizontal force'

    def __init__(self, omega, nu):
        self.omega = omega
        self.nu = nu

    def __str__(self):
        return self.OTYPE + (": omega = {0:.2g}, nu = {1:.2g}").format(
            self.omega, self.nu)

    def __call__(self, x, u=np.zeros(CONTROL_DIM), t=0.):
        f = np.empty(2, dtype='double')
        self.omega * self.omega
        f[0] = x[1]
        f[1] = (self.omega * self.omega * np.sin(x[0]) - self.nu * x[1]
                - u[0] * np.cos(x[0]))
        return f

class Control:
    """Base class for a control strategy"""
    PLANT_DIM = 0
    CONTROL_DIM = 0
    HIDDEN_DIM = 0
    CTYPE = 'Empty control'

    def __init__(self):
        pass

    def __str__(self):
        return self.CTYPE

    def __call__(self, x, v, t=0.):
        return np.empty(0, dtype='double')

    def control(self, x, v, t=0.):
        return np.empty(0, dtype='double')

class ZeroControl(Control):
    """Zero control strategy"""
    CTYPE = 'Zero control'
    
    def __init__(self, plant_dim=1, control_dim=1):
        self.PLANT_DIM = plant_dim
        self.CONTROL_DIM = control_dim

    def control(self, x, v, t=0.):
        return np.zeros(self.CONTROL_DIM, dtype='double')
    
class StateFeedbackControl(ZeroControl):
    """State feedback control strategy"""
    CTYPE = 'State feedback control'

    def __init__(self, plant_dim=1, control_dim=1, fun=None):
        ZeroControl.__init__(self, plant_dim, control_dim)
        if fun is not None:
            self.fun = fun
        else:
            self.fun = lambda x: np.zeros(self.CONTROL_DIM, dtype='double')
    
    def __str__(self):
        return self.CTYPE + str(self.fun)
    
    def control(self, x, v, t=0.):
        return self.fun(x)
    
class StateTimeFeedbackControl(ZeroControl):
    """State-time feedback control strategy"""
    CTYPE = 'State-time feedback control'

    def __init__(self, plant_dim=1, control_dim=1, fun=None):
        ZeroControl.__init__(self, plant_dim, control_dim)
        if fun is not None:
            self.fun = fun
        else:
            self.fun = lambda x, t: np.zeros(self.CONTROL_DIM, dtype='double')
    
    def __str__(self):
        return self.CTYPE + str(self.fun)
    
    def control(self, x, v, t=0.):
        return np.array(self.fun(x, t))

class LinearStateControl(ZeroControl):
    """Linear state feedback control strategy"""
    CTYPE = 'Linear state feedback control'

    def __init__(self, plant_dim=1, control_dim=1, k=None):
        ZeroControl.__init__(self, plant_dim, control_dim)
        if k is not None:
            self.k = np.atleast_2d(k)
        else:
            self.k = np.zeros((self.CONTROL_DIM, self.PLANT_DIM),
                              dtype='double')
    
    def __str__(self):
        return self.CTYPE + str(self.k)
    
    def control(self, x, v, t=0.):
        return self.k @ x

class StateDelayedFeedbackControl(ZeroControl):
    """State delayed feedback control strategy"""
    CTYPE = 'State delayed feedback control'

    def __init__(self, state_feedback_control, delay_times):
        ZeroControl.__init__(self, state_feedback_control.PLANT_DIM,
                             state_feedback_control.CONTROL_DIM)
        self.HIDDEN_DIM = state_feedback_control.CONTROL_DIM
        self.state_feedback_control = state_feedback_control
        self.inv_delay_times = 1 / np.array(delay_times)
    
    def __str__(self):
        return (self.CTYPE + str(self.state_feedback_control)
                + str(self.inv_delay_times))
    
    def control(self, x, v, t=0.):
        return v

    def __call__(self, x, v, t=0.):
        return ((self.state_feedback_control.control(x, None, t) - self.v)
                * self.inv_delay_times)

class DelayedStateFeedbackControl(ZeroControl):
    """Delayed state feedback control strategy"""
    CTYPE = 'Delayed state feedback control'

    def __init__(self, state_feedback_control, delay_times,
                 transform=lambda x, x_delayed: x_delayed):
        ZeroControl.__init__(self, state_feedback_control.PLANT_DIM,
                             state_feedback_control.CONTROL_DIM)
        self.HIDDEN_DIM = state_feedback_control.PLANT_DIM
        self.state_feedback_control = state_feedback_control
        self.inv_delay_times = 1 / np.array(delay_times)
        self.transform = transform
    
    def __str__(self):
        return (self.CTYPE + str(self.state_feedback_control)
                + str(self.inv_delay_times) + str(self.transform))
    
    def control(self, x, v, t=0.):
        return self.state_feedback_control.control(self.transform(x, v),
                                                   None, t)

    def __call__(self, x, v, t=0.):
        return (x - v) * self.inv_delay_times

class DelayedStateDelayedFeedbackControl(ZeroControl):
    """Delayed state delayed feedback control strategy"""
    CTYPE = 'Delayed state delayed feedback control'

    def __init__(self, state_feedback_control, state_delay_times,
                 feedback_delay_times,
                 transform=lambda x, x_delayed: x_delayed):
        ZeroControl.__init__(self, state_feedback_control.PLANT_DIM,
                             state_feedback_control.CONTROL_DIM)
        self.HIDDEN_DIM = (state_feedback_control.PLANT_DIM
                           + state_feedback_control.CONTROL_DIM)
        self.state_feedback_control = state_feedback_control
        self.inv_state_delay_times = 1 / np.array(state_delay_times)
        self.inv_feedback_delay_times = 1 / np.array(feedback_delay_times)
        self.transform = transform
    
    def __str__(self):
        return (self.CTYPE + str(self.state_feedback_control)
                + str(self.inv_state_delay_times) + str(self.transform)
                + str(self.inv_feedback_delay_times))
    
    def control(self, x, v, t=0.):
        return v[PLANT_DIM:]

    def __call__(self, x, v, t=0.):
        return np.hstack(((x - v) / self.state_delay_times,
                          (self.state_feedback_control
                           .control(self.transform(x, v[:PLANT_DIM]),
                                    None, t)
                           - v[PLANT_DIM:]) / self.feedback_delay_times))

d16 = 1 / 6

def rk(rhs, y0, t, N=1, stop_condition=None):
    """Solve system of ODEs using the Runge–Kutta method"""
    t = np.array(t)
    y0 = np.array(y0)
    
    if stop_condition is None:
        stop_condition = lambda y1, t: False

    y = np.empty((t.size,) + y0.shape, dtype='double')
    y[0] = y0
    if N > 1:
        yt = y0
        h = 0.
        for i in range(t.size - 1):
            h = (t[i + 1] - t[i]) / N
            yt = np.copy(y[i])
            for n in range(N):
                k1 = h * rhs(yt, t[i] + n * h)
                k2 = h * rhs(yt + 0.5 * k1, t[i] + (n + 0.5) * h)
                k3 = h * rhs(yt + 0.5 * k2, t[i] + (n + 0.5) * h)
                k4 = h * rhs(yt + k3, t[i] + (n + 1) * h)
                yt += d16 * (k1 + 2 * (k2 + k3) + k4)
            y[i + 1] = yt
            if stop_condition(y[i + 1], t[i + 1]):
                return t[:i + 2], y[:i + 2]
    else:
        th = 0.
        h = 0.
        for i in range(t.size - 1):
            th = 0.5 * (t[i] + t[i + 1])
            h = t[i + 1] - t[i]
            k1 = h * rhs(y[i], t[i])
            k2 = h * rhs(y[i] + 0.5 * k1, th)
            k3 = h * rhs(y[i] + 0.5 * k2, th)
            k4 = h * rhs(y[i] + k3, t[i + 1])
            y[i + 1] = y[i] + d16 * (k1 + 2 * (k2 + k3) + k4)
            if stop_condition(y[i + 1], t[i + 1]):
                return t[:i + 2], y[:i + 2]
    return t, y

def pcrhs(p, c):
    """Create function determining plant (with regulator) in the state
    space
    """
    n = p.PLANT_DIM
    return lambda y, t: np.hstack((p(y[:n], c.control(y[:n], y[n:], t), t),
                                   c(y[:n], y[n:], t)))

def control_output(c):
    """Create function calculating control
    """
    n = c.PLANT_DIM
    return lambda y, t: c.control(y[:n], y[n:], t)

def integrate(p, c, x0, v0, dt, T, N=1, method=rk, return_control=False,
              stop_condition=None):
    """Calculate temporal evolution of a plant for some initial state"""
    t = np.arange(0., T, dt)
    y0 = np.hstack((x0, v0))
    t, y = method(pcrhs(p, c), y0, t, N, stop_condition)
    if return_control:
        u = np.empty((t.size, c.CONTROL_DIM), dtype='double')
        for i in range(t.size):
            u[i] = control_output(c)(y[i], t[i])
        return t, y, u
    else:
        return t, y

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num,den)
        (num,den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$-\frac{%s}{%s}$' % (latex, den)
            elif num > 1:
                return r'$\frac{%s%s}{%s}$' % (num, latex,den)
            else:
                return r'$-\frac{%s%s}{%s}$' % (-num, latex,den)
    return _multiple_formatter

def animate_pendulum(t, state, phi_lims=(-np.pi, np.pi), cylinder_mode=True,
                     phase_portrait=None, resolution=(960, 540), dpi=108,
                     spacing=1, invsec=1.0, filename=None, codec=None,
                     progress=True):
    """Create Matplotlib animation of a pendulum"""

    with plt.style.context('default'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(resolution[0] / dpi,
                                                      resolution[1] / dpi),
                                       dpi=dpi)

        xmin1 = -1.4
        xmax1 = 1.4
        ymin1 = -2
        ymax1 = 2

        barline_level = -1.4
        hatch_level = -1.6
        
        xmin2 = phi_lims[0];
        xmax2 = phi_lims[1];
        ymin2 = 1.05 * np.amin(state[:, 1]) - 0.05 * np.amax(state[:, 1]);
        ymax2 = 1.05 * np.amax(state[:, 1]) - 0.05 * np.amin(state[:, 1]);

        ax1.axis([xmin1, xmax1, ymin1, ymax1])
        ax1.set_aspect('equal')
        ax1.set_axis_off()
        ax2.axis([xmin2, xmax2, ymin2, ymax2])
        if xmax2 - xmin2 >= np.pi / 2:
            if xmax2 - xmin2 < 4.9 * np.pi:
                minor_period = np.pi / 6
                major_period = np.pi / 2
            elif xmax2 - xmin2 < 9.9 * np.pi:
                minor_period = np.pi / 2
                major_period = np.pi
            else:
                minor_period = np.pi * int(np.ceil((xmax2 - xmin2) / (20 * pi)))
                major_period = 2 * minor_period
            ax2.xaxis.set_major_locator(plt.MultipleLocator(major_period))
            ax2.xaxis.set_minor_locator(plt.MultipleLocator(minor_period))
            ax2.xaxis.set_major_formatter(
                plt.FuncFormatter(multiple_formatter())
            )
        ax2.set_xlabel(r'Угол $\varphi$')
        ax2.set_ylabel(r'Угловая скорость $\dot\varphi$')

        if matplotlib.__version__ >= '3.6.0':
            fig.set_layout_engine('compressed')
        
        time_text = ax1.text(0.04, 0.9, '', transform=ax1.transAxes, zorder=1)

        barline, = ax1.plot([xmin1, xmax1], [barline_level, barline_level],
                            lw=3, color='C1', zorder=0)
        stand = patches.Polygon([[-0.3, -1.4], [0.3, -1.4], [0.1, 0.1],
                                 [-0.1, 0.1]],
                                color='C1', fill=True, ec=None, zorder=0)
        ax1.add_patch(stand)
        with plt.rc_context({'hatch.color': 'C1'}):
            ax1.fill_between([xmin1, xmax1], [hatch_level, hatch_level],
                             [barline_level, barline_level], hatch='//',
                             fc='white')

        main_line, = ax1.plot([], [], lw=5, color='C0', zorder=1)
        central_dot1, = ax1.plot([0.], [0.], color='C0', marker='o',
                                 markersize=10, zorder=1)
        central_dot2, = ax1.plot([0.], [0.], color='C1', marker='o',
                                 markersize=4, zorder=2)
        end_dot, = ax1.plot([], [], color='C0', marker='o', markersize=15,
                            zorder=1)


        if phase_portrait is not None:
            p = phase_portrait
            X, Y = np.meshgrid(np.linspace(xmin2, xmax2, 201),
                               np.linspace(ymin2, ymax2, 201))
            U = np.vectorize(lambda x, y:
                             p[0]([x, y],
                                  p[1].control([x, y],
                                               np.zeros(p[1].HIDDEN_DIM),
                                               t[0]),
                                  t[0])[0])(X, Y)
            V = np.vectorize(lambda x, y:
                             p[0]([x, y],
                                  p[1].control([x, y],
                                               np.zeros(p[1].HIDDEN_DIM),
                                               t[0]),
                                  t[0])[1])(X, Y)
            ax2.streamplot(X, Y, U, V, linewidth=0.3, color='grey', zorder=1)

        if cylinder_mode:
            min_shift = int(np.floor((np.amin(state[:, 0]) - phi_lims[1])
                                     / (2 * np.pi)))
            max_shift = int(np.ceil((np.amax(state[:, 0]) - phi_lims[0])
                                    / (2 * np.pi)))
        else:
            min_shift = 0
            max_shift = 0

        trajectories = [ax2.plot([], [], lw=1, color='C3', zorder=3)[0] for
                        i in range(min_shift, max_shift + 1)]
        trajectory_ends = [ax2.plot([], [], marker='o', markersize=4,
                                    color='C3', zorder=3)[0] for i in
                           range(min_shift, max_shift + 1)]
        
        def init():
            time_text.set_text('$t$ = {:2.1f}'.format(t[0]))
            main_line.set_data([0., np.sin(state[0, 0])],
                               [0., np.cos(state[0, 0])])
            end_dot.set_data([np.sin(state[0, 0])], [np.cos(state[0, 0])])
            for i in range(min_shift, max_shift + 1):
                trajectories[i - min_shift].set_data([state[0, 0]
                                                      - 2 * np.pi * i],
                                                     [state[0, 1]])
                trajectory_ends[i - min_shift].set_data([state[0, 0]
                                                         - 2 * np.pi * i],
                                                        [state[0, 1]])
            return ([time_text, main_line, end_dot] + trajectories
                    + trajectory_ends)

        def animate(i):
            l = i * spacing
            time_text.set_text('$t$ = {:2.1f}'.format(t[l]))
            main_line.set_data([0., np.sin(state[l, 0])],
                               [0., np.cos(state[l, 0])])
            end_dot.set_data([np.sin(state[l, 0])], [np.cos(state[l, 0])])
            for i in range(min_shift, max_shift + 1):
                trajectories[i - min_shift].set_data([state[:l, 0]
                                                      - 2 * np.pi * i],
                                                     [state[:l, 1]])
                trajectory_ends[i - min_shift].set_data([state[l, 0]
                                                         - 2 * np.pi * i],
                                                        [state[l, 1]])
            return ([time_text, main_line, end_dot] + trajectories
                    + trajectory_ends)

        anim = animation.FuncAnimation(fig, animate, frames=t.size // spacing,
                                       init_func=init, interval=(t[1] - t[0]) *
                                       1000 * spacing * invsec, blit=True,
                                       repeat=False)

        if filename is not None:
            if progress:
                print(filename + ', saving progress')
                pb = display.ProgressBar(len(t) // spacing)
                pb_iter = iter(pb)
                anim.save(filename, fps=30, codec=codec,
                          progress_callback=lambda i, n: next(pb_iter))
                display.clear_output(wait=True)
            else:
                anim.save(filename, fps=30, codec=codec)
        plt.close(fig)
    return anim

def axis_pi_ticks(axis, major=np.pi/2, minor=np.pi/6, denominator=2):
    axis.set_major_locator(plt.MultipleLocator(major))
    axis.set_minor_locator(plt.MultipleLocator(minor))
    axis.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator=
                                                                  denominator)))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle
from fast_control.controller_factory import simulate_sys

plt.style.use("seaborn-whitegrid")


def render(factory, controller, x_0, T, num_steps):
    """Render animation."""
    xs, _, _ = simulate_sys(factory, controller, x_0)
    dt = T / num_steps

    x_solution = np.zeros(len(xs))
    a_solution = xs[:, 0]
    b_solution = xs[:, 2]

    skip_frames = 5

    x_solution = x_solution[::skip_frames]
    a_solution = a_solution[::skip_frames]
    b_solution = b_solution[::skip_frames]

    frames = len(x_solution)

    j1_x = factory.l_1 * np.sin(a_solution) + x_solution
    j1_y = factory.l_1 * np.cos(a_solution)

    j2_x = factory.l_2 * np.sin(b_solution) + j1_x
    j2_y = factory.l_2 * np.cos(b_solution) + j1_y

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1))
    ax.set_aspect("equal")
    ax.grid()

    patch = ax.add_patch(
        Rectangle((0, 0), 0, 0, linewidth=1, edgecolor="k", facecolor="r")
    )

    (line,) = ax.plot([], [], "o-", lw=2)
    time_template = "time: %.1f s"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    cart_width = 0.15
    cart_height = 0.1

    def init():
        line.set_data([], [])
        time_text.set_text("")
        patch.set_xy((-cart_width / 2, -cart_height / 2))
        patch.set_width(cart_width)
        patch.set_height(cart_height)
        return line, time_text

    def animate(i):
        thisx = [x_solution[i], j1_x[i], j2_x[i]]
        thisy = [0, j1_y[i], j2_y[i]]

        line.set_data(thisx, thisy)
        now = i * skip_frames * dt
        time_text.set_text(time_template % now)

        patch.set_x(x_solution[i] - cart_width / 2)
        return line, time_text, patch

    ani = animation.FuncAnimation(
        fig, animate, frames=frames, interval=1, blit=True, init_func=init, repeat=False
    )
    plt.close(fig)
    return ani


def create_animation(factory, controllers, x_0=None, T=100, num_steps=1000):
    """Create animation,wrapper for render."""
    ani = render(
        factory,
        factory.system.qp_controller,
        "qp_controller",
        x_0,
        T=100,
        num_steps=1000,
    )
    ani.save("dip_qp.gif", writer=animation.PillowWriter(fps=24))
    for controller in controllers:
        ani = render(factory, controller, x_0, T, num_steps)
        ani.save(
            f"{controller.name}.gif", writer=animation.PillowWriter(fps=24)
        )  # %time

import ipywidgets
import numpy as np
from matplotlib import pyplot as plt

from IPython.display import display
from ensembler.potentials import OneD as pot
from ensembler.visualisation.plotConveyorBelt import plotEnsembler


def interactive_conveyor_belt(conveyorBelt=None, numsys: int = 8, nbins: int = 100, steps: int = 100):
    """
    This provides a nice widget for jupyter notebooks to play around with the conveyor belt.

    Parameters
    ----------
    conveyorBelt
    numsys
    nbins
    steps

    Returns
    -------

    """
    # if none given build cvb
    if isinstance(conveyorBelt, type(None)):
        lam = np.linspace(0, 1, nbins)
        ene = lam * np.sin(lam * np.pi) + lam**2
    else:

        (cvb_traj, systrajs) = conveyorBelt.get_trajs()

        if len(cvb_traj) == 0:
            raise IOError("Could not find any conveyor belt simulation in conveyorbelt traj. Please simulate first.")

        bins = np.zeros(nbins)
        dhdlbins = np.zeros(nbins)
        for i in systrajs:
            for j in range(systrajs[i].shape[0]):
                index = int(np.floor(systrajs[i].lam[j] / nbins))
                if index == nbins:
                    index = nbins - 1
                bins[index] += 1
                dhdlbins[index] += systrajs[i].dhdlam[j]
        for i, b in enumerate(bins):
            if b > 0:
                dhdlbins[i] /= b
        ene = np.cumsum(dhdlbins) / nbins

        lam = np.linspace(0, 1, nbins)

    def redraw(CapLam, M):
        plotEnsembler(lam, ene, CapLam=np.deg2rad(CapLam), M=M)

    # build layout and components

    player = ipywidgets.Play(value=0, min=0, max=360, step=1, description="rotate")
    capLam_slider = ipywidgets.IntSlider(
        value=0, min=0, max=360, step=1, orientation="vertical", description="Capital Lambda", continous_update=True
    )
    nReplicas_slider = ipywidgets.IntSlider(value=8, min=2, max=20, step=1, orientation="vertical", description="number of Replicas")

    ipywidgets.jslink((capLam_slider, "value"), (player, "value"))

    interactive_plot = ipywidgets.interactive_output(redraw, {"CapLam": capLam_slider, "M": nReplicas_slider})

    controls = ipywidgets.VBox([player, ipywidgets.HBox([capLam_slider, nReplicas_slider])])

    app = ipywidgets.AppLayout(
        header=None, left_sidebar=controls, center=interactive_plot, right_sidebar=None, footer=None, align_items="center"
    )

    return app

    """
    This provides a nice widget for jupyter notebooks to play around with the conveyor belt.

    Parameters
    ----------
    conveyorBelt
    numsys
    nbins
    steps

    Returns
    -------

    """
    # if none given build cvb
    if isinstance(conveyorBelt, type(None)):
        lam = np.linspace(0, 1, nbins)
        ene = lam * np.sin(lam * np.pi) + lam**2
    else:

        (cvb_traj, systrajs) = conveyorBelt.get_trajs()

        if len(cvb_traj) == 0:
            raise IOError("Could not find any conveyor belt simulation in conveyorbelt traj. Please simulate first.")

        bins = np.zeros(nbins)
        dhdlbins = np.zeros(nbins)
        for i in systrajs:
            for j in range(systrajs[i].shape[0]):
                index = int(np.floor(systrajs[i].lam[j] / nbins))
                if index == nbins:
                    index = nbins - 1
                bins[index] += 1
                dhdlbins[index] += systrajs[i].dhdlam[j]
        for i, b in enumerate(bins):
            if b > 0:
                dhdlbins[i] /= b
        ene = np.cumsum(dhdlbins) / nbins

        lam = np.linspace(0, 1, nbins)

    def redraw(CapLam, M):
        plotEnsembler(lam, ene, CapLam=np.deg2rad(CapLam), M=M)

    # build layout and components

    player = ipywidgets.Play(value=0, min=0, max=360, step=1, description="rotate")
    capLam_slider = ipywidgets.IntSlider(
        value=0, min=0, max=360, step=1, orientation="vertical", description="Capital Lambda", continous_update=True
    )
    nReplicas_slider = ipywidgets.IntSlider(value=8, min=2, max=20, step=1, orientation="vertical", description="number of Replicas")

    ipywidgets.jslink((capLam_slider, "value"), (player, "value"))

    interactive_plot = ipywidgets.interactive_output(redraw, {"CapLam": capLam_slider, "M": nReplicas_slider})

    controls = ipywidgets.VBox([player, ipywidgets.HBox([capLam_slider, nReplicas_slider])])

    app = ipywidgets.AppLayout(
        header=None, left_sidebar=controls, center=interactive_plot, right_sidebar=None, footer=None, align_items="center"
    )

    return app


class interactive_eds:
    """
    This provides a nice widget for jupyter notebooks to play around with the eds reference potetnial.

    """

    s = 1.35
    Eoffs = []
    V_is = []
    nstates = 2

    def __init__(self, nstates=2, s=100, Eoff=None, figsize=[12, 6]):
        self.nstates = nstates
        self.s = s
        plt.ion()
        if isinstance(Eoff, type(None)):
            self.Eoffs = [0 for state in range(self.nstates)]
        else:
            self.Eoffs = Eoff

        self.V_is = [pot.harmonicOscillatorPotential(x_shift=state * 4, k=10) for state in range(self.nstates)]
        self.eds_pot = pot.envelopedPotential(V_is=self.V_is, s=self.s, eoff=self.Eoffs)

        ##Parameters
        self.positions_state = np.arange(-4, 4 * self.nstates, 0.5)
        self.positions = np.arange(-4, 4 * self.nstates, 0.5)  # [x for x in  self.positions_state]

        energies = [V.ene(self.positions_state) for V in self.V_is]
        eds_enes = self.eds_pot.ene(self.positions)

        # plot
        if figsize is None:
            self.fig = plt.figure()  # dpi=300)
        else:
            self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot()
        ax.set_ylim([-50, 50])
        ax.set_xlim([-4, (4 * self.nstates)])
        ax.set_xlabel("x")
        ax.set_ylabel("V")

        ##init plots
        ax.plot(self.positions, energies[0], "C1", alpha=0.8, lw=5)
        ax.plot(self.positions, energies[1], "C2", alpha=0.8, lw=5)
        self.eds_line = ax.plot(self.positions, eds_enes, "C3", lw=2, zorder=100)[0]

        self.ax = ax

        # sliders
        ##states
        state_label = ipywidgets.Label("Number of States")
        state_slider = ipywidgets.IntSlider(value=2, min=2, max=10, step=1, orientation="horizontal")
        state_slider.observe(self.redraw_states, names="value")

        ##svals

        s_slider = ipywidgets.FloatSlider(value=100, min=0.1, max=101, step=1, orientation="horizontal", continous_update=True)
        self.s_label = ipywidgets.Label("smoothing Parameter:  " + str(np.log10(1 + (s_slider.value**1.5 / 1000))))
        s_slider.observe(self.redraw_s, names="value")

        player = ipywidgets.Play(value=100, min=0.1, max=100, step=1, description="s_values")

        ipywidgets.jslink((s_slider, "value"), (player, "value"))

        ##eoffs
        eoff_sliders = self.make_eoff_sliders(self.nstates)

        # listeners

        # layout
        state_slider = ipywidgets.HBox([state_label, state_slider])
        s_box = ipywidgets.HBox([self.s_label, player, s_slider])
        self.eoff_sliders_box = ipywidgets.HBox(eoff_sliders)

        controls = ipywidgets.VBox([state_slider, s_box, self.eoff_sliders_box])
        self.redraw_s({"new": 100})

        display(controls)
        self.fig.show()

    def redraw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def redraw_states(self, nstates_event):
        self.nstates = nstates_event["new"]

        # Plotting stuff

        for line in self.ax.lines:
            if line != self.eds_line:
                self.ax.lines.remove(line)
            del line

        self.positions_state = np.arange(-4, 4 * self.nstates, 0.5)
        self.positions = [[x] for x in self.positions_state]

        V_is = [pot.harmonicOscillatorPotential(x_shift=state * 4, k=10) for state in range(self.nstates)]
        for state_e in [V.ene(self.positions_state) for V in V_is]:
            self.ax.plot(self.positions_state, state_e, alpha=0.8, lw=5)
        self.ax.set_xlim([-4, (4 * self.nstates)])

        # pot
        self.Eoffs = self.eds_pot.Eoff
        if len(self.Eoffs) < self.nstates:
            for x in range(len(self.Eoffs), self.nstates):
                self.Eoffs.append(0)
        elif len(self.Eoffs) > self.nstates:
            self.Eoffs = self.Eoffs[: self.nstates]

        self.eoff_sliders_box.children = self.make_eoff_sliders(self.nstates)

        self.eds_pot = pot.envelopedPotential(V_is=V_is, s=np.log10(1 + (self.s**1.5 / 1000)), eoff=self.Eoffs)
        eds_enes = self.eds_pot.ene(self.positions)
        self.eds_line.set_data(self.positions, eds_enes)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def redraw_s(self, s):
        self.s = s["new"]
        self.eds_pot.s = np.log10(1 + (self.s**1.5 / 1000))
        self.s_label.value = "smoothing Parameter:  " + str(round(np.log10(1 + (self.s**1.5 / 1000)), 4))

        eds_enes = self.eds_pot.ene(self.positions)
        self.eds_line.set_data(self.positions, eds_enes)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def redraw_eoff(self, eoff_event):
        teoff = self.eds_pot.Eoff
        teoff[int(eoff_event["owner"].description.split("_")[-1])] = eoff_event["new"]
        self.Eoffs = teoff
        self.eds_pot.Eoff = teoff

        eds_enes = self.eds_pot.ene(self.positions)
        self.eds_line.set_data(self.positions, eds_enes)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def make_eoff_sliders(self, nstates):
        eoff_sliders = []
        for ind in range(nstates):
            tag = "Eoff_" + str(ind)
            eoffVi_slider = ipywidgets.FloatSlider(
                value=self.Eoffs[ind], min=-50, max=50, step=1, orientation="Vertical", description=tag, continous_update=True
            )
            eoff_sliders.append(eoffVi_slider)
            eoffVi_slider.observe(self.redraw_eoff, names="value")
        return eoff_sliders

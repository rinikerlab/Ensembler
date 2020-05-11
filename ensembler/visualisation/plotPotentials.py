import os, sys
import numpy as np
from matplotlib import pyplot as plt, colorbar

sys.path.append(os.path.dirname(__file__)+"/..")

from ensembler.potentials import OneD as pot, ND as nDPot
from ensembler.potentials._baseclasses import _potential1DCls, _perturbedPotentialNDCls

#UTIL FUNCTIONS
def significant_decimals(s:float)->float:
    significant_decimal=2
    if(s % 1 != 0):
        decimals = str(float(s)).split(".")[-1]
        for digit in decimals:
            if(digit == "0"):
                significant_decimal +=1
            else:
                return round(s, significant_decimal)
    else:
        return s

def plot_1DPotential(potential: _potential1DCls, positions:list,
                     x_range=None, y_range=None, title:str=None, ax=None):
    # generat Data
    energies = potential.ene(positions=positions)

    # is there already a figure?
    if (ax == None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = None

    # plot
    ax.plot(positions, energies)
    ax.set_xlim(min(x_range), max(x_range)) if (x_range!=None) else ax.set_xlim(min(positions), max(positions))
    ax.set_ylim(min(y_range), max(y_range)) if (y_range!=None) else ax.set_ylim(min(energies), max(energies))

    ax.set_xlabel('$x$')
    ax.set_ylabel('$Potential [kj]$')
    ax.set_title(title) if (title != None) else ax.set_title("Potential "+str(potential.name))

    if(ax != None):
        return fig, ax
    else:
        return ax
    pass

def plot_1DPotential_dhdpos(potential: _potential1DCls, positions:list,
                            x_range=None, y_range=None, title:str=None, ax=None):
    # generat Data
    energies = potential.dvdpos(positions=positions)

    # is there already a figure?
    if (ax == None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = None

    # plot
    ax.plot(positions, energies)
    ax.set_xlim(min(x_range), max(x_range)) if (x_range!=None) else ax.set_xlim(min(positions), max(positions))
    ax.set_ylim(min(y_range), max(y_range)) if (y_range!=None) else ax.set_ylim(min(energies), max(energies))

    ax.set_xlabel('$x$')
    ax.set_ylabel('$Potential [kj]$')
    ax.set_title(title) if (title != None) else ax.set_title("Potential "+str(potential.name))

    if(ax != None):
        return fig, ax
    else:
        return ax
    pass


def plot_1DPotential_Term(potential:_potential1DCls, positions: list,
                          x_range=None, y_range=None, title: str = None, ax=None):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plot_1DPotential(potential=potential, positions=positions, ax=axes[0], x_range=x_range, y_range=y_range, title="Pot")
    plot_1DPotential_dhdpos(potential=potential, positions=positions, ax=axes[1], x_range=x_range, y_range=y_range, title="dhdpos")
    fig.tight_layout()
    fig.suptitle(title) if(title!=None) else fig.suptitle("Potential "+str(potential.name))
    return fig, axes

def plot_1DPotential_Termoverlay(potential: _potential1DCls, positions:list,
                                 x_range=None, y_range=None, title: str = None, ax=None):
    #generate dat
    energies = potential.ene(positions=positions)
    dVdpos = potential.dhdpos(positions=positions)

    # is there already a figure?
    if (ax == None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = None

    ax.plot(positions, energies, label="V")
    ax.plot(positions, list(map(abs, dVdpos)), label="absdVdpos")
    ax.plot(positions, dVdpos, label="dVdpos")
    ax.set_xlim(min(x_range), max(x_range)) if (x_range!=None) else ax.set_xlim(min(positions), max(positions))
    ax.set_ylim(min(y_range), max(y_range)) if (y_range!=None) else ax.set_ylim(min([min(energies), min(dVdpos)]), max([max(energies), max(dVdpos)]))

    ax.ylabel("$Potential/kJ$")
    ax.xlabel("$x$")
    ax.legend()
    ax.set_title(title) if (title != None) else ax.set_title("Potential "+str(potential.__name__))

    if(ax != None):
        return fig, ax
    else:
        return ax

def plot_2DEnergy_landscape(potential1: _potential1DCls, potential2: _potential1DCls, positions1:list, positions2:list=None,
                            x_range=None, y_range=None, z_range=None, title:str=None, colbar:bool=False, ax=None, cmap:str="inferno"):
    #generat Data
    energy_map = []
    min_E, max_E = 0,0

    if(type(positions2)==type(None)):
        positions2 = positions1

    for pos in positions2:
        Va = potential2.ene(pos)[0]
        Vb = potential1.ene(positions1)
        Vtot = list(map(lambda x: x+Va, Vb))
        energy_map.append(Vtot)

        if(min(Vtot)<min_E):
            min_E = min(Vtot)
        if(max(Vtot)>max_E):
            max_E = max(Vtot)

    energy_map = np.array(energy_map)

    #is there already a figure?
    if(ax == None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colbar=True
    else:
        fig = None

    if(z_range==None):
        z_range = [min_E, max_E]

    #plot
    surf = ax.imshow(energy_map, cmap=cmap, interpolation="nearest",
                     origin='center', extent=[min(positions1), max(positions1), min(positions2), max(positions2)],  vmax=max(z_range), vmin=min(z_range), aspect="auto")

    if(colbar and fig != None):
        fig.colorbar(surf, aspect=5, label='Energy/kJ')

    if(x_range): ax.set_xlim(min(x_range), max(x_range))
    if(y_range): ax.set_ylim(min(y_range), max(y_range))

    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$')
    if(title): ax.set_title(title)
    return fig, ax, surf


def plot_2perturbedEnergy_landscape(potential:_perturbedPotentialNDCls, positions:list, lambdas:list,
                                    x_range=None, lam_range=None, title:str=None, colbar:bool=False, ax=None):

    energy_map_lin = []
    for y in lambdas:
        potential.set_lam(y)
        energy_map_lin.append(potential.ene(positions))
    energy_map_lin = np.array(energy_map_lin)

    if(ax == None):
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(111)
        colbar=True
    else:
        fig = None

    surf = ax.imshow(energy_map_lin, cmap="viridis", interpolation="nearest",
                     origin='center', extent=[min(positions), max(positions), min(lambdas), max(lambdas)],  vmax=100, vmin=0, aspect="auto")

    if(colbar):
        colorbar.Colorbar(ax, surf, label='Energy')

    if(x_range): ax.set_xlim(min(x_range), max(x_range))
    if(lam_range): ax.set_ylim(min(lam_range), max(lam_range))
    ax.set_xlabel('x')
    ax.set_ylabel('$\lambda$')
    if(title): ax.set_title(title)
    return fig, ax, surf

#show feature landscape per s
def envPot_differentS_overlay_min0_plot(eds_potential:nDPot.envelopedPotential, s_values:list, positions:list,
                                        y_range:tuple=None, hide_legend:bool=False, title:str=None, out_path:str=None):
    #generate energy values
    ys = []
    scale = 1 # 0.1
    for s in s_values:
        eds_potential.s=s
        enes = eds_potential.ene(positions)
        y_min =min(enes)
        y=list(map(lambda z: (z-y_min)*scale, enes))
        ys.append(y)

    #plotting
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    for s, y in reversed(list(zip(s_values, ys))):
        axes.plot(positions, y, label="s_"+str(significant_decimals(s)))

    if (y_range != None):
        axes.set_ylim(y_range)
    axes.set_xlim(min(positions),max(positions))

    #styling
    axes.set_ylabel("Vr/[kJ]")
    axes.set_xlabel("r")
    axes.set_title("different Vrs aligned at 0 with different s-values overlayed ")

    ##optionals
    if(not hide_legend): axes.legend()
    if(title):    fig.suptitle(title)
    if(out_path): fig.savefig(out_path)
    fig.show()

    return fig, axes

#show feature landscape per s
def envPot_differentS_overlay_plot(eds_potential:nDPot.envelopedPotential, s_values:list, positions:list,
                                   y_range:tuple=None, hide_legend:bool=False, title:str=None, out_path:str=None, axes=None):
    #generate energy values
    ys = []
    for s in s_values:
        eds_potential.s=s
        enes = eds_potential.ene(positions)
        ys.append(enes)

    #plotting
    if(axes == None):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    else:
        fig = None

    for s, y in reversed(list(zip(s_values, ys))):
        axes.plot(positions, y, label="s_"+str(significant_decimals(s)))

    #styling
    axes.set_xlim(min(positions),max(positions))
    axes.set_ylabel("Vr/[kJ]")
    axes.set_xlabel("r")
    if(title ==None):
        axes.set_title("different $V_{r}$s with different s-values overlayed ")
    else:
        axes.set_title(title)


    ##optionals
    if (y_range != None): axes.set_ylim(y_range)
    if(not hide_legend): axes.legend()
    if(title):    fig.suptitle(title)
    if(out_path): fig.savefig(out_path)
    if(fig!= None): fig.show()

    return fig, axes

def envPot_diffS_compare(eds_potential:nDPot.envelopedPotential, s_values:list, positions:list,
                         y_range:tuple=None,title:str=None, out_path:str=None):
    ##row/column ratio
    per_row =4
    n_rows = (len(s_values)//per_row)+1 if ((len(s_values)%per_row)>0) else (len(s_values)//per_row)

    ##plot
    fig, axes = plt.subplots(nrows=n_rows, ncols=per_row, figsize=(20,10))
    axes = [ax for ax_row in axes for ax in ax_row]

    for ax, s in zip( axes, s_values):
        eds_potential.s=s
        y=eds_potential.ene(positions)
        ax.plot(positions, y)

        #styling
        ax.set_xlim(min(positions), max(positions))
        ax.set_title("s_"+str(significant_decimals(s)))
        ax.set_ylabel("Vr/[kJ]")
        ax.set_xlabel("r")
        if (y_range != None): ax.set_ylim(y_range)

    ##optionals
    if(title):    fig.suptitle(title)
    if(out_path): fig.savefig(out_path)
    fig.show()
    return fig, axes

def plot_envelopedPotential_system(eds_potential:nDPot.envelopedPotential, positions:list, s_value:float=None, Eoffi:list=None,
                                   y_range:tuple=None,title:str=None, out_path:str=None):
    if(s_value!=None):
        eds_potential.s = s_value       #set new s
    if(Eoffi!=None):
        if(len(Eoffi) == len(eds_potential.V_is)):
            eds_potential.Eoff_i = Eoffi
        else:
            raise IOError("There are "+str(len(eds_potential.V_is))+" states and "+str(Eoffi)+", but the numbers have to be equal!")

    ##calc energies
    energy_Vr = eds_potential.ene(positions)
    energy_Vis = [state.ene(positions) for state in eds_potential.V_is]
    num_states = len(eds_potential.V_is)

    ##plot nicely
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes = [ax for ax_row in axes for ax in ax_row]
    y_values = energy_Vis + [energy_Vr]
    labels = ["state_"+str(ind) for ind in range(1,len(energy_Vis)+1)]+["refState"]

    for ax, y, label in zip(axes, y_values, labels):
        ax.plot(positions, y)
        ax.set_xlim(min(positions), max(positions))
        ax.set_ylim(y_range)
        ax.set_title(label)
        ax.set_ylabel("Vr/[kJ]")
        ax.set_xlabel("r_"+label)

    ##optionals
    if(title):    fig.suptitle(title)
    if(out_path): fig.savefig(out_path)
    fig.show()
    return fig, axes

def plot_envelopedPotential_2State_System(eds_potential: nDPot.envelopedPotential, positions:list, s_value:float=None, Eoffi:list=None,
                                          title:str=None, out_path:str=None, V_max:float=600, V_min:float=None):

    if(len(eds_potential.V_is)>2):
        raise IOError(__name__+" can only be used with two states in the potential!")

    if(s_value!=None):
        eds_potential.s = s_value

    if (Eoffi != None):
        if (len(Eoffi) == len(eds_potential.V_is)):
            eds_potential.Eoff_i = Eoffi
        else:
            raise IOError("There are " + str(len(eds_potential.V_is)) + " states and " + str(
                Eoffi) + ", but the numbers have to be equal!")

    #Calculate energies
    energy_Vr = eds_potential.ene(positions)
    energy_Vis = [state.ene(positions) for state in eds_potential.V_is]
    energy_map = []
    min_e = 0

    for x in positions:
        row = eds_potential.ene(list(map(lambda y:[[x], [y]], list(positions))))
        row_cut = list(map(lambda x:  V_max if(V_max != None and float(x) > V_max) else float(x), row))
        energy_map.append(row_cut)
        if(min(row)< min_e):
            min_e=min(row)

    if(V_min==None):
        V_min=min_e

    ##plot nicely
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes = [ax for ax_row in axes for ax in ax_row]
    y_values = energy_Vis + [energy_Vr]
    labels = ["State_" + str(ind) for ind in range(1, len(energy_Vis) + 1)] + ["State_R"]

    #plot the line potentials
    colors = ["steelblue", "orange", "forestgreen"]
    for ax, y, label,c in zip(axes, y_values, labels,colors):
        ax.plot(positions, y, c)
        ax.set_xlim(min(positions), max(positions))
        ax.set_ylim([V_min, V_max])
        ax.set_title("Potential $"+label+"$")
        ax.set_ylabel("$V/[kJ]$")
        ax.set_xlabel("$r_{ " + label+"} $")

    #plot phase space surface
    ax = axes[-1]
    surf = ax.imshow(energy_map, cmap="inferno", interpolation="nearest",
                    origin='center', extent=[min(positions), max(positions), min(positions), max(positions)],
                     vmax=V_max, vmin=V_min)
    ax.set_xlabel("$r_{"+labels[0]+"}$")
    ax.set_ylabel("$r_{"+labels[1]+"}$")
    ax.set_title("complete phaseSpace of $state_R$")
    #fig.colorbar(surf, aspect=5, label='Energy/kJ')

    ##optionals
    if(title):    fig.suptitle(title)
    if(out_path): fig.savefig(out_path)
    fig.show()
    return fig, axes


def envPot_diffS_2stateMap_compare(eds_potential: pot.envelopedPotential, s_values: list, positions: list,
                                   V_max: float = 500, V_min: float = None, title: str = None, out_path: str = None):
    ##row/column ratio
    per_row = 4
    n_rows = (len(s_values) // per_row) + 1 if ((len(s_values) % per_row) > 0) else (len(s_values) // per_row)

    ##plot
    fig, axes = plt.subplots(nrows=n_rows, ncols=per_row, figsize=(20, 10))
    axes = [ax for ax_row in axes for ax in ax_row]
    first = True

    for ax, s in zip(axes, s_values):
        eds_potential.s = s
        min_e = 0
        energy_map = []
        for x in positions:
            row = eds_potential.ene(list(map(lambda y: [[x], [y]], list(positions))))
            row_cut = list(map(lambda x: V_max if (V_max != None and float(x) > V_max) else float(x), row))
            energy_map.append(row_cut)
            if (min(row) < min_e):
                min_e = min(row)

        if (V_min == None and first):
            V_min = min_e
            first = False
            print("emin: ", min_e)

        # plot phase space surface
        surf = ax.imshow(energy_map, cmap="viridis", interpolation="nearest",
                         origin='center', extent=[min(positions), max(positions), min(positions), max(positions)],
                         vmax=V_max, vmin=V_min)
        ax.set_xlabel("$r_1$")
        ax.set_ylabel("$r_2$")
        ax.set_title("complete phaseSpace of $state_R$")
    fig.colorbar(surf, aspect=10, label='Energy/kJ')

    ##optionals
    if (title):    fig.suptitle(title)
    if (out_path): fig.savefig(out_path)
    fig.show()

    return fig, axes

if __name__ == "__main__":
    pass
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


def calc_lam(CapLam, i=0, numsys=8, w=0.1):
    ome = (CapLam + i * np.pi * 2.0 / numsys) % (2.0 * np.pi)
    if ome > np.pi:
        ome = 2.0 * np.pi - ome
    return ome / np.pi


def drawCirc(ax, radius, centX, centY, angle_, theta2_, lineWidth=3, color_="black"):
    # ========Line
    arc = patches.Arc(
        [centX, centY], radius, radius, angle=angle_, theta1=0, theta2=theta2_, capstyle="round", linestyle="-", lw=lineWidth, color=color_
    )
    ax.add_patch(arc)

    # ========Create the arrow head
    # endX=centX+(radius/2)*np.cos((theta2_+angle_)/180*np.pi) #Do trig to determine end position
    # endY=centY+(radius/2)*np.sin((theta2_+angle_)/180*np.pi)

    # ax.add_patch(                    #Create triangle as arrow head
    #     patches.RegularPolygon(
    #         (endX, endY),            # (x,y)
    #         3,                       # number of vertices
    #         radius/10,                # radius
    #         (angle_+theta2_)/180*np.pi,     # orientation
    #         color=color_
    #     )
    # )

    # ========Create the arrow head
    begX = centX + (radius / 2) * np.cos((angle_) / 180 * np.pi)  # Do trig to determine end position
    begY = centY + (radius / 2) * np.sin((angle_) / 180 * np.pi)

    ax.add_patch(  # Create triangle as arrow head
        patches.RegularPolygon(
            (begX, begY), 3, radius / 20, (180 + angle_) / 180 * np.pi, color=color_  # (x,y)  # number of vertices  # radius  # orientation
        )
    )
    ax.set_xlim([centX - radius, centY + radius]) and ax.set_ylim([centY - radius, centY + radius])


def drawFunicular(x, y, CapLam=0.1, M=2, drawArrows=False):
    pSize = 2.009
    goldRat = 1.618
    lineWidth = 1
    [path_effects.SimpleLineShadow(), path_effects.Normal()]
    fig = plt.figure(figsize=(pSize * goldRat, pSize))
    ax = fig.gca()
    fig.subplots_adjust(left=0.1, right=1.0 - 0.1, bottom=0.24, top=0.99)
    rx = 0.05
    ry = rx
    shifty = 0.75 / goldRat
    cvb_bot = np.zeros((90, 2))
    cvb_bot[:, 0] = np.linspace(calc_lam(CapLam, 1, numsys=2), 1.0 - rx, 90)
    cvb_bot[:, 1] = np.ones(90) * shifty
    cvb_top = np.zeros((90, 2))
    cvb_top[:, 0] = np.linspace(calc_lam(CapLam, 0, numsys=2), 1.0 - rx, 90)
    cvb_top[:, 1] = np.ones(90) * (shifty + 2.0 * ry)
    lamVals = x - x.min()
    lamVals /= lamVals.max()
    gVals = y - y.min()
    if gVals.max() != 0.0:
        gVals /= 2.0 * gVals.max() * goldRat
    else:
        gVals += 1 / (2.0 * goldRat)
    ax.plot(lamVals[2:], gVals[2:], "k", lw=lineWidth)

    l = CapLam
    numsys = M
    rotation = []
    y = []
    for i in range(M):
        if calc_lam(CapLam, i, numsys=M) > rx and calc_lam(CapLam, i, numsys=M) < (1.0 - rx):
            rotation.append(45)
            y.append(1.0)
        elif calc_lam(CapLam, i, numsys=M) < rx:
            alpha = np.arcsin((rx - calc_lam(CapLam, i, numsys=M)) / rx)
            rotation.append(45 - alpha / np.pi * 180.0)
            y.append(np.cos(alpha))
        else:
            alpha = np.arcsin((rx - (1 - calc_lam(CapLam, i, numsys=M))) / rx)
            rotation.append(45 - alpha / np.pi * 180.0)
            y.append(np.cos(alpha))
    shiftMarker = 0.02 * np.sqrt(2)

    ax.plot(cvb_bot[:, 0], cvb_bot[:, 1], "k", lw=lineWidth, zorder=1)
    ax.plot(cvb_top[:, 0], cvb_top[:, 1], "k", lw=lineWidth, zorder=1)
    #    ax.add_artist(patches.Arc((rx,shifty+ry), 2*rx, 2*ry, theta1=90, theta2=270, lw=lineWidth))
    ax.add_artist(patches.Arc((1.0 - rx, shifty + ry), 2 * rx, 2 * ry, theta1=270, theta2=90, lw=lineWidth))
    #    ax.add_artist(patches.Arc((rx,shifty+ry), 1.4*rx, 1.4*ry, lw=lineWidth))
    ax.add_artist(patches.Arc((1.0 - rx, shifty + ry), 1.4 * rx, 1.4 * ry, lw=lineWidth))
    # ax.annotate(r'$\Lambda=0$', xy=(-0.01, shifty+ry), xytext=(-0.05, shifty+ry), va='center', ha='right', arrowprops=dict(arrowstyle='-'))
    # ax.annotate(r'$\Lambda=\frac{\pi}{2}$', xy=(0.5,  shifty+2*ry+0.01), xytext=(0.5, shifty+2*ry+0.05), va='bottom', ha='center', arrowprops=dict(arrowstyle='-'))
    # ax.annotate(r'$\Lambda=\frac{3\pi}{2}$', xy=(0.5,  shifty-0.01), xytext=(0.5, shifty-0.05), va='top', ha='center', arrowprops=dict(arrowstyle='-'))
    # ax.annotate(r'$\Lambda=\pi$', xy=(1.01,  shifty+ry), xytext=(1.05, shifty+ry), va='center', ha='left', arrowprops=dict(arrowstyle='-'))
    # if np.fabs(rotation[0]-45)>0.0001:
    #     print(alpha)
    #     ax.annotate('Current state:\n$\Lambda={:.1f}$'.format(CapLam), xy=(calc_lam(CapLam, 0, numsys=M), shifty+ry+np.cos(alpha)*ry),
    #         xytext=(calc_lam(CapLam, 0, numsys=M)-np.sin(alpha)*1.5*rx, shifty+(1+np.cos(alpha)*2.5)*ry),
    #         arrowprops=dict(arrowstyle='<-', linewidth=3), va='center', ha='center', zorder=0)
    # else:
    #     ax.annotate('Current state:\n$\Lambda={:.1f}$'.format(CapLam), xy=(calc_lam(CapLam, 0, numsys=M), shifty+2.0*ry+shiftMarker),
    #         xytext=(calc_lam(CapLam, 0, numsys=M), shifty+3.5*ry),
    #         arrowprops=dict(arrowstyle='<-', linewidth=3), va='center', ha='center', zorder=0)
    # arrows in the conveyor belt
    #    drawCirc(ax,rx*0.8,rx,shifty+ry,45,270, color_='red')
    drawCirc(ax, rx * 0.8, 1.0 - rx, shifty + ry, 225, 270, lineWidth=lineWidth, color_="red")

    for i in range(int(M / 2)):
        x = calc_lam(CapLam, i, numsys=M) - np.sqrt(1 - y[i] ** 2) * shiftMarker
        ax.add_patch(  # Create triangle as arrow head
            patches.RegularPolygon(
                (x, shifty + ry + y[i] * ry),  # (x,y)
                4,  # number of vertices
                0.02,  # radius
                rotation[i] / 180.0 * np.pi,  # orientation
                color="red",
                zorder=10,
            )
        )
        ax.scatter(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker, s=30, marker="o", edgecolors="face", color="r", zorder=10)
        if drawArrows:
            ax.annotate(
                "",
                xy=(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker),
                xytext=(x + 0.1, gVals[np.abs(lamVals - x - 0.1).argmin()] + shiftMarker),
                arrowprops=dict(arrowstyle="<-", linewidth=lineWidth),
            )
        ax.plot([x, x], [gVals[np.abs(lamVals - x).argmin()], shifty + ry + y[i] * ry], color="0.8", lw=lineWidth, zorder=0)
    for i in range(int(M / 2)):
        x = calc_lam(CapLam, i + int(M / 2), numsys=M) - np.sqrt(1 - y[i] ** 2) * shiftMarker
        ax.add_patch(  # Create triangle as arrow head
            patches.RegularPolygon(
                (x, shifty),  # (x,y)
                4,  # number of vertices
                0.02,  # radius
                rotation[i] / 180.0 * np.pi,  # orientation
                color="red",
                zorder=10,
            )
        )
        ax.plot([x, x], [gVals[np.abs(lamVals - x).argmin()], shifty + (1.0 - y[i]) * ry], color="0.8", lw=lineWidth, zorder=0)
        ax.scatter(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker, s=30, marker="o", edgecolors="face", color="r", zorder=10)
        if drawArrows:
            ax.annotate(
                "",
                xy=(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker),
                xytext=(x - 0.1, gVals[np.abs(lamVals - x + 0.1).argmin()] + shiftMarker),
                arrowprops=dict(arrowstyle="<-", linewidth=lineWidth),
            )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1.2 / goldRat)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0\n(A)", r"$\sfrac{1}{2}$", "1\n(B)"])
    #    ax.text(lamVals[-1], gVals[-1]-0.05, 'Free energy profile', ha='right', va='top')
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_yticks([])
    ax.spines["left"].set_color("None")
    ax.spines["right"].set_color("None")
    ax.spines["top"].set_color("None")

    ax.annotate(
        "", xy=(0, 0), xytext=(0, 0.5 / goldRat), ha="center", va="bottom", arrowprops=dict(arrowstyle="<|-", facecolor="k", linewidth=1.5)
    )
    ax.text(-0.025, 0.25 / goldRat, "$G(\lambda)$", ha="right", va="center", fontsize=14)
    ax.text(1.025, 0.0, "$\lambda$", ha="left", va="center", fontsize=14)
    return fig


def plotEnsembler(x, y, CapLam=0.1, M=8, drawArrows=False):
    pSize = 6.027
    goldRat = 1.70
    lineWidth = 1
    [path_effects.SimpleLineShadow(), path_effects.Normal()]
    fig = plt.figure(figsize=(pSize * goldRat, pSize))
    ax = fig.gca()
    fig.subplots_adjust(left=0.1, right=1.0 - 0.1, bottom=0.25, top=0.964)
    rx = 0.05
    ry = rx
    shifty = 0.75 / goldRat
    cvb_bot = np.zeros((90, 2))
    cvb_bot[:, 0] = np.linspace(rx, 1.0 - rx, 90)
    cvb_bot[:, 1] = np.ones(90) * shifty
    cvb_top = np.zeros((90, 2))
    cvb_top[:, 0] = np.linspace(rx, 1.0 - rx, 90)
    cvb_top[:, 1] = np.ones(90) * (shifty + 2.0 * ry)
    lamVals = x - x.min()
    lamVals /= lamVals.max()
    gVals = y - y.min()
    if gVals.max() != 0.0:
        gVals /= 2.0 * gVals.max() * goldRat
    else:
        gVals += 1 / (2.0 * goldRat)
    ax.plot(lamVals[2:], gVals[2:], "k", lw=lineWidth)

    l = CapLam
    numsys = M
    rotation = []
    y = []
    # replicas boxes
    for i in range(M):
        if calc_lam(CapLam, i, numsys=M) > rx and calc_lam(CapLam, i, numsys=M) < (1.0 - rx):
            rotation.append(45)
            y.append(1.0)
        elif calc_lam(CapLam, i, numsys=M) < rx:
            alpha = np.arcsin((rx - calc_lam(CapLam, i, numsys=M)) / rx)
            if (CapLam + i * 2 * np.pi / float(M)) % (2.0 * np.pi) < np.pi:
                rotation.append(45 + alpha / np.pi * 180.0)
            else:
                rotation.append(45 - alpha / np.pi * 180.0)
            y.append(np.cos(alpha))
        else:
            alpha = np.arcsin((rx - (1 - calc_lam(CapLam, i, numsys=M))) / rx)
            if (CapLam + i * 2 * np.pi / float(M)) % (2.0 * np.pi) < np.pi:
                rotation.append(45 - alpha / np.pi * 180.0)
            else:
                rotation.append(45 + alpha / np.pi * 180.0)
            y.append(np.cos(alpha))
    shiftMarker = 0.02 * np.sqrt(2)

    # funicular
    ax.plot(cvb_bot[:, 0], cvb_bot[:, 1], "k", lw=lineWidth)
    ax.plot(cvb_top[:, 0], cvb_top[:, 1], "k", lw=lineWidth)
    ax.add_artist(patches.Arc((rx, shifty + ry), 2 * rx, 2 * ry, theta1=90, theta2=270, lw=lineWidth))
    ax.add_artist(patches.Arc((1.0 - rx, shifty + ry), 2 * rx, 2 * ry, theta1=270, theta2=90, lw=lineWidth))
    ax.add_artist(patches.Arc((rx, shifty + ry), 1.4 * rx, 1.4 * ry, lw=lineWidth))
    ax.add_artist(patches.Arc((1.0 - rx, shifty + ry), 1.4 * rx, 1.4 * ry, lw=lineWidth))
    ax.annotate(
        r"$\Lambda=0$",
        xy=(0.01, shifty + ry),
        xytext=(-0.05, shifty + ry),
        va="center",
        ha="right",
        fontsize="small",
        arrowprops=dict(arrowstyle="-", linewidth=lineWidth),
    )
    ax.annotate(
        r"$\Lambda=\frac{\pi}{2}$",
        xy=(0.5, shifty + 2 * ry + 0.01),
        xytext=(0.5, shifty + 2 * ry + 0.05),
        va="bottom",
        ha="center",
        fontsize="small",
        arrowprops=dict(arrowstyle="-", linewidth=lineWidth),
    )
    ax.annotate(
        r"$\Lambda=\frac{3\pi}{2}$",
        xy=(0.5, shifty - 0.01),
        xytext=(0.5, shifty - 0.05),
        va="top",
        ha="center",
        fontsize="small",
        arrowprops=dict(arrowstyle="-", linewidth=lineWidth),
    )
    ax.annotate(
        r"$\Lambda=\pi$",
        xy=(0.99, shifty + ry),
        xytext=(1.05, shifty + ry),
        va="center",
        ha="left",
        fontsize="small",
        arrowprops=dict(arrowstyle="-", linewidth=lineWidth),
    )
    if drawArrows:
        if np.fabs(rotation[0] - 45) > 0.0001:
            ax.annotate(
                "Current state:\n$\Lambda={:.1f}$".format(CapLam),
                xy=(calc_lam(CapLam, 0, numsys=M), shifty + ry + np.cos(alpha) * (ry + shiftMarker)),
                xytext=(calc_lam(CapLam, 0, numsys=M) - np.sin(alpha) * 2 * rx, shifty + (1 + np.cos(alpha) * 5) * ry),
                fontsize="small",
                arrowprops=dict(arrowstyle="<-", linewidth=1.0, shrinkA=0.0),
                va="top",
                ha="center",
                zorder=0,
                bbox=dict(pad=-0.1, lw=0.0, color="None"),
            )
        else:
            ax.annotate(
                "Current state:\n$\Lambda={:.1f}$".format(CapLam),
                xy=(calc_lam(CapLam, 0, numsys=M), shifty + 2.0 * ry + shiftMarker),
                xytext=(calc_lam(CapLam, 0, numsys=M), shifty + 6 * ry),
                arrowprops=dict(arrowstyle="<-", linewidth=1.0, shrinkA=0.0),
                fontsize="small",
                va="top",
                ha="center",
                zorder=0,
                bbox=dict(pad=-0.1, lw=0.0, color="None"),
            )

    # arrows in the conveyor belt
    drawCirc(ax, rx * 0.8, rx, shifty + ry, 45, 270, lineWidth=1.0, color_="red")
    drawCirc(ax, rx * 0.8, 1.0 - rx, shifty + ry, 225, 270, lineWidth=1.0, color_="red")

    # lines and markers for Epot
    for i in range(M):
        x = calc_lam(CapLam, i, numsys=M)
        if x < rx:
            rx -= np.sqrt(1 - y[i] ** 2) * shiftMarker
        elif x > 1 - rx:
            rx += np.sqrt(1 - y[i] ** 2) * shiftMarker
        if (CapLam + i * 2 * np.pi / float(M)) % (2.0 * np.pi) < np.pi:
            ax.add_patch(  # Create triangle as arrow head
                patches.RegularPolygon(
                    (x, shifty + ry + y[i] * ry + y[i] * shiftMarker),  # (x,y)
                    4,  # number of vertices
                    0.02,  # radius
                    rotation[i] / 180.0 * np.pi,  # orientation
                    color="red",
                    zorder=10,
                )
            )
            ax.scatter(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker, s=30, marker="o", edgecolors="face", color="r", zorder=10)
            if drawArrows:
                ax.annotate(
                    "",
                    xy=(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker),
                    xytext=(x + 0.1, gVals[np.abs(lamVals - x - 0.1).argmin()] + shiftMarker),
                    arrowprops=dict(arrowstyle="<-", linewidth=lineWidth),
                )
            ax.plot(
                [x, x],
                [gVals[np.abs(lamVals - x).argmin()], shifty + ry + y[i] * ry + y[i] * shiftMarker],
                color="0.8",
                lw=lineWidth,
                zorder=0,
            )
        else:
            ax.add_patch(  # Create triangle as arrow head
                patches.RegularPolygon(
                    (x, shifty - y[i] * shiftMarker),  # (x,y)
                    4,  # number of vertices
                    0.02,  # radius
                    rotation[i] / 180.0 * np.pi,  # orientation
                    color="red",
                    zorder=10,
                )
            )
            ax.plot(
                [x, x],
                [gVals[np.abs(lamVals - x).argmin()], shifty + (1.0 - y[i]) * ry - y[i] * shiftMarker],
                color="0.8",
                lw=lineWidth,
                zorder=0,
            )
            ax.scatter(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker, s=30, marker="o", edgecolors="face", color="r", zorder=10)
            if drawArrows:
                ax.annotate(
                    "",
                    xy=(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker),
                    xytext=(x - 0.1, gVals[np.abs(lamVals - x + 0.1).argmin()] + shiftMarker),
                    arrowprops=dict(arrowstyle="<-", linewidth=lineWidth),
                )

    # formatting
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1.2 / goldRat)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0\n(A)", r"$\sfrac{1}{2}$", "1\n(B)"])
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_yticks([])
    ax.spines["left"].set_color("None")
    ax.spines["right"].set_color("None")
    ax.spines["top"].set_color("None")
    ax.set_title("Conveyor Belt over simulated Free Energy Landscape")

    ax.annotate(
        "", xy=(0, 0), xytext=(0, 0.5 / goldRat), ha="center", va="bottom", arrowprops=dict(arrowstyle="<|-", facecolor="k", linewidth=1.5)
    )
    ax.text(-0.025, 0.25 / goldRat, "$G(\lambda)$", ha="right", va="center", fontsize=14)
    ax.text(1.025, 0.0, "$\lambda$", ha="left", va="center", fontsize=14)
    return fig, ax


def updateEnsembler(x, y, ax, CapLam=0.1, M=8, drawArrows=False):
    pSize = 6.027
    goldRat = 1.70
    lineWidth = 1
    [path_effects.SimpleLineShadow(), path_effects.Normal()]

    rx = 0.05
    ry = rx
    shifty = 0.75 / goldRat
    cvb_bot = np.zeros((90, 2))
    cvb_bot[:, 0] = np.linspace(rx, 1.0 - rx, 90)
    cvb_bot[:, 1] = np.ones(90) * shifty
    cvb_top = np.zeros((90, 2))
    cvb_top[:, 0] = np.linspace(rx, 1.0 - rx, 90)
    cvb_top[:, 1] = np.ones(90) * (shifty + 2.0 * ry)
    lamVals = x - x.min()
    lamVals /= lamVals.max()
    gVals = y - y.min()
    if gVals.max() != 0.0:
        gVals /= 2.0 * gVals.max() * goldRat
    else:
        gVals += 1 / (2.0 * goldRat)
    ax.plot(lamVals[2:], gVals[2:], "k", lw=lineWidth)

    rotation = []
    y = []
    # buildBox
    for i in range(M):
        if calc_lam(CapLam, i, numsys=M) > rx and calc_lam(CapLam, i, numsys=M) < (1.0 - rx):
            rotation.append(45)
            y.append(1.0)
        elif calc_lam(CapLam, i, numsys=M) < rx:
            alpha = np.arcsin((rx - calc_lam(CapLam, i, numsys=M)) / rx)
            if (CapLam + i * 2 * np.pi / float(M)) % (2.0 * np.pi) < np.pi:
                rotation.append(45 + alpha / np.pi * 180.0)
            else:
                rotation.append(45 - alpha / np.pi * 180.0)
            y.append(np.cos(alpha))
        else:
            alpha = np.arcsin((rx - (1 - calc_lam(CapLam, i, numsys=M))) / rx)
            if (CapLam + i * 2 * np.pi / float(M)) % (2.0 * np.pi) < np.pi:
                rotation.append(45 - alpha / np.pi * 180.0)
            else:
                rotation.append(45 + alpha / np.pi * 180.0)
            y.append(np.cos(alpha))
    shiftMarker = 0.02 * np.sqrt(2)

    # arrow
    if drawArrows:
        if np.fabs(rotation[0] - 45) > 0.0001:
            ax.annotate(
                "Current state:\n$\Lambda={:.1f}$".format(CapLam),
                xy=(calc_lam(CapLam, 0, numsys=M), shifty + ry + np.cos(alpha) * (ry + shiftMarker)),
                xytext=(calc_lam(CapLam, 0, numsys=M) - np.sin(alpha) * 2 * rx, shifty + (1 + np.cos(alpha) * 5) * ry),
                fontsize="small",
                arrowprops=dict(arrowstyle="<-", linewidth=1.0, shrinkA=0.0),
                va="top",
                ha="center",
                zorder=0,
                bbox=dict(pad=-0.1, lw=0.0, color="None"),
            )
        else:
            ax.annotate(
                "Current state:\n$\Lambda={:.1f}$".format(CapLam),
                xy=(calc_lam(CapLam, 0, numsys=M), shifty + 2.0 * ry + shiftMarker),
                xytext=(calc_lam(CapLam, 0, numsys=M), shifty + 6 * ry),
                arrowprops=dict(arrowstyle="<-", linewidth=1.0, shrinkA=0.0),
                fontsize="small",
                va="top",
                ha="center",
                zorder=0,
                bbox=dict(pad=-0.1, lw=0.0, color="None"),
            )

    # arrows in the conveyor belt
    drawCirc(ax, rx * 0.8, rx, shifty + ry, 45, 270, lineWidth=1.0, color_="red")
    drawCirc(ax, rx * 0.8, 1.0 - rx, shifty + ry, 225, 270, lineWidth=1.0, color_="red")

    # box arrow?
    for i in range(M):
        x = calc_lam(CapLam, i, numsys=M)
        if x < rx:
            rx -= np.sqrt(1 - y[i] ** 2) * shiftMarker
        elif x > 1 - rx:
            rx += np.sqrt(1 - y[i] ** 2) * shiftMarker
        if (CapLam + i * 2 * np.pi / float(M)) % (2.0 * np.pi) < np.pi:
            ax.add_patch(  # Create triangle as arrow head
                patches.RegularPolygon(
                    (x, shifty + ry + y[i] * ry + y[i] * shiftMarker),  # (x,y)
                    4,  # number of vertices
                    0.02,  # radius
                    rotation[i] / 180.0 * np.pi,  # orientation
                    color="red",
                    zorder=10,
                )
            )
            ax.scatter(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker, s=30, marker="o", edgecolors="face", color="r", zorder=10)
            if drawArrows:
                ax.annotate(
                    "",
                    xy=(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker),
                    xytext=(x + 0.1, gVals[np.abs(lamVals - x - 0.1).argmin()] + shiftMarker),
                    arrowprops=dict(arrowstyle="<-", linewidth=lineWidth),
                )
            ax.plot(
                [x, x],
                [gVals[np.abs(lamVals - x).argmin()], shifty + ry + y[i] * ry + y[i] * shiftMarker],
                color="0.8",
                lw=lineWidth,
                zorder=0,
            )
        else:
            ax.add_patch(  # Create triangle as arrow head
                patches.RegularPolygon(
                    (x, shifty - y[i] * shiftMarker),  # (x,y)
                    4,  # number of vertices
                    0.02,  # radius
                    rotation[i] / 180.0 * np.pi,  # orientation
                    color="red",
                    zorder=10,
                )
            )
            ax.plot(
                [x, x],
                [gVals[np.abs(lamVals - x).argmin()], shifty + (1.0 - y[i]) * ry - y[i] * shiftMarker],
                color="0.8",
                lw=lineWidth,
                zorder=0,
            )
            ax.scatter(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker, s=30, marker="o", edgecolors="face", color="r", zorder=10)
            if drawArrows:
                ax.annotate(
                    "",
                    xy=(x, gVals[np.abs(lamVals - x).argmin()] + shiftMarker),
                    xytext=(x - 0.1, gVals[np.abs(lamVals - x + 0.1).argmin()] + shiftMarker),
                    arrowprops=dict(arrowstyle="<-", linewidth=lineWidth),
                )

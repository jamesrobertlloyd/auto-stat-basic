#!/usr/bin/python

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

import matplotlib as mpl
import matplotlib.pyplot as plt
import make_graphs as gr
import seaborn.apionly as sns


def best_jsd(means, covars, weights):
    """Go through all the axis pairs and return a list of the JS divergences"""
    # limits are at 95% - speeds up integral
    maxes = np.amax(means + np.sqrt(covars * 5.991), axis=0)
    mins = np.amin(means - np.sqrt(covars * 5.991), axis=0)
    jsds = []
    axes = np.identity(len(means[0]))
    weights = np.array(weights)

    for j in range(len(means[0])):
        for i in range(j):
            ax = axes[:, [i, j]]

            # create a list of scipy 2D gaussian distributions
            rv_list = []
            for mean, cov in zip(means, covars):
                mean = np.dot(mean, ax)
                cov = np.dot(cov, ax)
                # cov = np.diagflat(cov) # making covariance 2D appears to be unnecessary
                rv_list.append(multivariate_normal(mean=mean, cov=cov))

                # find sum of entropy * weight over all gaussians
                # N.B. scipy uses entropy with base e
            term2 = np.dot([rv.entropy() for rv in rv_list], weights.T)
            #print [rv.entropy() for rv in rv_list]

            def sumpdf(x, y):  # integrand for differential entropy integral
                fofx = np.dot([rv2.pdf([x, y]) for rv2 in rv_list], weights.T)
                if fofx == 0:  # hack to avoid log(0)
                    return 0
                else:
                    return -1 * fofx * np.log(fofx)

            # limits for integral
            axmin = np.dot(mins, ax)
            axmax = np.dot(maxes, ax)
            # integrate to get the differential entropy of the set of gaussians
            # term1_full = dblquad(sumpdf, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
            term1 = dblquad(sumpdf, axmin[1], axmax[1], lambda x: axmin[0], lambda x: axmax[0])

            jsd = (term1[0] - term2) * np.log(2)  # convert to log2 from ln
            if jsd < 0:
                jsd = 0
            jsds.append((jsd, i, j))
            # print (i, j),  jsd, term1[0], term1_full[0], term2

    jsds.sort(reverse=True)
    # print "JSDs :", jsds
    return jsds


def js_graphs(means, covars, weights, outdir, data, labels, columns, ind2order):
    """Draw the three best projections as found by JS divergence"""
    all_jsd = best_jsd(means, covars, weights)
    best_projs = all_jsd[0:4]

    column_labels = range(means.shape[1])
    row_labels = range(means.shape[1])
    griddata = np.zeros((means.shape[1], means.shape[1]))
    for jsd, i, j in all_jsd:
        griddata[i, j] = griddata[j, i] = jsd
    fig, ax = plt.subplots()
    cmap = sns.light_palette(color=gr.palette(0), as_cmap=True)
    heatmap = ax.pcolor(griddata, cmap=cmap, vmin=0)  # vmax=1)
    fig.colorbar(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(griddata.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(griddata.shape[1])+0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_title('JSD for different projections')
    ax.set_xlabel('Column index for feature')
    ax.set_ylabel('Column index for feature')

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    savefile = outdir + '/figures/js_heatmap.png'
    fig.savefig(savefile)
    plt.close(fig)

    pal = gr.cb_palette
    colours = [pal(x) for x in labels]
    with mpl.rc_context({'figure.autolayout': False, 'font.size': 20}):

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9))

        classes = np.unique(labels)
        recs = []
        for i in classes:
            recs.append(mpl.patches.Rectangle((0, 0), 1, 1, fc=pal(i)))
        plt.figlegend(recs, classes, loc='center right')

        for ax, proj in zip([(0, 0), (0, 1), (1, 0), (1, 1)], best_projs):
            ax = axes[ax]
            x = proj[1]
            y = proj[2]

            #ax.set_title(columns[y] + ' against ' + columns[x])
            ax.set_xlabel(columns[x])
            ax.set_ylabel(columns[y])

            ax.scatter(data[:, x], data[:, y], c=colours)

            for k, mean in enumerate(means):
                ell = mpl.patches.Ellipse((mean[x], mean[y]), 2 * np.sqrt(covars[k][x] * 5.991),
                                          2 * np.sqrt(covars[k][y] * 5.991),
                                          color=pal(ind2order[k]), alpha=0.5)
                ax.add_artist(ell)
                ell.set_clip_box(ax.bbox)

        fig.tight_layout()
        fig.subplots_adjust(right=0.86)  # gives space for legend

        savefile = outdir + '/figures/js_scatter.png'

        fig.savefig(savefile)
        plt.close(fig)

    return all_jsd





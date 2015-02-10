import textwrap

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

import matplotlib as mpl
import matplotlib.pyplot as plt
import make_graphs as gr
import seaborn.apionly as sns


def calc_jsd(means, covars, weights, mins, maxes):
    rv_list = [multivariate_normal(mean=mean, cov=cov) for mean, cov in zip(means, covars)]

    # find sum of entropy * weight over all gaussians
    # N.B. scipy uses entropy with base e
    term2 = np.dot([rv.entropy() for rv in rv_list], weights.T)

    def sumpdf(x, y):  # integrand for differential entropy integral
        fofx = np.dot([rv2.pdf([x, y]) for rv2 in rv_list], weights.T)
        if fofx == 0:  # hack to avoid log(0)
            return 0
        else:
            return -1 * fofx * np.log(fofx)

    # limits for integral
    # integrate to get the differential entropy of the set of gaussians
    epsabs = 1e-1
    epsrel = 1e-1
    term1 = dblquad(sumpdf, mins[1], maxes[1], lambda x: mins[0], lambda x: maxes[0],
                    epsabs=epsabs, epsrel=epsrel
                    )

    jsd = (term1[0] - term2) * np.log(2)  # convert to log2 from ln
    if jsd < 0:
        jsd = 0
    return jsd


def best_jsd(means, covars, weights):
    """Go through all the axis pairs and return a list of the JS divergences"""
    # limits are at 95% - speeds up integral
    error90 = 4.605
    error95 = 5.991
    maxes = np.amax(means + np.sqrt(covars * error90), axis=0)
    mins = np.amin(means - np.sqrt(covars * error90), axis=0)
    jsds = []
    n_clusters, n_features = means.shape
    axes = np.identity(n_features)
    weights = np.array(weights)

    for j in range(len(means[0])):
        for i in range(j):
            ax = axes[:, [i, j]]
            means_s = np.dot(means, ax)  # shape is (n_clusters, n_scale)
            n_clusters, n_features = covars.shape
            covars_s = np.dot(ax.T, np.reshape(covars, (n_clusters, n_features, 1)) * ax)
                # shape is (n_scale, n_clusters, n_scale)
            covars_s = np.swapaxes(covars_s, 0, 1)  # shape is (n_clusters, n_scale, n_scale)

            jsd = calc_jsd(means_s, covars_s, weights, mins[[i, j]], maxes[[i, j]])
            jsds.append((jsd / np.log2(n_clusters), i, j))  # scale by number of clusters

    jsds.sort(reverse=True)
    # print "JSDs :", jsds
    return jsds


def js_graphs(data, columns, labels, outdir, means, covars, weights):
    """Draw the three best projections as found by JS divergence"""
    all_jsd = best_jsd(means, covars, weights)

    if len(all_jsd) < 6:
        return all_jsd

    n_clusters, n_features = means.shape
    griddata = np.zeros((n_features, n_features))
    for jsd, i, j in all_jsd:
        griddata[i, j] = griddata[j, i] = jsd

    with mpl.rc_context(gr.scatterplot):
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = sns.light_palette(color=gr.palette(0), as_cmap=True)
        heatmap = ax.pcolor(griddata, cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(heatmap)

        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(n_features)+0.5, minor=False)
        ax.set_yticks(np.arange(n_features)+0.5, minor=False)

        ax.invert_yaxis()
        ax.xaxis.tick_top()

        wrapped_cols = [textwrap.fill(x, 10) for x in columns]
        ax.set_xticklabels(wrapped_cols, minor=False)
        ax.set_yticklabels(columns, minor=False)
        savefile = outdir + '/figures/js_heatmap.png'
        fig.savefig(savefile)
        plt.close(fig)

    colours = [gr.palette(x) for x in labels]
    with mpl.rc_context(gr.two_cols):
        fig, ax = plt.subplots()
        # good_projs = [x for x in all_jsd if x[0] > 0.1]
        good_projs = all_jsd[0:4]
        for i, proj in enumerate(good_projs):
            ax.clear()
            x = proj[1]
            y = proj[2]

            ax.set_title("JSD = {0:.2g}".format(proj[0]))
            ax.set_xlabel(textwrap.fill(columns[x], 45))
            ax.set_ylabel(textwrap.fill(columns[y], 30))
            ax.scatter(data[:, x], data[:, y], c=colours, edgecolor='none', zorder=2)

            for k, mean in enumerate(means):
                ell = mpl.patches.Ellipse((mean[x], mean[y]), 2 * np.sqrt(covars[k][x] * 5.991),
                                          2 * np.sqrt(covars[k][y] * 5.991),
                                          color=gr.palette(k), alpha=0.5)
                ax.add_artist(ell)
                ell.set_clip_box(ax.bbox)

            gr.labelformatting(fig)
            savefile = outdir + '/figures/js_scatter_{}.png'.format(i)
            fig.savefig(savefile)

        if len(good_projs) > 0:  # add legend to last graph
            classes = np.unique(labels)
            recs = []
            for i in classes:
                recs.append(mpl.patches.Rectangle((0, 0), 1, 1, fc=gr.palette(i)))
            fig.legend(recs, classes, loc='center right')

            fig.set_tight_layout(False)
            fig.subplots_adjust(right=0.8)
            fig.savefig(savefile)
        plt.close(fig)

    return all_jsd
#!/usr/bin/python
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn.apionly as sns

import pydot
from sklearn.lda import LDA
import numpy as np
from scipy.stats import norm
from scipy import linalg

two_cols = {'xtick.labelsize': 20, 'ytick.labelsize': 20,
            'axes.labelsize': 20, 'axes.titlesize': 20}  # font sizes for graphs side-by-side


def palette(i):
    pal = sns.color_palette("deep")
    return pal[i % len(pal)]


def graded_palette(i):
    pal = sns.light_palette(color=palette(0), reverse=True, n_colors=10)  # dark to light blues
    return pal[i % len(pal)]


def cb_palette(i):
    #pal = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
    #       '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd']  # pastels
    pal = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',  # paired bright colours
           '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
    return pal[i % len(pal)]


def dag(inlabels, outlabel, outdir):
    """DAG for linear regression"""
    dot_object = pydot.Dot(graph_name="main_graph", rankdir="LR", labelloc='b',
                           labeljust='r', ranksep=1)
    dot_object.set_node_defaults(shape='circle', fixedsize='false',
                                 height=1, width=1, fontsize=12)

    outnode = pydot.Node(name=outlabel, texlbl=outlabel, label=outlabel)
    dot_object.add_node(outnode)

    for i, label in enumerate(inlabels):
        innode = pydot.Node(name=label + str(i), texlbl=label, label=label)
        dot_object.add_node(innode)
        dot_object.add_edge(pydot.Edge(innode, outnode))

    dot_object.write_png(outdir + '/figures/pgm_graph.png', prog='dot')


def learning_curve(all_messages, senders, outdir):
    """Make graph of how score varies with amount of data"""
    savefile = outdir + '/figures/learning_curve.png'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Learning Curve')
    ax.set_xlabel('Number of datapoints')
    ax.set_ylabel('Cross-Validated Log Predictive Density')

    shortdescs, xs, ys, xarr = {}, {x: [] for x in senders}, {x: [] for x in senders}, {x: [] for x in senders}
    for message in all_messages:
        sender = message['sender']
        distribution = message['distribution']
        shortdescs[sender] = distribution.shortdescrip
        xs[sender].append(distribution.data_size)
        ys[sender].append(distribution.avscore)
        xarr[sender].append(sorted(distribution.scores))

    for i, sender in enumerate(shortdescs.keys()):
        ax.plot(xs[sender], ys[sender], label=shortdescs[sender])
        sns.boxplot(xarr[sender], positions=xs[sender], whis=np.inf,
                    ax=ax, color=palette(i), widths=10)

    ax.autoscale(enable=True)
    ax.margins(x=0.1, y=0.1, tight=False)

    loc = plticker.AutoLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_ticklabels([int(x) for x in ax.xaxis.get_majorticklocs()])
    ax.xaxis.grid(True)
    ax.legend(loc='best')

    fig.savefig(savefile)
    plt.close(fig)


def scatterplot_matrix(data, labels, names, outdir, means, covars, ind2order):
    """Plots a scatterplot matrix of subplots."""
    with mpl.rc_context({'figure.autolayout': False, 'font.size': 20}):
        pal = cb_palette  # so I only have to set it once
        numdata, numvars = data.shape

        colours = [pal(x) for x in labels]
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(10, 9), sharex='col', sharey='row')

        classes = np.unique(labels)
        recs = []
        for i in classes:
            recs.append(mpl.patches.Rectangle((0, 0), 1, 1, fc=pal(i)))
        plt.figlegend(recs, classes, loc='center right')

        # Plot the data.
        for i in range(numvars):
            for j in range(numvars):
                ax = axes[i, j]
                if i < j:
                    ax.scatter(data[:, j], data[:, i], c=colours)
                elif i == j:
                    ax.annotate(textwrap.fill(names[i], 14), (0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center',
                                font_properties=mpl.font_manager.FontProperties(family='monospace', size=14))
                else:
                    for k, mean in enumerate(means):
                        # 95% confidence ellipse
                        ell = mpl.patches.Ellipse((mean[j], mean[i]), 2 * np.sqrt(covars[k][j] * 5.991),
                                                  2 * np.sqrt(covars[k][i] * 5.991),
                                                  color=pal(ind2order[k]), alpha=0.5)
                        ax.add_artist(ell)
                        ell.set_clip_box(ax.bbox)
                    ax.scatter(data[:, j], data[:, i], c=colours, label=labels)

        # Adjust display after plotting
        for i in range(numvars):
            for j in range(numvars):
                ax = axes[i, j]
                xticker = ax.xaxis.get_major_locator()
                yticker = ax.yaxis.get_major_locator()
                xticker.set_params(prune='upper', nbins=7-(numvars+1)/2)
                yticker.set_params(prune='upper', nbins=7)
                if i == 0:
                    ax.xaxis.set_ticks_position('top')
                elif i == numvars - 1:
                    ax.xaxis.set_ticks_position('bottom')
                if j == numvars - 1:
                    ax.yaxis.set_ticks_position('right')
                elif j == 0:
                    ax.yaxis.set_ticks_position('left')
                if i == j:
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.8)  # gives space for legend

        fig.savefig(outdir + '/figures/scatter_matrix.png')
        plt.close(fig)


def lda_graph(train_data, labels, outdir, means, covars, ind2order):
    pal = cb_palette
    clf = LDA()
    clf.fit(train_data, labels)

    scalings = clf.scalings_[:, :2]  # n_scale = 1 or 2. shape is (n_features, n_scale)
    train_data_s = np.dot(train_data, scalings)
    means_s = np.dot(means, scalings)  # shape is (n_clusters, n_scale)
    n_clusters, n_features = covars.shape
    covars_s = np.dot(scalings.T, np.reshape(covars, (n_clusters, n_features, 1)) * scalings)
    # shape is (n_scale, n_clusters, n_scale)
    covars_s = np.swapaxes(covars_s, 0, 1)  # shape is (n_clusters, n_scale, n_scale)

    with mpl.rc_context({'figure.autolayout': False}):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Clusters projected using LDA algorithm")
        ax.set_xlabel("Linear combination 1")

        if scalings.shape[1] == 1:  # do a 1D graph
            cl0 = np.array([x[0] for x in zip(train_data_s, labels) if x[1] == 0])
            cl1 = np.array([x[0] for x in zip(train_data_s, labels) if x[1] == 1])
            rv0 = norm(loc=means_s[0], scale=covars_s[0])
            rv1 = norm(loc=means_s[1], scale=covars_s[1])

            n, bins, patches = ax.hist([cl0, cl1], 50, normed=True, stacked=True, rwidth=1.0)
            x = np.arange(bins[0], bins[-1], (bins[1] - bins[0])/2.0)
            ax.plot(x, rv0.pdf(x)/2, color=pal(ind2order[0]), label='Cluster ' + str(ind2order[0]))
            ax.plot(x, rv1.pdf(x)/2, color=pal(ind2order[1]), label='Cluster ' + str(ind2order[1]))

            ax.legend(loc='best')
        else:
            colours = [pal(lab) for lab in labels]
            ax.scatter(train_data_s[:, 0], train_data_s[:, 1], c=colours)
            ax.set_ylabel("Linear combination 2")

            classes = np.unique(labels)
            recs = []
            for i in classes:  # make legend
                recs.append(mpl.patches.Rectangle((0, 0), 1, 1, fc=pal(i)))
            plt.figlegend(recs, classes, loc='center right')

            for k in range(means.shape[0]):  # for each cluster, draw 95% confidence ellipse
                v, w = linalg.eigh(covars_s[k])
                u = w[0] / linalg.norm(w[0])
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi

                ell = mpl.patches.Ellipse(means_s[k], 2 * np.sqrt(v[0] * 5.991),
                                          2 * np.sqrt(v[1] * 5.991),
                                          angle=angle, color=pal(ind2order[k]), alpha=0.5)
                ax.add_artist(ell)
                ell.set_clip_box(ax.bbox)

            fig.tight_layout()
            fig.subplots_adjust(right=0.85)  # gives space for legend

        fig.savefig(outdir + '/figures/lda_scatter.png')
        plt.close(fig)
        return scalings


def reg_graphs(xy_data, i, residuals, partial_residuals, outdir):
    with mpl.rc_context(two_cols):
        input_data = xy_data.arrays['X'][:, [i]]
        output = xy_data.arrays['Y']

        fig = plt.figure()
        ax = fig.add_subplot(111)

        savefile = outdir + '/figures/scatter_graph_{}.png'.format(i)
        ax.plot(input_data, output, 'o')
        ax.set_title('Output (' + xy_data.labels['Y'][0]
                     + ') against '
                     + xy_data.labels['X'][i])
        ax.set_xlabel(xy_data.labels['X'][i])
        ax.set_ylabel(xy_data.labels['Y'][0])
        fig.savefig(savefile)
        ax.clear()

        savefile = outdir + '/figures/partial_residuals_graph_{}.png'.format(i)
        ax.plot(input_data, partial_residuals, 'o')
        ax.set_title('Partial residuals against '
                     + xy_data.labels['X'][i])
        ax.set_xlabel(xy_data.labels['X'][i])
        ax.set_ylabel('Partial residuals')
        fig.savefig(savefile)
        ax.clear()

        savefile = outdir + '/figures/residuals_graph_{}.png'.format(i)
        ax.plot(input_data, residuals, 'o')
        ax.set_title('Residuals against ' +
                     xy_data.labels['X'][i])
        ax.set_xlabel(xy_data.labels['X'][i])
        ax.set_ylabel('Residuals')
        fig.savefig(savefile)
        plt.close(fig)


def histogram(train_data, means, stds, outdir):
    with mpl.rc_context(two_cols):
        for i in range(train_data.arrays['X'].shape[1]):
            input_data = train_data.arrays['X'][:, i]
            n, bins, patches = plt.hist(input_data, 50, normed=True)
            rv = norm(loc=means[i], scale=stds[i])
            x = np.arange(bins[0], bins[-1], bins[1] - bins[0])

            plt.plot(x, rv.pdf(x))
            plt.xlabel(train_data.labels['X'][i])
            plt.title('Histogram and 1D gaussian model for ' + train_data.labels['X'][i])
            savefile = outdir + '/figures/histo_graph_{}.png'.format(i)
            plt.savefig(savefile)
            plt.close()
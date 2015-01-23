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


def fs(i):
    tw = 10.0
    return '{}, {}'.format(tw * i, tw * i * 0.75)

two_cols = {'figure.figsize': fs(0.5)}  # rc params for graphs side-by-side
three_cols = {'figure.figsize': fs(0.33)}  # rc params for graphs side-by-side
full_box = {'figure.figsize': '10, 10'}
width_3o4 = {'figure.figsize': fs(0.75)}


def palette(i):
    """Not all plotting functions use default colour cycle - this function takes a number and returns the
    colour associated with it.
    """
    pal = sns.color_palette(name=None)
    return pal[i % len(pal)]


def dag(inlabels, outlabel, outdir):
    """DAG for linear regression"""
    dot_object = pydot.Dot(graph_name="main_graph", rankdir="LR", labelloc='b',
                           labeljust='r', ranksep=1)
    dot_object.set_node_defaults(shape='circle', fixedsize='false',
                                 height=1, width=1, fontsize=12)

    outnode = pydot.Node(name=outlabel, label=textwrap.fill(outlabel, 14))
    dot_object.add_node(outnode)

    for i, label in enumerate(inlabels):
        innode = pydot.Node(name=label + str(i), label=textwrap.fill(label, 14))
        dot_object.add_node(innode)
        dot_object.add_edge(pydot.Edge(innode, outnode))

    dot_object.write_png(outdir + '/figures/pgm_graph.png', prog='dot')


def learning_curve(all_messages, senders, outdir):
    """Make graph of how score varies with amount of data"""
    savefile = outdir + '/figures/learning_curve.png'

    with mpl.rc_context(width_3o4):
        fig, ax = plt.subplots()
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

        for i, sender in enumerate(senders):
            ax.plot(xs[sender], ys[sender], label=shortdescs[sender])
            sns.boxplot(np.array(xarr[sender]).T, positions=xs[sender], whis=np.inf,
                        ax=ax, color=palette(i), widths=20)

        ax.autoscale(enable=True, axis='x')
        ax.margins(x=0.1, y=0.1, tight=False)

        limits = ax.axis()
        if limits[2] < -200:
            allscores = [x for scores in xarr for y in scores for x in y]
            newmin = min([x for x in allscores if x > -200])
            newmin = (newmin / 10) * 10 - 10  # round to nearest 10 below
            ax.set_ylim(newmin, limits[3])

        loc = plticker.AutoLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_ticklabels([int(x) for x in ax.xaxis.get_majorticklocs()])
        ax.xaxis.grid(True)
        ax.legend(loc='best')

        fig.savefig(savefile)
        plt.close(fig)


def method_boxplots(all_messages, outdir):
    """Boxplot for range of methods (used when data doubling not done)"""
    with mpl.rc_context(width_3o4):
        savefile = outdir + '/figures/learning_curve.png'
        fig, ax = plt.subplots()
        ax.set_title('Scores for different methods')
        ax.set_ylabel('Cross-Validated Log Predictive Density')

        scores, names = [], []
        for i, message in enumerate(all_messages):
            scores.append(message['distribution'].scores)
            names.append(textwrap.fill(message['distribution'].shortdescrip, 14))

        scores = np.array(scores).T
        sns.boxplot(scores, whis=np.inf,
                    ax=ax, names=names)

        limits = ax.axis()
        if limits[2] < -200:
            newmin = min([x for x in scores[0] if x > -200])
            newmin = (newmin / 10) * 10 - 10  # round to nearest 10 below
            ax.set_ylim(newmin, limits[3])

        fig.savefig(savefile)
        plt.close(fig)


def scatterplot_matrix(data, labels, names, outdir, means, covars, ind2order):
    """Plots a scatterplot matrix of subplots."""
    with mpl.rc_context(full_box):
        numdata, numvars = data.shape

        colours = [palette(x) for x in labels]
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars, sharex='col', sharey='row',
                                 tight_layout=False)

        classes = np.unique(labels)
        recs = []
        for i in classes:
            recs.append(mpl.patches.Rectangle((0, 0), 1, 1, fc=palette(i)))
        fig.legend(recs, classes, loc='center right')

        # Plot the data.
        for i in range(numvars):
            for j in range(numvars):
                ax = axes[i, j]
                if i < j:
                    ax.scatter(data[:, j], data[:, i], c=colours)
                elif i == j:
                    ax.annotate(textwrap.fill(names[i], 14), (0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center',)
                                #font_properties=mpl.font_manager.FontProperties(family='monospace', size=14))
                else:
                    for k, mean in enumerate(means):
                        # 95% confidence ellipse
                        ell = mpl.patches.Ellipse((mean[j], mean[i]), 2 * np.sqrt(covars[k][j] * 5.991),
                                                  2 * np.sqrt(covars[k][i] * 5.991),
                                                  color=palette(ind2order[k]), alpha=0.5)
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


def scatterplot_matrix_no_ellipse(data, names, outdir):
    """Plots a scatterplot matrix of subplots."""
    with mpl.rc_context(full_box):
        numdata, numvars = data.shape

        fig, axes = plt.subplots(nrows=numvars, ncols=numvars, sharex='col', sharey='row',
                                 tight_layout=False)

        # Plot the data.
        for i in range(numvars):
            for j in range(numvars):
                ax = axes[i, j]
                if i < j or i > j:
                    ax.scatter(data[:, j], data[:, i], c=palette(0), edgecolors='none')
                elif i == j:
                    ax.annotate(textwrap.fill(names[i], 14), (0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center')  # ,
                                #font_properties=mpl.font_manager.FontProperties(family='monospace', size=14))

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
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        fig.savefig(outdir + '/figures/scatter_matrix_no_ellipse.png')
        plt.close(fig)


def lda_graph(train_data, labels, outdir, means, covars, ind2order):
    clf = LDA()
    clf.fit(train_data, labels)

    scalings = clf.scalings_[:, :2]  # n_scale = 1 or 2. shape is (n_features, n_scale)
    train_data_s = np.dot(train_data, scalings)
    means_s = np.dot(means, scalings)  # shape is (n_clusters, n_scale)
    n_clusters, n_features = covars.shape
    covars_s = np.dot(scalings.T, np.reshape(covars, (n_clusters, n_features, 1)) * scalings)
    # shape is (n_scale, n_clusters, n_scale)
    covars_s = np.swapaxes(covars_s, 0, 1)  # shape is (n_clusters, n_scale, n_scale)

    with mpl.rc_context(width_3o4):
        fig, ax = plt.subplots()
        ax.set_title("Clusters projected using LDA algorithm")
        ax.set_xlabel("Linear combination 1")

        if scalings.shape[1] == 1:  # do a 1D graph
            cl0 = np.array([x[0] for x in zip(train_data_s, labels) if x[1] == 0])
            cl1 = np.array([x[0] for x in zip(train_data_s, labels) if x[1] == 1])
            rv0 = norm(loc=means_s[0], scale=covars_s[0])
            rv1 = norm(loc=means_s[1], scale=covars_s[1])

            n_bins = min(50, max(20, int(train_data.shape[0])/20))
            n, bins, patches = ax.hist([cl0, cl1], n_bins, normed=True, stacked=True, rwidth=1.0)
            x = np.arange(bins[0], bins[-1], (bins[-1] - bins[0])/100.0)
            ax.plot(x, rv0.pdf(x)[0]/2.0, color=palette(0), label='Cluster ' + str(ind2order[0]))
            ax.plot(x, rv1.pdf(x)[0]/2.0, color=palette(1), label='Cluster ' + str(ind2order[1]))

            ax.legend(loc='best')
        else:
            colours = [palette(lab) for lab in labels]
            ax.scatter(train_data_s[:, 0], train_data_s[:, 1], c=colours)
            ax.set_ylabel("Linear combination 2")

            classes = np.unique(labels)
            recs = []
            for i in classes:  # make legend
                recs.append(mpl.patches.Rectangle((0, 0), 1, 1, fc=palette(i)))
            fig.legend(recs, classes, loc='center right')

            for k in range(means.shape[0]):  # for each cluster, draw 95% confidence ellipse
                v, w = linalg.eigh(covars_s[k])
                u = w[0] / linalg.norm(w[0])
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi

                ell = mpl.patches.Ellipse(means_s[k], 2 * np.sqrt(v[0] * 5.991),
                                          2 * np.sqrt(v[1] * 5.991),
                                          angle=angle, color=palette(ind2order[k]), alpha=0.5)
                ax.add_artist(ell)
                ell.set_clip_box(ax.bbox)

            fig.set_tight_layout(False)
            fig.tight_layout()
            fig.subplots_adjust(right=0.85)  # gives space for legend

        fig.savefig(outdir + '/figures/lda_scatter.png')
        plt.close(fig)

        # calculate importance
        imp = np.zeros(scalings.T.shape)
        for k in range(scalings.shape[1]):
            v1 = train_data_s[:, [k]] - train_data * scalings[:, k].T
            imp[k] = 1 - np.var(v1, axis=0) / np.var(train_data_s[:, k])

        return scalings, imp.T


def reg_graphs(xy_data, i, i_coeff, residuals, partial_residuals, outdir):
    with mpl.rc_context(three_cols):
        input_data = xy_data.arrays['X'][:, [i]]
        output = xy_data.arrays['Y']

        fig, ax = plt.subplots()

        savefile = outdir + '/figures/scatter_graph_{}.png'.format(i)
        ax.plot(input_data, output, 'o')
        ax.set_title(textwrap.fill('Output (' + xy_data.labels['Y'][0]
                                   + ') against '
                                   + xy_data.labels['X'][i], 30))
        ax.set_xlabel(textwrap.fill(xy_data.labels['X'][i], 30))
        ax.set_ylabel(textwrap.fill(xy_data.labels['Y'][0], 20))
        fig.savefig(savefile)
        ax.clear()

        savefile = outdir + '/figures/partial_residuals_graph_{}.png'.format(i)
        ax.plot(input_data, partial_residuals, 'o')
        xdata = np.linspace(min(input_data), max(input_data))
        ax.plot(xdata, i_coeff * xdata)
        ax.set_title(textwrap.fill('Partial residuals against '
                                   + xy_data.labels['X'][i], 30))
        ax.set_xlabel(textwrap.fill(xy_data.labels['X'][i], 30))
        ax.set_ylabel('Partial residuals')
        fig.savefig(savefile)
        ax.clear()

        savefile = outdir + '/figures/residuals_graph_{}.png'.format(i)
        ax.plot(input_data, residuals, 'o')
        ax.set_title(textwrap.fill('Residuals against ' +
                                   xy_data.labels['X'][i], 30))
        ax.set_xlabel(textwrap.fill(xy_data.labels['X'][i], 30))
        ax.set_ylabel('Residuals')
        fig.savefig(savefile)
        plt.close(fig)


def histogram(train_data, means, stds, outdir):
    with mpl.rc_context(three_cols):
        for i in range(train_data.arrays['X'].shape[1]):
            input_data = train_data.arrays['X'][:, i]
            n_bins = min(50, max(10, int(train_data.arrays['X'].shape[0])/20))

            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(input_data, n_bins, normed=True)
            rv = norm(loc=means[i], scale=stds[i])
            x = np.arange(bins[0], bins[-1], (bins[-1] - bins[0])/50.0)

            ax.plot(x, rv.pdf(x))
            ax.set_xlabel(textwrap.fill(train_data.labels['X'][i], 30))
            ax.set_title(textwrap.fill('Histogram and 1D gaussian model for ' + train_data.labels['X'][i], 30))
            savefile = outdir + '/figures/histo_graph_{}.png'.format(i)
            fig.savefig(savefile)
            plt.close(fig)
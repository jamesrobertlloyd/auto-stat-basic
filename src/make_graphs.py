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
import js_div as jsd

# rc params for different sizes of graph
font_size = 16
scatterplot = {'figure.figsize': '10, 10', 'font.size': font_size,  # graph width should be 10 - height varies
               'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
               'axes.labelsize': font_size, 'axes.titlesize': font_size, 'legend.fontsize': font_size}
font_size = 18
width_3o4 = {'figure.figsize': '8, 6', 'font.size': font_size,
             'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
             'axes.labelsize': font_size, 'axes.titlesize': font_size, 'legend.fontsize': font_size}
font_size = 24
two_cols = {'figure.figsize': '8, 6', 'font.size': font_size,
            'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
            'axes.labelsize': font_size, 'axes.titlesize': font_size, 'legend.fontsize': font_size}
font_size = 30
three_cols = {'figure.figsize': '8, 6', 'font.size': font_size,
              'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
              'axes.labelsize': font_size, 'axes.titlesize': font_size, 'legend.fontsize': font_size}


def palette(i):
    """Not all plotting functions use default colour cycle - this function takes a number and returns the
    colour associated with it.
    """
    pal = sns.color_palette(name=None, n_colors=10)
    return pal[i % len(pal)]


def dag(inlabels, outlabel, outdir):
    """DAG for linear regression"""
    dot_object = pydot.Dot(graph_name="main_graph", rankdir="LR", labelloc='b',
                           labeljust='r', ranksep=1)
    dot_object.set_node_defaults(shape='circle', fixedsize='false',
                                 height=1, width=1, fontsize=12, fontname='Inconsolata')

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

        for i, sender in enumerate(sorted(ys, key=ys.get, reverse=True)):
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

        scores, names = np.empty((5, len(all_messages)), dtype=float), []
        wrap_length = 54 / len(all_messages)
        for i, message in enumerate(all_messages):
            scores[:, i] = sorted(message['distribution'].scores)
            names.append(textwrap.fill(message['distribution'].shortdescrip, wrap_length))

        orders = np.argsort(scores[2])
        scores = scores[:, orders[::-1]]
        names = [names[i] for i in orders[::-1]]
        sns.boxplot(scores, whis=np.inf,
                    ax=ax, names=names)

        limits = ax.axis()
        if limits[2] < -200:
            newmin = [x for x in scores[0] if x > -200]
            if newmin:
                newmin = (min(newmin) / 10) * 10 - 10  # round to nearest 10 below
            else:
                newmin = -200
            ax.set_ylim(newmin, limits[3])

        fig.savefig(savefile)
        plt.close(fig)


def scatterplot_matrix(data, col_labels, labels, outdir, means, covars):
    """Plots a scatterplot matrix of subplots."""
    with mpl.rc_context(scatterplot):
        numdata, numvars = data.shape

        colours = [palette(x) for x in labels]
        fig, axes = plt.subplots(figsize=(10, 8.5), nrows=numvars, ncols=numvars, sharex='col', sharey='row',
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
                    ax.scatter(data[:, j], data[:, i], c=colours, edgecolors='none', zorder=2)
                elif j < i:
                    ax.scatter(data[:, j], data[:, i], c=colours, edgecolors='none', zorder=2)
                    for k, mean in enumerate(means):
                        # 95% confidence ellipse
                        ell = mpl.patches.Ellipse((mean[j], mean[i]), 2 * np.sqrt(covars[k][j] * 5.991),
                                                  2 * np.sqrt(covars[k][i] * 5.991),
                                                  color=palette(k), alpha=0.5)
                        ax.add_artist(ell)
                        ell.set_clip_box(ax.bbox)

        maxes = [None] * numvars
        for i in range(numvars):
            for j in range(numvars):
                ax = axes[i, j]
                set_ticks(ax.xaxis, prune=True)
                set_ticks(ax.yaxis, prune=True)

                if i == 0:
                    labelnosci(ax.xaxis)
                    ax.xaxis.set_ticks_position('top')
                    ax.set_xlabel('')
                elif i == numvars - 1:
                    labelnosci(ax.xaxis)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.set_xlabel('')
                if j == numvars - 1:
                    labelnosci(ax.yaxis)
                    ax.yaxis.set_ticks_position('right')
                    ax.set_ylabel('')
                elif j == 0:
                    maxes[i] = labelnosci(ax.yaxis)
                    ax.yaxis.set_ticks_position('left')
                    ax.set_ylabel('')
                if i == j:
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    label = col_labels[i]
                    if maxes[i] is not None:
                        label = label + ' / 1e' + str(maxes[i])
                    ax.annotate(textwrap.fill(label, 14), (0.5, 0.5),
                                xycoords='axes fraction', ha='center', va='center',)
                ax.set_xticklabels([x.get_text() for x in ax.xaxis.get_ticklabels()], rotation=90)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.85)  # gives space for legend

        fig.savefig(outdir + '/figures/scatter_matrix.png')
        plt.close(fig)


def scatterplot_matrix_no_ellipse(data, names, outdir):
    """Plots a scatterplot matrix of subplots."""
    with mpl.rc_context(scatterplot):
        numdata, numvars = data.shape

        fig, axes = plt.subplots(figsize=(10, 10), nrows=numvars, ncols=numvars, sharex='col', sharey='row',
                                 tight_layout=False)

        # Plot the data.
        for i in range(numvars):
            for j in range(numvars):
                ax = axes[i, j]
                if i < j or i > j:
                    ax.scatter(data[:, j], data[:, i], c=palette(0), edgecolors='none')

        # Adjust display after plotting
        maxes = [None] * numvars
        for i in range(numvars):
            for j in range(numvars):
                ax = axes[i, j]
                set_ticks(ax.xaxis, prune=True)
                set_ticks(ax.yaxis, prune=True)

                if i == 0:
                    labelnosci(ax.xaxis)
                    ax.xaxis.set_ticks_position('top')
                    ax.set_xlabel('')
                elif i == numvars - 1:
                    labelnosci(ax.xaxis)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.set_xlabel('')
                if j == numvars - 1:
                    labelnosci(ax.yaxis)
                    ax.yaxis.set_ticks_position('right')
                    ax.set_ylabel('')
                elif j == 0:
                    maxes[i] = labelnosci(ax.yaxis)
                    ax.yaxis.set_ticks_position('left')
                    ax.set_ylabel('')
                if i == j:
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    label = names[i]
                    if maxes[i] is not None:
                        label = label + ' / 1e' + str(maxes[i])
                    ax.annotate(textwrap.fill(label, 14), (0.5, 0.5),
                                xycoords='axes fraction', ha='center', va='center',)
                ax.set_xticklabels([x.get_text() for x in ax.xaxis.get_ticklabels()], rotation=90)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        fig.savefig(outdir + '/figures/scatter_matrix_no_ellipse.png')
        plt.close(fig)


def lda_graph(train_data, labels, outdir, means, covars, weights):
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
            ax.plot(x, rv0.pdf(x)[0]/2.0, color=palette(0), label='Cluster ' + str(0))
            ax.plot(x, rv1.pdf(x)[0]/2.0, color=palette(1), label='Cluster ' + str(1))

            ax.legend(loc='best')
        else:
            colours = [palette(lab) for lab in labels]
            ax.scatter(train_data_s[:, 0], train_data_s[:, 1], c=colours, edgecolors='none', zorder=2)
            ax.set_ylabel("Linear combination 2")

            classes = np.unique(labels)
            recs = []
            for i in classes:  # make legend
                recs.append(mpl.patches.Rectangle((0, 0), 1, 1, fc=palette(i)))
            fig.legend(recs, classes, loc='center right')

            for k in range(n_clusters):  # for each cluster, draw 95% confidence ellipse
                e_vals, e_vecs = linalg.eigh(covars_s[k])
                u = e_vecs[0] / linalg.norm(e_vecs[0])
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi
                ell = mpl.patches.Ellipse(means_s[k], 2 * np.sqrt(e_vals[0] * 5.991),
                                          2 * np.sqrt(e_vals[1] * 5.991),
                                          angle=angle, color=palette(k), alpha=0.5)
                ax.add_artist(ell)
                ell.set_clip_box(ax.bbox)

            fig.set_tight_layout(False)
            fig.tight_layout()
            fig.subplots_adjust(right=0.85)  # gives space for legend

        fig.savefig(outdir + '/figures/lda_scatter.png')
        plt.close(fig)

        # calculate JSD
        if scalings.shape[1] == 2:
            mins = [-np.inf, -np.inf]
            maxes = [np.inf, np.inf]
            jsd1 = jsd.calc_jsd(means_s, covars_s, weights, mins, maxes)

            jsd1 = jsd1 / np.log2(n_clusters)
        else:
            jsd1 = None

        # calculate importance
        imp = np.zeros(scalings.T.shape)
        for k in range(scalings.shape[1]):
            v1 = train_data_s[:, [k]] - train_data * scalings[:, k].T
            imp[k] = 1 - np.var(v1, axis=0) / np.var(train_data_s[:, k])

        lda_dicts = [{'coeff': x, 'importance': y} for x, y in zip(scalings, imp.T)]

        return lda_dicts, jsd1


def reg_graphs(xy_data, i, i_coeff, input_index, residuals, partial_residuals, outdir):
    with mpl.rc_context(three_cols):
        input_data = xy_data.arrays['X'][:, [i]]
        output = xy_data.arrays['Y']

        fig, ax = plt.subplots()
        savefile = outdir + '/figures/scatter_graph_{}.png'.format(input_index)
        ax.plot(input_data, output, 'o')
        ax.set_title(textwrap.fill(xy_data.labels['Y'][0]
                                   + ' against '
                                   + xy_data.labels['X'][i], 25))
        ax.set_xlabel(textwrap.fill(xy_data.labels['X'][i], 25))
        ax.set_ylabel(textwrap.fill(xy_data.labels['Y'][0], 20))
        labelformatting(fig)
        fig.savefig(savefile)
        plt.close(fig)

        fig, ax = plt.subplots()
        savefile = outdir + '/figures/partial_residuals_graph_{}.png'.format(input_index)
        ax.plot(input_data, partial_residuals, 'o')
        xdata = np.linspace(min(input_data), max(input_data))
        ax.plot(xdata, i_coeff * xdata)
        ax.set_title(textwrap.fill('Partial residuals against '
                                   + xy_data.labels['X'][i], 25))
        ax.set_xlabel(textwrap.fill(xy_data.labels['X'][i], 25))
        ax.set_ylabel('Partial residuals')
        labelformatting(fig)
        fig.savefig(savefile)
        plt.close(fig)

        fig, ax = plt.subplots()
        savefile = outdir + '/figures/residuals_graph_{}.png'.format(input_index)
        ax.plot(input_data, residuals, 'o')
        ax.set_title(textwrap.fill('Residuals against ' +
                                   xy_data.labels['X'][i], 25))
        ax.set_xlabel(textwrap.fill(xy_data.labels['X'][i], 25))
        ax.set_ylabel('Residuals')
        labelformatting(fig)
        fig.savefig(savefile)
        plt.close(fig)


def histogram(train_data, means, stds, input_indices, outdir, dist):
    with mpl.rc_context(three_cols):
        for i, index in enumerate(input_indices):
            input_data = train_data.arrays['X'][:, i]
            n_bins = min(50, max(10, int(train_data.arrays['X'].shape[0]/20)))

            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(input_data, n_bins, normed=True)

            if dist == 'gaussian':
                rv = norm(loc=means[i], scale=stds[i])
                x = np.arange(bins[0], bins[-1], (bins[-1] - bins[0])/50.0)
                ax.plot(x, rv.pdf(x))
                ax.set_title(textwrap.fill('Histogram and 1D gaussian model for ' + train_data.labels['X'][i], 25))
            elif dist == 'uniform':
                lefts = means
                rights = stds
                ax.plot([lefts[i], lefts[i], rights[i], rights[i]],
                        [0, 1.0/(rights[i] - lefts[i]), 1.0/(rights[i] - lefts[i]), 0])
                ax.set_title(textwrap.fill('Histogram and 1D uniform model for ' + train_data.labels['X'][i], 25))

            ax.set_ylabel('Frequency')
            ax.set_xlabel(textwrap.fill(train_data.labels['X'][i], 25))
            ax.autoscale(enable=True, axis='x', tight=True)
            labelformatting(fig)
            savefile = outdir + '/figures/histo_graph_{}.png'.format(index)
            fig.savefig(savefile)
            plt.close(fig)


def fa_plots(data, outdir):
    fig, ax = plt.subplots()
    ax.plot(range(data.shape[0]), data)
    savefile = outdir + '/figures/fa_graph.png'
    fig.savefig(savefile)
    plt.close(fig)


def fa_heatmap(mat, columns, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = sns.diverging_palette(145, 280, as_cmap=True)
    vmax = max(np.max(mat), abs(np.min(mat)))
    heatmap = ax.pcolor(mat, cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(heatmap)
            # put the major ticks at the middle of each cell
    n_features = mat.shape[1]
    ax.set_xticks(np.arange(n_features)+0.5, minor=False)
    ax.set_yticks(np.arange(n_features)+0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    wrapped_cols = [textwrap.fill(x, 10) for x in columns]
    ax.set_xticklabels(wrapped_cols, minor=False)
    ax.set_yticklabels(range(n_features), minor=False)
    savefile = outdir + '/figures/fa_heatmap.png'
    fig.savefig(savefile)
    plt.close(fig)


def labelformatting(fig):
    ax = fig.gca()
    set_ticks(ax.xaxis)
    set_ticks(ax.yaxis)
    labelnosci(ax.xaxis)
    labelnosci(ax.yaxis)
    xlabelfilter(ax)


def labelnosci(ax):
    """Remove scientific notation from axes as it overlaps with axis labels"""
    alllabels = [x.get_text().replace(u'\u2212', '-') for x in ax.get_ticklabels()]
    axmax = max([int(np.log10(abs(float(x)))) for x in alllabels if x != '' and float(x) != 0], key=abs)
    if abs(axmax) > 3:
        alllabels = [float(x)/pow(10, axmax) if x != '' else '' for x in alllabels]
        alllabels = [str(x) for x in alllabels]
        ax.set_ticklabels(alllabels, visible=True)
        ax.set_label_text(ax.get_label_text() + ' / 1e' + str(axmax))
        return axmax
    else:
        return None


def set_ticks(ax, prune=False):
    """Gets rid of invisible ticks beyond the axis limits and sets the tick labels (which mpl wouldn't do until
    fig.canvas.draw is called)"""
    allxlocs = ax.get_majorticklocs()
    xlims = ax.get_view_interval()
    newlocs = [x for x in allxlocs if xlims[0] <= x <= xlims[1]]
    newlabels = [str(x) for x in newlocs]
    if prune:  # remove highest tick
        newlabels[-1] = ''
    ax.set_ticks(newlocs)
    ax.set_ticklabels(newlabels)


def xlabelfilter(ax):
    """Remove x tick labels when there are more than 4"""
    allxlabels = [x.get_text() for x in ax.xaxis.get_ticklabels()]
    allxlocs = ax.xaxis.get_majorticklocs()

    xlims = ax.get_xlim()
    lefthide = len([x for x in allxlocs if xlims[0] > x])
    righthide = len([x for x in allxlocs if xlims[1] < x])
    shownxlabels = [x for x, y in zip(allxlabels, allxlocs) if xlims[0] <= y <= xlims[1]]
    if len(shownxlabels) < 5:
        return
    filtlabels = ["" for _ in shownxlabels]
    filtlabels[0] = shownxlabels[0]
    if len(shownxlabels) == 5:
        filtlabels[2] = shownxlabels[2]
        filtlabels[-1] = shownxlabels[-1]
    elif len(shownxlabels) == 6:
        filtlabels[2] = shownxlabels[2]
        filtlabels[4] = shownxlabels[4]
    else:
        filtlabels[int((len(filtlabels)-1)/3.0 + 0.5)] = shownxlabels[int((len(filtlabels)-1)/3.0 + 0.5)]
        filtlabels[int(((len(filtlabels)-1)*2)/3.0 + 0.5)] = shownxlabels[int(((len(filtlabels)-1)*2)/3.0 + 0.5)]
        filtlabels[-1] = shownxlabels[-1]

    allxlabels[lefthide:len(allxlabels)-righthide] = filtlabels[:]
    ax.xaxis.set_ticklabels(allxlabels, visible=True)

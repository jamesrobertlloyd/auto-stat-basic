#!/usr/bin/python
import matplotlib as mpl
mpl.use('Agg', warn=False)  # Use a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn.apionly as sns

sns.set(context='paper', style='whitegrid', palette='deep', font_scale=1)
#plt.xkcd()  # to turn on xkcd graphs
#plt.rcdefaults() # to reset defaults

import pydot

from sklearn.lda import LDA
import numpy as np
from scipy.stats import norm
from scipy import linalg

def palette(i):
    pal = sns.color_palette("deep")
    return pal[i % len(pal)]

def dag(inlabels,outlabel,outdir):
    '''DAG for linear regression'''
    dot_object = pydot.Dot(graph_name="main_graph",rankdir="LR", labelloc='b', 
                           labeljust='r', ranksep=1)
    dot_object.set_node_defaults(shape='circle', fixedsize='false',
                                 height=1, width=1, fontsize=12)

    outnode = pydot.Node(name=outlabel, texlbl=outlabel, label=outlabel)
    dot_object.add_node(outnode)

    for i,label in enumerate(inlabels):
        innode = pydot.Node(name=label + str(i), texlbl=label, label=label)
        dot_object.add_node(innode)
        dot_object.add_edge(pydot.Edge(innode,outnode))

    dot_object.write_png(outdir + '/figures/pgm_graph.png', prog='dot')


def learning_curve(xs, xarr, shortdescs, outdir):
    '''Make graph of how score varies with amount of data'''
    savefile = outdir + '/figures/learning_curve.png'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Learning Curve')
    ax.set_xlabel('Number of datapoints')
    ax.set_ylabel('Cross-Validated Predicted Density')

    for i,sender in enumerate(xs):
        ys = [np.median(x) for x in xarr[sender]]
        ax.plot(xs[sender],ys, label=shortdescs[sender])
        sns.boxplot(xarr[sender], positions=xs[sender], whis=np.inf, 
                    ax=ax, color=palette(i), widths = 10)

    #plt.getp(ax)
    #plt.getp(ax.xaxis)
    ax.autoscale(enable=True)
    ax.margins(x=0.1,y=0.1, tight=False)

    loc = plticker.AutoLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_ticklabels([int(x) for x in ax.xaxis.get_majorticklocs()])
    ax.xaxis.grid(True) 
    ax.legend()

    fig.savefig(savefile)
    plt.close(fig)


def scatterplot_matrix(data, labels, names, outdir, means, covars, weights):
    """Plots a scatterplot matrix of subplots."""
    #mpl.rc('xtick', labelsize=12) 
    #mpl.rc('ytick', labelsize=12)
    if True: #with mpl.rc_context({'xtick.labelsize':14, 'ytick.labelsize':14}):
        numdata, numvars  = data.shape

        colours = [palette(x) for x in labels]
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(14,12), sharex='col', sharey='row')

        classes = np.unique(labels)
        recs = []
        for i in classes:
            recs.append(mpl.patches.Rectangle((0,0),1,1,fc=palette(i)))
        plt.figlegend(recs,classes,loc='center right')

        # Plot the data.
        for i in range(numvars):
            for j in range(numvars):
                ax = axes[i,j]
                if i < j:
                    ax.scatter(data[:,j], data[:,i], c=colours)
                elif i==j:
                    ax.annotate(names[i], (0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center')
                else:
                    for k,mean in enumerate(means):
                        # 95% confidence ellipse
                        ell = mpl.patches.Ellipse((mean[j],mean[i]), 2*np.sqrt(covars[k][j]*5.991), 
                                                  2*np.sqrt(covars[k][i]*5.991), 
                                                  color = palette(k), alpha=0.5)
                        ax.add_artist(ell)
                        ell.set_clip_box(ax.bbox)
                    ax.scatter(data[:,j], data[:,i], c=colours, label=labels)


        # Adjust display after plotting
        for i in range(numvars):
            for j in range(numvars):
                ax = axes[i,j]

                if i==0:
                    ax.xaxis.set_ticks_position('top')
                    if j!= numvars - 1:
                        xticks = ax.xaxis.get_major_ticks()
                        xticks[-1].label2.set_visible(False)
                elif i==numvars - 1:
                    ax.xaxis.set_ticks_position('bottom')
                    if j!= numvars - 2:
                        xticks = ax.xaxis.get_major_ticks()
                        xticks[-1].label1.set_visible(False)
                if j==numvars - 1:
                    ax.yaxis.set_ticks_position('right')
                    if i!=0:
                        yticks = ax.yaxis.get_major_ticks()
                        yticks[-1].label2.set_visible(False)
                elif j == 0:
                    ax.yaxis.set_ticks_position('left')
                    if i!=1:
                        yticks = ax.yaxis.get_major_ticks()
                        yticks[-1].label1.set_visible(False)
                if i==j:
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                #if i+j == numvars-1:
                #ax.margins(x=0.05,y=0.05, tight=False)
                #ax.autoscale(enable=True)
                #ax.margins(x=0.1,y=0.1, tight=False)

                #loc = plticker.AutoLocator()
                #ax.xaxis.set_major_locator(loc)
                #loc2 = plticker.AutoLocator()
                #ax.yaxis.set_major_locator(loc2)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.86) # gives space for legend

        fig.savefig(outdir + '/figures/scatter_matrix.png')
        plt.close(fig)


def lda_graph(train_data,labels,outdir, means, covars):
    clf = LDA()
    clf.fit(train_data,labels)
    xnew = np.dot(train_data,clf.scalings_[:,[0,1]])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colours = [palette(x) for x in labels]
    ax.scatter(xnew[:,0],xnew[:,1], c=colours)
    ax.set_title("Clusters projected using LDA algorithm")
    ax.set_xlabel("Features {}".format(clf.scalings_[:,0]))
    ax.set_ylabel("Features {}".format(clf.scalings_[:,1]))

    classes = np.unique(labels)
    recs = []
    for i in classes:
        recs.append(mpl.patches.Rectangle((0,0),1,1,fc=palette(i)))
    plt.figlegend(recs,classes,loc='center right')

    #print clf.means_
    #print clf.scalings_

    for k,mean in enumerate(means):
        # 95% confidence ellipse
        covar = np.dot(clf.scalings_[:,[0,1]].T,
                       np.dot(np.diagflat(covars[k]),clf.scalings_[:,[0,1]])
                       )
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi 
        newmean = np.dot(mean,clf.scalings_[:,[0,1]])

        ell = mpl.patches.Ellipse(newmean, 2*np.sqrt(v[0]*5.991), 
                                  2*np.sqrt(v[1]*5.991), 
                                  angle = angle, color = palette(k), alpha=0.5)
        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)

    fig.savefig(outdir + '/figures/lda_scatter.png')
    plt.close(fig)


def reg_graphs(XY_data, i, residuals, reduced, outdir):
    input_data = XY_data.arrays['X'][:,[i]]
    output = XY_data.arrays['Y']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    savefile = outdir + '/figures/scatter_graph_{}.png'.format(i)
    ax.plot(input_data,output, 'o')
    ax.set_title('Output (' + XY_data.labels['Y'][0] + \
                           ') against ' + \
                           XY_data.labels['X'][i])
    ax.set_xlabel(XY_data.labels['X'][i])
    ax.set_ylabel(XY_data.labels['Y'][0])
    fig.savefig(savefile)
    ax.clear()

    savefile = outdir + '/figures/residuals_graph_{}.png'.format(i)
    ax.plot(input_data,residuals, 'o')
    ax.set_title('Partial residuals against ' + \
                           XY_data.labels['X'][i])
    ax.set_xlabel(XY_data.labels['X'][i])
    ax.set_ylabel('Partial residuals')
    fig.savefig(savefile)
    ax.clear()

    savefile = outdir + '/figures/reduced_graph_{}.png'.format(i)
    ax.plot(input_data,reduced, 'o')
    ax.set_title('Residuals against ' + \
                           XY_data.labels['X'][i])
    ax.set_xlabel(XY_data.labels['X'][i])
    ax.set_ylabel('Residuals')
    fig.savefig(savefile)
    plt.close(fig)

def histogram(train_data, means, stds, outdir):
    for i in range(train_data.arrays['X'].shape[1]):
        input_data = train_data.arrays['X'][:,i]
        n, bins, patches = plt.hist(input_data, 50, normed=True)
        rv = norm(loc = means[i], scale = stds[i])
        x = np.arange(bins[0], bins[-1], bins[1] - bins[0])

        plt.plot(x, rv.pdf(x))
        plt.xlabel(train_data.labels['X'][i])
        plt.title('Histogram and 1D gaussian model for ' + train_data.labels['X'][i])
        savefile = outdir + '/figures/histo_graph_{}.png'.format(i)
        plt.savefig(savefile)
        plt.close()



if __name__ == '__main__':
    import re

    outdir = './tmp/'
 
    datafl = '../data/test-lin/iris_labels.csv'

    data = np.loadtxt(datafl, delimiter=',', skiprows=1, ndmin=2)
    with open(datafl, 'r') as data_file:
        labels = [re.sub('[.]', '', label) for label in data_file.readline().strip().split(',')]

    
    clusters = [x / (int(data.shape[0]/4)+1) for x in range(data.shape[0])]
    #fig = scatterplot_matrix(data, clusters, labels)
    #fig.savefig(outdir + '/figures/scatter_matrix.png')
    #plt.close(fig)
    print clusters

    means = np.array([[ 6.21358287,  3.10006579,  4.61993707,  1.53159768],
             [ 5.00333333,  3.39     ,   1.48666667,  0.25666667],
             [ 5.61027539,  2.56467607,  4.02785824,  1.25595022],
             [ 6.6636493 ,  3.0579262 ,  5.58996574,  2.05736395]])
    covars = np.array([[ 0.13631331,  0.02666539,  0.02505668,  0.0200721 ],
              [ 0.11865556,  0.13723333,  0.02948889,  0.01078889],
              [ 0.14964476,  0.05346434,  0.12761418,  0.03912362],
              [ 0.29390576,  0.07086452,  0.16778024,  0.0569962 ]])
    lda_graph(data,clusters,outdir,means,covars)
 

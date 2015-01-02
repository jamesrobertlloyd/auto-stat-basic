#!/usr/bin/python

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
from scipy import linalg

import itertools

import matplotlib as mpl
mpl.use('Agg', warn=False)  # Use a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn.apionly as sns
sns.set(context='poster', style='whitegrid', palette='deep', font_scale=1)
import make_graphs as gr


def best_jsd(means,covars,weights):
    '''Go through all the axis pairs and return a list of the JS divergences'''
    # limits are at 95% - speeds up integral
    maxes = np.amax(means + np.sqrt(covars*5.991), axis=0) 
    mins = np.amin(means - np.sqrt(covars*5.991), axis=0) 
    jsds = []
    axes = np.identity(len(means[0]))
    weights = np.array(weights)

    for j in range(len(means[0])):
        for i in range(j):
            ax = axes[:,[i,j]]

            # create a list of scipy 2D gaussian distributions
            rv_list = []
            for mean, cov in zip(means,covars):
                mean = np.dot(mean,ax)
                cov = np.dot(cov,ax)
                #cov = np.diagflat(cov) # making covariance 2D appears to be unnecessary
                rv_list.append(multivariate_normal(mean=mean, cov=cov))

                # find sum of entropy * weight over all gaussians
                # N.B. scipy uses entropy with base e
            term2 = np.dot([rv.entropy() for rv in rv_list], weights.T)

            def sumpdf(x,y): # integrand for differential entropy integral
                fofx = np.dot([rv.pdf([x,y]) for rv in rv_list], weights.T)
                if fofx == 0: # hack to avoid log(0)
                    return 0
                else:
                    return -1 * fofx * np.log(fofx) # use base e here to match scipy

            # limits for integral
            axmin = np.dot(mins, ax)
            axmax = np.dot(maxes, ax)
            # integrate to get the differential entropy of the set of gaussians
            #term1_full = dblquad(sumpdf, -np.inf, np.inf,lambda x:-np.inf, lambda x:np.inf)
            term1 = dblquad(sumpdf, axmin[1], axmax[1],lambda x:axmin[0], lambda x:axmax[0])

            jsd = term1[0] - term2
            jsds.append((jsd,i,j))
            #print ax,jsd, term1, term2

    jsds.sort(reverse=True)
    #print "JSDs :", jsds
    return jsds[0:3]


def js_graphs(means,covars,weights,outdir, data, labels, columns):
    '''Draw the three best projections as found by JS divergence'''
    best_projs = best_jsd(means,covars,weights)

    colours = [gr.palette(x) for x in labels]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    classes = np.unique(labels)
    recs = []
    for i in classes:
        recs.append(mpl.patches.Rectangle((0,0),1,1,fc=gr.palette(i)))
    plt.figlegend(recs,classes,loc='center right')

    for i, proj in enumerate(best_projs):
        x = proj[1]
        y = proj[2]

        ax.set_title(columns[y] + ' against ' + columns[x])
        ax.set_xlabel(columns[x])
        ax.set_ylabel(columns[y])

        savefile = outdir + '/figures/js_scatter{}.png'.format(i)
        ax.scatter(data[:,x], data[:,y], c=colours)

        for k,mean in enumerate(means):
            ell = mpl.patches.Ellipse((mean[x],mean[y]), 2*np.sqrt(covars[k][x]*5.991), 
                                      2*np.sqrt(covars[k][y]*5.991), 
                                      color = gr.palette(k), alpha=0.5)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)

        fig.savefig(savefile)
        ax.cla()

    plt.close(fig)



if __name__ == '__main__':
    # iris dataset as clustered by MoGLearner
    weights = np.array([ 0.14409516,  0.4,         0.17468047,  0.28122437])
    means = np.array([[ 6.21358287,  3.10006579,  4.61993707,  1.53159768],
             [ 5.00333333,  3.39     ,   1.48666667,  0.25666667],
             [ 5.61027539,  2.56467607,  4.02785824,  1.25595022],
             [ 6.6636493 ,  3.0579262 ,  5.58996574,  2.05736395]])
    covars = np.array([[ 0.13631331,  0.02666539,  0.02505668,  0.0200721 ],
              [ 0.11865556,  0.13723333,  0.02948889,  0.01078889],
              [ 0.14964476,  0.05346434,  0.12761418,  0.03912362],
              [ 0.29390576,  0.07086452,  0.16778024,  0.0569962 ]])

    print best_jsd(means,covars,weights)



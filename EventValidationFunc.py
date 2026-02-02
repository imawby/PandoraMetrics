import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions
        
##############################################################################################
##############################################################################################

def PlotVertices(event_branches, vertex_string, color, detector, true_nu_vtx_boundary, mask, ax) :
    # Draw XY
    ax[0].scatter(ak.to_numpy(event_branches[(vertex_string + 'X')][mask]), ak.to_numpy(event_branches[(vertex_string + 'Y')][mask]), color=color, s=5, label=vertex_string)
    ax[0].hlines(detector['Y'][0], detector['X'][0], detector['X'][1], color='black', label='DUNE LArTPC')
    ax[0].hlines(detector['Y'][1], detector['X'][0], detector['X'][1], color='black', label='_nolegend_')
    ax[0].vlines(detector['X'][0], detector['Y'][0], detector['Y'][1], color='black', label='_nolegend_')
    ax[0].vlines(detector['X'][1], detector['Y'][0], detector['Y'][1], color='black', label='_nolegend_')
    ax[0].hlines(true_nu_vtx_boundary['Y'][0], true_nu_vtx_boundary['X'][0], true_nu_vtx_boundary['X'][1], color='violet', label='True Nu Boundary')
    ax[0].hlines(true_nu_vtx_boundary['Y'][1], true_nu_vtx_boundary['X'][0], true_nu_vtx_boundary['X'][1], color='violet', label='_nolegend_')
    ax[0].vlines(true_nu_vtx_boundary['X'][0], true_nu_vtx_boundary['Y'][0], true_nu_vtx_boundary['Y'][1], color='violet', label='_nolegend_')
    ax[0].vlines(true_nu_vtx_boundary['X'][1], true_nu_vtx_boundary['Y'][0], true_nu_vtx_boundary['Y'][1], color='violet', label='_nolegend_')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].legend()
    
    # # Draw YZ
    ax[1].scatter(ak.to_numpy(event_branches[(vertex_string + 'Z')][mask]), ak.to_numpy(event_branches[(vertex_string + 'Y')][mask]), color=color, s=5, label=vertex_string)
    ax[1].vlines(detector['Z'][0], detector['Y'][0], detector['Y'][1], color='black', label='DUNE LArTPC')
    ax[1].vlines(detector['Z'][1], detector['Y'][0], detector['Y'][1], color='black', label='_nolegend_')
    ax[1].hlines(detector['Y'][0], detector['Z'][0], detector['Z'][1], color='black', label='_nolegend_')
    ax[1].hlines(detector['Y'][1], detector['Z'][0], detector['Z'][1], color='black', label='_nolegend_')
    ax[1].vlines(true_nu_vtx_boundary['Z'][0], true_nu_vtx_boundary['Y'][0], true_nu_vtx_boundary['Y'][1], color='violet', label='True Nu Boundary')
    ax[1].vlines(true_nu_vtx_boundary['Z'][1], true_nu_vtx_boundary['Y'][0], true_nu_vtx_boundary['Y'][1], color='violet', label='_nolegend_')
    ax[1].hlines(true_nu_vtx_boundary['Y'][0], true_nu_vtx_boundary['Z'][0], true_nu_vtx_boundary['Z'][1], color='violet', label='_nolegend_')
    ax[1].hlines(true_nu_vtx_boundary['Y'][1], true_nu_vtx_boundary['Z'][0], true_nu_vtx_boundary['Z'][1], color='violet', label='_nolegend_')
    ax[1].set_xlabel('Z')
    ax[1].set_ylabel('Y')
    ax[1].legend()
    
    # # Draw XZ
    ax[2].scatter(ak.to_numpy(event_branches[(vertex_string + 'Z')][mask]), ak.to_numpy(event_branches[(vertex_string + 'X')][mask]), color=color, s=5, label=vertex_string)
    ax[2].hlines(detector['X'][0], detector['Z'][0], detector['Z'][1], color='black', label='DUNE LArTPC')
    ax[2].hlines(detector['X'][1], detector['Z'][0], detector['Z'][1], color='black', label='_nolegend_')
    ax[2].vlines(detector['Z'][0], detector['X'][0], detector['X'][1], color='black', label='_nolegend_')
    ax[2].vlines(detector['Z'][1], detector['X'][0], detector['X'][1], color='black', label='_nolegend_')
    ax[2].hlines(true_nu_vtx_boundary['X'][0], true_nu_vtx_boundary['Z'][0], true_nu_vtx_boundary['Z'][1], color='violet', label='True Nu Boundary')
    ax[2].hlines(true_nu_vtx_boundary['X'][1], true_nu_vtx_boundary['Z'][0], true_nu_vtx_boundary['Z'][1], color='violet', label='_nolegend_')
    ax[2].vlines(true_nu_vtx_boundary['Z'][0], true_nu_vtx_boundary['X'][0], true_nu_vtx_boundary['X'][1], color='violet', label='_nolegend_')
    ax[2].vlines(true_nu_vtx_boundary['Z'][1], true_nu_vtx_boundary['X'][0], true_nu_vtx_boundary['X'][1], color='violet', label='_nolegend_')
    ax[2].set_xlabel('Z')
    ax[2].set_ylabel('X')
    ax[2].legend()

##############################################################################################
##############################################################################################

def PlotVertexCumulativeDR(event_branches, indices_or_mask, is_target_or_reco, string, color, ax, fig) :

    accuracy = event_branches['RecoNu_VertexAcc_Pass2'][indices_or_mask]
    positive_entry = (accuracy > 0.0)
    
    n_entries = len(accuracy)
    sample_points = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,15,20]
    
    cum_dR = [float(ak.sum(positive_entry & (accuracy < i))) / n_entries for i in sample_points]

    ax.plot(sample_points, cum_dR, color=color, linewidth=1, label='dR')
    ax.legend()
    ax.set_title(string)
    ax.set_xlabel('dR (cm)')
    ax.set_ylabel('Frac. of true nu' if is_target_or_reco else 'Frac. of reco nu')
    ax.grid(True)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelbottom=True, bottom=True, labelleft=True, left=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.95)#, hspace=0.3)

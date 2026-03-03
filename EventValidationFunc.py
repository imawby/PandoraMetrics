import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions
import Variables
import ValidationFunc

##############################################################################################
##############################################################################################

def CreateGraphs(plot_dir_path, int_masks, event_branches) :

    for int_type in Definitions.ints :
        int_mask = int_masks[int_type]
    
        target_mask = int_mask
        reco_mask = target_mask & (event_branches['RecoNu_VertexZ'] > -9000)

        file_name = f'{Definitions.int_file_strings[int_type]}'
        
        # Plot MCP_var distributions
        for plot_var in Variables.Event_MCP_plotting_vars :
            fig, ax = plt.subplots()
            ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_strings[int_type], pdg_string=Definitions.pdg_strings[-1])
            ValidationFunc.PlotVariable(target_mask, event_branches, plot_var, ax, Definitions.int_strings[int_type], Definitions.int_color[int_type])
            plt.close(fig)            
            fig.savefig(f'{plot_dir_path}/MCP/{plot_var.tree_name}/{file_name}.pdf', bbox_inches='tight')          
    
        # Plot BM_var distributions
        for plot_var in Variables.Event_Reco_plotting_vars :
            fig, ax = plt.subplots()
            ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_strings[int_type], pdg_string=Definitions.pdg_strings[-1])
            ValidationFunc.PlotVariable(reco_mask, event_branches, plot_var, ax, Definitions.int_strings[int_type], Definitions.int_color[int_type])
            plt.close(fig) 
            fig.savefig(f'{plot_dir_path}/Reco/{plot_var.tree_name}/{file_name}.pdf', bbox_inches='tight')           
    
        # Plot diff_vars
        for plot_var in Variables.Event_diff_plotting_vars :
            fig, ax = plt.subplots()
            ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_strings[int_type], pdg_string=Definitions.pdg_strings[-1])
            ValidationFunc.PlotDiffVariable(reco_mask, event_branches, plot_var, ax, Definitions.int_strings[int_type], Definitions.int_color[int_type])
            plt.close(fig)
            fig.savefig(f'{plot_dir_path}/Diff/{plot_var.true_tree_name}_{plot_var.reco_tree_name}/{file_name}.pdf', bbox_inches='tight')             
    
        # Plot vertex dR plots
        fig, ax = plt.subplots()
        PlotVertexCumulativeDR(event_branches, target_mask, True,  Definitions.int_strings[int_type], Definitions.int_color[int_type], ax, fig)
        plt.close(fig)
        fig.savefig(f'{plot_dir_path}/Reco/{file_name}_CumulativeDR_All.pdf', bbox_inches='tight')  
    
        fig, ax = plt.subplots()
        PlotVertexCumulativeDR(event_branches, reco_mask, False,  Definitions.int_strings[int_type], Definitions.int_color[int_type], ax, fig)
        plt.close(fig)
        fig.savefig(f'{plot_dir_path}/Reco/{file_name}_CumulativeDR_OnlyReco.pdf', bbox_inches='tight') 


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

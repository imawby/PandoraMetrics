import awkward as ak
import matplotlib.pyplot as plt
import Definitions
import Variables
import ValidationFunc

class PlotConfig :
    def __init__(self, file_name, int_string, tier_string, pdg_string, color) :
        self.file_name = file_name
        self.int_string = int_string
        self.tier_string = tier_string        
        self.pdg_string = pdg_string
        self.color = color

##############################################################################################
##############################################################################################

def save_plot(fig, path) :
    plt.close(fig)      
    fig.savefig(path, bbox_inches='tight')

def create_event_plots(target_mask, event_branches, plot_func, plot_vars, sub_dir, plot_config) :
    for plot_var in plot_vars :
        fig, ax = plt.subplots()
        ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=plot_config.int_string, pdg_string=plot_config.pdg_string)
        plot_func(target_mask, event_branches, plot_var, ax, plot_config.int_string, plot_config.color)
        save_plot(fig, f'{sub_dir}/{plot_var.dir_name}/{plot_config.file_name}.pdf')

def plot_cumulative_dr(indices_or_mask, event_branches, plot_var, ax, string, color) :
    accuracy = event_branches[plot_var.tree_name][indices_or_mask]
    positive_entry = (accuracy > 0.0)
    n_entries = len(accuracy)
    sample_points = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,15,20]
    cum_dR = [float(ak.sum(positive_entry & (accuracy < i))) / n_entries for i in sample_points]
    ax.plot(sample_points, cum_dR, color=color, linewidth=1, label='dR')
    ax.legend()
    ax.set_ylim(0.0, 1.0)        
    
##############################################################################################
##############################################################################################

def run_event_validation(plot_dir_path, int_masks, event_branches) :

    for int_type in Definitions.ints :
        # Get masks
        target_mask = int_masks[int_type]
        reco_mask = target_mask & (event_branches['RecoNu_VertexZ'] > -9000)

        # Plot_config
        plot_config = PlotConfig(Definitions.int_file_strings[int_type], Definitions.int_strings[int_type], '', Definitions.pdg_strings[-1], Definitions.int_color[int_type])
        
        # Plot MCP_var distributions
        create_event_plots(target_mask, event_branches, ValidationFunc.PlotVariable, Variables.Event_MCP_plotting_vars, f'{plot_dir_path}/MC', plot_config)

        # Plot BM_var distributions
        create_event_plots(reco_mask, event_branches, ValidationFunc.PlotVariable, Variables.Event_Reco_plotting_vars, f'{plot_dir_path}/Reco', plot_config)        

        # Plot diff_var distributions
        create_event_plots(reco_mask, event_branches, ValidationFunc.PlotDiffVariable, Variables.Event_diff_plotting_vars, f'{plot_dir_path}/Diff', plot_config)
        create_event_plots(target_mask, event_branches, plot_cumulative_dr, [Variables.vtx_dr_all], f'{plot_dir_path}/Diff', plot_config)
        create_event_plots(reco_mask, event_branches, plot_cumulative_dr, [Variables.vtx_dr_only_reco], f'{plot_dir_path}/Diff', plot_config)

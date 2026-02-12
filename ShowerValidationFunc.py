import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions
import ValidationFunc

#####################################################################################################################################################
####################################################################################################################################################

def CreateGraphs(plot_dir_path, config_target_mask, config_reco_mask, tier_masks, int_masks, pdg_masks, shower_branches) :

    for tier in Definitions.tiers :
        tier_mask = tier_masks[tier]
        
        for int_type in Definitions.ints :
            int_mask = int_masks[int_type]
    
            for i_pdg in range(len(Definitions.pdgs)) :
                pdg = Definitions.pdgs[i_pdg]
                pdg_mask = pdg_masks[pdg]
    
                target_mask = config_target_mask & tier_mask & int_mask & pdg_mask
                reco_mask = target_mask & config_reco_mask
    
                # Plot MCP_var distributions
                for plot_var in ValidationFunc.Shower_MCP_plotting_vars :
                    fig, ax = plt.subplots()
                    ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, plot_var)                
                    ValidationFunc.PlotVariable(target_mask, shower_branches, plot_var, ax, Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                    plt.close(fig)
                    file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                    fig.savefig(f'{plot_dir_path}/MCP/{plot_var.tree_name}/{file_name}.pdf', bbox_inches='tight')                 
    
                # Plot BM_var distributions
                for plot_var in ValidationFunc.Shower_BM_plotting_vars :
                    fig, ax = plt.subplots()
                    ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, plot_var)
                    ValidationFunc.PlotVariable(reco_mask, shower_branches, plot_var, ax, Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                    plt.close(fig)
                    file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                    fig.savefig(f'{plot_dir_path}/BM/{plot_var.tree_name}/{file_name}.pdf', bbox_inches='tight')                     
    
                # Plot diff_vars
                for plot_var in ValidationFunc.Shower_diff_plotting_vars :
                    fig, ax = plt.subplots()
                    ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, plot_var)
                    ValidationFunc.PlotDiffVariable(reco_mask, shower_branches, plot_var, ax, Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                    plt.close(fig)
                    fig.savefig(f'{plot_dir_path}/Diff/{plot_var.true_tree_name}_{plot_var.reco_tree_name}/{file_name}.pdf', bbox_inches='tight')         



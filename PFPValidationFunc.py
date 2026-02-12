import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions
import ValidationFunc



#####################################################################################################################################################
####################################################################################################################################################

def CreateGraphs(plot_dir_path, config_target_mask, config_reco_mask, tier_masks, int_masks, pdg_masks, pfp_branches) :

    for tier in Definitions.tiers :     
        tier_mask = tier_masks[tier]
    
        # Efficiency file
        efficiency_file_name = f'EfficiencyTables_{Definitions.tier_strings[tier]}'
    
        with open(f"{plot_dir_path}/Efficiency/{efficiency_file_name}.txt", "w") as f_efficiency:
            for int_type in Definitions.ints :
                int_mask = int_masks[int_type]
        
                ValidationFunc.PrintEfficiencyTableHeader(int_type, f_efficiency)
        
                # Global track-shower confusion matrices
                fig, ax = plt.subplots()
                TrackShowerClassification(config_reco_mask, pdg_masks, pfp_branches, fig, ax, int_type, tier)
                plt.close(fig)
                file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}'
                fig.savefig(f'{plot_dir_path}/TrackShower/{file_name}.pdf', bbox_inches='tight')    
        
                for i_pdg in range(len(Definitions.pdgs)) :
                    pdg = Definitions.pdgs[i_pdg]
                    pdg_mask = pdg_masks[pdg]
                    target_mask = tier_mask & int_mask & pdg_mask & config_target_mask
                    reco_mask = target_mask & config_reco_mask               
        
                    # Efficiency Metrics
                    efficiency_metrics = ValidationFunc.CalculateEfficiencyMetrics(target_mask, reco_mask, True)
                    ValidationFunc.PrintEfficiencyTableEntry(tier, pdg, efficiency_metrics, f_efficiency)
                    
                    # Track-shower plots
                    for i_var in range(len(ValidationFunc.PFP_track_shower_plotting_vars)) :
                        fig, ax = plt.subplots()
                        ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, ValidationFunc.PFP_track_shower_plotting_vars[i_var])
                        ValidationFunc.TrackShowerAsAFunctionOf(reco_mask, pfp_branches, ValidationFunc.PFP_track_shower_plotting_vars[i_var], fig, ax, Definitions.pdg_strings[pdg])
                        plt.close(fig)
                        file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                        fig.savefig(f'{plot_dir_path}/TrackShower/{ValidationFunc.PFP_track_shower_plotting_vars[i_var].tree_name}/{file_name}.pdf', bbox_inches='tight')                    
        
                    # Plot MCP_var distributions
                    for i_var in range(len(ValidationFunc.PFP_MCP_plotting_vars)) :
                        fig, ax = plt.subplots()
                        ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, ValidationFunc.PFP_MCP_plotting_vars[i_var])
                        ValidationFunc.PlotVariable(target_mask, pfp_branches, ValidationFunc.PFP_MCP_plotting_vars[i_var], ax, Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                        plt.close(fig)
                        file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                        fig.savefig(f'{plot_dir_path}/MCP/{ValidationFunc.PFP_MCP_plotting_vars[i_var].tree_name}/{file_name}.pdf', bbox_inches='tight')                        
        
                    # Plot BM_var distributions
                    for i_var in range(len(ValidationFunc.PFP_BM_plotting_vars)) :
                        fig, ax = plt.subplots()                    
                        ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, ValidationFunc.PFP_BM_plotting_vars[i_var])
                        ValidationFunc.PlotVariable(reco_mask, pfp_branches, ValidationFunc.PFP_BM_plotting_vars[i_var], ax, Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                        plt.close(fig)
                        file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                        fig.savefig(f'{plot_dir_path}/BM/{ValidationFunc.PFP_BM_plotting_vars[i_var].tree_name}/{file_name}.pdf', bbox_inches='tight')                             
                    # Plot ALT_var distributions                        
                    for i_var in range(len(ValidationFunc.PFP_ALT_plotting_vars)) :
                        fig, ax = plt.subplots()
                        ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, ValidationFunc.PFP_ALT_plotting_vars[i_var])
                        ValidationFunc.PlotVariable(target_mask, pfp_branches, ValidationFunc.PFP_ALT_plotting_vars[i_var], ax, Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                        plt.close(fig)
                        file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                        fig.savefig(f'{plot_dir_path}/ALT/{ValidationFunc.PFP_ALT_plotting_vars[i_var].tree_name}/{file_name}.pdf', bbox_inches='tight')  
                        # Now segment
                        for seg_var in ValidationFunc.ALT_seg_vars :
                            fig, ax = plt.subplots()
                            ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, ValidationFunc.PFP_ALT_plotting_vars[i_var])
                            SegmentAltVar(target_mask, pfp_branches, ValidationFunc.PFP_ALT_plotting_vars[i_var], seg_var, ax)
                            file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                            plt.close(fig)
                            fig.savefig(f'{plot_dir_path}/ALT/{ValidationFunc.PFP_ALT_plotting_vars[i_var].tree_name}/Seg/{file_name}_{seg_var.label}.pdf', bbox_inches='tight')
                        
                    # Plot efficiency
                    for i_var in range(len(ValidationFunc.PFP_efficiency_vars)) :
                        fig, ax = plt.subplots()
                        ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, ValidationFunc.PFP_efficiency_vars[i_var])
                        ValidationFunc.PlotEfficiency(target_mask, reco_mask, pfp_branches, ValidationFunc.PFP_efficiency_vars[i_var], fig, ax, Definitions.pdg_color[pdg], Definitions.pdg_strings[pdg])   
                        plt.close(fig)
                        file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                        fig.savefig(f'{plot_dir_path}/Efficiency/{ValidationFunc.PFP_efficiency_vars[i_var].tree_name}/{file_name}.pdf', bbox_inches='tight')                        
        
                    # Plot diff_vars
                    for i_var in range(len(ValidationFunc.PFP_diff_plotting_vars)) :
                        fig, ax = plt.subplots()
                        ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, pdg, ValidationFunc.PFP_diff_plotting_vars[i_var])
                        ValidationFunc.PlotDiffVariable(reco_mask, pfp_branches, ValidationFunc.PFP_diff_plotting_vars[i_var], ax, Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                        plt.close(fig)
                        file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                        fig.savefig(f'{plot_dir_path}/Diff/{ValidationFunc.PFP_diff_plotting_vars[i_var].true_tree_name}_{ValidationFunc.PFP_diff_plotting_vars[i_var].reco_tree_name}/{file_name}.pdf', bbox_inches='tight')                          
                    
    
                ValidationFunc.PrintEfficiencyTableFooter(f_efficiency)

#####################################################################################################################################################
####################################################################################################################################################

def TrackShowerClassification(reco_mask, pdg_masks, pfp_branches, fig, ax, int_type, tier) :
    conf_matrix_eff = []

    for pdg in Definitions.pdgs :
        # Only look at those that have been reconstructed
        target_mask = pdg_masks[pdg] & reco_mask
        n_particle = ak.sum(target_mask)
        n_track = ak.sum(pfp_branches['BM_IsTrack'][target_mask] == 1)
        n_shower = ak.sum(pfp_branches['BM_IsShower'][target_mask] == 1)
        conf_matrix_eff.append([round(n_track / n_particle, 2), round(n_shower / n_particle, 2)])
        
    conf_matrix_eff = np.array(conf_matrix_eff)
    im = ax.imshow(conf_matrix_eff, cmap='Blues')
    # Axis ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Track", "Shower"])
    ax.set_yticks(range(len(Definitions.pdgs)))
    ax.set_yticklabels([str(p) for p in Definitions.pdgs])
    # Axis labels and title
    ax.set_xlabel("Reco Classification")
    ax.set_ylabel("True PDG")
    ax.set_title(f'{Definitions.int_strings[int_type]}: {Definitions.tier_strings[tier]}')    
    
    # Add text inside cells
    for i in range(conf_matrix_eff.shape[0]):
        for j in range(conf_matrix_eff.shape[1]):
            ax.text(j, i, conf_matrix_eff[i, j],
                    ha="center", va="center", color=("white" if conf_matrix_eff[i, j] > 0.5 else "black"))
    
    plt.tight_layout()

#####################################################################################################################################################
####################################################################################################################################################

def SegmentAltVar(target_mask, pfp_branches, plot_var, seg_var, ax):

    n_entries = ak.count_nonzero(target_mask)
    
    for index in range(len(seg_var.options)) :
        mask = target_mask & (pfp_branches[seg_var.tree_name] == seg_var.options[index])
        ValidationFunc.PlotVariable(mask, pfp_branches, plot_var, ax, f'{seg_var.options[index]}', seg_var.colors[index], fill=False, n_entries=n_entries)

#####################################################################################################################################################
#####################################################################################################################################################

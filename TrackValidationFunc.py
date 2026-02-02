import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions
import ValidationFunc


#####################################################################################################################################################
####################################################################################################################################################

def CreateGraphs(plot_dir_path, config_target_mask, config_reco_mask, tier_masks, int_masks, pdg_masks, hierarchy_branches, pfp_branches, track_branches, michel_type) :
    efficiency_file_name = f'MichelEfficiencyTables'
    hierarchy_file_name = f'MichelHierarchyTables'

    michel_type_mask = (track_branches['MCP_MichelFromMuon'] == 1) if michel_type == 0 else (track_branches['MCP_MichelFromMuon'] == 0) if michel_type == 1 else ak.ones_like(track_branches['MCP_MichelFromMuon'], dtype=bool)
    
    with open(f"{plot_dir_path}/Efficiency/{efficiency_file_name}.txt", "w") as f_efficiency, open(f"{plot_dir_path}/Hierarchy/{hierarchy_file_name}.txt", "w") as f_hierarchy:
        for int_type in Definitions.ints :
            int_mask = int_masks[int_type]
    
            ValidationFunc.PrintHierarchyTableHeader(int_type, f_hierarchy)
            ValidationFunc.PrintEfficiencyTableHeader(int_type, f_efficiency)
            
            for tier in Definitions.tiers :
                file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}Parent_{Definitions.pdg_strings[777]}'
                tier_mask = tier_masks[tier]
                
                # Get target muon = target muon that has been reco'd
                target_muon_mask = tier_mask & int_mask & config_target_mask & config_reco_mask & michel_type_mask & (track_branches['MCP_HasTargetMichel'] == 1)
                target_muon_michel_indices = track_branches['MCP_MichelIndex'][target_muon_mask]
                
                # Make sure that michel is reco target
                target_michel_indices = target_muon_michel_indices[config_target_mask[target_muon_michel_indices]]
                reco_michel_indices = target_muon_michel_indices[config_target_mask[target_muon_michel_indices] & config_reco_mask[target_muon_michel_indices]]
    
                # Hierarchy Metrics
                hierarchy_metrics = ValidationFunc.CalculateHierarchyMetrics(hierarchy_branches, reco_michel_indices)
                ValidationFunc.PrintHierarchyTableEntry(tier, hierarchy_metrics, f_hierarchy)         
    
                # Efficiency Metrics
                efficiency_metrics = ValidationFunc.CalculateEfficiencyMetrics(target_michel_indices, reco_michel_indices, False)
                ValidationFunc.PrintEfficiencyTableEntry(tier, 777, efficiency_metrics, f_efficiency)
                
                # Plot MCP_var distributions
                for i_var in range(len(ValidationFunc.Track_MCP_plotting_vars)) :
                    fig, ax = plt.subplots()
                    ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, 777, ValidationFunc.Track_MCP_plotting_vars[i_var])
                    ValidationFunc.PlotVariable(target_michel_indices, pfp_branches, ValidationFunc.Track_MCP_plotting_vars[i_var], ax, Definitions.pdg_strings[777], Definitions.pdg_color[777])
                    plt.close(fig)                    
                    fig.savefig(f'{plot_dir_path}/MCP/{ValidationFunc.Track_MCP_plotting_vars[i_var].tree_name}/{file_name}.pdf', bbox_inches='tight')                     

                # Plot track/shower classifications
                for i_var in range(len(ValidationFunc.Track_track_shower_vars)) :
                    fig, ax = plt.subplots()
                    ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, 777, ValidationFunc.Track_track_shower_vars[i_var])
                    ValidationFunc.TrackShowerAsAFunctionOf(reco_michel_indices, pfp_branches, ValidationFunc.Track_track_shower_vars[i_var], fig, ax, Definitions.pdg_strings[777])
                    plt.close(fig)
                    fig.savefig(f'{plot_dir_path}/TrackShower/{ValidationFunc.Track_track_shower_vars[i_var].tree_name}/{file_name}.pdf', bbox_inches='tight')                       
    
                # Plot efficiency
                for i_var in range(len(ValidationFunc.Track_efficiency_vars)) :
                    fig, ax = plt.subplots()                    
                    ValidationFunc.ConfigurePlot(fig, ax, int_type, tier, 777, ValidationFunc.Track_efficiency_vars[i_var])
                    ValidationFunc.PlotEfficiency(target_michel_indices, reco_michel_indices, pfp_branches, ValidationFunc.Track_efficiency_vars[i_var], fig, ax, Definitions.pdg_color[777], Definitions.pdg_strings[777])
                    plt.close(fig)
                    fig.savefig(f'{plot_dir_path}/Efficiency/{ValidationFunc.Track_efficiency_vars[i_var].tree_name}/{file_name}.pdf', bbox_inches='tight')                        
                        
            ValidationFunc.PrintHierarchyTableFooter(f_hierarchy)
            ValidationFunc.PrintEfficiencyTableFooter(f_efficiency)


    
##############################################################################################
##############################################################################################

def plot_michel_efficiency(target_michel_indices, reco_michel_indices, pfp_branches, plot_var, fig, ax) :
    target_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][target_michel_indices]))
    reco_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][reco_michel_indices]))
    
    hist_target, edges = np.histogram(target_entries, bins=plot_var.n_bins, range=plot_var.range)
    hist_reco, _ = np.histogram(reco_entries, bins=plot_var.n_bins, range=plot_var.range)
    efficiency = np.divide(hist_reco, hist_target, 
                           out=np.zeros_like(hist_reco, dtype=float), 
                           where=hist_target > 0)
    
    # Binomial efficiency uncertainty
    efficiency_err = np.zeros_like(efficiency)
    valid = hist_target > 0
    efficiency_err[valid] = np.sqrt(
        efficiency[valid] * (1.0 - efficiency[valid]) / hist_target[valid]
    )

    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    ax.errorbar(bin_centers, efficiency, yerr=efficiency_err, fmt='x-', color='black', capsize=3, label=f' Michel ')
    ax.legend()

##############################################################################################
##############################################################################################

def plot_michel_var(target_michel_indices, pfp_branches, plot_var, fig, ax) :
    target_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][target_michel_indices]))
    n_target_entries = len(target_entries)

    if (n_target_entries == 0) :
        return
    
    weights = np.ones(n_target_entries) * (1.0 / n_target_entries)
    
    ax.hist(target_entries, bins=plot_var.n_bins, range=plot_var.range, weights=weights, histtype='step', color='black', linewidth=1, label=(f' Michel '))
    ax.legend()


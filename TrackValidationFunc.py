import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions
import ValidationFunc
import Variables

#####################################################################################################################################################
####################################################################################################################################################

def CreateTrackGraphs(plot_dir_path, config_target_mask, config_reco_mask, tier_masks, int_masks, pdg_masks, track_branches) :

    for int_type in Definitions.ints :
        int_mask = int_masks[int_type]
            
        for tier in Definitions.tiers :
            tier_mask = tier_masks[tier]

            for i_pdg in range(len(Definitions.pdgs)) :
                pdg = Definitions.pdgs[i_pdg]
                file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                
                pdg_mask = pdg_masks[pdg]
                target_mask = tier_mask & int_mask & pdg_mask & config_target_mask
                reco_mask = target_mask & config_reco_mask   
                
                # Plot MCP_var distributions
                for plot_var in Variables.Track_MCP_plotting_vars :
                    fig, ax = plt.subplots()
                    ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_file_strings[int_type],  tier_string=Definitions.tier_strings[tier], pdg_string=Definitions.pdg_strings[pdg])
                    ValidationFunc.PlotVariable(target_mask, track_branches,plot_var, ax, Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                    plt.close(fig)                    
                    fig.savefig(f'{plot_dir_path}/MCP/{plot_var.tree_name}/{file_name}.pdf', bbox_inches='tight') 

                # Plot BM_var distributions
                for plot_var in Variables.Track_BM_plotting_vars :
                    fig, ax = plt.subplots()
                    ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_file_strings[int_type],  tier_string=Definitions.tier_strings[tier], pdg_string=Definitions.pdg_strings[pdg])
                    ValidationFunc.PlotVariable(reco_mask, track_branches, plot_var, ax, Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                    plt.close(fig)
                    fig.savefig(f'{plot_dir_path}/BM/{plot_var.tree_name}/{file_name}.pdf', bbox_inches='tight')                     

#####################################################################################################################################################
####################################################################################################################################################

def CreateMichelGraphs(plot_dir_path, config_target_mask, config_reco_mask, tier_masks, int_masks, pdg_masks, hierarchy_branches, pfp_branches, track_branches, michel_type) :
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
                for plot_var in Variables.Michel_MCP_plotting_vars :
                    fig, ax = plt.subplots()
                    ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_file_strings[int_type],  pdg_string=Definitions.pdg_strings[777])
                    ValidationFunc.PlotVariable(target_michel_indices, pfp_branches, plot_var, ax, Definitions.pdg_strings[777], Definitions.pdg_color[777])
                    plt.close(fig)                    
                    fig.savefig(f'{plot_dir_path}/MCP/{plot_var.tree_name}/{file_name}.pdf', bbox_inches='tight')                     

                # Plot track/shower classifications
                for plot_var in Variables.Michel_track_shower_vars :
                    fig, ax = plt.subplots()
                    ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_file_strings[int_type],  pdg_string=Definitions.pdg_strings[777])
                    ValidationFunc.TrackShowerAsAFunctionOf(reco_michel_indices, pfp_branches, plot_var, fig, ax, Definitions.pdg_strings[777])
                    plt.close(fig)
                    fig.savefig(f'{plot_dir_path}/TrackShower/{plot_var.tree_name}/{file_name}.pdf', bbox_inches='tight')                       
    
                # Plot efficiency
                for plot_var in Variables.Michel_efficiency_vars :
                    fig, ax = plt.subplots()                    
                    ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_file_strings[int_type],  pdg_string=Definitions.pdg_strings[777])
                    ValidationFunc.PlotEfficiency(target_michel_indices, reco_michel_indices, pfp_branches, plot_var, fig, ax, Definitions.pdg_color[777], Definitions.pdg_strings[777])
                    plt.close(fig)
                    fig.savefig(f'{plot_dir_path}/Efficiency/{plot_var.tree_name}/{file_name}.pdf', bbox_inches='tight')                        
                        
            ValidationFunc.PrintHierarchyTableFooter(f_hierarchy)
            ValidationFunc.PrintEfficiencyTableFooter(f_efficiency)
    
# ##############################################################################################
# ##############################################################################################



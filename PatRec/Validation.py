# Imports
import argparse
import uproot
import numpy as np
import awkward as ak

import Definitions
import Variables
#import ValidationFunc
import EventValidationFunc
import HierarchyValidationFunc

import os

def main(args) :

    #########################
    # Handle file
    #########################
    file_name = f'{args.validation_file}'
    file = uproot.open(file_name)

    #########################
    # Get branches
    #########################
    event_tree = file['EventTree']
    pfp_tree = file['PFPTree']
    track_tree = file['TrackTree']
    shower_tree = file['ShowerTree']
    hierarchy_tree = file['HierarchyTree']

    event_branches = event_tree.arrays(['Run', 'Subrun', 'Event', 'MCInt_IsCC', 'MCNu_PDG', 'MCNu_Energy', 'MCNu_VisEnergy',
                                        'MCNu_VertexX', 'MCNu_VertexY', 'MCNu_VertexZ',
                                        'RecoNu_VertexX', 'RecoNu_VertexY', 'RecoNu_VertexZ',
                                        'RecoNu_VertexAcc_Pass2'], library="ak")

    pfp_branches = pfp_tree.arrays(['MCP_TruePDG', 'BM_VertexAcc', 'MCP_TrueVisEnergy', 
                                    'MCP_NMCHits2D', 'MCP_NMCHitsU', 'MCP_NMCHitsV', 'MCP_NMCHitsW',
                                    'BM_Completeness', 'BM_Purity'], library="ak")

    shower_branches = shower_tree.arrays(['BM_InitialCompleteness'], library="ak")

    track_branches = track_tree.arrays(['BM_EndpointAcc'], library="ak")

    hierarchy_branches = hierarchy_tree.arrays(['MC_HierarchyTier', 'MC_ParentIndex', 'BM_HierarchyTier', 'BM_ParentIndex'], library="ak")

    ###################################
    # Create interaction/tier/PDG masks
    ###################################
    int_masks = Definitions.GetIntMasks(event_branches, pfp_branches={}, broadcast=False)
    int_masks_broadcast = Definitions.GetIntMasks(event_branches, pfp_branches)
    pdg_masks = Definitions.GetPDGMasks(pfp_branches)
    tier_masks = Definitions.GetTierMasks(hierarchy_branches)

    ##########################################################################################################
    # Implement custom definitions of reconstructable and reconstructed, to be used in correctness definitions
    ##########################################################################################################
    # Apply custom def of reconstructable and reconstructed
    # The tree constains some particles that we reconstructed, even if not initially deemed to be a target
    pfp_target_mask = Definitions.GetIsTargetMask(pfp_branches)
    pfp_reco_mask = Definitions.GetIsRecoMask(pfp_target_mask, pfp_branches)

    # Event Validation Plots
    event_plot_dir = f'{args.plot_dir}/EventValidation/'
    EventValidationFunc.run_event_validation(event_plot_dir, int_masks, event_branches)

    # Hierarchy Validation Plots
    hierarchy_plot_dir = f'{args.plot_dir}/HierarchyValidation/'
    HierarchyValidationFunc.run_hierarchy_validation(hierarchy_plot_dir, pfp_target_mask, pfp_reco_mask, int_masks_broadcast, tier_masks, pdg_masks, hierarchy_branches, pfp_branches)
    
##########################################################################################################
##########################################################################################################
    
def create_directory_structure(plot_dir) :
    if not os.path.isdir(plot_dir) :
        os.makedirs(plot_dir)

    # EventTree
    create_tree_directory(plot_dir, 'EventValidation')
    create_subdirectory(f'{plot_dir}/EventValidation', 'MC', Variables.Event_MCP_plotting_vars)
    create_subdirectory(f'{plot_dir}/EventValidation', 'Reco', Variables.Event_Reco_plotting_vars)
    create_subdirectory(f'{plot_dir}/EventValidation', 'Diff', Variables.Event_diff_plotting_vars + [Variables.vtx_dr_all, Variables.vtx_dr_only_reco])

    # HierarchyTree
    create_tree_directory(plot_dir, 'HierarchyValidation')


##########################################################################################################
##########################################################################################################
    
def create_tree_directory(root_dir, tree_name) :
    if not os.path.isdir(f'{root_dir}/{tree_name}') :
        os.makedirs(f'{root_dir}/{tree_name}')
    
##########################################################################################################
##########################################################################################################
    
def create_subdirectory(root_dir, sub_dir, plot_vars) :
    if not os.path.isdir(f'{root_dir}/{sub_dir}') :
        os.makedirs(f'{root_dir}/{sub_dir}/')
    for plot_var in plot_vars :
        if not os.path.isdir(f'{root_dir}/{sub_dir}/{plot_var.dir_name}') :
            os.makedirs(f'{root_dir}/{sub_dir}/{plot_var.dir_name}')
            
##########################################################################################################
##########################################################################################################
            
def parse_cli():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Validation script for Pandora.")
    parser.add_argument("--plot_dir", type=str, required=True, help="Directory for storing plots.")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to the validation file.")
    return parser.parse_args()

##########################################################################################################
##########################################################################################################

if __name__ == "__main__":
    args = parse_cli()
    create_directory_structure(args.plot_dir)
    main(args)

# Imports
import argparse
import uproot
import numpy as np
import awkward as ak

import Definitions
import Variables
#import ValidationFunc
import EventValidationFunc

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

    hierarchy_branches = hierarchy_tree.arrays(['MC_HierarchyTier'], library="ak")

    ###################################
    # Create interaction/tier/PDG masks
    ###################################
    int_masks = Definitions.GetIntMasks(event_branches, pfp_branches={}, broadcast=False)
    int_masks_broadcast = Definitions.GetIntMasks(event_branches, pfp_branches)
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
    EventValidationFunc.create_graphs(event_plot_dir, int_masks, event_branches)

##########################################################################################################
##########################################################################################################
    
def create_directory_structure(plot_dir) :
    # Make directory structure for plots
    if not os.path.isdir(plot_dir) :
        os.makedirs(plot_dir)

    # EventTree plots
    event_plot_dir = f'{plot_dir}/EventValidation/'
    if not os.path.isdir(f'{event_plot_dir}') :
        os.makedirs(f'{event_plot_dir}')

    #  MCP_var distributions
    if not os.path.isdir(f'{event_plot_dir}/MC/') :
        os.makedirs(f'{event_plot_dir}/MC/')
    for plot_var in Variables.Event_MCP_plotting_vars :
        if not os.path.isdir(f'{event_plot_dir}/MC/{plot_var.dir_name}') :
            os.makedirs(f'{event_plot_dir}/MC/{plot_var.dir_name}')

    #  Reco distributions
    if not os.path.isdir(f'{event_plot_dir}/Reco/') :
        os.makedirs(f'{event_plot_dir}/Reco/')
    for plot_var in Variables.Event_Reco_plotting_vars :
        if not os.path.isdir(f'{event_plot_dir}/Reco/{plot_var.dir_name}') :
            os.makedirs(f'{event_plot_dir}/Reco/{plot_var.dir_name}')
            
    #  Diff distributions
    if not os.path.isdir(f'{event_plot_dir}/Diff/') :
        os.makedirs(f'{event_plot_dir}/Diff/')
    for plot_var in Variables.Event_diff_plotting_vars :
        if not os.path.isdir(f'{event_plot_dir}/Diff/{plot_var.dir_name}') :
            os.makedirs(f'{event_plot_dir}/Diff/{plot_var.dir_name}')
    for plot_var in [Variables.vtx_dr_all, Variables.vtx_dr_only_reco] :
        if not os.path.isdir(f'{event_plot_dir}/Diff/{plot_var.dir_name}') :
            os.makedirs(f'{event_plot_dir}/Diff/{plot_var.dir_name}')            

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

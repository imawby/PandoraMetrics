#!/usr/bin/env python
# coding: utf-8

# ## Event Validation
# 
# written by Isobel Mawby (i.mawby1@lancaster.ac.uk)

# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Imports
# </div>

# In[ ]:


import random
import uproot
import numpy as np
import math
import matplotlib.pyplot as plt
import awkward as ak
import Definitions
import Variables
import ValidationFunc
import EventValidationFunc
import os


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     File
# </div>

# In[ ]:


file_name = "/exp/dune/data/users/imawby/warwickWorkshop/shower_HD_LBL/validation_shower_HD_LBL.root"


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Lets open the file...
# </div>

# In[ ]:


file = uproot.open(file_name)


# In[ ]:


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


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Is Track and Shower
# </div>

# In[ ]:


is_mc_track = (abs(pfp_branches['MCP_TruePDG']) != 11) & (pfp_branches['MCP_TruePDG'] != 22) & (pfp_branches['MCP_TruePDG'] != 111) & (pfp_branches['MCP_TruePDG'] != 777)

pfp_branches = ak.with_field(
    pfp_branches,
    ak.values_astype(is_mc_track, np.int32),
    "MCP_IsTrack"
)


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Summary
# </div>

# In[ ]:


int_masks = Definitions.GetIntMasks(event_branches, pfp_branches={}, broadcast=False)
tier_masks = Definitions.GetTierMasks(hierarchy_branches)
mc_track_shower_masks = Definitions.GetTrackShowerMasks(pfp_branches)

# Apply custom def of reconstructable and reconstructed
# The tree constains some particles that we reconstructed, even if not initially deemed to be a target
pfp_target_mask = Definitions.GetIsTargetMask(pfp_branches)
pfp_reco_mask = Definitions.GetIsRecoMask(pfp_target_mask, pfp_branches)


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Add VisEnergyFraction
# </div>

# In[ ]:


pfp_vis = pfp_branches['MCP_TrueVisEnergy']
nu_vis =  ak.broadcast_arrays(event_branches['MCNu_VisEnergy'], pfp_vis)[0]
pfp_vis_frac = ak.where(nu_vis > 0.0,  pfp_vis / nu_vis, 0.0)

pfp_branches = ak.with_field(
    pfp_branches,
    pfp_vis_frac,
    "MCP_VisEnergyFrac"
)


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Add IsCorrect
# </div>

# In[ ]:


#is_correct_track = (abs(pfp_branches['BM_VertexAcc']) < 5.0) & (abs(track_branches['BM_EndpointAcc']) < 5.0)
is_correct_track = (shower_branches['BM_InitialCompleteness'] > 0.8) & (abs(track_branches['BM_EndpointAcc']) < 5.0)
is_correct_shower = (abs(pfp_branches['BM_VertexAcc']) < 3.0) & (pfp_branches['BM_Completeness'] > 0.8) & (pfp_branches['BM_Purity'] > 0.8)
is_correct = ak.where((is_mc_track & is_correct_track) | (~is_mc_track & is_correct_shower), True, False)

pfp_branches = ak.with_field(
    pfp_branches,
    is_correct,
    "MCP_IsCorrect"
)


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Add event-level PFP-correctness
# </div>

# In[ ]:


for tier in Definitions.tiers :
    for track_shower_type in Definitions.track_shower_types :

        target = pfp_target_mask & tier_masks[tier] & mc_track_shower_masks[track_shower_type]
        correct = target & pfp_branches['MCP_IsCorrect']

        n_target = ak.sum(pfp_branches['MCP_VisEnergyFrac'][target], axis=1)
        n_correct = ak.sum(pfp_branches['MCP_VisEnergyFrac'][correct], axis=1)

        correct_frac = ak.where(n_target > 0.0001, n_correct / n_target, -1.0)

        target_string = f'MC_N{Definitions.tier_strings[tier]}{Definitions.track_shower_strings[track_shower_type]}'
        frac_string = f'MC_FracCorrect{Definitions.tier_strings[tier]}{Definitions.track_shower_strings[track_shower_type]}'

        event_branches = ak.with_field(event_branches, correct_frac, frac_string)

# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Add event-level event-correctness
# </div>

# In[ ]:


previous_tier_correct = ak.ones_like(event_branches["Run"])

for tier in Definitions.tiers :

    tier_mask = tier_masks[tier]
    target = pfp_target_mask & tier_mask
    correct = target & pfp_branches['MCP_IsCorrect']
    
    n_target = ak.sum(target, axis=1)
    n_correct = ak.sum(correct, axis=1)
    is_event_correct = ak.where(n_target > 0, n_target == n_correct, -1)
    # update only if the previous tier was correct, or if no longer valid i.e. no targets in tier
    is_event_correct = ak.where((is_event_correct == -1) | (previous_tier_correct == 1), is_event_correct, previous_tier_correct) 
    # Save in tree..
    tree_string = f'MCNu_IsCorrectTo{Definitions.tier_strings[tier]}'
    event_branches = ak.with_field(event_branches, is_event_correct, tree_string)

    # Record for next tier
    previous_tier_correct = is_event_correct


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Make directories
# </div>

# In[ ]:

if not os.path.isdir(f'{os.environ["PLOT_DIR"]}/Plots/') :
    os.makedirs(f'{os.environ["PLOT_DIR"]}/Plots/')

plot_dir = os.environ["PLOT_DIR"]+'/Plots/EventValPlots'
if not os.path.isdir(f'{plot_dir}') :
    os.makedirs(f'{plot_dir}')

#  MCP_var distributions
if not os.path.isdir(f'{plot_dir}/MCP/') :
    os.makedirs(f'{plot_dir}/MCP/')
for plot_var in Variables.Event_MCP_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/MCP/{plot_var.tree_name}') :
        os.makedirs(f'{plot_dir}/MCP/{plot_var.tree_name}')

#  Reco distributions
if not os.path.isdir(f'{plot_dir}/Reco/') :
    os.makedirs(f'{plot_dir}/Reco/')
for plot_var in Variables.Event_Reco_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/Reco/{plot_var.tree_name}') :
        os.makedirs(f'{plot_dir}/Reco/{plot_var.tree_name}')
        
#  Diff distributions
if not os.path.isdir(f'{plot_dir}/Diff/') :
    os.makedirs(f'{plot_dir}/Diff/')
for plot_var in Variables.Event_diff_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/Diff/{plot_var.true_tree_name}_{plot_var.reco_tree_name}') :
        os.makedirs(f'{plot_dir}/Diff/{plot_var.true_tree_name}_{plot_var.reco_tree_name}')

#  IsCorrect
if not os.path.isdir(f'{plot_dir}/IsCorrect/') :
    os.makedirs(f'{plot_dir}/IsCorrect/')

for string in ['PFP', 'Event'] :
    if not os.path.isdir(f'{plot_dir}/IsCorrect/{string}') :
        os.makedirs(f'{plot_dir}/IsCorrect/{string}')

    # XProfile
    if not os.path.isdir(f'{plot_dir}/XProfile/{string}/') :
        os.makedirs(f'{plot_dir}/XProfile/{string}/')
    for plot_var in Variables.Event_MCP_plotting_vars :
        if not os.path.isdir(f'{plot_dir}/XProfile/{string}/{plot_var.tree_name}') :
            os.makedirs(f'{plot_dir}/XProfile/{string}/{plot_var.tree_name}')


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Event-level pfp-correctness distributions
# </div>

# In[ ]:


for int_type in Definitions.ints :
    for tier in Definitions.tiers :
        for track_shower_type in Definitions.track_shower_types :
    
            tree_string = f'MC_FracCorrect{Definitions.tier_strings[tier]}{Definitions.track_shower_strings[track_shower_type]}'

            mask = int_masks[int_type] & (event_branches[tree_string] > -0.01)
            
            plot_var = Variables.PlotVar(tree_string, f'{Variables.event_pfp_correctness_frac.x_label} {Definitions.track_shower_strings[track_shower_type]}',
                                         Variables.event_pfp_correctness_frac.y_label, Variables.event_pfp_correctness_frac.range, Variables.event_pfp_correctness_frac.n_bins)
    
            fig, ax = plt.subplots()
            ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_strings[int_type], tier_string=Definitions.tier_strings[tier])
            ValidationFunc.PlotVariable(mask, event_branches, plot_var, ax, tree_string, 'blue')
            plt.close(fig)
            fig.savefig(f'{plot_dir}/IsCorrect/PFP/{Definitions.int_file_strings[int_type]}_{tree_string}.pdf', bbox_inches='tight')


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Event-level PFP-correctness profiles
# </div>

# In[ ]:


for int_type in Definitions.ints :
    for tier in Definitions.tiers :
        for track_shower_type in Definitions.track_shower_types :
    
            tree_string = f'MC_FracCorrect{Definitions.tier_strings[tier]}{Definitions.track_shower_strings[track_shower_type]}'

            mask = int_masks[int_type] & (event_branches[tree_string] > -0.01)
            
            correctness_var = Variables.PlotVar(tree_string, f'{Variables.event_pfp_correctness_frac.x_label} {Definitions.track_shower_strings[track_shower_type]}',
                                         Variables.event_pfp_correctness_frac.y_label, Variables.event_pfp_correctness_frac.range, Variables.event_pfp_correctness_frac.n_bins)

            for plot_var in Variables.Event_MCP_plotting_vars :
                profile_var = Variables.ProfileVar(plot_var, correctness_var)
                fig, ax = plt.subplots()
                ValidationFunc.ConfigurePlot(fig, ax, correctness_var, Definitions.int_strings[int_type], Definitions.tier_strings[tier], Definitions.track_shower_strings[track_shower_type])
                ValidationFunc.PlotProfileX(mask, event_branches, profile_var, ax, tree_string, 'blue')
                plt.close(fig)
                fig.savefig(f'{plot_dir}/XProfile/PFP/{plot_var.tree_name}/{Definitions.int_file_strings[int_type]}_{tree_string}.pdf', bbox_inches='tight')


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Event-correctness profiles
# </div>

# In[ ]:

for int_type in Definitions.ints :
    for tier in Definitions.tiers :

        tree_string = f'MCNu_IsCorrectTo{Definitions.tier_strings[tier]}'
                
        # Ignore 'invalid cases' where the tier has no particles... 
        mask = int_masks[int_type] & (event_branches[tree_string] > -0.1)
    
                
        plot_var = Variables.PlotVar(tree_string, f'{Variables.event_correctness.x_label}',
                                     Variables.event_correctness.y_label, Variables.event_correctness.range, Variables.event_correctness.n_bins)
        
        fig, ax = plt.subplots()
        ValidationFunc.ConfigurePlot(fig, ax, plot_var, int_string=Definitions.int_strings[int_type], tier_string=Definitions.tier_strings[tier])
        ValidationFunc.PlotVariable(mask, event_branches, plot_var, ax, tree_string, 'blue')
        plt.close(fig)
        fig.savefig(f'{plot_dir}/IsCorrect/Event/{Definitions.int_file_strings[int_type]}_{tree_string}.pdf', bbox_inches='tight')

# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Get plots + tables
# </div>

# In[ ]:

EventValidationFunc.CreateGraphs(plot_dir, int_masks, event_branches)

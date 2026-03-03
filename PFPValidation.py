#!/usr/bin/env python
# coding: utf-8

# ## PFP Validation
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
import PFPValidationFunc
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
hierarchy_tree = file['HierarchyTree']

event_branches = event_tree.arrays(['Run', 'Subrun', 'Event', 'MCInt_IsCC', 'MCNu_PDG'], library="ak")

pfp_branches = pfp_tree.arrays(['Run', 'Subrun', 'Event',
                                'MCP_TruePDG', 'MCP_TrueEnergy', 'MCP_TrueVisEnergy', 'MCP_TrueThetaXZ', 'MCP_TrueThetaYZ',
                                'MCP_NMCHits2D', 'MCP_NMCHitsU', 'MCP_NMCHitsV', 'MCP_NMCHitsW',
                                'MCP_HasMatch', 'MCP_Length', 'MCP_Displacement',
                                'BM_IsTrack', 'BM_IsShower',
                                'BM_Completeness', 'BM_CompletenessU', 'BM_CompletenessV', 'BM_CompletenessW',
                                'BM_Purity', 'BM_PurityU', 'BM_PurityV', 'BM_PurityW',
                                'BM_VertexAcc', 'BM_Length', 'BM_Displacement', 
                                'ALT_Completeness', 'ALT_Purity', 'ALT_PDG', 'ALT_IsUpstreamHierarchy', 'ALT_IsSameMC'], library="ak")

track_branches = track_tree.arrays(['BM_EndpointAcc'], library="ak")
hierarchy_branches = hierarchy_tree.arrays(['MC_HierarchyTier'], library="ak")


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Make directories
# </div>

# In[ ]:

if not os.path.isdir(f'{os.environ["PLOT_DIR"]}/Plots/') :
    os.makedirs(f'{os.environ["PLOT_DIR"]}/Plots/')

plot_dir = os.environ["PLOT_DIR"]+'/Plots/PFPValPlots'
if not os.path.isdir(f'{plot_dir}') :
    os.makedirs(f'{plot_dir}')

#  MCP_var distributions
if not os.path.isdir(f'{plot_dir}/MCP/') :
    os.makedirs(f'{plot_dir}/MCP/')
for plot_var in Variables.PFP_MCP_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/MCP/{plot_var.tree_name}') :
        os.makedirs(f'{plot_dir}/MCP/{plot_var.tree_name}')

# BM_var distributions
if not os.path.isdir(f'{plot_dir}/BM/') :
    os.makedirs(f'{plot_dir}/BM/')
for plot_var in Variables.PFP_BM_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/BM/{plot_var.tree_name}') :
        os.makedirs(f'{plot_dir}/BM/{plot_var.tree_name}')

# ALT_var distributions
if not os.path.isdir(f'{plot_dir}/ALT/') :
    os.makedirs(f'{plot_dir}/ALT/')
for plot_var in Variables.PFP_ALT_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/ALT/{plot_var.tree_name}') :
        os.makedirs(f'{plot_dir}/ALT/{plot_var.tree_name}')
    if not os.path.isdir(f'{plot_dir}/ALT/{plot_var.tree_name}/Seg') :
        os.makedirs(f'{plot_dir}/ALT/{plot_var.tree_name}/Seg')

# Diff distributions
if not os.path.isdir(f'{plot_dir}/Diff/') :
    os.makedirs(f'{plot_dir}/Diff/')
for plot_var in Variables.PFP_diff_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/Diff/{plot_var.true_tree_name}_{plot_var.reco_tree_name}') :
        os.makedirs(f'{plot_dir}/Diff/{plot_var.true_tree_name}_{plot_var.reco_tree_name}')

# Track-shower
if not os.path.isdir(f'{plot_dir}/TrackShower/') :
    os.makedirs(f'{plot_dir}/TrackShower/')
for plot_var in Variables.PFP_track_shower_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/TrackShower/{plot_var.tree_name}') :
        os.makedirs(f'{plot_dir}/TrackShower/{plot_var.tree_name}')

# Efficiency
if not os.path.isdir(f'{plot_dir}/Efficiency/') :
    os.makedirs(f'{plot_dir}/Efficiency/')
for plot_var in Variables.PFP_efficiency_vars :
    if not os.path.isdir(f'{plot_dir}/Efficiency/{plot_var.tree_name}') :
        os.makedirs(f'{plot_dir}/Efficiency/{plot_var.tree_name}')

# Profile vars
if not os.path.isdir(f'{plot_dir}/XProfile/') :
    os.makedirs(f'{plot_dir}/XProfile/')
for profile_var in Variables.PFP_profile_vars :
    if not os.path.isdir(f'{plot_dir}/XProfile/{profile_var.plot_var_y.tree_name}_{profile_var.plot_var_x.tree_name}') :
        os.makedirs(f'{plot_dir}/XProfile/{profile_var.plot_var_y.tree_name}_{profile_var.plot_var_x.tree_name}')



# In[ ]:

# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Summary
# </div>

# In[ ]:


int_masks = Definitions.GetIntMasks(event_branches, pfp_branches)
pdg_masks = Definitions.GetPDGMasks(pfp_branches)
tier_masks = Definitions.GetTierMasks(hierarchy_branches)

# Apply custom def of reconstructable and reconstructed
# The tree constains some particles that we reconstructed, even if not initially deemed to be a target
pfp_target_mask = Definitions.GetIsTargetMask(pfp_branches)
pfp_reco_mask = Definitions.GetIsRecoMask(pfp_target_mask, pfp_branches)


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Add multiplicity
# </div>

# In[ ]:


multiplicity = ak.sum(pfp_target_mask, axis=1)
shower_multiplicity = ak.sum(pfp_target_mask & ((abs(pfp_branches['MCP_TruePDG']) == 11) | (pfp_branches['MCP_TruePDG'] == 22) | (pfp_branches['MCP_TruePDG'] == 777) | (pfp_branches['MCP_TruePDG'] == 111)), axis=1)
track_multiplicity = ak.sum(pfp_target_mask & (abs(pfp_branches['MCP_TruePDG']) != 11) & (pfp_branches['MCP_TruePDG'] != 22) & (pfp_branches['MCP_TruePDG'] != 777) & (pfp_branches['MCP_TruePDG'] != 111), axis=1)

# Match shape to PFP jagged arra
multiplicity = ak.broadcast_arrays(multiplicity, pfp_branches['MCP_TruePDG'])[0]
shower_multiplicity = ak.broadcast_arrays(shower_multiplicity, pfp_branches['MCP_TruePDG'])[0]
track_multiplicity = ak.broadcast_arrays(track_multiplicity, pfp_branches['MCP_TruePDG'])[0]

pfp_branches = ak.with_field(
    pfp_branches,
    multiplicity,
    "MCNu_Multiplicity"
)

pfp_branches = ak.with_field(
    pfp_branches,
    shower_multiplicity,
    "MCNu_ShowerMultiplicity"
)

pfp_branches = ak.with_field(
    pfp_branches,
    track_multiplicity,
    "MCNu_TrackMultiplicity"
)

multiplicity_mask = Definitions.GetMultiplicityMasks(pfp_branches) 


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Add correctness
# </div>

# In[ ]:


is_mc_track = (abs(pfp_branches['MCP_TruePDG']) != 11) & (pfp_branches['MCP_TruePDG'] != 22) & (pfp_branches['MCP_TruePDG'] != 111) & (pfp_branches['MCP_TruePDG'] != 777)
is_correct_track = (abs(pfp_branches['BM_VertexAcc']) < 3.0) & (abs(track_branches['BM_EndpointAcc']) < 3.0)
is_mc_shower = (abs(pfp_branches['MCP_TruePDG']) == 11) | (pfp_branches['MCP_TruePDG'] == 22) | (pfp_branches['MCP_TruePDG'] == 777) | (pfp_branches['MCP_TruePDG'] == 111)
is_correct_shower = (abs(pfp_branches['BM_VertexAcc']) < 3.0) & (pfp_branches['BM_Completeness'] > 0.8) & (pfp_branches['BM_Purity'] > 0.8)
is_correct = ak.where((is_mc_track & is_correct_track) | (is_mc_shower & is_correct_shower), True, False)

pfp_branches = ak.with_field(
    pfp_branches,
    is_correct,
    "MCP_IsCorrect"
)

# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Variables to plot
# </div>

# In[ ]:


PFPValidationFunc.CreateGraphs(plot_dir, pfp_target_mask, pfp_reco_mask, tier_masks, int_masks, pdg_masks, pfp_branches)


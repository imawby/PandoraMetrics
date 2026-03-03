#!/usr/bin/env python
# coding: utf-8

# ## Track Validation
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
import ValidationFunc
import TrackValidationFunc
import Variables
import os


# In[ ]:

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
                                'MCP_TruePDG', 'MCP_TrueVisEnergy', 'MCP_HasMatch',
                                'MCP_NMCHits2D', 'MCP_NMCHitsU', 'MCP_NMCHitsV', 'MCP_NMCHitsW',
                                'BM_Completeness', 'BM_Purity', 'BM_IsTrack', 'BM_IsShower'], library="ak")

track_branches = track_tree.arrays(['BM_EndpointAcc', 'BM_EndpointCompleteness', 'BM_EndpointPurity', 
                                    'MCP_HasMichel', 'MCP_HasTargetMichel', 'BM_IsMichelRecod', 'MCP_MichelIndex', 
                                    'BM_MichelIsChild', 'BM_MichelIsShower', 'MCP_MichelFromMuon', 'MCP_EndpointsMCHits'], library="ak")

hierarchy_branches = hierarchy_tree.arrays(['MC_HierarchyTier', 'MC_ParentIndex', 'BM_HierarchyTier', 'BM_ParentIndex'], library="ak")


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Get masks
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
#     Get directory structure
# </div>

# In[ ]:

if not os.path.isdir(f'{os.environ["PLOT_DIR"]}/Plots/') :
    os.makedirs(f'{os.environ["PLOT_DIR"]}/Plots/')

plot_dir = os.environ["PLOT_DIR"]+'/Plots/TrackValPlots'
if not os.path.isdir(f'{plot_dir}') :
    os.makedirs(f'{plot_dir}')

    
# Track directories
track_plot_dir = f'{plot_dir}/Track/'

#  MCP_var distributions
if not os.path.isdir(f'{track_plot_dir}/MCP/') :
    os.makedirs(f'{track_plot_dir}/MCP/')
for plot_var in Variables.Track_MCP_plotting_vars :
    if not os.path.isdir(f'{track_plot_dir}/MCP/{plot_var.tree_name}') :
        os.makedirs(f'{track_plot_dir}/MCP/{plot_var.tree_name}')

#  BM_var distributions
if not os.path.isdir(f'{track_plot_dir}/BM/') :
    os.makedirs(f'{track_plot_dir}/BM/')
for plot_var in Variables.Track_BM_plotting_vars :
    if not os.path.isdir(f'{track_plot_dir}/BM/{plot_var.tree_name}') :
        os.makedirs(f'{track_plot_dir}/BM/{plot_var.tree_name}')        

# Michel directories
michel_plot_dir = f'{plot_dir}/Michel/'

for michel_type in Definitions.michel_types :

    plot_dir_michel_type = f'{michel_plot_dir}/{Definitions.michel_type_strings[michel_type]}/'
    
    #  MCP_var distributions
    if not os.path.isdir(f'{plot_dir_michel_type}/MCP/') :
        os.makedirs(f'{plot_dir_michel_type}/MCP/')
    for plot_var in Variables.Michel_MCP_plotting_vars :
        if not os.path.isdir(f'{plot_dir_michel_type}/MCP/{plot_var.tree_name}') :
            os.makedirs(f'{plot_dir_michel_type}/MCP/{plot_var.tree_name}')
    
    # Hierarchy
    if not os.path.isdir(f'{plot_dir_michel_type}/Hierarchy/') :
        os.makedirs(f'{plot_dir_michel_type}/Hierarchy/')   
        
    # Track-shower
    if not os.path.isdir(f'{plot_dir_michel_type}/TrackShower/') :
        os.makedirs(f'{plot_dir_michel_type}/TrackShower/')
    for plot_var in Variables.Michel_track_shower_vars :
        if not os.path.isdir(f'{plot_dir_michel_type}/TrackShower/{plot_var.tree_name}') :
            os.makedirs(f'{plot_dir_michel_type}/TrackShower/{plot_var.tree_name}')
    
    # Efficiency
    if not os.path.isdir(f'{plot_dir_michel_type}/Efficiency/') :
        os.makedirs(f'{plot_dir_michel_type}/Efficiency/')
    for plot_var in Variables.Michel_efficiency_vars :
        if not os.path.isdir(f'{plot_dir_michel_type}/Efficiency/{plot_var.tree_name}') :
            os.makedirs(f'{plot_dir_michel_type}/Efficiency/{plot_var.tree_name}')


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Get plots + tables
# </div>

# In[ ]:


# Track
TrackValidationFunc.CreateTrackGraphs(track_plot_dir, pfp_target_mask, pfp_reco_mask, tier_masks, int_masks, pdg_masks, track_branches)

# Michel
for michel_type in Definitions.michel_types :
    plot_dir_michel_type = f'{michel_plot_dir}/{Definitions.michel_type_strings[michel_type]}/'
    TrackValidationFunc.CreateMichelGraphs(plot_dir_michel_type, pfp_target_mask, pfp_reco_mask, tier_masks, int_masks, pdg_masks, hierarchy_branches, pfp_branches, track_branches, michel_type)


# In[ ]:





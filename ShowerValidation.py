#!/usr/bin/env python
# coding: utf-8

# ## Shower Validation
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
import ShowerValidationFunc
import Variables
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
shower_tree = file['ShowerTree']
hierarchy_tree = file['HierarchyTree']

event_branches = event_tree.arrays(['Run', 'Subrun', 'Event', 'MCInt_IsCC', 'MCNu_PDG'], library="ak")
pfp_branches = pfp_tree.arrays(['Run', 'Subrun', 'Event',
                                'MCP_TruePDG', 'MCP_HasMatch',
                                'MCP_NMCHits2D', 'MCP_NMCHitsU', 'MCP_NMCHitsV', 'MCP_NMCHitsW',
                                'BM_Completeness', 'BM_Purity', 'BM_IsTrack', 'BM_IsShower'], library="ak")

shower_branches = shower_tree.arrays(['MCP_TrueCoreLengthFromU', 'MCP_TrueCoreLengthFromV', 'MCP_TrueCoreLengthFromW',
                                      'BM_RecoCoreLength', 'BM_RecoLength', 'BM_MoliereRadius',
                                      'BM_DirAcc',
                                      'MCP_InitialMCHits', 'MCP_InitialMCHitsU', 'MCP_InitialMCHitsV', 'MCP_InitialMCHitsW',
                                      'BM_InitialPfoHits', 'BM_InitialPfoHitsU', 'BM_InitialPfoHitsV', 'BM_InitialPfoHitsW',
                                      'BM_InitialCompleteness', 'BM_InitialCompletenessU', 'BM_InitialCompletenessV', 'BM_InitialCompletenessW',
                                      'BM_InitialPurity', 'BM_InitialPurityU', 'BM_InitialPurityV', 'BM_InitialPurityW'], library="ak")

hierarchy_branches = hierarchy_tree.arrays(['MC_HierarchyTier'], library="ak")


# In[ ]:


#print(shower_tree.keys())


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Get Masks
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

plot_dir = os.environ["PLOT_DIR"]+'/Plots/ShowerValPlots'
if not os.path.isdir(f'{plot_dir}') :
    os.makedirs(f'{plot_dir}')


#  MCP_var distributions
if not os.path.isdir(f'{plot_dir}/MCP/') :
    os.makedirs(f'{plot_dir}/MCP/')
for plot_var in Variables.Shower_MCP_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/MCP/{plot_var.tree_name}') :
        os.makedirs(f'{plot_dir}/MCP/{plot_var.tree_name}')

#  BM_var distributions
if not os.path.isdir(f'{plot_dir}/BM/') :
    os.makedirs(f'{plot_dir}/BM/')
for plot_var in Variables.Shower_BM_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/BM/{plot_var.tree_name}') :
        os.makedirs(f'{plot_dir}/BM/{plot_var.tree_name}')      

# Diff distributions
if not os.path.isdir(f'{plot_dir}/Diff/') :
    os.makedirs(f'{plot_dir}/Diff/')
for plot_var in Variables.Shower_diff_plotting_vars :
    if not os.path.isdir(f'{plot_dir}/Diff/{plot_var.true_tree_name}_{plot_var.reco_tree_name}') :
        os.makedirs(f'{plot_dir}/Diff/{plot_var.true_tree_name}_{plot_var.reco_tree_name}')


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Get plots + tables
# </div>

# In[ ]:


ShowerValidationFunc.CreateGraphs(plot_dir, pfp_target_mask, pfp_reco_mask, tier_masks, int_masks, pdg_masks, shower_branches)


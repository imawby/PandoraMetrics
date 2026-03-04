#!/usr/bin/env python
# coding: utf-8

# ## Hierarchy Validation
# 
# I think can only look at 'correct parent?' metrics and 'Tier efficiency', reco tier is very hard to interpret.
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
import os
import Definitions
import HierarchyValidationFunc


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
hierarchy_tree = file['HierarchyTree']

event_branches = event_tree.arrays(['Run', 'Subrun', 'Event', 'MCInt_IsCC', 'MCNu_PDG'], library="ak")
pfp_branches = pfp_tree.arrays(['MCP_TruePDG', 'MCP_HasMatch', 'BM_IsShower', 'BM_Completeness', 'MCP_NMCHits2D', 
                                'MCP_NMCHitsU', 'MCP_NMCHitsV', 'MCP_NMCHitsW',
                                'BM_Completeness', 'BM_Purity'], library="ak")
hierarchy_branches = hierarchy_tree.arrays(['MC_HierarchyTier', 'MC_ParentIndex', 'BM_HierarchyTier', 'BM_ParentIndex'], library="ak")

n_entries = len(event_branches['Run'])


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


# In[ ]:


#Definitions.PrintEventSummary(int_masks, hierarchy_branches, pfp_branches)


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Get directory structure
# </div>

# In[ ]:


if not os.path.isdir(f'{os.environ["PLOT_DIR"]}/Plots/') :
    os.makedirs(f'{os.environ["PLOT_DIR"]}/Plots/')

plot_dir = os.environ["PLOT_DIR"]+'/Plots/HierarchyValPlots'
if not os.path.isdir(f'{plot_dir}') :
    os.makedirs(f'{plot_dir}')

# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Overall stats:
# </div>

# In[ ]:


# DEMAND_PARENT_HAS_MATCH = True, SPLIT_BY_PDG = True
HierarchyValidationFunc.CreateHierarchyTableMetrics(plot_dir, int_masks, tier_masks, pfp_target_mask, pfp_reco_mask, pdg_masks, hierarchy_branches, pfp_branches, True, True)
# DEMAND_PARENT_HAS_MATCH = False, SPLIT_BY_PDG = True
HierarchyValidationFunc.CreateHierarchyTableMetrics(plot_dir, int_masks, tier_masks, pfp_target_mask, pfp_reco_mask, pdg_masks, hierarchy_branches, pfp_branches, False, True)
# DEMAND_PARENT_HAS_MATCH = True, SPLIT_BY_PDG = False
HierarchyValidationFunc.CreateHierarchyTableMetrics(plot_dir, int_masks, tier_masks, pfp_target_mask, pfp_reco_mask, pdg_masks, hierarchy_branches, pfp_branches, True, False)
# DEMAND_PARENT_HAS_MATCH = False, SPLIT_BY_PDG = False
HierarchyValidationFunc.CreateHierarchyTableMetrics(plot_dir, int_masks, tier_masks, pfp_target_mask, pfp_reco_mask, pdg_masks, hierarchy_branches, pfp_branches, False, False)


# In[ ]:


# # Cache awkward arrays locally
# index = 0

# mc_tier      = hierarchy_branches['MC_HierarchyTier'][index]
# bm_tier      = hierarchy_branches['BM_HierarchyTier'][index]
# mc_parent    = hierarchy_branches['MC_ParentIndex'][index]
# bm_parent    = hierarchy_branches['BM_ParentIndex'][index]
# bm_istrack   = pfp_branches['BM_IsTrack'][index]

# # mc_tier      = mc_tier[bm_istrack != -1]
# # bm_tier      = bm_tier[bm_istrack != -1]
# # mc_parent    = mc_parent[bm_istrack != -1]
# # bm_parent    = bm_parent[bm_istrack != -1]
# # bm_istrack   = bm_istrack[bm_istrack != -1]


# print('mc_tier:', mc_tier)
# print('bm_tier:', bm_tier)
# print('mc_parent:', mc_parent)
# print('bm_parent:', bm_parent)
# print('bm_istrack:', bm_istrack)

# # Get primary masks
# is_child_reco = bm_istrack != -1
# true_primaries = mc_tier == 1
# primary_mask = is_child_reco & true_primaries
# # Count primary 
# n_primaries = ak.sum(primary_mask)
# n_correct_primaries = ak.sum(bm_tier[primary_mask] == 1)

# # Get 'other' masks
# other_mask = is_child_reco & (mc_tier != 1)
# other_fp_mask = other_mask & (bm_tier == 1) # reco'd as primary
# other_np_mask = other_mask & (bm_tier != 1) & (bm_istrack[mc_parent] != -1) # reco'd as not primary - make sure that the true parent actually exists! (can remove this demand?)
# n_other = ak.sum(other_fp_mask) + ak.sum(other_np_mask)

# # Pre-mask once to avoid repeated masked indexing
# bm_tier_o   = bm_tier[other_np_mask]
# bm_parent_o = bm_parent[other_np_mask]
# mc_parent_o = mc_parent[other_np_mask]

# n_not_best_match = ak.sum(bm_parent_o == -1)
# n_false_primary  = ak.sum(other_fp_mask)
# n_correct_parent = ak.sum(bm_parent_o == mc_parent_o)
# n_false_parent = ak.sum((bm_parent_o != -1) & (bm_parent_o != mc_parent_o))

# print('n_primaries:', n_primaries)
# print('n_correct_primaries:', n_correct_primaries)
# print('')
# print('n_other:', n_other)
# print('n_not_best_match:', n_not_best_match)
# print('n_false_primary:', n_false_primary)
# print('n_correct_parent:', n_correct_parent)
# print('n_false_parent:', n_false_parent)


# <div class="alert alert-block alert-info" style="font-size: 18px;">
#     Branch Efficiency:
# </div>

# In[ ]:


# mc_tier      = hierarchy_branches['MC_HierarchyTier']#[0:2]
# bm_tier      = hierarchy_branches['BM_HierarchyTier']#[0:2]
# mc_parent    = hierarchy_branches['MC_ParentIndex']#[0:2]
# bm_parent    = hierarchy_branches['BM_ParentIndex']#[0:2]
# has_match    = (pfp_branches['MCP_HasMatch'] == 1)#[0:2]
# has_match    = ak.values_astype(has_match, 'int')

# particle_indices = ak.local_index(mc_parent)

# num_tiers = 4
# jam = [None] * num_tiers
# correct = [None] * num_tiers

# for i in reversed(range(num_tiers)):
#     tier = i + 1
#     this_tier_mask = (mc_tier == tier)
#     this_tier_indices = particle_indices[this_tier_mask]

#     if i == 3:
#         jam[i] = this_tier_indices
#     else :
#         jam[i] = ak.concatenate(
#             [this_tier_indices, mc_parent[jam[i + 1]]] , axis=1
#         )
    
#     correct[i] = has_match[jam[i]]

# max_len = ak.max(ak.num(correct, axis=-1))  # max number of elements in any innermost array
# padded = ak.pad_none(correct, max_len, axis=-1)  # pad with None
# padded = ak.fill_none(padded, 0)

# cumulative = ak.Array([padded[0],
#     padded[1] * padded[0],
#     padded[2] * padded[1] * padded[0],
#     padded[3] * padded[2] * padded[1] * padded[0]])

# flat_jam = ak.flatten(jam, axis=2)
# flat_correct = ak.flatten(cumulative, axis=2)

# for i in range(4):
#     count = len(flat_jam[i])
#     correct_count = ak.count_nonzero(flat_correct[i])
#     print('count: ', count)
#     print('correct_count: ', correct_count)
#     print(round(float(correct_count) / float(count), 2))


# In[ ]:


## Make reconstruction efficiency tables with different primary pdgs.. 

# energy regieme

# multiplicty on x axis

# deposited energy from edepsim 

# initial region - for all pdgs
# charge dominates in completeness


# In[ ]:


# # for each particle in this tier, get its parent cumulative match
# parent_cumulative = cumulative[i-1][mc_parent[jam[i]]]
# # cumulative match = parent_cumulative AND this particle
# cumulative.append(correct[i] * parent_cumulative)

# print(jam)
# print(ak.flatten(jam, axis=2))


# In[ ]:





# In[ ]:


# mc_tier = hierarchy_branches['MC_HierarchyTier']
# mc_parent = hierarchy_branches['MC_ParentIndex']
# has_match = (pfp_branches['MCP_HasMatch'] == 1)
# particle_indices = ak.local_index(mc_parent)

# # Build tiered_branches
# num_tiers = 4
# tiered_branches = [None] * num_tiers
# correct = [None] * num_tiers

# for i in reversed(range(num_tiers)):
#     tier = i + 1
#     this_tier_mask = (mc_tier == tier)
#     this_tier_indices = particle_indices[this_tier_mask]

#     if (i == num_tiers - 1):
#         tiered_branches[i] = this_tier_indices
#     else:
#         # get parents of next tier's particles
#         tiered_branches[i] = ak.concatenate([mc_parent[tiered_branches[i + 1]], this_tier_indices], axis=1)

#     # Could also demand that the parent is correct? and that track/shower characterisation is correct?
#     correct[i] = has_match[tiered_branches[i]]

# # Pad to rectangular NumPy arrays
# max_len = ak.max(ak.num(correct, axis=-1))
# padded = ak.pad_none(correct, max_len, axis=-1)
# padded = ak.fill_none(padded, 0)
# padded_np = ak.to_numpy(padded)  # shape: (tiers, events, max_len_particles)

# # Cumulative product across tiers
# cumulative = np.ones_like(padded_np)
# cumulative[0] = padded_np[0]
# for i in range(1, num_tiers):
#     cumulative[i] = cumulative[i-1] * padded_np[i]

# # Metrics
# flat_tiered_branches = ak.flatten(tiered_branches, axis=2)
# flat_correct = cumulative.reshape(cumulative.shape[0], cumulative.shape[1], -1)
# for i in range(num_tiers):
#     count = len(flat_tiered_branches[i])
#     correct_count = ak.count_nonzero(flat_correct[i])
#     # print('count: ', count)
#     # print('correct_count: ', correct_count)
#     print(f' Efficiency to tier {i}: {round(float(correct_count) / float(count), 2)}')


# In[ ]:





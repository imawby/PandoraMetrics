import awkward as ak

##############################################################################################
##############################################################################################

ints = [0, 1, 2]

int_strings = {
    0: "CC \u03BD\u03BC",
    1: "CC \u03BDe",
    2: "NC"
}

pdgs = [13, 2212, 211, 22, 11]

pdg_strings = {
    13   : "Muon",
    2212 : "Proton",
    211  : "ChPion", 
    22   : "Photon",
    11   : "Electron"
}

pdg_color = {
    13   : "Blue",
    2212 : "tab:green",
    211  : "tab:pink", 
    22   : "tab:orange",
    11   : "Red"
}

tiers = [0, 1, 2]

tier_strings = {
    0  : "Primary",
    1  : "Secondary",
    2  : "Other"
}

tier_style = {
    0  : "solid",
    1  : "dashed",
    2  : "dotted"
}


##############################################################################################
##############################################################################################

def GetIntMasks(event_branches, pfp_branches) :

    mc_pdg = pfp_branches['MCP_TruePDG']
    
    # Match PFP jagged array
    mc_iscc = ak.broadcast_arrays(event_branches['MCInt_IsCC'], mc_pdg)[0]
    mc_nupdg = ak.broadcast_arrays(event_branches['MCNu_PDG'], mc_pdg)[0]
    
    int_masks = {0 : (mc_iscc == 1) & (abs(mc_nupdg) == 14),
                 1 : (mc_iscc == 1) & (abs(mc_nupdg) == 12),
                 2 : (mc_iscc == 0)}

    #(int_mask_broadcast, _) = ak.broadcast_arrays(int_mask, pfp_branches['MCP_TruePDG'])

    return int_masks

##############################################################################################
##############################################################################################

def GetPDGMasks(pfp_branches) :

    mc_pdg = abs(pfp_branches['MCP_TruePDG'])
    
    pfp_masks = {
        13   : (mc_pdg == 13),
        2212 : (mc_pdg == 2212),
        211  : (mc_pdg == 211), 
        22   : (mc_pdg == 22),
        11   : (mc_pdg == 11)
    }

    return pfp_masks

##############################################################################################
##############################################################################################

def GetTierMasks(hierarchy_branches) :

    mc_tier = hierarchy_branches['MC_HierarchyTier']
    
    tier_masks = {
        0  : (mc_tier == 1),
        1  : (mc_tier == 2),
        2  : (mc_tier > 2)
    }

    return tier_masks    
    

##############################################################################################
##############################################################################################

def PrintEventSummary(int_masks, hierarchy_branches, pfp_branches) :

    mc_tier = hierarchy_branches['MC_HierarchyTier']
    mc_pdg = abs(pfp_branches['MCP_TruePDG'])
    mc_has_match = pfp_branches['MCP_HasMatch']


    # CC numu [0], CC nue [1], NC [2]
    for int_type in [0, 1, 2] :


        print('--------------------------------------------------------------------------')
        print(int_strings[int_type] + str(' '* (10 - len(int_strings[int_type]))) + '|         1         |          2          |          3+        |')
        print('--------------------------------------------------------------------------')

        int_mask = int_masks[int_type]
        (int_mask_broadcast, _) = ak.broadcast_arrays(int_mask, mc_pdg)
        target_mask_1 = int_mask & (mc_tier == 1)
        target_mask_2 = int_mask & (mc_tier == 2)
        target_mask_3 = int_mask & (mc_tier > 2)
        reco_mask = int_mask & (mc_has_match == 1)
        reco_mask_1 = reco_mask & (mc_tier == 1)
        reco_mask_2 = reco_mask & (mc_tier == 2)
        reco_mask_3 = reco_mask & (mc_tier > 2)        

        for pdg in pdgs :

            pdg_mask = mc_pdg == pdg
            
            string_1 = f" MC: {ak.sum(target_mask_1 & pdg_mask)}, BM: {ak.sum(reco_mask_1 & pdg_mask)}"
            string_2 = f" MC: {ak.sum(target_mask_2 & pdg_mask)}, BM: {ak.sum(reco_mask_2 & pdg_mask)}"
            string_3 = f" MC: {ak.sum(target_mask_3 & pdg_mask)}, BM: {ak.sum(reco_mask_3 & pdg_mask)}"

            print(str(pdg) + str(' '* (10 - len(str(pdg)))) + \
                                    '|' + string_1 + str(' '* (19 - len(string_1))) + \
                                    '|' + string_2 + str(' '* (21 - len(string_2))) + \
                                    '|' + string_3 + str(' '* (20 - len(string_3))) + \
                                    '|')
        print('--------------------------------------------------------------------------')


        tot_string_1 = f" MC: {ak.sum(target_mask_1)}, BM: {ak.sum(reco_mask_1)}"
        tot_string_2 = f" MC: {ak.sum(target_mask_2)}, BM: {ak.sum(reco_mask_2)}"
        tot_string_3 = f" MC: {ak.sum(target_mask_3)}, BM: {ak.sum(reco_mask_3)}"

        print('          ' + \
                       '|' + tot_string_1 + str(' '* (19 - len(tot_string_1))) + \
                       '|' + tot_string_2 + str(' '* (21 - len(tot_string_2))) + \
                       '|' + tot_string_3 + str(' '* (20 - len(tot_string_3))) + \
                       '|')

        print('--------------------------------------------------------------------------')

        # print('------------------------------------------------------------')
        # print(('TRACK' if isTrack else 'SHOWER'))
        # print('------------------------------------------------------------')
        # print('NEW - True Gen   | Primary | Secondary | Tertiary | Higher |')
        # print('------------------------------------------------------------')

        # print('Correct parent WT |' + str(n_correct_parent_wrong_tier_primary_frac) + str(' '* (9 - len(str(n_correct_parent_wrong_tier_primary_frac)))) + \
        #                         '|' + str(n_correct_parent_wrong_tier_secondary_frac) + str(' '* (11 - len(str(n_correct_parent_wrong_tier_secondary_frac)))) + \
        #                         '|' + str(n_correct_parent_wrong_tier_tertiary_frac) + str(' '* (10 - len(str(n_correct_parent_wrong_tier_tertiary_frac)))) + \
        #                         '|' + str(n_correct_parent_wrong_tier_higher_frac) + str(' '* (8 - len(str(n_correct_parent_wrong_tier_higher_frac)))) + \
        #                         '|')
        # print('False primary     |' + str(n_tagged_as_primary_primary_frac) + str(' '* (9 - len(str(n_tagged_as_primary_primary_frac)))) + \
        #                         '|' + str(n_tagged_as_primary_secondary_frac) + str(' '* (11 - len(str(n_tagged_as_primary_secondary_frac)))) + \
        #                         '|' + str(n_tagged_as_primary_tertiary_frac) + str(' '* (10 - len(str(n_tagged_as_primary_tertiary_frac)))) + \
        #                         '|' + str(n_tagged_as_primary_higher_frac) + str(' '* (8 - len(str(n_tagged_as_primary_higher_frac)))) + \
        #                         '|')
        # print('Incorrect parent  |' + str(n_incorrect_parent_primary_frac) + str(' '* (9 - len(str(n_incorrect_parent_primary_frac)))) + \
        #                         '|' + str(n_incorrect_parent_secondary_frac) + str(' '* (11 - len(str(n_incorrect_parent_secondary_frac)))) + \
        #                         '|' + str(n_incorrect_parent_tertiary_frac) + str(' '* (10 - len(str(n_incorrect_parent_tertiary_frac)))) + \
        #                         '|' + str(n_incorrect_parent_higher_frac) + str(' '* (8 - len(str(n_incorrect_parent_higher_frac)))) + \
        #                         '|')
        # print('Not tagged        |' + str(n_not_tagged_primary_frac) + str(' '* (9 - len(str(n_not_tagged_primary_frac)))) + \
        #                         '|' + str(n_not_tagged_secondary_frac) + str(' '* (11 - len(str(n_not_tagged_secondary_frac)))) + \
        #                         '|' + str(n_not_tagged_tertiary_frac) + str(' '* (10 - len(str(n_not_tagged_tertiary_frac)))) + \
        #                         '|' + str(n_not_tagged_higher_frac) + str(' '* (8 - len(str(n_not_tagged_higher_frac)))) + \
        #                         '|')
        # print('------------------------------------------------------------')
        # print('Total             |' + str(n_true_primary) + str(' '* (9 - len(str(n_true_primary)))) + \
        #                         '|' + str(n_true_secondary) + str(' '* (11 - len(str(n_true_secondary)))) + \
        #                         '|' + str(n_true_tertiary) + str(' '* (10 - len(str(n_true_tertiary)))) + \
        #                         '|' + str(n_true_higher) + str(' '* (8 - len(str(n_true_higher)))) + \
        #                         '|')
        # print('------------------------------------------------------------')
        # print('')














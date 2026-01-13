import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions

##############################################################################################
##############################################################################################

class PlotVar :
    def __init__(self, tree_name, x_label, y_label, range, n_bins):
        self.tree_name = tree_name
        self.x_label = x_label
        self.y_label = y_label
        self.range = range
        self.n_bins = n_bins

# PFP
completeness_var = PlotVar('BM_Completeness', 'Completeness', 'Frac. of MCParticles', [0,1.0], 20)
purity_var = PlotVar('BM_Purity', 'Purity', 'Frac. of MCParticles', [0,1.0], 20)
n_mc_hits_2d_var = PlotVar('MCP_NMCHits2D', 'NMCHits2D', 'Frac. of PFParticles', [0,3000], 30)
theta_xz_var = PlotVar('MCP_TrueThetaXZ', 'ThetaXZ', 'Frac. of PFParticles', [-3.5, 3.5], 25)
theta_yz_var = PlotVar('MCP_TrueThetaYZ', 'ThetaYZ', 'Frac. of PFParticles', [-1.6, 1.6], 12)
pfo_energy_var = PlotVar('MCP_TrueEnergy', 'True Particle Energy', 'Frac. of PFParticles', [0,3], 50)
pfo_signed_vertex_acc_var = PlotVar('BM_VertexAcc', 'Signed Vertex deltaR [cm]', 'Frac. of PFParticles', [-50,50], 25)
        
# Michel
michel_n_mc_hits_2d_var = PlotVar('MCP_NMCHits2D', 'Michel NMCHits2D', 'Frac. of Michels', [0,100], 30)
michel_completeness_var = PlotVar('BM_Completeness', 'Completeness', 'Frac. of Michels', [0,1.0], 20)
michel_purity_var = PlotVar('BM_Purity', 'Purity', 'Frac. of Michels', [0,1.0], 20)

class PlotDiffVar :
    def __init__(self, true_tree_name, reco_tree_name, x_label, y_label, range, n_bins):
        self.true_tree_name = true_tree_name
        self.reco_tree_name = reco_tree_name
        self.x_label = x_label
        self.y_label = y_label
        self.range = range
        self.n_bins = n_bins        

# PFP
length_diff_var = PlotDiffVar('MCP_Length', 'BM_Length', 'TrueRecoLength', 'Frac. of PFParticles', [-100, 100], 50)
displacement_diff_var = PlotDiffVar('MCP_Displacement', 'BM_Displacement', 'True-Reco Displacement', 'Frac. of PFParticles', [-20, 20], 20)

##############################################################################################
##############################################################################################

def ConfigurePlot(fig, ax, int_type, tier, plot_var) :
    is_y_label_index = (tier == 0)
    ax.set_title(f'       {Definitions.int_strings[int_type]}: {Definitions.tier_strings[tier]}')
    ax.set_xlabel(plot_var.x_label)
    ax.set_ylabel(plot_var.y_label if is_y_label_index else '')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.tick_params(labelbottom=True, bottom=True, labelleft=is_y_label_index, left=is_y_label_index)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, hspace=0.3)

##############################################################################################
##############################################################################################

def TrackShowerAsAFunctionOf(pfp_indices, pfp_branches, plot_var, fig, ax) :
    
    n_hits = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][pfp_indices]))
    is_track = ak.to_numpy(ak.flatten(pfp_branches['BM_IsTrack'][pfp_indices]))
    is_shower = ak.to_numpy(ak.flatten(pfp_branches['BM_IsShower'][pfp_indices]))

    hist_all, edges = np.histogram(n_hits, bins=plot_var.n_bins, range=plot_var.range)
    hist_track, _ = np.histogram(n_hits[is_track == 1], bins=plot_var.n_bins, range=plot_var.range)
    hist_shower, _ = np.histogram(n_hits[is_shower == 1], bins=plot_var.n_bins, range=plot_var.range)
    
    proportion_track = np.divide(hist_track, hist_all, out=np.zeros_like(hist_track, dtype=float), where=hist_all > 0)
    proportion_shower = np.divide(hist_shower, hist_all, out=np.zeros_like(hist_shower, dtype=float), where=hist_all > 0)
    err_track = np.sqrt(proportion_track * (1.0 - proportion_track) / np.maximum(hist_all, 1))
    err_shower = np.sqrt(proportion_shower * (1.0 - proportion_shower) / np.maximum(hist_all, 1))
    
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    ax.errorbar(bin_centers, proportion_shower, yerr=err_shower, marker='x', capsize=2, label='Shower')
    ax.errorbar(bin_centers, proportion_track, yerr=err_track, marker='x', capsize=2, label='Track')
    ax.legend()
    
##############################################################################################
##############################################################################################

def plot_var(pfp_indices, pfp_branches, plot_var, fig, ax, label) :
    target_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][pfp_indices]))
    n_target_entries = len(target_entries)

    if (n_target_entries == 0) :
        return
    
    weights = np.ones(n_target_entries) * (1.0 / n_target_entries)
    ax.hist(target_entries, bins=plot_var.n_bins, range=plot_var.range, weights=weights, histtype='step', color='black', linewidth=1, label=(f' {label} '))
    ax.legend()

##############################################################################################
##############################################################################################

def PrintHierarchyTableHeader(int_type, file) :
    print(f'{Definitions.int_strings[int_type]}', file=file)
    print('------------------------------------------------------------------------------------', file=file)
    print('           | Correct Parent | False Primary | Wrong Parent | Parent Not Best Match |', file=file)
    print('------------------------------------------------------------------------------------', file=file) 

##############################################################################################
##############################################################################################

def CalculateHierarchyMetrics(hierarchy_branches, reco_michel_indices) :
    mc_tier         = hierarchy_branches['MC_HierarchyTier']
    bm_tier         = hierarchy_branches['BM_HierarchyTier']
    mc_parent       = hierarchy_branches['MC_ParentIndex']
    bm_parent       = hierarchy_branches['BM_ParentIndex']

    mc_tier_michel      = ak.to_numpy(ak.flatten(mc_tier[reco_michel_indices]))
    bm_tier_michel      = ak.to_numpy(ak.flatten(bm_tier[reco_michel_indices]))
    mc_parent_michel    = ak.to_numpy(ak.flatten(mc_parent[reco_michel_indices]))
    bm_parent_michel    = ak.to_numpy(ak.flatten(bm_parent[reco_michel_indices]))

    n_michel = mc_tier_michel.shape[0]
    n_not_best_match = np.count_nonzero((bm_tier_michel != 1) & (bm_parent_michel == -1))
    n_false_primary  = np.count_nonzero(bm_tier_michel == 1)
    n_correct_parent = np.count_nonzero((bm_tier_michel != 1) & (bm_parent_michel == mc_parent_michel))
    n_false_parent = np.count_nonzero((bm_tier_michel != 1) & (bm_parent_michel != -1) & (bm_parent_michel != mc_parent_michel))

    hierarchy_metrics = {}
    hierarchy_metrics['frac_not_best_match'] = round(0.0 if n_michel == 0 else float(n_not_best_match) / float(n_michel), 2)
    hierarchy_metrics['frac_false_primary'] = round(0.0 if n_michel == 0 else float(n_false_primary) / float(n_michel), 2)
    hierarchy_metrics['frac_correct_parent'] = round(0.0 if n_michel == 0 else float(n_correct_parent) / float(n_michel), 2)
    hierarchy_metrics['frac_false_parent'] = round(0.0 if n_michel == 0 else float(n_false_parent) / float(n_michel), 2)
    return hierarchy_metrics

##############################################################################################
##############################################################################################

def PrintHierarchyTableEntry(tier, hierarchy_metrics, file) :
    print(' ' + str(Definitions.tier_strings[tier]) + str(' '* (10 - len(str(Definitions.tier_strings[tier])))) +
                                            '|' + str(hierarchy_metrics['frac_correct_parent']) + str(' '* (16 - len(str(hierarchy_metrics['frac_correct_parent'])))) + \
                                            '|' + str(hierarchy_metrics['frac_false_primary']) + str(' '* (15 - len(str(hierarchy_metrics['frac_false_primary'])))) + \
                                            '|' + str(hierarchy_metrics['frac_false_parent']) + str(' '* (14 - len(str(hierarchy_metrics['frac_false_parent'])))) + \
                                            '|' + str(hierarchy_metrics['frac_not_best_match']) + str(' '* (23 - len(str(hierarchy_metrics['frac_not_best_match'])))) + \
                                            '|', file=file)

##############################################################################################
##############################################################################################

def PrintHierarchyTableFooter(file) :
    print('------------------------------------------------------------------------------------', file=file)
    print('', file=file)

##############################################################################################
##############################################################################################

def PrintEfficiencyTableHeader(int_type, file) :
    print(f'{Definitions.int_strings[int_type]}', file=file)
    print('-----------------------------------------------------------------', file=file)
    print('           |      NTarget      |       NReco       | Efficiency |', file=file)
    print('-----------------------------------------------------------------', file=file) 

##############################################################################################
##############################################################################################

def CalculateEfficiencyMetrics(target_mask, reco_mask) :
    n_targets = ak.sum(target_mask)
    n_reco = ak.sum(reco_mask)
    reco_efficiency = 0 if n_targets == 0 else round(float(n_reco) / n_targets, 2)    
    
    efficiency_metrics = {}
    efficiency_metrics['NTarget'] = n_targets
    efficiency_metrics['NReco'] = n_reco
    efficiency_metrics['Efficiency'] = reco_efficiency
    return efficiency_metrics

##############################################################################################
##############################################################################################

def PrintEfficiencyTableEntry(tier, efficiency_metrics, file) :
    print(' ' + str(Definitions.tier_strings[tier]) + str(' '* (10 - len(str(Definitions.tier_strings[tier])))) +
                                            '|' + str(efficiency_metrics['NTarget']) + str(' '* (19 - len(str(efficiency_metrics['NTarget'])))) + \
                                            '|' + str(efficiency_metrics['NReco']) + str(' '* (19 - len(str(efficiency_metrics['NReco'])))) + \
                                            '|' + str(efficiency_metrics['Efficiency']) + str(' '* (12 - len(str(efficiency_metrics['Efficiency'])))) + \
                                            '|', file=file)

##############################################################################################
##############################################################################################

def PrintEfficiencyTableFooter(file) :
    print('-----------------------------------------------------------------', file=file)
    print('', file=file)
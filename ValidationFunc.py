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

# Event
nu_true_energy = PlotVar('MCNu_Energy', 'True Nu Energy [GeV]', 'Frac. of True Nu', [0,10.0], 40)
nu_vis_true_energy = PlotVar('MCNu_VisEnergy', 'True Nu Energy [GeV]', 'Frac. of True Nu', [0,10.0], 40)
nu_vertex_accuracy = PlotVar('RecoNu_VertexAcc_Pass2', 'NuVertexAccuracy [cm]', 'Frac. of Reco Nu', [0,10.0], 50)
Event_MCP_plotting_vars = [nu_true_energy, nu_vis_true_energy]
Event_Reco_plotting_vars = [nu_vertex_accuracy]

# PFP
completeness_var = PlotVar('BM_Completeness', 'Completeness', 'Frac. of MCParticles', [0,1.0], 20)
purity_var = PlotVar('BM_Purity', 'Purity', 'Frac. of MCParticles', [0,1.0], 20)
alt_completeness_var = PlotVar('ALT_Completeness', 'AltCompleteness', 'Frac. of MCParticles', [-1.01, 1.0], 40)
alt_purity_var = PlotVar('ALT_Purity', 'AltPurity', 'Frac. of MCParticles', [-1.01, 1.0], 40)
n_mc_hits_2d_var = PlotVar('MCP_NMCHits2D', 'NMCHits2D', 'Frac. of PFParticles', [0, 1500], 30)
mc_displacement_var = PlotVar('MCP_Displacement', 'MC Displacement [cm]', 'Frac. of PFParticles', [0, 100], 10)
theta_xz_var = PlotVar('MCP_TrueThetaXZ', 'ThetaXZ', 'Frac. of PFParticles', [-3.5, 3.5], 25)
theta_yz_var = PlotVar('MCP_TrueThetaYZ', 'ThetaYZ', 'Frac. of PFParticles', [-1.6, 1.6], 12)
pfo_energy_var = PlotVar('MCP_TrueEnergy', 'True Particle Energy', 'Frac. of PFParticles', [0,3], 50)
true_vis_energy = PlotVar('MCP_TrueVisEnergy', 'True Visible Energy', 'Frac. of PFParticles', [0,5], 50)
pfo_signed_vertex_acc_var = PlotVar('BM_VertexAcc', 'Signed Vertex deltaR [cm]', 'Frac. of PFParticles', [-25,25], 25)
multiplicity_var = PlotVar('MCNu_Multiplicity', 'N Reco Targets in Event', 'Frac. of PFParticles', [0,50], 50)    
shower_multiplicity_var = PlotVar('MCNu_ShowerMultiplicity', 'N Reco Targets in Event', 'Frac. of PFParticles', [0,100], 50)
PFP_MCP_plotting_vars = [completeness_var, purity_var, n_mc_hits_2d_var, true_vis_energy, multiplicity_var, shower_multiplicity_var]
PFP_BM_plotting_vars = [pfo_signed_vertex_acc_var]
PFP_ALT_plotting_vars = [alt_completeness_var, alt_purity_var]
PFP_track_shower_plotting_vars = [n_mc_hits_2d_var, theta_xz_var, theta_yz_var]
PFP_efficiency_vars = [n_mc_hits_2d_var, theta_xz_var, theta_yz_var, true_vis_energy, multiplicity_var, shower_multiplicity_var, mc_displacement_var]

# Michel
michel_true_vis_energy = PlotVar('MCP_TrueVisEnergy', 'True Visible Energy', 'Frac. of Michels', [0,0.1], 20)
michel_n_mc_hits_2d_var = PlotVar('MCP_NMCHits2D', 'Michel NMCHits2D', 'Frac. of Michels', [Definitions.MIN_TOTAL_HITS, 150], 15)
michel_completeness_var = PlotVar('BM_Completeness', 'Completeness', 'Frac. of Michels', [0,1.0], 20)
michel_purity_var = PlotVar('BM_Purity', 'Purity', 'Frac. of Michels', [0,1.0], 20)
Michel_MCP_plotting_vars = [michel_n_mc_hits_2d_var, michel_completeness_var, michel_purity_var, michel_true_vis_energy]
Michel_track_shower_vars = [michel_n_mc_hits_2d_var]
Michel_efficiency_vars = [michel_n_mc_hits_2d_var, michel_true_vis_energy]

# Track
track_n_endpoint_mc_hits_var = PlotVar('MCP_EndpointsMCHits', 'NEndpointsMCHits', 'Frac. of PFParticles', [0,50], 20)
track_endpoint_completeness_var = PlotVar('BM_EndpointCompleteness', 'Endpoint Completeness', 'Frac. of PFParticles', [-1.5,1.0], 50)
track_endpoint_purity_var = PlotVar('BM_EndpointPurity', 'EndpointPurity', 'Frac. of PFParticles', [-1.5,1.0], 50)
track_signed_endpoint_acc_var = PlotVar('BM_EndpointAcc', 'Signed Endpoint deltaR [cm]', 'Frac. of PFParticles', [-25,25], 25)
Track_MCP_plotting_vars = [track_endpoint_completeness_var, track_endpoint_purity_var, track_n_endpoint_mc_hits_var]
Track_BM_plotting_vars = [track_signed_endpoint_acc_var]

# Shower
shower_initial_MC_hits = PlotVar('MCP_InitialMCHits', 'Initial NMCHits2D', 'Frac. of True Showers', [0,150], 30)
shower_initial_PFP_hits = PlotVar('BM_InitialPfoHits', 'Initial n Pfo Hits2D', 'Frac. of True Showers', [0,150], 30)
# -1 == no MC hits in initial region
shower_initial_completeness = PlotVar('BM_InitialCompleteness', 'Initial Completeness', 'Frac. of True Showers', [-1.05,1.0], 20)
# -1 == no MC hits in initial region
shower_initial_purity = PlotVar('BM_InitialPurity', 'Initial Purity', 'Frac. of True Showers', [-1.05,1.0], 20)                   
shower_dir_acc = PlotVar('BM_DirAcc', 'True-Reco Dir Opening Angle [radians]', 'Frac. of Reco Showers', [-1.1, 3.2], 50)  
shower_moliere = PlotVar('BM_MoliereRadius', 'Moliere Radius', 'Frac. of Reco Showers', [-1.0, 20], 22)  
Shower_MCP_plotting_vars = [shower_initial_MC_hits, shower_initial_PFP_hits, shower_initial_completeness, shower_initial_purity]
Shower_BM_plotting_vars = [shower_dir_acc, shower_moliere]

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
PFP_diff_plotting_vars = [length_diff_var, displacement_diff_var]

# Shower
core_length_diff = PlotDiffVar('MCP_TrueCoreLengthFromW', 'BM_RecoCoreLength', 'True-Reco Core Length (from W) [cm]', 'Frac. of Reco Showers', [-50, 50], 50)
Shower_diff_plotting_vars = [core_length_diff]

# Event
nu_vtx_delta_x = PlotDiffVar('MCNu_VertexX', 'RecoNu_VertexX', 'TrueX-RecoX [cm]', 'Frac. of Reco Nu', [-10,10.0], 100)
nu_vtx_delta_y = PlotDiffVar('MCNu_VertexY', 'RecoNu_VertexY', 'TrueY-RecoY [cm]', 'Frac. of Reco Nu', [-10,10.0], 100)
nu_vtx_delta_z = PlotDiffVar('MCNu_VertexZ', 'RecoNu_VertexZ', 'TrueZ-RecoZ [cm]', 'Frac. of Reco Nu', [-10,10.0], 100)
Event_diff_plotting_vars = [nu_vtx_delta_x, nu_vtx_delta_y, nu_vtx_delta_z]

class SegVar :
    def __init__(self, tree_name, label, options, colors):
        self.tree_name = tree_name
        self.label = label
        self.options = options
        self.colors = colors

# ALT
alt_pfp_seg_var = SegVar('ALT_PDG', 'PDG', [13, 2212, 211, 777, 22, 11, 111], ["Blue", "tab:green", "tab:pink", 'tab:purple', "tab:orange", "Red", "tab:olive"])
alt_is_up_hierarchy_seg_var = SegVar('ALT_IsUpstreamHierarchy', 'IsUpHierarchy', [-1, 0, 1], ["tab:gray", "red", "green"]) 
alt_is_same_mc_var = SegVar('ALT_IsSameMC', 'IsSameMC', [-1, 0, 1], ["tab:gray", "red", "green"]) 
ALT_seg_vars = [alt_pfp_seg_var, alt_is_up_hierarchy_seg_var, alt_is_same_mc_var]

##############################################################################################
##############################################################################################

def ConfigurePlot(fig, ax, int_type, tier, pdg, plot_var) :
    is_y_label_index = (tier == 0)
    ax.set_title(f'       {Definitions.int_strings[int_type]} - {Definitions.tier_strings[tier]} - {Definitions.pdg_strings[pdg]}')
    ax.set_xlabel(plot_var.x_label)
    ax.set_ylabel(plot_var.y_label)
    #ax.set_ylabel(plot_var.y_label if is_y_label_index else '')
    #ax.set_ylim(0.0, 1.05)
    ax.grid(True)
    #ax.tick_params(labelbottom=True, bottom=True, labelleft=is_y_label_index, left=is_y_label_index)
    ax.tick_params(labelbottom=True, bottom=True, labelleft=True, left=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.95, hspace=0.4, wspace=0.4)

##############################################################################################
##############################################################################################

def ConfigureTightPlot(fig, ax, int_type, tier, i_pdg, plot_var) :

    ax.set_ylim(0.0, 1.0)
    is_x_label_index = True
    ax.set_xlabel(plot_var.x_label if is_x_label_index else '')
    ax.tick_params(labelbottom=is_x_label_index, bottom=is_x_label_index)
    is_y_label_index = (i_pdg == 0)
    ax.set_title(f'       {Definitions.int_strings[int_type]}: {Definitions.tier_strings[tier]}' if is_y_label_index else '')
    ax.set_ylabel(plot_var.y_label if is_y_label_index else '')
    ax.tick_params(labelleft=is_y_label_index, left=is_y_label_index)
    ax.grid(True)
    fig.subplots_adjust(left=0.08, bottom=0.08, wspace=0.0, hspace=0.4)    

##############################################################################################
##############################################################################################

def TrackShowerAsAFunctionOf(pfp_indices, pfp_branches, plot_var, fig, ax, legend_string) :
    
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
    ax.errorbar(bin_centers, proportion_shower, yerr=err_shower, marker='x', capsize=2, label=(f'{legend_string} - Shower'))
    ax.errorbar(bin_centers, proportion_track, yerr=err_track, marker='x', capsize=2, label=(f'{legend_string} - Track'))
    ax.legend(loc='center right')
    
##############################################################################################
##############################################################################################

# def PlotVariable(indices_or_mask, branches, plot_var, ax, label, color) :
#     target_entries = branches[plot_var.tree_name][indices_or_mask]

#     if (target_entries.ndim > 1) :
#         target_entries = ak.flatten(target_entries)

#     target_entries = ak.to_numpy(target_entries).reshape(-1)
#     n_target_entries = len(target_entries)
#     weights = np.ones(n_target_entries) * (1.0 / n_target_entries)
#     ax.hist(target_entries, bins=plot_var.n_bins, range=plot_var.range, weights=weights, histtype='step', color=color, linewidth=1, label=(f' {label} '))
#     ax.legend()

def PlotVariable(indices_or_mask, branches, plot_var, ax, label, color, fill=True, n_entries=0):
    target_entries = branches[plot_var.tree_name][indices_or_mask]

    if target_entries.ndim > 1:
        target_entries = ak.flatten(target_entries)

    target_entries = ak.to_numpy(target_entries).reshape(-1)
    
    if (n_entries == 0) : 
        n_entries = len(target_entries)        
        if (n_entries == 0) :
            return
    
    hist_counts, bin_edges = np.histogram(
        target_entries, bins=plot_var.n_bins, range=plot_var.range
    )
    hist_fraction = hist_counts / n_entries
    
    # Plot with error == sqrt(n_i)/N
    hist_error = np.sqrt(hist_counts) / n_entries
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])    
    ax.step(bin_centers, hist_fraction, where='mid', color=color, linewidth=1, label=f'{label}')
    if (fill) :
        ax.fill_between(bin_centers,hist_fraction,step='mid',color=color,alpha=0.3)
    ax.errorbar(bin_centers, hist_fraction, yerr=hist_error, fmt='none', ecolor=color, capsize=2)

    if (len(label) != 0) :
        ax.legend()

##############################################################################################
##############################################################################################

def Plot2DHist(indices_or_mask, branches, plot_var_x, plot_var_y, ax, label):
    target_x_entries = branches[plot_var_x.tree_name][indices_or_mask]
    target_y_entries = branches[plot_var_y.tree_name][indices_or_mask]

    if target_x_entries.ndim > 1:
        target_x_entries = ak.flatten(target_x_entries)

    if target_y_entries.ndim > 1:
        target_y_entries = ak.flatten(target_y_entries)        

    target_x_entries = ak.to_numpy(target_x_entries).reshape(-1)
    target_y_entries = ak.to_numpy(target_y_entries).reshape(-1)
    
    # Compute 2D histogram
    hist_counts, x_edges, y_edges = np.histogram2d(
        target_x_entries,
        target_y_entries,
        bins=[plot_var_x.n_bins, plot_var_y.n_bins],
        range=[plot_var_x.range, plot_var_y.range]
    )

    n_entries = len(target_x_entries)
    hist_fraction = hist_counts / n_entries

    # Plot as colored mesh
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        hist_fraction.T,   # transpose needed for correct orientation
        cmap='viridis'
    )
    mesh.set_clim(0.0, 0.1) 
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(plot_var_x.y_label)
    ax.set_xlabel(plot_var_x.x_label)
    ax.set_ylabel(plot_var_y.x_label)

        
##############################################################################################
##############################################################################################

def PlotDiffVariable(indices_or_mask, branches, plot_diff_var, ax, label, color) :
    metric = branches[plot_diff_var.true_tree_name] - branches[plot_diff_var.reco_tree_name]
    target_entries = metric[indices_or_mask]
    
    if (target_entries.ndim > 1) :
        target_entries = ak.flatten(target_entries)
    
    target_entries = ak.to_numpy(target_entries).reshape(-1)
    n_target_entries = len(target_entries)

    if (n_target_entries == 0) :
        return
    
    weights = np.ones(n_target_entries) * (1.0 / n_target_entries)
    ax.hist(target_entries, bins=plot_diff_var.n_bins, range=plot_diff_var.range, weights=weights, histtype='step', color=color, linewidth=1, label=(f' {label} '))
    ax.legend()

##############################################################################################
##############################################################################################

def PlotEfficiency(target_mask_or_indices, reco_mask_or_indices, pfp_branches, plot_var, fig, ax, color, legend_string) :
    
    target_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][target_mask_or_indices]))
    reco_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][reco_mask_or_indices]))
    
    hist_target, edges = np.histogram(target_entries, bins=plot_var.n_bins, range=plot_var.range)
    hist_reco, _ = np.histogram(reco_entries, bins=plot_var.n_bins, range=plot_var.range)
    efficiency = np.divide(hist_reco, hist_target, 
                           out=np.zeros_like(hist_reco, dtype=float), 
                           where=hist_target > 0)


    
    
    ax2 = ax.twinx()
    PlotVariable(target_mask_or_indices, pfp_branches, plot_var, ax2, 'Dist', color)

    # Binomial efficiency uncertainty
    efficiency_err = np.zeros_like(efficiency)
    valid = hist_target > 0
    efficiency_err[valid] = np.sqrt(
        efficiency[valid] * (1.0 - efficiency[valid]) / hist_target[valid]
    )

    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    ax.errorbar(bin_centers, efficiency, yerr=efficiency_err, fmt='x-', color='black', capsize=3, label=f' {legend_string} ')
    ax.set_ylabel('Efficiency')
    ax2.set_ylabel(plot_var.y_label, color=color)
    ax2.tick_params(axis='y', colors=color)
    ax2.spines['right'].set_color(color)
    ax.legend()

    # ax.scatter(bin_centers, efficiency, color=color, label=f' {legend_string} ')

        # file_name = f'Efficiency_{plot_var.tree_name}_{Definitions.tier_strings[tier]}'
        # fig.savefig(f'{plot_dir}{file_name}.pdf', bbox_inches='tight')

##############################################################################################
##############################################################################################

def Plot2DEfficiency(target_mask_or_indices, reco_mask_or_indices, branches, plot_var_x, plot_var_y, ax, label):
    target_x_entries = branches[plot_var_x.tree_name][target_mask_or_indices]
    target_y_entries = branches[plot_var_y.tree_name][target_mask_or_indices]
    reco_x_entries = branches[plot_var_x.tree_name][reco_mask_or_indices]
    reco_y_entries = branches[plot_var_y.tree_name][reco_mask_or_indices]

    if target_x_entries.ndim > 1:
        target_x_entries = ak.flatten(target_x_entries)
    if reco_x_entries.ndim > 1:
        reco_x_entries = ak.flatten(reco_x_entries)         
    if target_y_entries.ndim > 1:
        target_y_entries = ak.flatten(target_y_entries)
    if reco_y_entries.ndim > 1:
        reco_y_entries = ak.flatten(reco_y_entries)         

    target_x_entries = ak.to_numpy(target_x_entries).reshape(-1)
    target_y_entries = ak.to_numpy(target_y_entries).reshape(-1)
    reco_x_entries = ak.to_numpy(reco_x_entries).reshape(-1)
    reco_y_entries = ak.to_numpy(reco_y_entries).reshape(-1)    
    
    hist_target, x_edges, y_edges = np.histogram2d(
        target_x_entries, target_y_entries,
        bins=[plot_var_x.n_bins, plot_var_y.n_bins],
        range=[plot_var_x.range, plot_var_y.range]
    )
    hist_reco, _, _ = np.histogram2d(
        reco_x_entries, reco_y_entries,
        bins=[plot_var_x.n_bins, plot_var_y.n_bins],
        range=[plot_var_x.range, plot_var_y.range]
    )

    # Efficiency
    efficiency = np.divide(
        hist_reco, hist_target,
        out=np.zeros_like(hist_reco, dtype=float),
        where=hist_target > 0
    )

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        efficiency.T,   # transpose for correct orientation
        cmap='viridis',
        vmin=0, vmax=1
    )
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(plot_var_x.y_label)
    ax.set_xlabel(plot_var_x.x_label)
    ax.set_ylabel(plot_var_y.x_label)
    
    
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
    print('------------------------------------------------------------------------', file=file)
    print('                  |      NTarget      |       NReco       | Efficiency |', file=file)
    print('------------------------------------------------------------------------', file=file)     

##############################################################################################
##############################################################################################

def CalculateEfficiencyMetrics(target_mask_or_indices, reco_mask_or_indices, is_mask) :

    if (is_mask) :
        n_targets = ak.sum(target_mask_or_indices)
        n_reco = ak.sum(reco_mask_or_indices)
    else :
        n_targets = len(ak.flatten(target_mask_or_indices))
        n_reco = len(ak.flatten(reco_mask_or_indices))
    
    reco_efficiency = 0 if n_targets == 0 else round(float(n_reco) / n_targets, 2)    
    
    efficiency_metrics = {}
    efficiency_metrics['NTarget'] = n_targets
    efficiency_metrics['NReco'] = n_reco
    efficiency_metrics['Efficiency'] = reco_efficiency
    return efficiency_metrics

##############################################################################################
##############################################################################################

# def PrintEfficiencyTableEntry(tier, efficiency_metrics, file) :

#     print(' ' + str(Definitions.tier_strings[tier]) + str(' '* (10 - len(str(Definitions.tier_strings[tier])))) +
#                                             '|' + str(efficiency_metrics['NTarget']) + str(' '* (19 - len(str(efficiency_metrics['NTarget'])))) + \
#                                             '|' + str(efficiency_metrics['NReco']) + str(' '* (19 - len(str(efficiency_metrics['NReco'])))) + \
#                                             '|' + str(efficiency_metrics['Efficiency']) + str(' '* (12 - len(str(efficiency_metrics['Efficiency'])))) + \
#                                             '|', file=file)

##############################################################################################
##############################################################################################

def PrintEfficiencyTableEntry(tier, pdg, efficiency_metrics, file) :
    title_string = f'{Definitions.tier_strings[tier]} {Definitions.pdg_strings[pdg]}'
    print(' ' + title_string + str(' '* (18 - len(title_string))) +
                                            '|' + str(efficiency_metrics['NTarget']) + str(' '* (19 - len(str(efficiency_metrics['NTarget'])))) + \
                                            '|' + str(efficiency_metrics['NReco']) + str(' '* (19 - len(str(efficiency_metrics['NReco'])))) + \
                                            '|' + str(efficiency_metrics['Efficiency']) + str(' '* (12 - len(str(efficiency_metrics['Efficiency'])))) + \
                                            '|', file=file)

##############################################################################################
##############################################################################################

def PrintEfficiencyTableFooter(file) :
    print('------------------------------------------------------------------------', file=file)
    print('', file=file)
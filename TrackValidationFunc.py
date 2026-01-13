import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions
        
##############################################################################################
##############################################################################################


def plot_michel_efficiency(target_michel_indices, reco_michel_indices, pfp_branches, plot_var, fig, ax) :
    target_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][target_michel_indices]))
    reco_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][reco_michel_indices]))
    
    hist_target, edges = np.histogram(target_entries, bins=plot_var.n_bins, range=plot_var.range)
    hist_reco, _ = np.histogram(reco_entries, bins=plot_var.n_bins, range=plot_var.range)
    efficiency = np.divide(hist_reco, hist_target, 
                           out=np.zeros_like(hist_reco, dtype=float), 
                           where=hist_target > 0)
    
    # Binomial efficiency uncertainty
    efficiency_err = np.zeros_like(efficiency)
    valid = hist_target > 0
    efficiency_err[valid] = np.sqrt(
        efficiency[valid] * (1.0 - efficiency[valid]) / hist_target[valid]
    )

    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    ax.errorbar(bin_centers, efficiency, yerr=efficiency_err, fmt='o-', color='black', capsize=3, label=f' Michel ')
    ax.legend()

##############################################################################################
##############################################################################################

def plot_michel_var(target_michel_indices, pfp_branches, plot_var, fig, ax) :
    target_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][target_michel_indices]))
    n_target_entries = len(target_entries)

    if (n_target_entries == 0) :
        return
    
    weights = np.ones(n_target_entries) * (1.0 / n_target_entries)
    
    ax.hist(target_entries, bins=plot_var.n_bins, range=plot_var.range, weights=weights, histtype='step', color='black', linewidth=1, label=(f' Michel '))
    ax.legend()


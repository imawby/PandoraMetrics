import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions

plot_dir = '/Users/isobel/Desktop/DUNE/2026/PandoraValidation/PFPValPlots/'

##############################################################################################
##############################################################################################

class PlotVar :
    def __init__(self, tree_name, x_label, range, n_bins):
        self.tree_name = tree_name
        self.x_label = x_label
        self.range = range
        self.n_bins = n_bins

n_mc_hits_2d_var = PlotVar('MCP_NMCHits2D', 'NMCHits2D', [0,3000], 30)
theta_xz_var = PlotVar('MCP_TrueThetaXZ', 'ThetaXZ', [-3.5, 3.5], 25)
theta_yz_var = PlotVar('MCP_TrueThetaYZ', 'ThetaYZ', [-1.6, 1.6], 12)
pfo_energy_var = PlotVar('MCP_TrueEnergy', 'True Particle Energy', [0,3], 50)
        
##############################################################################################
##############################################################################################

def TrackShowerClassification(int_masks, tier_masks, pdg_masks, pfp_branches) :
    
    for tier in Definitions.tiers :
    
        confMatrix_eff = [[], [], []]
        tier_mask = tier_masks[tier]
        
        for int_type in Definitions.ints :
            
            int_mask = int_masks[int_type]
    
            for pdg in Definitions.pdgs :
                
                # Only look at those that have been reconstructed
                target_mask = int_mask & pdg_masks[pdg] & tier_mask & (pfp_branches['BM_IsTrack'] != -1)
                n_particle = ak.sum(target_mask)
                n_track = ak.sum(pfp_branches['BM_IsTrack'][target_mask] == 1)
                n_shower = ak.sum(pfp_branches['BM_IsShower'][target_mask] == 1)
                confMatrix_eff[int_type].append([round(n_track / n_particle, 2), round(n_shower / n_particle, 2)])
        
        ## Draw
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
        confMatrix_eff = np.array(confMatrix_eff)
        
        for int_type in Definitions.ints :
        
            # Draw confusion
            im = ax[int_type].imshow(confMatrix_eff[int_type], cmap='Blues')
            
            # Axis ticks
            ax[int_type].set_xticks([0, 1])
            ax[int_type].set_xticklabels(["Track", "Shower"])
            ax[int_type].set_yticks(range(len(Definitions.pdgs)))
            ax[int_type].set_yticklabels([str(p) for p in Definitions.pdgs])
            
            # Axis labels and title
            ax[int_type].set_xlabel("Reco Classification")
            ax[int_type].set_ylabel("True PDG")
            ax[int_type].set_title(f'{Definitions.int_strings[int_type]}: {Definitions.tier_strings[tier]}')
            
            # Colorbar
            cbar = fig.colorbar(im, ax=ax[int_type])
            cbar.set_label("Counts")
            
            # Add text inside cells
            for i in range(confMatrix_eff[int_type].shape[0]):
                for j in range(confMatrix_eff[int_type].shape[1]):
                    ax[int_type].text(j, i, confMatrix_eff[int_type][i, j],
                            ha="center", va="center", color="black")
            
        plt.tight_layout()
        plt.show()
    
        file_name = f'TrackShowerClassification_{Definitions.tier_strings[tier]}'
        fig.savefig(f'{plot_dir}{file_name}.pdf', bbox_inches='tight')


#####################################################################################################################################################
#####################################################################################################################################################

def CompletenessPurity(int_masks, tier_masks, pdg_masks, pfp_branches, metric_string) :

    mc_metric = pfp_branches[metric_string]
    
    for tier in Definitions.tiers :

        tier_mask = tier_masks[tier]
    
        # Save plots for each tier in a different file
        fig, ax = plt.subplots(ncols=len(Definitions.pdgs), nrows=len(Definitions.ints), figsize=(14, 10))
        
        for int_type in Definitions.ints :
            
            int_mask = int_masks[int_type]
    
            for iPDG in range(len(Definitions.pdgs)) :
    
                pdg = Definitions.pdgs[iPDG]
                pdg_string = Definitions.pdg_strings[pdg]
                colour = Definitions.pdg_color[pdg]
                    
                target_mask = int_mask & pdg_masks[pdg] & tier_mask
                reconstructable_entries = ak.to_numpy(ak.flatten(mc_metric[target_mask]))
                n_reconstructable_entries = len(reconstructable_entries)
                weights = np.ones(n_reconstructable_entries) * (1.0 / n_reconstructable_entries)
                ax[int_type][iPDG].hist(reconstructable_entries, bins=20, range=[0.0,1.0], weights=weights, histtype='step', color=colour, linewidth=1, label=(f' {pdg_string} '))
                ax[int_type][iPDG].legend()
                ax[int_type][iPDG].set_ylim(0.0, 1.0)
                is_x_label_index = int_type == (len(Definitions.ints) - 1)
                ax[int_type][iPDG].set_xlabel(metric_string if is_x_label_index else '')
                ax[int_type][iPDG].tick_params(labelbottom=is_x_label_index, bottom=is_x_label_index)
                is_y_label_index = (iPDG == 0)
                ax[int_type][iPDG].set_title(f'       {Definitions.int_strings[int_type]}: {Definitions.tier_strings[tier]} {metric_string}' if is_y_label_index else '')
                ax[int_type][iPDG].set_ylabel('Frac. of MCParticles' if is_y_label_index else '')
                ax[int_type][iPDG].tick_params(labelleft=is_y_label_index, left=is_y_label_index)
                ax[int_type][iPDG].grid(True)
                fig.tight_layout(pad=0)
                fig.subplots_adjust(left=0.08, bottom=0.08) 
                
        file_name = f'{metric_string}_{Definitions.tier_strings[tier]}'
        fig.savefig(f'{plot_dir}{file_name}.pdf', bbox_inches='tight')

#####################################################################################################################################################
#####################################################################################################################################################

def RecoEfficiency(int_masks, tier_masks, pdg_masks, pfp_branches, plot_var) :

    mc_metric = pfp_branches[plot_var.tree_name]
    mc_has_match = pfp_branches['MCP_HasMatch']
    
    for tier in Definitions.tiers :

        tier_mask = tier_masks[tier]
    
        # Save plots for each tier in a different file
        fig, ax = plt.subplots(ncols=len(Definitions.pdgs), nrows=len(Definitions.ints), figsize=(14, 10))

        for int_type in Definitions.ints :
        
            int_mask = int_masks[int_type]
    
            for iPDG in range(len(Definitions.pdgs)) :
    
                pdg = Definitions.pdgs[iPDG]
                pdg_string = Definitions.pdg_strings[pdg]
                colour = Definitions.pdg_color[pdg]

                target_mask = int_mask & pdg_masks[pdg] & tier_mask
                reco_mask = target_mask & (mc_has_match == 1)

                target_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][target_mask]))
                reco_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][reco_mask]))
                
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
                ax[int_type][iPDG].errorbar(bin_centers, efficiency, yerr=efficiency_err, fmt='o-', color=colour, capsize=3, label=f' {pdg_string} ')
                #ax[int_type][iPDG].plot(bin_centers, efficiency, 'o-', color=colour, label=(f' {pdg_string} '))
                ax[int_type][iPDG].legend()
                ax[int_type][iPDG].set_ylim(0.0, 1.0)
                is_x_label_index = int_type == (len(Definitions.ints) - 1)
                ax[int_type][iPDG].set_xlabel(plot_var.x_label if is_x_label_index else '')
                ax[int_type][iPDG].tick_params(labelbottom=is_x_label_index, bottom=is_x_label_index)
                is_y_label_index = (iPDG == 0)
                ax[int_type][iPDG].set_title(f'       {Definitions.int_strings[int_type]}: {Definitions.tier_strings[tier]} {plot_var.tree_name}' if is_y_label_index else '')
                ax[int_type][iPDG].set_ylabel('Efficiency' if is_y_label_index else '')
                ax[int_type][iPDG].tick_params(labelleft=is_y_label_index, left=is_y_label_index)
                ax[int_type][iPDG].grid(True)
                fig.tight_layout(pad=0)
                fig.subplots_adjust(left=0.08, bottom=0.08)

        file_name = f'Efficiency_{plot_var.tree_name}_{Definitions.tier_strings[tier]}'
        fig.savefig(f'{plot_dir}{file_name}.pdf', bbox_inches='tight')
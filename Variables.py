import Definitions

class PlotVar :
    def __init__(self, tree_name, x_label, y_label, range, n_bins):
        self.tree_name = tree_name
        self.x_label = x_label
        self.y_label = y_label
        self.range = range
        self.n_bins = n_bins

class PlotDiffVar :
    def __init__(self, true_tree_name, reco_tree_name, x_label, y_label, range, n_bins):
        self.true_tree_name = true_tree_name
        self.reco_tree_name = reco_tree_name
        self.x_label = x_label
        self.y_label = y_label
        self.range = range
        self.n_bins = n_bins  

class SegVar :
    def __init__(self, tree_name, label, options, colors):
        self.tree_name = tree_name
        self.label = label
        self.options = options
        self.colors = colors

class ProfileVar :
    def __init__(self, plot_var_x, plot_var_y):
        self.plot_var_x = plot_var_x
        self.plot_var_y = plot_var_y     
        
###############################################
# PlotVar
###############################################        
        
# Event
nu_true_energy = PlotVar('MCNu_Energy', 'True Nu Energy [GeV]', 'Frac. of True Nu', [0,10.0], 40)
nu_vis_true_energy = PlotVar('MCNu_VisEnergy', 'True Vis Nu Energy [GeV]', 'Frac. of True Nu', [0,10.0], 40)
nu_vertex_accuracy = PlotVar('RecoNu_VertexAcc_Pass2', 'NuVertexAccuracy [cm]', 'Frac. of Reco Nu', [0,10.0], 50)
event_pfp_correctness_frac = PlotVar('-', 'Fraction of Correct', 'Frac. of Events', [0.0, 1.0], 20) # This will be modified
event_correctness = PlotVar('-', 'IsCorrect', 'Frac. of Events', [0.0, 1.0], 20) # This will be modified
Event_MCP_plotting_vars = [nu_true_energy, nu_vis_true_energy]
Event_Reco_plotting_vars = [nu_vertex_accuracy]

# PFP
completeness_var = PlotVar('BM_Completeness', 'Completeness', 'Frac. of MCParticles', [0,1.0], 20)
purity_var = PlotVar('BM_Purity', 'Purity', 'Frac. of MCParticles', [0,1.0], 20)
alt_completeness_var = PlotVar('ALT_Completeness', 'AltCompleteness', 'Frac. of MCParticles', [-1.01, 1.0], 40)
alt_purity_var = PlotVar('ALT_Purity', 'AltPurity', 'Frac. of MCParticles', [-1.01, 1.0], 40)
n_mc_hits_2d_var = PlotVar('MCP_NMCHits2D', 'NMCHits2D', 'Frac. of PFParticles', [0, 1500], 30)
#n_mc_hits_2d_var = PlotVar('MCP_NMCHits2D', 'NMCHits2D', 'Frac. of PFParticles', [0, 200], 20)
mc_displacement_var = PlotVar('MCP_Displacement', 'MC Displacement [cm]', 'Frac. of PFParticles', [0, 100], 10)
theta_xz_var = PlotVar('MCP_TrueThetaXZ', 'ThetaXZ', 'Frac. of PFParticles', [-3.5, 3.5], 25)
theta_yz_var = PlotVar('MCP_TrueThetaYZ', 'ThetaYZ', 'Frac. of PFParticles', [-1.6, 1.6], 12)
pfo_energy_var = PlotVar('MCP_TrueEnergy', 'True Particle Energy', 'Frac. of PFParticles', [0,3], 50)
true_vis_energy = PlotVar('MCP_TrueVisEnergy', 'True Visible Energy', 'Frac. of PFParticles', [0,5], 50)
pfo_signed_vertex_acc_var = PlotVar('BM_VertexAcc', 'Signed Vertex deltaR [cm]', 'Frac. of PFParticles', [-25,25], 25)
multiplicity_var = PlotVar('MCNu_Multiplicity', 'N Reco Targets in Event', 'Frac. of PFParticles', [0,50], 50)    
shower_multiplicity_var = PlotVar('MCNu_ShowerMultiplicity', 'N Reco Targets in Event', 'Frac. of PFParticles', [0,20], 20)
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

###############################################
# PlotDiffVar
############################################### 

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

###############################################
# SegVar
############################################### 

# ALT
alt_pfp_seg_var = SegVar('ALT_PDG', 'PDG', [13, 2212, 211, 777, 22, 11, 111], ["Blue", "tab:green", "tab:pink", 'tab:purple', "tab:orange", "Red", "tab:olive"])
alt_is_up_hierarchy_seg_var = SegVar('ALT_IsUpstreamHierarchy', 'IsUpHierarchy', [-1, 0, 1], ["tab:gray", "red", "green"]) 
alt_is_same_mc_var = SegVar('ALT_IsSameMC', 'IsSameMC', [-1, 0, 1], ["tab:gray", "red", "green"]) 
ALT_seg_vars = [alt_pfp_seg_var, alt_is_up_hierarchy_seg_var, alt_is_same_mc_var]

###############################################
# ProfileVar
###############################################
pfp_completeness_true_vis_energy = ProfileVar(true_vis_energy, completeness_var)
pfp_completeness_multiplicity = ProfileVar(multiplicity_var, completeness_var)
pfp_completeness_shower_multiplicity = ProfileVar(shower_multiplicity_var, completeness_var)
pfp_completeness_n_mc_hits = ProfileVar(n_mc_hits_2d_var, completeness_var)
pfp_purity_true_vis_energy = ProfileVar(true_vis_energy, purity_var)
pfp_purity_multiplicity = ProfileVar(multiplicity_var, purity_var)
pfp_purity_shower_multiplicity = ProfileVar(shower_multiplicity_var, purity_var)
pfp_purity_n_mc_hits = ProfileVar(n_mc_hits_2d_var, purity_var)

PFP_profile_vars = [pfp_completeness_true_vis_energy, pfp_completeness_multiplicity, pfp_completeness_shower_multiplicity, pfp_completeness_n_mc_hits,
                    pfp_purity_true_vis_energy, pfp_purity_multiplicity, pfp_purity_shower_multiplicity, pfp_purity_n_mc_hits]





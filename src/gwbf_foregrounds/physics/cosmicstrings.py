import numpy as np
from scipy.special import zeta
import sys
import os

from .cosmo import *
from .data_processing import *

import jax.numpy as jnp
from jax.scipy.special import zeta
from jax import jit
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)
from jax import debug

# --- Default string parameters ---
canonical_string_params_VOS = {
    'q': 4. / 3.,                 # spectral index, for VOS, q=
    'c_tilde': 0.23,            # Loop formation efficiency (taken from Nambu-Goto string simulations in an expanding Universe)
    'gamma': jnp.sqrt(2.0),     # Typical Lorentz boost of loops
    'curly_F_a': 0.1,           # Fraction of the total number of loops in the NG simulations that contributes significantly to GW production
    'Gamma':    50.,            # Sum of all P_j
    'type_id':  0,                # 0 --> VOS (semianalytical) or 1 --> BOS (simulation-based)
    'Gmu':      10**-10,              # Gravitational constant times string tension
    'T_i':      10**10,         # GeV, Energy scale at formation
    'alpha_L':  0.1/0.27            # 
}

# --- Default string parameters ---
canonical_string_params_BOS = {
    'q': 4 / 3,                 # spectral index, for VOS, q=
    'c_tilde': 0.23,            # Loop formation efficiency (taken from Nambu-Goto string simulations in an expanding Universe)
    'gamma': jnp.sqrt(2.0),     # Typical Lorentz boost of loops
    'curly_F_a': 0.1,           # Fraction of the total number of loops in the NG simulations that contributes significantly to GW production
    'Gamma':    51.43,             # Sum of all P_j
    'type_id':  1,                # 0 --> VOS (semianalytical) or 1 --> BOS (simulation-based)
    'Gmu':      10**-11,              # Gravitational constant times string tension
    'T_i':      10**10,         # GeV, Energy scale at formation
    'alpha_L':  0.1/0.27            # 
}

import jax.numpy as jnp
from jax.scipy.special import zeta
from jax import jit

@jit
def P_j(j, string_params=canonical_string_params_VOS):
    """
    Efficient JAX-compatible harmonic power function P_j.

    Parameters:
    -----------
    j : int or array_like
        Harmonic mode(s)

    string_params : dict
        - 'type_id': 0 for VOS, 1 for BOS
        - 'q': spectral index 
        - 'Gamma': total GW power (both models)

    Returns:
    --------
    P_j : float or array
        GW power in mode j, normalized to ∑P_j = Gamma.
    """
    j = jnp.asarray(j)
    type_id = string_params.get('type_id', 0)
    Gamma = string_params.get('Gamma', 51.43)

    def P_j_vos(j):
        q = string_params.get('q', 4/3)
        return (Gamma / zeta(q, 1.0)) * j**(-q)

    def P_j_bos(j):
        def bos_shape(j):
            bump = (1 + (j / 3.2)**4)**(-0.18)
            flatten = (1 + j / 70)**(-0.7)
            return 10.8 * j**(-4/3) * bump * flatten

        j_norm = jnp.arange(1, 2000)
        norm = Gamma / jnp.sum(bos_shape(j_norm))
        raw = bos_shape(j)
        override = 14.0 / j**(4/3)
        return jnp.where(j == 1, override, norm * raw)

    return jnp.where(type_id == 0, P_j_vos(j), P_j_bos(j))



import jax
import jax.numpy as jnp
from jax import jit, grad

# --- ξ(t) and ξ'(t) (replace with dynamic form later) ---
@jit
def xi(t, string_params=canonical_string_params_VOS):
    return 0.27 # Placeholder for now, valid for radiation domination

@jit
def xi_dot(t, string_params=canonical_string_params_VOS):
    """
    Time derivative of ξ(t). Currently returns zero (placeholder).
    """
    return jnp.zeros_like(t)

@jit
def l_of_f(f, te, j, string_params=canonical_string_params_VOS):
    """
    Compute loop length l from frequency f, time te, and harmonic j.
    """
    a0 = 1.
    ae = a_of_t(te)  # Use pre-compiled cosmology!
    return (2*j/f) * (ae/a0)


def li_of_t(ti, string_params=canonical_string_params_VOS):
    alpha_L = string_params['alpha_L']
    xi_i = xi(ti, string_params=string_params)
    return alpha_L * xi_i * ti


#@partial(jit, static_argnames=['tol', 'max_iter', 'use_tabulated_data'])
@jit
def ti_of_l_VOS(l, te, string_params=canonical_string_params_VOS): #, tol=1e-10, max_iter=100, use_tabulated_data=True):
    """
    Compute formation time t_i from loop length l in VOS model.
    Uses iterative solution of the implicit equation.
    """
    tol=1e-10
    max_iter=100    
    Γ   = string_params['Gamma']
    Gμ  = string_params['Gmu']
    αL  = string_params['alpha_L']
    a0 = 1.
    ae = a_of_t(te)  # Use pre-compiled cosmology!

    # Initial guess: same form but assuming constant ξ
    ξ0 = xi(te, string_params)
    t_i = (l + Γ * Gμ * te) / (αL * ξ0 + Γ * Gμ)

    def body_fn(i, t_i):
        ξ_val = xi(t_i, string_params)
        denom = αL * ξ_val + Γ * Gμ 
        return (l + Γ * Gμ * te) / denom

    # Iterate fixed-point update
    #t_i_final = jax.lax.fori_loop(0, max_iter, body_fn, t_i)
    #return t_i_final # Uncomment for fixed-point iteration when xi depends on t
    return t_i

#Derivtive of ti_of_l_VOS with respect to l using JAX build in derivative function
@jit
def ti_of_l_prime(l, te, string_params=canonical_string_params_VOS):
    # Define a function that computes the gradient for a single value
    def single_grad(l_single, te_single):
        return grad(lambda l: ti_of_l_VOS(l, te_single, string_params))(l_single)
    
    # Broadcast te to match the size of l
    te_broadcasted = jnp.broadcast_to(te, l.shape)
    
    # Vectorize the gradient computation
    return jax.vmap(single_grad)(l, te_broadcasted)

#@jit
#def ti_of_f(f, te, j, string_params=canonical_string_params_VOS, tol=1e-10, max_iter=100):
#    """
#    Solve for t_i satisfying:
#        l = α_L ξ(t_i) t_i - Γ Gμ (t - t_i)
#    using fixed-point iteration:
#        t_i^{n+1} = (l + Γ Gμ t) / (α_L ξ(t_i^n) + Γ Gμ)
#    """
#    Γ   = string_params['Gamma']
#    Gμ  = string_params['Gmu']
#    αL  = string_params['alpha_L']
#    a0 = 1.
#    ae = a_of_t(te)
#    l = (2*j/f) * (ae/a0)   # CHECKED! Note: l is interpreted as loop length at emission time te
#
#            
#    # 1) Build a network time grid that covers BOTH [min(t_i), max(t)]
#    t_min = jnp.min(te)
#    t_max = jnp.max(te)
#    # use log spacing to resolve early times
#    N_net = 2048
#    log_t_grid = jnp.linspace(jnp.log(t_min * 0.999), jnp.log(t_max * 1.001), N_net)
#    t_net = jnp.exp(log_t_grid)
#    # 2) Integrate VOS on t_net
#    #    ICs at the START of t_net (radiation-like guesses are fine)
#    t0 = t_net[0]
#    L0 = 0.27 * t0
#    v0 = 0.66215023
#    v_net, L_net = vbar_L_of_t(t_net, L0, v0, c_tilde=string_params['c_tilde'])
#    # 3) Build xi(t)=L(t)/t and its time derivative on t_net
#    xi_net = L_net / t_net
#    xi_net_interp = jnp.interp(te, t_net, xi_net)
#
#    # Initial guess: same form but assuming constant ξ
#    ξ0 = xi_net[-1] #xi(te, string_params)
#    t_i = (l + Γ * Gμ * te) / (αL * ξ0 + Γ * Gμ)
#
#    def body_fn(i, t_i):
#        ξ_val = xi_net_interp[i] #xi(t_i, string_params)
#        denom = αL * ξ_val + Γ * Gμ 
#        return (l + Γ * Gμ * te) / denom
#
#    # Iterate fixed-point update
#    t_i_final = jax.lax.fori_loop(0, max_iter, body_fn, t_i)
#    t_i_final = (l + Γ * Gμ * te) / (0.1 + Γ * Gμ) # !!! This is the simpler version assuming αL * xi(t_i) = 0.1
#    return t_i_final

@jit
def ti_of_f(f, te, j, string_params=canonical_string_params_VOS):
    Γ, Gμ = string_params['Gamma'], string_params['Gmu']
    a0 = 1.0
    ae = a_of_t(te)
    l  = (2.0*j/f) * (ae/a0)

    # constant-alpha case: alpha = 0.1 exactly
    alpha = 0.1
    ti = (l + Γ*Gμ*te) / (alpha + Γ*Gμ)
    return ti


# --- Loop number density ---
@jit
def n_loops(f, a, j, string_params=canonical_string_params_VOS, VOSeq=False):
    """
    Cosmic string loop number density n(l,a) in VOS or BOS model.

    Parameters
    ----------
    l : float or array
        Loop length 
    a : float or array
        Scale factor.
    string_params : dict
        Includes type_id, xi_r, alpha_L, gamma, Gmu, Gamma, curly_F_a
    """
    #l = jnp.atleast_1d(l)
    f = jnp.atleast_1d(f)
    a = jnp.atleast_1d(a)
    j = jnp.atleast_1d(j)

    # Unpack
    type_id = string_params['type_id']
    debug.print("type_id = {x}", x=type_id) 
    γ       = string_params['gamma']
    ℱα      = string_params['curly_F_a']
    Gmu     = string_params['Gmu']
    Γ       = string_params['Gamma']
    αL      = string_params['alpha_L']
    c_tilde = string_params['c_tilde']

    # Cosmology - use pre-compiled functions for speed
    t = t_of_a(a, use_tabulated_data=True) # These are times (of emmission) at scale factors a

    # --- VOS branch ---
    def vos_branch():
        # --- formation times & scale factors at emission ---
        t_i = ti_of_f(f, t, j, string_params=string_params)   # shape ~ a (or broadcast)
        a_i = a_of_t(t_i)

        # --- "simple" (your existing placeholders) ---
        ξ_ti_simple    = xi(t_i, string_params=string_params)      # scalar 0.27 in your stub
        ξdot_ti_simple = xi_dot(t_i, string_params=string_params)  # zeros
        Ceff_simple    = 5.4

        # === Complex (VOSeq=True): evolve VOS and sample at t_i ===

        # 1) Build a network time grid that covers BOTH [min(t_i), max(t)]
        t_min = jnp.minimum(jnp.min(t_i), jnp.min(t))
        t_max = jnp.maximum(jnp.max(t_i), jnp.max(t))
        # use log spacing to resolve early times
        N_net = 2048
        log_t_grid = jnp.linspace(jnp.log(t_min * 0.999), jnp.log(t_max * 1.001), N_net)
        t_net = jnp.exp(log_t_grid)

        # 2) Integrate VOS on t_net
        #    ICs at the START of t_net (radiation-like guesses are fine)
        t0 = t_net[0]
        L0 = 0.27 * t0
        v0 = 0.66215023
        v_net, L_net = vbar_L_of_t(t_net, L0, v0, c_tilde=c_tilde)

        # 3) Build xi(t)=L(t)/t and its time derivative on t_net
        xi_net = L_net / t_net
        # gradient w.r.t. t (central where possible)
        # derivative wrt ln t, then convert to d/dt
        dln_t = log_t_grid[1] - log_t_grid[0]              # constant step
        dxi_dln_t = jnp.gradient(xi_net, dln_t)            # OK: constant spacing
        dxi_dt_net = dxi_dln_t / t_net # d/dt = (1/t) d/dln t


        # 4) Interpolate xi, dxi/dt, and v to the required times t_i
        #    (and also v to the "observer" grid t if you want it)
        ξ_ti_complex    = jnp.interp(t_i, t_net, xi_net)
        ξdot_ti_complex = jnp.interp(t_i, t_net, dxi_dt_net)
        v_ti_complex    = jnp.interp(t_i, t_net, v_net)
        #γ = 1/jnp.sqrt(1-v_ti_complex**2) # Uncomment if you want to use the relativistic gamma factor computed from vbar

        # 5) C_eff from VOS:  c_tilde * vbar / (gamma * xi^3)
        Ceff_complex = c_tilde * v_ti_complex / (γ * ξ_ti_complex**3)

        # --- choose simple vs complex with VOSeq flag and broadcast safely ---
        Ceff    = jnp.where(VOSeq, Ceff_complex,    Ceff_simple)
        ξ_ti    = jnp.where(VOSeq, ξ_ti_complex,    ξ_ti_simple)
        ξdot_ti = jnp.where(VOSeq, ξdot_ti_complex, ξdot_ti_simple)

        # denominator with alpha = 0.1 (constant-alpha case)
        denom = 0.1 * (0.1 + Γ * Gmu)

        # formation time of the network
        t_form = t_of_a(a_of_T(string_params['T_i']))

        # loops contribute only if  t_form ≤ t_i ≤ t
        valid = (t_i >= t_form) & (t_i <= t)

        expr = ℱα * Ceff / (denom * t_i**4) * (a_i / a)**3
        return jnp.where(valid, expr, 0.0)
        #return ℱα * Ceff / (denom * t_i**4) * (a_i / a)**3
    


    # --- BOS branch ---
    #def bos_branch():
    #    # Get formation time t_i
    #    ti = ti_of_f(f, t, j, string_params=string_params)
    #    #ti_of_l_VOS(l, te, string_params=string_params)
    #    li = li_of_t(ti, string_params=string_params)
    #    tilde_dH = li/0.05        
    #    ti_dH       = ti_of_l_VOS(tilde_dH, ti, string_params=string_params)
    #    #ti_dH_prime = ti_of_l_prime(tilde_dH, ti, string_params=string_params)
    #    ti_dH_prime = 0.05 * ti_of_l_prime (tilde_dH, ti, string_params=string_params) # ti_of_dH_prime(tilde_dH) #0# 0.05 is for derivatie wrt dH not dl, with dl = 0.05 ddH
    #    ai    = a_of_t(ti) #, use_tabulated_data=True)
    #    ai_dH       = a_of_t(ti_dH) #, use_tabulated_data=True)
#
    #    
    #    diagnotics = {}
    #    #diagnotics['t'] = t
    #    diagnotics['ti'] = ti
    #    diagnotics['li'] = li
    #    diagnotics['ai'] = ai
    #    #diagnotics['tilde_dH'] = tilde_dH
    #    diagnotics['ti_dH'] = ti_dH
    #    diagnotics['ti_dH_prime'] = ti_dH_prime
    #    diagnotics['ai_dH'] = ai_dH
    #    
    #    # Plot all diagnostics in one figure with many subplots
    #    fig, axes = plt.subplots(figsize=(20, 4), ncols=len(diagnotics), nrows=1)
    #    for i, (key, value) in enumerate(diagnotics.items()):
    #        # if lenght of a is >1, then use a for x
    #        if len(a) > 1: 
    #            if key == 'ti_dH_prime':
    #                axes[i].loglog(li, ti_dH_prime)
    #                axes[i].set_xlabel('li')
    #                axes[i].set_ylabel('ti_dH_prime')
    #            else:
    #                axes[i].loglog(a, value, label = 'a')
    #                axes[i].loglog(t, value, label = 't')
    #                axes[i].set_xlabel('a')
    #                axes[i].set_ylabel(key)
    #        elif len(f) > 1:
    #            axes[i].loglog(f,value)
    #            axes[i].set_xlabel('f')
    #            axes[i].set_ylabel(key)
    #        else:
    #            axes[i].loglog(t, value)
    #            axes[i].set_xlabel('t')
    #            axes[i].set_ylabel(key)
    #    plt.tight_layout()
    #    plt.show()
    #    
    #    return 1842.52 * tilde_dH**-4 * (ai_dH / a)**3 * ti_dH_prime
    #
    #    # --- BOS branch ---
    def bos_branch():
        # Get formation time t_i
        ti = ti_of_f(f, t, j, string_params=string_params)
        #li = l_of_f(f, ti, j)
        li = li_of_t(ti, string_params=string_params)
        dH = 20 * li
        t_i_star = t_of_dH(dH) #, cosmo=LCDM, t_bounds=(7.709107749996553e-16, 2.428330008027432e+42), rel_tol=1e-10):
        ti_star_prime = ti_of_dH_prime(dH)
        print("In BOS branch, ti_star_prime = ", ti_star_prime)
        ai_star = a_of_t(t_i_star) #, use_tabulated_data=True)
    
        return 20 * 92.126 / dH**4  * ti_star_prime * (ai_star / a)**3


    # --- Dispatch ---
    #return jnp.where(type_id == 0, vos_branch(), bos_branch())
    return jax.lax.cond(type_id == 0, lambda _: vos_branch(), lambda _: bos_branch(), None)
    #if type_id == 0:
    #    print("In VOS branch; type_id = ", type_id)
    #    return vos_branch()
    #else:
    #    print("In BOS branch; type_id = ", type_id)
    #    return bos_branch()



#@partial(jit, static_argnames=['use_tabulated_data'])
@jit    
def gw_integrand(a, f, j, string_params=canonical_string_params_VOS, a_0=1.0, VOSeq=False): #, use_tabulated_data=True):
    use_tabulated_data=True
    """
    Compute the integrand for the gravitational wave spectrum:
    (1/H(a)) * (a/a_0)^4 * n[2j/f * a/a_0, t(a)]
    
    Parameters:
    -----------
    a : float or array_like
        Scale factor
    f : float
        Frequency in GeV
    j : int
        Harmonic mode number
    string_params : dict
        String network parameters
    a_0 : float
        Present-day scale factor (default: 1.0)
    use_tabulated_data : bool (static)
        Whether to use tabulated cosmological functions
    
    Returns:
    --------
    integrand : float or array_like
        Value of the integrand at scale factor a
    """
    # Ensure inputs are JAX arrays
    a = jnp.asarray(a)
    
    # Cosmic history factor: (1/H(a)) * (a/a_0)^4
    H = H_of_a(a) #, use_tabulated_data=use_tabulated_data)  # Use pre-compiled cosmology!
    
    
    # Loop length: l = 2j/f * a/a_0
    #loop_length = (2.0 * j / f) * (a / a_0)
    
    # Time corresponding to scale factor a
    #time_a = t_of_a(a) # , use_tabulated_data=use_tabulated_data)  # Use pre-compiled cosmology!
    
    # Loop number density n(l, t)
    # Note: n_loops expects (f, a, j) but we need n(l, t)
    # We'll call n_loops with the appropriate arguments
    n_density = n_loops(f, a, j, string_params=string_params, VOSeq=VOSeq)
    
    return 1/H * (a / a_0)**4 * n_density

# --- Original Omega_GW function ---
@partial(jit, static_argnames=['j_min', 'j_max', 'N_a', 'VOSeq']) #, 'use_tabulated_data'])
def Omega_GW_no_trick(f, string_params=canonical_string_params_VOS, j_min=1, j_max=100, N_a=100, VOSeq=False): #, use_tabulated_data=True):
    """
    Compute the gravitational wave spectrum Ω_GW(f) from cosmic strings.
    
    Ω_GW(f) = (1/(3 H₀² m_Pl²)) * Σⱼ (2j/f)(Gμ²Pⱼ) * ∫ da (1/H(a))(a/a₀)⁴ n[2j/f·a/a₀, t(a)]
    
    Parameters:
    -----------
    f : float or array_like
        Frequency in GeV
    string_params : dict
        String network parameters
    j_min : int
        Minimum harmonic mode to include
    j_max : int
        Maximum harmonic mode to include
    N_a : int
        Number of points for scale factor integration
    use_tabulated_data : bool
        Whether to use tabulated cosmological data
    
    Returns:
    --------
    Omega_GW : float or array_like
        Gravitational wave spectrum at frequency f
    """
    # Convert input to JAX array
    f = jnp.asarray(f)
    
    # Physical constants 
    m_Pl = constants['m_pl']
    G    = constants['G']
    a_0=1.0
    # Use pre-compiled cosmology - no cosmo parameter needed!
    H0 = LCDM['H0']  # Hubble constant from default cosmology
    Gmu = string_params['Gmu']  # String tension parameter
    T_ini = string_params['T_i']
    a_ini = a_of_T(T_ini)  # Use pre-compiled cosmology!
    
    # Create integration grid in scale factor
    a_grid = jnp.exp(jnp.linspace(jnp.log(a_ini), jnp.log(a_0), N_a))
    
    # Create array of harmonic modes
    j_modes = jnp.arange(j_min, j_max + 1)
    
    def single_mode_contribution(j, f_val):
        """Compute contribution from a single harmonic mode j at frequency f_val"""
        in_sum_factor = (2.0 * j / f_val) * (Gmu**2/G * P_j(j, string_params=string_params))
        integrand_vals = gw_integrand(a_grid, f_val, j, string_params=string_params, a_0=a_0, VOSeq=VOSeq) #, use_tabulated_data=use_tabulated_data)
        integral = jnp.trapezoid(integrand_vals, a_grid)
        
        return in_sum_factor * integral
    
    def compute_single_frequency(f_val):
        """Compute Ω_GW for a single frequency by summing over all modes"""
        # Compute contributions from all modes
        mode_contributions = jax.vmap(lambda j: single_mode_contribution(j, f_val))(j_modes)
        
        # Sum over all modes
        total_contribution = jnp.sum(mode_contributions)
    
        return total_contribution
    
    
    # Vectorize over frequencies if f is an array
    if f.ndim == 0: total = compute_single_frequency(f)           # Single frequency
    else:           total = jax.vmap(compute_single_frequency)(f) # Array of frequencies
    
    # Apply prefactor: 1/(3 H₀² m_Pl²)
    prefactor = 1.0 / (3.0 * H0**2 * m_Pl**2) 
    
    return prefactor * total


# --- Omega_GW function with "trick" ---
def Omega_GW(f, string_params=canonical_string_params_VOS, j_min=1, j_max=1, N_a=1000, N_f=1000, VOSeq=False): #, use_tabulated_data=True, return_interpolation_function=False):
    """
    Compute the gravitational wave spectrum Ω_GW(f) from cosmic strings using the scaling trick.
    
    The trick is that Ω_GW^(j)(f) = P_j * Ω_GW^(1)(f/j), which allows us to compute the spectrum
    for all harmonics by just computing j=1 once and scaling appropriately.
    
    Parameters:
    -----------
    f : float or array_like
        Frequency in GeV
    string_params : dict
        String network parameters
    j_min : int
        Minimum harmonic mode to include
    j_max : int
        Maximum harmonic mode to include
    N_a : int
        Number of points for scale factor integration
    N_f : int
        Number of points in frequency grid for interpolation.
        Should be large enough to ensure sufficient points per octave:
        N_f > N_points_per_octave * log10(f_max/(f_min/j_max))
    use_peera_spectra : bool
        Whether to use Peera's tabulated spectra data
    
    Returns:
    --------
    Omega_GW : float or array_like
        Gravitational wave spectrum at frequency f
    """
            
    # Convert input to JAX array
    f = jnp.asarray(f)
    
    # Create frequency grid that covers all needed frequencies (f/j_max to f)
    #f_min = f[0]/j_max
    #f_max = f[-1]

    f_obs_min = jnp.min(f)
    f_obs_max = jnp.max(f)

    # Cover all needed f/j AND add a safety margin to avoid edge clamping
    pad = 2.0
    f_min = f_obs_min / j_max / pad
    f_max = f_obs_max * pad

    f_log_grid = jnp.logspace(jnp.log10(f_min), jnp.log10(f_max), N_f)

    # Compute base spectrum for j=1
    tilde_Omega_GW_1 = Omega_GW_no_trick(f_log_grid, string_params=string_params, j_min=1, j_max=1, N_a=N_a, VOSeq=VOSeq)/ P_j(1, string_params=string_params)
    
    # Create interpolation function in log space
    log_f = jnp.log10(f_log_grid)
    log_Omega = jnp.log10(tilde_Omega_GW_1)
    # A JAX-pure linear interpolator: works on scalars or arrays
    loglog_Omega_interp = jax.jit(
        lambda x: jnp.interp(x, log_f, log_Omega)
    )
    
    def compute_j_contribution(j, f_val):
        """Compute contribution from a single harmonic j at frequency f_val"""
        # Get P_j for this harmonic
        p_j = P_j(j, string_params=string_params)
        # Evaluate base spectrum at f/j in log space and convert back
        log_f_j = jnp.log10(f_val/j)
        return p_j * 10.0**loglog_Omega_interp(log_f_j)
    
    def compute_single_frequency(f_val):
        """Compute total Ω_GW for a single frequency by summing over all modes"""
        j_modes = jnp.arange(j_min, j_max + 1)
        # Compute contributions from all modes
        mode_contributions = jax.vmap(lambda j: compute_j_contribution(j, f_val))(j_modes)
        # Sum over all modes
        return jnp.sum(mode_contributions)
    
    # Vectorize over frequencies if f is an array
    if f.ndim == 0:
        return compute_single_frequency(f)  # Single frequency
    #elif return_interpolation_function:
    #    return jax.jit(lambda f: 10.0**loglog_Omega_interp(jnp.log10(f)))
    else:
        return jax.vmap(compute_single_frequency)(f)  # Array of frequencies
    

from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=['j_min','j_max','N_a','N_f','j_d','N_tail'])
def Omega_GW_mixed(f,
                       string_params=canonical_string_params_VOS,
                       j_min=1,
                       j_max=100000,
                       N_a=1000,
                       N_f=1000,
                       VOSeq=False,
                       j_d=10000,        # split between sum and integral
                       N_tail=1024,      # #points for the tail integral
                       pad=2.0):

    # --------- PREAMBLE: same as your Omega_GW ---------
    f = jnp.asarray(f)

    f_obs_min = jnp.min(f)
    f_obs_max = jnp.max(f)

    # frequency support for the j=1 base spectrum (with pad)
    f_min = f_obs_min / j_max / pad
    f_max = f_obs_max * pad
    f_log_grid = jnp.logspace(jnp.log10(f_min), jnp.log10(f_max), N_f)

    # base j=1 spectrum, de-weight by P_1 to get tildeΩ^(1)
    tilde_Omega_GW_1 = Omega_GW_no_trick(
        f_log_grid, string_params=string_params, j_min=1, j_max=1, N_a=N_a, VOSeq=VOSeq
    ) / P_j(1, string_params=string_params)

    log_f = jnp.log10(f_log_grid)
    log_Om = jnp.log10(tilde_Omega_GW_1)

    # simple log–log interpolator (same as yours)
    @jax.jit
    def loglog_Omega_interp(x_log10):
        x_log10 = jnp.clip(x_log10, log_f[0], log_f[-1])
        return jnp.interp(x_log10, log_f, log_Om)

    def compute_j_contribution(j, f_val):
        pj = P_j(j, string_params=string_params)
        return pj * 10.0 ** loglog_Omega_interp(jnp.log10(f_val / j))

    # --------- NEW: set up tail grid once (concrete sizes!) ---------
    # Use python min() for concrete upper bound of the discrete sum
    J_STOP = int(min(j_d, j_max + 1))
    j_discrete = jnp.arange(j_min, J_STOP, dtype=jnp.int32)  # OK: static bounds

    # Tail grid in j-space (log-spaced is robust), integrate in *linear j* with trapezoid
    if j_max >= j_d:
        j_tail_grid = jnp.logspace(jnp.log10(float(j_d)), jnp.log10(float(j_max)), N_tail)
    else:
        j_tail_grid = jnp.array([], dtype=jnp.float32)

    def tail_integral(f_val):
        if j_tail_grid.size == 0:
            return 0.0
        # integrand(j) = P_j(j) * tildeΩ^(1)(f/j)
        pj = jax.vmap(lambda jj: P_j(jj, string_params=string_params))(j_tail_grid)
        tilde = 10.0 ** loglog_Omega_interp(jnp.log10(f_val / j_tail_grid))
        integrand = pj * tilde
        # trapezoid rule in linear j (works fine on a log-spaced grid too)
        return jnp.trapezoid(integrand, j_tail_grid)

    def compute_single_frequency(f_val):
        # discrete sum
        disc = (jnp.sum(jax.vmap(lambda jj: compute_j_contribution(jj, f_val))(j_discrete))
                if j_discrete.size > 0 else 0.0)
        # continuous tail
        tail = tail_integral(f_val)
        return disc + tail

    return jax.vmap(compute_single_frequency)(f) if f.ndim else compute_single_frequency(f)



# --- Omega_GW function with "trick" ---
# Global cache for the interpolation function (keyed by VOS_of_BOS)
_peera_interp_cache = {}

def h2Omega_GW_Peera_interp(VOS_or_BOS=None):
    """
    Return the interpolation function for Peera's tabulated spectra data.

    Parameters
    ----------
    VOS_or_BOS : str, optional
        Which tabulated spectra to use: ``"BOS"`` for BOS_saikawa.dat (default),
        or ``"VOS"`` for vos_saikawa.dat.

    Returns
    -------
    Omega_GeV_interp : function
        Interpolation function for Peera's tabulated spectra data.
    """
    global _peera_interp_cache
    if VOS_or_BOS is None:
        raise ValueError("VOS_or_BOS must be provided")

    choice = str(VOS_or_BOS).strip().upper()
    if choice not in ("VOS", "BOS"):
        raise ValueError(f"VOS_or_BOS must be 'VOS' or 'BOS', got {VOS_or_BOS!r}")

    # Return cached function if available for this choice (cache must be a dict)
    if isinstance(_peera_interp_cache, dict) and choice in _peera_interp_cache:
        return _peera_interp_cache[choice]

    # METHOD USING DATA FROM PEERA AND INTERPOLATION BETWEEN SPECTRA
    # This branch cannot be JIT-compiled because it returns a JAX function.
    # Tabulated notebook input data now lives under Work/data/notebook_inputs.
    _physics_dir = os.path.dirname(os.path.abspath(__file__))
    # DELETE this line: _repo_root = os.path.dirname(_physics_dir)
    filename = "vos_saikawa.dat" if choice == "VOS" else "BOS_saikawa.dat"
    path_data = os.path.join(
        _physics_dir,          # ← use this, not _repo_root
        "tabulated_spectra",
        filename,
    )
    bundle = load_peera_spectra(path_data)
    
    Omega_GeV_interp = build_interpolator_from_spectra_bundle(bundle)
    
    # Change from logGmu to Gmu
    
    def Omega_GeV_interp_Gmu(f_GeV, string_params=None, j_min=None, j_max=None, j_d=None, N_a=None, N_f=None, N_tail=None, VOSeq=True, use_tabulated_data=None):
        # Use same keyword arguments to ensure compatibility with original GW spectra fucntion
        # Convert to JAX array and use JAX log10 (not NumPy) for JIT compatibility
        Gmu_val = jnp.asarray(string_params['Gmu'])
        logGmu = jnp.log10(Gmu_val)
        return Omega_GeV_interp(f_GeV, logGmu)
    
    # Cache the function for this choice
    if not isinstance(_peera_interp_cache, dict):
        _peera_interp_cache = {}
    _peera_interp_cache[choice] = Omega_GeV_interp_Gmu

    return Omega_GeV_interp_Gmu






import jax
import jax.numpy as jnp
from jax import jit, lax

# --- VOS 'k(v)' for Y=0 ---
@jit
def k_of_v(v):
    v = jnp.clip(v, 0.0, 1.0 - 1e-12)
    num = 1.0 - 8.0 * v**6
    den = 1.0 + 8.0 * v**6
    return (2.0*jnp.sqrt(2.0)/jnp.pi) * (num/den) * (1.0 - v**2) * (1.0 + 2.0*jnp.sqrt(2.0)*v**3)

# --- RHS of VOS (Y=0): dL/dt and dv/dt ---
@jit
def _vos_rhs(t, y, c_tilde):
    L, v = y
    a = a_of_t(t)            # you already have these tabulated/JITed
    H = H_of_a(a)
    kv = k_of_v(v)
    dLdt = H * L * (1.0 + v*v) + 0.5 * c_tilde * v
    dvdt = (1.0 - v*v) * (kv / jnp.maximum(L, 1e-60) - 2.0 * H * v)
    return jnp.array([dLdt, dvdt])

# --- RK4 step (time-variable dt) ---
@jit
def _rk4_step(t_prev, y_prev, t_next, c_tilde):
    dt = t_next - t_prev
    k1 = _vos_rhs(t_prev,              y_prev,                 c_tilde)
    k2 = _vos_rhs(t_prev + 0.5*dt,     y_prev + 0.5*dt*k1,     c_tilde)
    k3 = _vos_rhs(t_prev + 0.5*dt,     y_prev + 0.5*dt*k2,     c_tilde)
    k4 = _vos_rhs(t_prev + dt,         y_prev + dt*k3,         c_tilde)
    y_next = y_prev + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

    # Enforce physical bounds
    L_next = jnp.maximum(y_next[0], 1e-30)
    v_next = jnp.clip(y_next[1], 0.0, 1.0 - 1e-12)
    return jnp.array([L_next, v_next])

# --- Integrate VOS over a monotonic t-grid ---
@jit
def vbar_L_of_t(t_grid, L0, v0, c_tilde=0.23):
    """
    Integrate the Y=0 VOS system along a (strictly increasing) time grid t_grid [GeV^-1].
    Inputs
    ------
    t_grid : (N,) array, increasing in time
    L0, v0 : initial conditions at t_grid[0]
    c_tilde: loop-chopping efficiency (~0.23)

    Returns
    -------
    v_of_t : (N,) array of vbar(t)
    L_of_t : (N,) array of L(t)
    """
    t_grid = jnp.asarray(t_grid)
    y0 = jnp.array([L0, jnp.clip(v0, 0.0, 1.0 - 1e-12)])

    def scan_step(carry, t_next):
        t_prev, y_prev = carry
        y_next = _rk4_step(t_prev, y_prev, t_next, c_tilde)
        return (t_next, y_next), y_next

    # Run the scan over t_grid[1:]
    (_, _), ys = lax.scan(scan_step, (t_grid[0], y0), t_grid[1:])

    # Prepend the initial state
    y_all = jnp.vstack([y0, ys])
    L_all = y_all[:, 0]
    v_all = y_all[:, 1]
    return v_all, L_all

# --- Convenience wrapper to get only vbar(t) ---
@jit
def vbar_of_t(t_grid, L0, v0, c_tilde=0.23):
    v_all, _ = vbar_L_of_t(t_grid, L0, v0, c_tilde)
    return v_all

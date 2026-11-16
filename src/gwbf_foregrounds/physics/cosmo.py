# --- Default ΛCDM Parameters ---
LCDM = {
    'H0': 1.4482e-42,              # Hubble constant today in GeV
    'h': 0.6789,                   # 
    'Omega_rad0': 9.20692e-5,       # Radiation density today
    'Omega_mat0': 0.308,        # Matter density today
    'Omega_de0': 0.692,         # Dark energy today
    'T0': 2.34871e-13              # CMB temperature today in GeV
}

units = {
    # GeV to other units
    'GeV_to_s': 6.582119569e-25,
    'GeV_to_m': 1.973270522e-16,
    'GeV_to_km': 1.973270522e-13,
    'GeV_to_cm': 1.973270522e-14,
    'GeV_to_Mpc': 6.39558868e-41,
    'GeV_to_Hz': 1.519267447e24,  # 1/ℏ

    # other units to GeV
    's_to_GeV': 1.519267447e24,   # 1/ℏ
    'm_to_GeV': 5.0677307e15,
    'km_to_GeV': 5.0677307e12,
    'cm_to_GeV': 5.0677307e13,
    'Mpc_to_GeV': 1.5637e40,
    'Hz_to_GeV': 6.582119569e-25  # ℏ
}
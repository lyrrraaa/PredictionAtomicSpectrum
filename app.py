import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------
# Physical Constants
# ----------------------------
h = 6.626e-34
c = 3.0e8
eV = 1.602e-19
d_default = 1.0 / (600e3)  # 600 lines/mm

# ----------------------------
# Utility Functions
# ----------------------------
def wavelength_to_rgb(wl):
    wl = float(wl)
    if wl < 380 or wl > 780:
        return (0.0, 0.0, 0.0)
    if wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    factor = 1.0 if 420 <= wl <= 700 else 0.3
    return (max(0.0, r * factor), max(0.0, g * factor), max(0.0, b * factor))

def photon_energy_eV(wl_nm):
    wl_m = wl_nm * 1e-9
    E_J = h * c / wl_m
    return E_J / eV

def zone_label(wl, zones):
    for color, (low, high) in zones.items():
        if low <= wl < high:
            return color
    return "None"

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="LED Spectrum Visualizer", layout="wide")

# ----------------------------
# Custom Style (Bright Theme)
# ----------------------------
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        color: #222;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-color: #ffffff;
    }
    h1, h2, h3, h4 {
        color: #222;
        font-weight: 600;
    }
    .css-1d391kg {
        background-color: #ffffff !important;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.title("LED Spectrum Visualizer")
st.caption("Interactive visualization of LED emission spectrum, photon energy, and diffraction geometry")

# ----------------------------
# Sidebar Configuration
# ----------------------------
st.sidebar.header("Configuration")
d_input = st.sidebar.number_input("Grating spacing (d, m)", value=float(d_default), format="%.6e")
sigma_nm = st.sidebar.slider("Gaussian width (σ, nm)", 2.0, 80.0, 12.0)
normalize = st.sidebar.checkbox("Normalize intensity", value=True)
show_peaks = st.sidebar.checkbox("Show λ markers", value=True)
st.sidebar.markdown("---")

# Editable zones
st.sidebar.subheader("Spectrum Zones (Editable)")
zones = {
    "Violet": [380, 420],
    "Blue": [421, 490],
    "Green": [491, 530],
    "Yellow": [531, 580],
    "Orange": [581, 620],
    "Red": [621, 780]
}

for color in zones:
    cols = st.sidebar.columns(2)
    with cols[0]:
        zones[color][0] = st.number_input(f"{color} min", 380, 780, zones[color][0])
    with cols[1]:
        zones[color][1] = st.number_input(f"{color} max", 380, 780, zones[color][1])

# ----------------------------
# Diffraction Input Data
# ----------------------------
st.markdown("### Input Diffraction Data")
default = pd.DataFrame({"X (mm)": [20, 25, 30, 35, 40], "Y (mm)": [100]*5})
data = st.data_editor(default, num_rows="dynamic", use_container_width=True)

# ----------------------------
# Calculations
# ----------------------------
X_m = data["X (mm)"].values * 1e-3
Y_m = data["Y (mm)"].values * 1e-3
theta_rad = np.arctan2(X_m, Y_m)
lambda_m = d_input * np.sin(theta_rad)
lambda_nm = np.where(lambda_m > 0, lambda_m * 1e9, np.nan)
E_eV = [photon_energy_eV(wl) if not np.isnan(wl) else np.nan for wl in lambda_nm]
color_zones = [zone_label(wl, zones) for wl in lambda_nm]

results = pd.DataFrame({
    "X (mm)": data["X (mm)"],
    "Y (mm)": data["Y (mm)"],
    "θ (°)": np.degrees(theta_rad),
    "λ (nm)": lambda_nm,
    "E (eV)": E_eV,
    "Color Zone": color_zones
})

st.markdown("### Computed Results")
st.dataframe(results.style.format({"θ (°)": "{:.2f}", "λ (nm)": "{:.1f}", "E (eV)": "{:.2f}"}), use_container_width=True)

# ----------------------------
# Premium Emission Spectrum (Full Gradient)
# ----------------------------
st.markdown("### Emission Spectrum")

wavelengths = np.linspace(380, 780, 1000)
I = np.zeros_like(wavelengths)
valid_wls = [wl for wl in lambda_nm if not np.isnan(wl)]

# Buat distribusi Gaussian untuk setiap puncak
for wl in valid_wls:
    I += np.exp(-0.5 * ((wavelengths - wl) / sigma_nm) ** 2)

if normalize and I.max() > 0:
    I /= I.max()

# Buat warna spektrum kontinu (violet ke merah)
colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for (r,g,b) in [wavelength_to_rgb(w) for w in wavelengths]]

# Plotly figure
fig = go.Figure()

# Tambahkan area spektrum dengan gradasi
fig.add_trace(go.Scatter(
    x=wavelengths,
    y=I,
    mode='lines',
    line=dict(width=0),
    fill='tozeroy',
    fillcolor='rgba(255,255,255,0)',
    name='Emission Spectrum',
))

# Tambahkan warna gradasi spektrum
for i in range(len(wavelengths)-1):
    fig.add_shape(
        type="rect",
        x0=wavelengths[i], x1=wavelengths[i+1],
        y0=0, y1=I[i],
        fillcolor=colors[i],
        line=dict(width=0)
    )

# Tambahkan garis batas atas agar terlihat premium
fig.add_trace(go.Scatter(
    x=wavelengths, y=I,
    mode='lines',
    line=dict(width=3, color='rgba(255,255,255,0.8)'),
    name='Spectral Envelope'
))

# Tambahkan label tiap puncak
if show_peaks:
    for wl in valid_wls:
        clr = f'rgb({int(wavelength_to_rgb(wl)[0]*255)}, {int(wavelength_to_rgb(wl)[1]*255)}, {int(wavelength_to_rgb(wl)[2]*255)})'
        fig.add_vline(x=wl, line=dict(color=clr, width=2))
        fig.add_annotation(
            x=wl, y=1.05,
            text=f"{wl:.0f} nm",
            showarrow=False,
            font=dict(color=clr, size=12, family="Poppins")
        )

# Layout mewah dan terang
fig.update_layout(
    xaxis_title="Wavelength (nm)",
    yaxis_title="Relative Intensity",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#111", family="Poppins"),
    xaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
    yaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
    height=450,
    margin=dict(l=50, r=30, t=40, b=50)
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Premium Photon Energy vs Wavelength
# ----------------------------
st.markdown("### Photon Energy vs Wavelength")

energies = [photon_energy_eV(w) for w in wavelengths]
fig2 = go.Figure()

# Kurva energi (smooth inverse, glowing style)
fig2.add_trace(go.Scatter(
    x=wavelengths,
    y=energies,
    mode="lines",
    line=dict(
        width=4,
        color="rgba(255,0,100,0.7)"
    ),
    name="E = hc/λ"
))

# Highlight setiap data hasil eksperimen
for wl, E in zip(results["λ (nm)"], results["E (eV)"]):
    if not np.isnan(wl):
        clr = f'rgb({int(wavelength_to_rgb(wl)[0]*255)}, {int(wavelength_to_rgb(wl)[1]*255)}, {int(wavelength_to_rgb(wl)[2]*255)})'
        fig2.add_trace(go.Scatter(
            x=[wl], y=[E],
            mode="markers+text",
            marker=dict(
                size=14,
                color=clr,
                line=dict(width=2, color="white"),
                symbol="circle"
            ),
            text=[f"{wl:.0f} nm<br>{E:.2f} eV"],
            textposition="top center",
            textfont=dict(color=clr, size=11, family="Poppins"),
            showlegend=False
        ))

# Styling premium dan terang
fig2.update_layout(
    xaxis_title="Wavelength (nm)",
    yaxis_title="Photon Energy (eV)",
    plot_bgcolor="rgba(255,255,255,0.9)",
    paper_bgcolor="white",
    font=dict(color="#222", family="Poppins"),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=False
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=False
    ),
    height=420,
    margin=dict(l=50, r=30, t=50, b=50),
    showlegend=False,
    annotations=[
        dict(
            xref="paper", yref="paper",
            x=0.5, y=1.12,
            text="Photon Energy vs Wavelength",
            showarrow=False,
            font=dict(size=18, color="#111", family="Poppins", weight="bold")
        )
    ]
)

# Efek highlight "glow" lembut di sekitar kurva
fig2.add_trace(go.Scatter(
    x=wavelengths,
    y=np.array(energies) + 0.05,
    mode="lines",
    line=dict(width=12, color="rgba(255,120,180,0.1)"),
    showlegend=False
))

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Interpretation
# ----------------------------
st.markdown("---")
st.markdown("""
#### Simulation Notes
- LED color corresponds to its emission wavelength and photon energy.
- X (mm) and Y (mm) represent diffraction geometry (distance & offset).
- Wavelengths calculated from the grating equation: λ = d·sinθ.
- Gaussian peaks simulate spectral distribution.
- The photon energy–wavelength plot shows an inverse relationship: shorter λ → higher energy.
""")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
import matplotlib.animation as animation
import time

st.set_page_config(page_title="Gravitational Lensing Simulator", layout="wide")

st.markdown(
    """
    <style>
    body, .stApp {
        background: radial-gradient(ellipse at center, #0b1e36 0%, #01060f 100%);
        color: #eaeaea;
    }
    .stButton>button {background-color: #1a2340; color:#eaeaea;}
    .stSlider>div {background-color: #142139;}
    .stCheckbox>div {background-color: #142139;}
    </style>
    """, unsafe_allow_html=True
)

st.title("🌌 Gravitational Lensing Simulator")

st.markdown("""
Simulate gravitational lensing with an observer (Earth), a light source (star 1), a massive celestial body (star 2), and optionally a planet orbiting star 2.  
Adjust the parameters, then press **Start** to see the lensing animation and brightness variation!
""")

with st.sidebar:
    st.header("Parameters")
    mass_star1 = st.slider("Mass of Star 1 (Light Source) [Solar Masses]", 0.1, 20.0, 1.0, 0.1)
    radius_star1 = st.slider("Radius of Star 1 [Solar Radii]", 0.1, 10.0, 1.0, 0.1)
    mass_star2 = st.slider("Mass of Star 2 (Lens) [Solar Masses]", 0.1, 40.0, 10.0, 0.1)
    radius_star2 = st.slider("Radius of Star 2 [Solar Radii]", 0.5, 20.0, 2.0, 0.1)
    distance_star2 = st.slider("Distance of Star 2 from Earth [ly]", 1.0, 20.0, 10.0, 0.1)
    distance_star1 = st.slider("Distance of Star 1 from Earth [ly]", distance_star2+1, 100.0, 50.0, 0.1)
    planet_orbit = st.checkbox("Star 2 has a planet?")
    if planet_orbit:
        planet_mass = st.slider("Planet Mass [Earth Masses]", 0.1, 10.0, 1.0, 0.1)
        planet_radius = st.slider("Planet Radius [Earth Radii]", 0.1, 2.0, 1.0, 0.1)
        planet_orbit_radius = st.slider("Planet Orbit Radius [AU]", 0.2, 5.0, 1.0, 0.1)

start_sim = st.button("Start Simulation")

# Physical constants
G = 6.67430e-11 # m^3/kg/s^2
c = 3e8 # m/s
Msun = 1.9885e30
Rsun = 6.96e8
Mearth = 5.972e24
Rearth = 6.371e6
ly = 9.461e15
AU = 1.496e11

# Layout
col1, col2 = st.columns([2,1])

# Simulation setup
def get_positions(t, total_steps, with_planet):
    # Earth at (0,0)
    # Star 2 moves horizontally between earth and star 1
    x_star2 = ((t/total_steps) - 0.5) * (distance_star2*ly*0.7) # move in range [-L/2, +L/2]
    y_star2 = distance_star2 * ly
    x_star1 = 0
    y_star1 = distance_star1 * ly
    positions = {"earth": (0,0), "star2": (x_star2, y_star2), "star1": (x_star1, y_star1)}
    if with_planet:
        # Planet orbits star 2
        theta = 2*np.pi * (t/total_steps)
        x_planet = x_star2 + planet_orbit_radius*AU * np.cos(theta)
        y_planet = y_star2 + planet_orbit_radius*AU * np.sin(theta)
        positions["planet"] = (x_planet, y_planet)
    return positions

def lensing_angle(M, b):
    # Einstein angle approximation (arcsec)
    theta_E = np.sqrt(4*G*M/(c**2*b))
    return theta_E

def brightness_factor(M, b):
    # Magnification factor (simplified for point lens)
    u = b / (np.sqrt(4*G*M/(c**2)))
    A = (u**2 + 2)/(u*np.sqrt(u**2 + 4))
    return A

def draw_sim(t, total_steps, with_planet):
    positions = get_positions(t, total_steps, with_planet)
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_facecolor("#0b1e36")
    # Draw earth
    ax.add_patch(Circle(positions["earth"], 0.2*ly, color="#3fa9f5", label="Earth", zorder=6))
    ax.text(positions["earth"][0], positions["earth"][1]-0.3*ly, "🌍", ha='center', fontsize=16, color="#eaeaea", zorder=7)
    # Draw star 1
    ax.add_patch(Circle(positions["star1"], radius_star1*Rsun, color="#ffe066", label="Star 1", zorder=5))
    ax.text(positions["star1"][0], positions["star1"][1]+radius_star1*Rsun+0.2*ly, "⭐", ha='center', fontsize=18, color="#ffe066", zorder=7)
    # Draw star 2
    ax.add_patch(Circle(positions["star2"], radius_star2*Rsun, color="#ff5e5b", label="Star 2", alpha=0.8, zorder=5))
    ax.text(positions["star2"][0], positions["star2"][1]+radius_star2*Rsun+0.2*ly, "⚫", ha='center', fontsize=18, color="#ff5e5b", zorder=7)
    # Draw planet
    if with_planet:
        ax.add_patch(Circle(positions["planet"], planet_radius*Rearth, color="#7cdb86", label="Planet", alpha=0.9, zorder=8))
        ax.text(positions["planet"][0], positions["planet"][1]+planet_radius*Rearth+0.1*ly, "🪐", ha='center', fontsize=14, color="#7cdb86", zorder=9)
    # Draw light path
    # Calculate lensing: path bends near star2
    x0, y0 = positions["star1"]
    x1, y1 = positions["earth"]
    x2, y2 = positions["star2"]
    # Impact parameter (closest approach to lens)
    # For simplicity, it's the distance from star2 to the straight line connecting star1 and earth
    def dist_point_line(px, py, x0, y0, x1, y1):
        a = np.array([x1-x0, y1-y0])
        b = np.array([px-x0, py-y0])
        proj = np.dot(a, b)/np.dot(a, a)
        closest = np.array([x0, y0]) + proj*a
        return np.linalg.norm(np.array([px, py])-closest)
    b = dist_point_line(x2, y2, x0, y0, x1, y1)
    # Lensing bend
    ang = lensing_angle(mass_star2*Msun, max(b, 1e9))
    # Show as a curved arrow
    num_points = 50
    light_x = np.linspace(x0, x1, num_points)
    light_y = np.linspace(y0, y1, num_points)
    # Apply bend near the lens
    for i in range(num_points):
        # if close to star2 horizontally, apply lensing
        d = np.hypot(light_x[i]-x2, light_y[i]-y2)
        if d < 2*radius_star2*Rsun:
            light_x[i] += ang * 1e11
    ax.plot(light_x, light_y, color="#ffe066", lw=2.5, zorder=10, label="Light Path")
    ax.arrow(light_x[-2], light_y[-2], light_x[-1]-light_x[-2], light_y[-1]-light_y[-2],
             head_width=0.2*ly, head_length=0.2*ly, fc='#ffe066', ec='#ffe066', zorder=12)
    # Universe theme
    for _ in range(44):
        ax.scatter(np.random.uniform(-distance_star1*ly/2, distance_star1*ly/2), 
                   np.random.uniform(0, distance_star1*ly*1.05), 
                   color="#eaeaea", alpha=0.3, s=np.random.uniform(20, 80), zorder=1)
    ax.set_xlim(-distance_star1*ly/2, distance_star1*ly/2)
    ax.set_ylim(-0.5*ly, distance_star1*ly*1.1)
    ax.axis('off')
    return fig, b

def simulate(total_steps=100):
    brightness = []
    times = []
    with col1:
        sim_placeholder = st.empty()
    with col2:
        graph_placeholder = st.empty()
    for t in range(total_steps):
        fig, b = draw_sim(t, total_steps, planet_orbit)
        # Calculate brightness factor
        mag = brightness_factor(mass_star2*Msun, max(b, 1e9))
        # If planet is present, modulate the brightness occasionally
        if planet_orbit:
            positions = get_positions(t, total_steps, planet_orbit)
            x_p, y_p = positions["planet"]
            # If planet passes near light path, add a dip
            dist_to_path = np.abs(x_p) + np.abs(y_p - (distance_star1*ly))
            if dist_to_path < 0.3*ly:
                mag *= 0.7
        brightness.append(mag)
        times.append(t)
        sim_placeholder.pyplot(fig)
        graph_placeholder.pyplot(plot_brightness(times, brightness))
        time.sleep(0.07)
    # Final plot
    sim_placeholder.pyplot(fig)
    graph_placeholder.pyplot(plot_brightness(times, brightness))

def plot_brightness(times, brightness):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(times, brightness, color="#ffe066", lw=2)
    ax.set_facecolor("#142139")
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative Brightness")
    ax.set_title("Observed Brightness of Star 1")
    ax.grid(True, alpha=0.2)
    return fig

if start_sim:
    simulate(100)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Constants (can be adjusted for visual effect) ---
# Gravitational constant (m^3 kg^-1 s^-2)
G = 6.67430e-11
# Speed of light (m/s)
c = 2.998e8

# Conversion factors for astronomical units to SI units
M_SUN = 1.989e30  # Mass of the Sun in kg
R_SUN = 6.957e8   # Radius of the Sun in meters
M_JUPITER = 1.898e27 # Mass of Jupiter in kg
R_JUPITER = 6.9911e7 # Radius of Jupiter in meters

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Gravitational Lensing Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for universe theme and element styling
st.markdown(
    """
    <style>
    /* Main app background with a plain black color */
    .stApp {
        background-color: black; /* Changed to plain black */
        color: white; /* Default text color */
    }
    /* Sidebar background with some transparency */
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.7);
    }
    /* Main content area background with some transparency and rounded corners */
    .css-1lcbmhc {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }
    /* Ensure all text elements are white */
    h1, h2, h3, h4, h5, h6, .stMarkdown, label, .stSlider > div > div > div {
        color: white !important;
    }
    /* Style for the slider fill */
    .stSlider > div > div > div > div {
        background-color: #4CAF50; /* Green color for the slider track */
    }
    /* Style for the start simulation button */
    .stButton > button {
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        border-radius: 10px; /* Rounded corners */
        padding: 10px 20px; /* Padding inside the button */
        font-size: 1.2em; /* Larger font size */
        border: none; /* No border */
        cursor: pointer; /* Pointer on hover */
        transition: background-color 0.3s ease; /* Smooth transition on hover */
    }
    .stButton > button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    /* Style for the checkbox label */
    .stCheckbox > label {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌌 Gravitational Lensing Simulator")
st.markdown("Explore how massive objects bend light and magnify distant stars.")

st.sidebar.header("Simulation Parameters")

# --- Sliders and Checkbox for user input ---

# Slider for the mass of the lensing star (Star 2)
mass_star2_solar = st.sidebar.slider(
    "Mass of Lensing Star (Star 2) (Solar Masses)",
    min_value=0.1, max_value=10.0, value=1.0, step=0.1,
    help="Adjust the mass of the star causing the lensing effect."
)
# Slider for the radius of the lensing star (Star 2)
radius_star2_solar = st.sidebar.slider(
    "Radius of Lensing Star (Star 2) (Solar Radii)",
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="Adjust the visual size of the lensing star. (Does not affect lensing physics in this simplified model)."
)

# Slider for the distance of Star 2 from Earth (as a fraction of Earth-Star1 distance)
distance_lens_fraction = st.sidebar.slider(
    "Distance of Star 2 from Earth (Fraction of Earth-Star1 distance)",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Position Star 2 between Earth and Star 1. 0.5 means halfway."
)

# Checkbox to decide if Star 2 has an orbiting planet
has_planet = st.sidebar.checkbox("Star 2 has an orbiting Planet?", value=False)

mass_planet_jupiter = 0.0
radius_planet_jupiter = 0.0
if has_planet:
    # Sliders for planet mass and radius, visible only if has_planet is checked
    mass_planet_jupiter = st.sidebar.slider(
        "Mass of Planet (Jupiter Masses)",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Adjust the mass of the orbiting planet."
    )
    radius_planet_jupiter = st.sidebar.slider(
        "Radius of Planet (Jupiter Radii)",
        min_value=0.1, max_value=2.0, value=1.0, step=0.1,
        help="Adjust the visual size of the orbiting planet."
    )
    st.sidebar.info("Note: Planet's orbit is fixed for simplicity. Its lensing effect is a simplified perturbation.")

st.sidebar.write("---")
start_simulation = st.sidebar.button("🚀 Start Simulation")

# --- Simulation Setup ---

# Define fixed conceptual distances for the simulation visualization
# These are not real-world units but represent relative distances for plotting
D_S = 1000  # Conceptual distance from Earth to Star 1 (Source)
D_L = D_S * distance_lens_fraction  # Conceptual distance from Earth to Star 2 (Lens)
D_LS = D_S - D_L # Conceptual distance from Star 2 to Star 1

# Convert slider values to SI units for physical calculations
M_lens_SI = mass_star2_solar * M_SUN
R_lens_SI = radius_star2_solar * M_SUN # Using M_SUN for scaling radius for visual effect
M_planet_SI = mass_planet_jupiter * M_JUPITER
R_planet_SI = radius_planet_jupiter * M_JUPITER # Using M_JUPITER for scaling radius for visual effect

# --- Gravitational Lensing Physics Functions ---

def einstein_radius(M, D_L_conceptual, D_S_conceptual, D_LS_conceptual):
    """
    Calculates the Einstein radius in radians.
    Uses conceptual distances for the ratio, but SI mass.
    """
    # To use the formula correctly, D_L, D_S, D_LS must be in consistent units.
    # We'll assume the conceptual units are consistent, and the ratio is what matters.
    # For actual physical calculation, these distances should be in meters.
    # For a visual simulation, we can use the ratio of conceptual distances.
    # Let's assume conceptual units are 'light-years' for example, and the formula
    # still holds with M in kg, G, c in SI.
    
    # Avoid division by zero if distances are zero (though sliders prevent this)
    if D_L_conceptual == 0 or D_S_conceptual == 0 or D_LS_conceptual == 0:
        return 0.0
    
    # The formula requires consistent units for distances. Since our D_S, D_L, D_LS
    # are conceptual for plotting, we need to ensure the ratio is correct.
    # For a realistic calculation, these would be in meters.
    # Let's assume D_S_conceptual is in some unit, and the ratio (D_LS / (D_L * D_S))
    # is dimensionless, so the result is correct in radians.
    
    # A more robust way would be to define a conceptual unit to meter conversion.
    # For simplicity of demonstration, we proceed with the conceptual units directly
    # in the ratio part of the formula.
    
    # This factor scales the Einstein radius for visualization purposes.
    # The actual Einstein radius is tiny. We need to make it visible.
    scaling_factor_for_vis = 1e10 # Arbitrary scaling to make the effect visible in plot
    
    return np.sqrt((4 * G * M / c**2) * (D_LS_conceptual / (D_L_conceptual * D_S_conceptual))) * scaling_factor_for_vis

def magnification(u):
    """
    Calculates the magnification for a point source/lens.
    u is the dimensionless impact parameter (beta / Einstein_radius).
    """
    if u == 0: # Handle exact alignment, theoretically infinite magnification
        return 100.0 # Cap for visualization purposes
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))

# --- Simulation Visualization Area ---
# Use columns to arrange the simulation and graph side-by-side
col1, col2 = st.columns([2, 1]) # Column 1 is wider for the simulation

with col1:
    st.header("Visual Simulation")
    # Placeholder for the animation plot, allows real-time updates
    simulation_placeholder = st.empty()

with col2:
    st.header("Brightness Curve (Magnification)")
    # Placeholder for the brightness graph, allows real-time updates
    brightness_placeholder = st.empty()

# --- Run Simulation on Button Click ---
if start_simulation:
    st.toast("Simulation started! ✨", icon="🚀")

    # Simulation parameters for the animation loop
    total_steps = 200 # Number of frames in the animation
    time_duration = 5 # Conceptual time units for the animation - Made even faster
    dt = time_duration / total_steps # Time step per frame

    # How far the lens moves perpendicular to the line of sight
    # This determines the "impact parameter" for the lensing event
    cross_distance_max = 0.05 * D_S

    # Data lists to store values for the brightness curve
    times = []
    magnifications_data = []

    # Setup Matplotlib figures and axes for simulation and brightness plots
    fig_sim, ax_sim = plt.subplots(figsize=(10, 5))
    fig_bright, ax_bright = plt.subplots(figsize=(8, 4))

    # Configure simulation plot (ax_sim) for a space theme
    ax_sim.set_facecolor('black') # Black background for space
    fig_sim.patch.set_facecolor('black') # Match figure background
    ax_sim.set_xlim(-0.1 * D_S, D_S * 1.1) # X-axis limits
    ax_sim.set_ylim(-cross_distance_max * 1.5, cross_distance_max * 1.5) # Y-axis limits
    ax_sim.set_aspect('equal', adjustable='box') # Maintain aspect ratio
    ax_sim.axis('off') # Hide axes for a cleaner look

    # Configure brightness plot (ax_bright)
    ax_bright.set_facecolor('black') # Black background
    fig_bright.patch.set_facecolor('black') # Match figure background
    ax_bright.set_xlabel("Time (conceptual units)", color='white') # X-axis label
    ax_bright.set_ylabel("Magnification", color='white') # Y-axis label
    ax_bright.tick_params(axis='x', colors='white') # White tick labels for x-axis
    ax_bright.tick_params(axis='y', colors='white') # White tick labels for y-axis
    # White spines (borders) for the plot
    ax_bright.spines['bottom'].set_color('white')
    ax_bright.spines['left'].set_color('white')
    ax_bright.spines['top'].set_color('white')
    ax_bright.spines['right'].set_color('white')
    ax_bright.set_title("Apparent Brightness of Star 1", color='white') # Plot title
    ax_bright.grid(True, linestyle='--', alpha=0.5, color='gray') # Grid for readability

    # --- Animation Loop ---
    for i in range(total_steps):
        current_time = i * dt
        # Calculate the linear movement of Star 2 across the line of sight
        y_lens = -cross_distance_max + (2 * cross_distance_max / total_steps) * i
        x_lens = D_L # Star 2's X position remains constant (at lens distance)

        # Calculate planet's position if it exists (simplified circular orbit)
        x_planet, y_planet = x_lens, y_lens # Default to same as star if no planet or no orbit
        if has_planet:
            planet_orbit_radius = 0.015 * D_S # Increased conceptual orbit radius for visualization
            # Angle of the planet in its orbit (5 full orbits during the simulation)
            planet_angle = (current_time / time_duration) * 2 * np.pi * 5
            x_planet = x_lens + planet_orbit_radius * np.cos(planet_angle)
            y_planet = y_lens + planet_orbit_radius * np.sin(planet_angle)

        # Calculate angular separation (beta) for the main lensing star (Star 2)
        # Beta is the angular separation between the source and the lens as seen from the observer
        # without lensing. It's approximated by the perpendicular distance divided by the lens distance.
        beta_star2 = np.abs(y_lens) / D_L # Angular separation in radians

        # Calculate Einstein Radius for Star 2 using its mass in SI units
        theta_E_star2 = einstein_radius(M_lens_SI, D_L, D_S, D_LS)

        # Calculate dimensionless impact parameter (u) for Star 2
        u_star2 = beta_star2 / theta_E_star2 if theta_E_star2 > 0 else float('inf')

        # Calculate magnification for Star 2
        mag_star2 = magnification(u_star2)

        # If a planet exists, calculate its additional lensing effect
        mag_planet_effect = 1.0 # Default to no additional effect
        if has_planet:
            # Simplified calculation for planet's effect:
            # Treat planet as a separate lens. This is a simplification for visualization
            # and does not fully represent the complex binary lens equation.
            beta_planet = np.abs(y_planet) / D_L # Angular separation of planet from line of sight
            theta_E_planet = einstein_radius(M_planet_SI, D_L, D_S, D_LS)
            u_planet = beta_planet / theta_E_planet if theta_E_planet > 0 else float('inf')
            
            # If the planet is very close to the line of sight (u_planet < 2),
            # its individual magnification contributes.
            if u_planet < 2:
                mag_planet_effect = magnification(u_planet)
            else:
                mag_planet_effect = 1.0 # No significant effect if far away

        # Combine magnifications (simplified: multiply effects)
        total_magnification = mag_star2 * mag_planet_effect
        
        # Add current time and magnification to data lists
        times.append(current_time)
        magnifications_data.append(total_magnification)

        # --- Update Visual Simulation Plot ---
        ax_sim.clear() # Clear the previous frame
        ax_sim.set_facecolor('black') # Reset background color
        ax_sim.set_xlim(-0.1 * D_S, D_S * 1.1)
        ax_sim.set_ylim(-cross_distance_max * 1.5, cross_distance_max * 1.5)
        ax_sim.set_aspect('equal', adjustable='box')
        ax_sim.axis('off')

        # Plot Earth (Observer)
        ax_sim.plot(0, 0, 'o', color='blue', markersize=10, label="Earth (Observer)")
        ax_sim.text(0, -0.08 * D_S, "Earth", color='white', ha='center', fontsize=10)

        # Plot Star 1 (Light Source)
        ax_sim.plot(D_S, 0, '*', color='yellow', markersize=15, label="Star 1 (Source)")
        ax_sim.text(D_S, -0.08 * D_S, "Star 1", color='white', ha='center', fontsize=10)

        # Plot Star 2 (Lensing Body)
        # Increased scaling factor for visual size of Star 2
        star2_circle = plt.Circle((x_lens, y_lens), radius=R_lens_SI * 2e-9, color='red', alpha=0.8)
        ax_sim.add_patch(star2_circle)
        ax_sim.text(x_lens, y_lens + R_lens_SI * 2e-9 + 0.005 * D_S, "Star 2", color='white', ha='center', fontsize=10)

        # Plot Planet (if exists)
        if has_planet:
            # Increased scaling factor for visual size of the Planet
            planet_circle = plt.Circle((x_planet, y_planet), radius=R_planet_SI * 2e-9, color='cyan', alpha=0.8)
            ax_sim.add_patch(planet_circle)
            ax_sim.text(x_planet, y_planet + R_planet_SI * 2e-9 + 0.005 * D_S, "Planet", color='white', ha='center', fontsize=10)

        # --- Light Path Visualization ---
        # Show a bent light path if the lens is close enough to the line of sight
        if u_star2 < 5: # Arbitrary threshold for "significant lensing"
            # Draw two segments: Star 1 to Star 2, and Star 2 to Earth
            ax_sim.plot([D_S, x_lens], [0, y_lens], 'w--', alpha=0.5) # Dashed line
            ax_sim.plot([x_lens, 0], [y_lens, 0], 'w--', alpha=0.5) # Dashed line
            # Add arrows to indicate light direction
            ax_sim.arrow(D_S, 0, x_lens - D_S, y_lens, head_width=0.005*D_S, head_length=0.01*D_S, fc='white', ec='white', length_includes_head=True)
            ax_sim.arrow(x_lens, y_lens, 0 - x_lens, 0 - y_lens, head_width=0.005*D_S, head_length=0.01*D_S, fc='white', ec='white', length_includes_head=True)
        else:
            # Draw a straight line if no significant lensing effect
            ax_sim.plot([D_S, 0], [0, 0], 'w-', alpha=0.5) # Solid line
            ax_sim.arrow(D_S, 0, -D_S, 0, head_width=0.005*D_S, head_length=0.01*D_S, fc='white', ec='white', length_includes_head=True)

        # Update the simulation plot in the placeholder
        simulation_placeholder.pyplot(fig_sim)

        # --- Update Brightness Curve Plot ---
        ax_bright.clear() # Clear the previous brightness plot
        ax_bright.set_facecolor('black') # Reset background color
        ax_bright.plot(times, magnifications_data, color='lime', linewidth=2) # Plot the curve
        ax_bright.set_xlabel("Time (conceptual units)", color='white')
        ax_bright.set_ylabel("Magnification", color='white')
        ax_bright.tick_params(axis='x', colors='white')
        ax_bright.tick_params(axis='y', colors='white')
        ax_bright.spines['bottom'].set_color('white')
        ax_bright.spines['left'].set_color('white')
        ax_bright.spines['top'].set_color('white')
        ax_bright.spines['right'].set_color('white')
        ax_bright.set_title("Apparent Brightness of Star 1", color='white')
        ax_bright.grid(True, linestyle='--', alpha=0.5, color='gray')
        # Dynamically adjust y-axis limits for the brightness plot
        ax_bright.set_ylim(0.8, max(magnifications_data) * 1.2 if magnifications_data else 2.0)

        # Update the brightness plot in the placeholder
        brightness_placeholder.pyplot(fig_bright)

        time.sleep(0.02) # Control animation speed (0.02 seconds per frame)

    # Close Matplotlib figures to free up memory after the simulation is complete
    plt.close(fig_sim)
    plt.close(fig_bright)

    st.toast("Simulation finished! 🎉", icon="✅")

# --- Footer / Instructions ---
st.markdown(
    """
    <div style="text-align: center; margin-top: 30px; color: gray;">
        <p>This simulator provides a simplified visualization of gravitational lensing.</p>
        <p>Adjust the parameters in the sidebar to observe how the mass and position of a foreground star (Star 2) and its planet affect the apparent brightness of a background star (Star 1).</p>
    </div>
    """,
    unsafe_allow_html=True
)

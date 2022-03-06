"""
Several Low Lunar orbiters will be propagated over the simulation time as described in Simulation_Setup.py.
A low Lunar orbiter is experiencing a symmetric gravity field, so this body problem is reduced to a two-body problem,
including solar radiation pressure and. There is no drag due to no atmosphere to cause it.

Goal is to propagate a LLO circular orbiter, the Lunar PathFinder and LRO.

Hopefully ephemeris' can be used
"""
import Simulation_setup
import numpy as np
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
import matplotlib.pyplot as plt
spice_interface.load_standard_kernels()
print("Running [LowLunarOrbiter.py]")
### Simulation parameters ###
simulation_start_epoch = Simulation_setup.simulation_start_epoch
simulation_end_epoch = Simulation_setup.simulation_end_epoch
### Environment ###
initial_time = Simulation_setup.simulation_start_epoch
final_time = Simulation_setup.simulation_end_epoch
fixed_time_step = Simulation_setup.fixed_time_step
bodies_to_create = ["Moon", "Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
global_frame_origin = "Moon"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings_time_limited(
    bodies_to_create, initial_time, final_time, global_frame_origin, global_frame_orientation, fixed_time_step)

body_system = environment_setup.create_system_of_bodies(body_settings)

### Adding satellites ###
body_system.create_empty_body("CircSAT1")
body_system.create_empty_body("CircSAT2")
body_system.create_empty_body("Pathfinder")
body_system.get("CircSAT1").mass = 280
body_system.get("CircSAT2").mass = 280
body_system.get("Pathfinder").mass = 280

bodies_to_propagate = ["CircSAT1", "CircSAT2", "Pathfinder"]
central_bodies = ["Moon", "Moon", "Moon"]

moon_gravitational_parameter = body_system.get("Moon").gravitational_parameter
##############################Initial States of the Low Lunar Orbiters #############################################
### Circular Orbiter ###
initial_state_circ1 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=moon_gravitational_parameter,
    semi_major_axis=60E3,
    eccentricity=0.0,
    inclination=np.deg2rad(0.0),
    argument_of_periapsis=np.deg2rad(0.0),
    longitude_of_ascending_node=np.deg2rad(0.0),
    true_anomaly=np.deg2rad(0.0)
)
initial_state_circ2 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=moon_gravitational_parameter,
    semi_major_axis=60E3,
    eccentricity=0.0,
    inclination=np.deg2rad(90.0),
    argument_of_periapsis=np.deg2rad(0.0),
    longitude_of_ascending_node=np.deg2rad(0.0),
    true_anomaly=np.deg2rad(0.0)
)

initial_state_pathfinder = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=moon_gravitational_parameter,
    semi_major_axis=5737.4E3,
    eccentricity=0.61,
    inclination=np.deg2rad(57.82),
    argument_of_periapsis=np.deg2rad(90),
    longitude_of_ascending_node=np.rad2deg(61.552),
    true_anomaly=np.deg2rad(30)
)
initial_states = np.vstack((np.vstack((np.transpose([initial_state_circ1]), np.transpose([initial_state_circ2]))),
                            np.transpose([initial_state_pathfinder])))

### Acceleration Setup ###
# SRP
reference_area_radiation = 1.0
radiation_pressure_coefficient = 1.0
occulting_bodies = ["Moon"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)
environment_setup.add_radiation_pressure_interface(body_system,"CircSAT1", radiation_pressure_settings)
environment_setup.add_radiation_pressure_interface(body_system,"CircSAT2", radiation_pressure_settings)
environment_setup.add_radiation_pressure_interface(body_system,"Pathfinder", radiation_pressure_settings)

acceleration_settings_CircSAT1 = dict(
    Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(40,40)],
    Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(40,40)],
    Sun=[propagation_setup.acceleration.point_mass_gravity(),
         propagation_setup.acceleration.cannonball_radiation_pressure()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_CircSAT2 = dict(
    Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(40,40)],
    Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(40,40)],
    Sun=[propagation_setup.acceleration.point_mass_gravity(),
         propagation_setup.acceleration.cannonball_radiation_pressure()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_Pathfinder = dict(
    Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(40,40)],
    Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(40,40)],
    Sun=[propagation_setup.acceleration.point_mass_gravity(),
         propagation_setup.acceleration.cannonball_radiation_pressure()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings = {
    "CircSAT1": acceleration_settings_CircSAT1,
    "CircSAT2": acceleration_settings_CircSAT2,
    "Pathfinder": acceleration_settings_Pathfinder
}

acceleration_models = propagation_setup.create_acceleration_models(
    body_system, acceleration_settings, bodies_to_propagate, central_bodies
)

### Savings ###
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("CircSAT1"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.cannonball_radiation_pressure_type, "CircSAT1", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "CircSAT1", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "CircSAT1", "Moon"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT1", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT1", "Mercury"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT1", "Venus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT1", "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT1", "Jupiter"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT1", "Saturn"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT1", "Uranus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT1", "Neptune"
    ),
    propagation_setup.dependent_variable.total_acceleration("CircSAT2"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.cannonball_radiation_pressure_type, "CircSAT2", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "CircSAT2", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "CircSAT2", "Moon"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT2", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT2", "Mercury"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT2", "Venus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT2", "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT2", "Jupiter"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT2", "Saturn"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT2", "Uranus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "CircSAT2", "Neptune"
    ),
    propagation_setup.dependent_variable.total_acceleration("Pathfinder"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.cannonball_radiation_pressure_type, "Pathfinder", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "Pathfinder", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "Pathfinder", "Moon"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Pathfinder", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Pathfinder", "Mercury"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Pathfinder", "Venus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Pathfinder", "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Pathfinder", "Jupiter"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Pathfinder", "Saturn"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Pathfinder", "Uranus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Pathfinder", "Neptune"
    )
]

### Propagation settings ###
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
propagation_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
    termination_condition,
    output_variables=dependent_variables_to_save
)
### Integrator ###
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_time_step
)

### Dynamic Simulator ###
dynamic_simulator = numerical_simulation.SingleArcSimulator(
    body_system, integrator_settings, propagation_settings
)

time = Simulation_setup.simulation_span

output_dict = dynamic_simulator.dependent_variable_history
states_dict = dynamic_simulator.state_history
output = np.vstack(list(output_dict.values()))
states = np.vstack(list(states_dict.values()))



print("[LowLunarOrbiter.py] ran successfully")
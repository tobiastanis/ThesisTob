"""
Propagation of LUMIO using the initial state at MJD 60390.00000.
"""
import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import propagation
import Datapack_to_initial_CRTBP
import LUMIO_States_reader as lsr
import Input as I
import matplotlib.pyplot as plt
spice_interface.load_standard_kernels()

##### Environment set-up #####
bodies_to_create = ["Sun", "Earth", "Moon", "Venus", "Mars", "Jupiter"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

bodies = environment_setup.create_system_of_bodies(body_settings)

##### System of bodies #####
bodies.create_empty_body("LUMIO")
bodies.get("LUMIO").mass = 22.3
bodies.get("Moon").mass = spice_interface.get_body_gravitational_parameter("Moon")/constants.GRAVITATIONAL_CONSTANT

bodies_to_propagate = ["Moon", "LUMIO"]
central_bodies = ["Earth", "Earth"]

##### Acceleration Settings #####
# Radiation pressure settings
# Radiation pressure
reference_area_radiation = 0.85
radiation_pressure_coefficient = 1.2
occulting_bodies = ["Moon"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)

environment_setup.add_radiation_pressure_interface(bodies,"LUMIO", radiation_pressure_settings)

acceleration_settings_Moon = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()]
    )
acceleration_settings_LUMIO = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],        # spherical harmonic gravity as well as for moon
    Sun=[propagation_setup.acceleration.point_mass_gravity(),
         propagation_setup.acceleration.cannonball_radiation_pressure()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()]          # Spherical harmonic gravity
    )

acceleration_settings = {
    "Moon": acceleration_settings_Moon,
    "LUMIO": acceleration_settings_LUMIO
}
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

###### Propagation Settings ######

#initial_state_Moon = np.transpose([Datapack_to_initial_CRTBP.X_Moon])
initial_state_Moon = spice_interface.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", float(lsr.LUMIOdata_timespan[0, 1]))
initial_state_Moon = np.transpose([initial_state_Moon])
initial_state_LUMIO = np.transpose([lsr.state_LUMIO[0,:]])*10**3
#initial_state_LUMIO = np.transpose([Datapack_to_initial_CRTBP.LUMIO_initial_value_data])
initial_states = np.concatenate((initial_state_Moon, initial_state_LUMIO), axis=0)

###### Savings ######
Moon_dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("Moon"),
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("Moon", "Earth"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Moon", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Moon", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Moon", "Venus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Moon", "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Moon", "Jupiter"
    )
]
LUMIO_dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("LUMIO"),
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("LUMIO", "Earth"),
    propagation_setup.dependent_variable.radiation_pressure("LUMIO", "Sun"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Venus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Jupiter"
    )
]

Saved_variables = Moon_dependent_variables_to_save + LUMIO_dependent_variables_to_save

###### Propagating ######
termination_condition = propagation_setup.propagator.time_termination(I.simulation_end_epoch)
propagation_setup = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
    termination_condition,
    output_variables=Saved_variables
)

###### Integrating ######
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_4(
    I.simulation_start_epoch, I.fixed_step_size
)

# Simulating
dynamic_simulator = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagation_setup
)

states = dynamic_simulator.state_history
######### Rewritting Data #############
width = states[0.0].size          # This is the length of np array
height = len(states)                # This is the amount of key/value pairs fo dict

states_array = np.empty(shape=(height,width))        # Ini 2d matrix
# Loop over entries in dictionair getting both key and value
for x, (key, np_array) in enumerate(states.items()):
    # Looping over elements in the np array
    for y, np_value in enumerate(np_array):
        #print("i {}: key: {}, np.array {}".format(x, key, np_value))
        states_array[x, y] = np_value

states_Moon = states_array[:, 0:6]
states_LUMIO = states_array[:, 6:12]

if all(states_Moon[0, :]) == all(initial_state_Moon) and all(states_LUMIO[0, :]) == all(initial_state_LUMIO):
    print('jeej')
else:
    print('error')



plt.figure()
plt.plot(states_Moon[:,0], states_Moon[:,1])
plt.plot(states_LUMIO[:,0], states_LUMIO[:, 1])

plt.show()

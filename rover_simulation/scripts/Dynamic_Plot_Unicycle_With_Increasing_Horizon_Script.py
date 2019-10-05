import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import ltv_contouring_mpc as ltvcmpc
import math


# Path following using ltv contouring mpc

# Constant terms
N_STATES = 4
N_INPUTS = 3

N = 50 # Set horizon

# Spline and path files

# Use these filenames for custom path
# track_points_filename = 'data/track_shortest_dist_v8.mat'
# track_spline_filename = 'data/path_track_spline_v2.mat' # Filename for spline for
# # path/track

# Use these filenames for center of track path
track_points_filename = 'data/center_path_track_points.mat'
track_spline_filename = 'data/center_path_track_spline.mat' # Filename for spline for
# path/track

# Load track and check if it has been loaded properly
track_points = ltvcmpc.load_track(track_points_filename)

# Generate spline for path, center and outer of track
cycles = 2
track_spline = ltvcmpc.generate_track_spline(track_points,cycles)

# Simulation settings
Ts = 0.05 # Sampling period
M = 1 # Number of iterations to repeat LTV for single time

# Weight matrices
weights = {
        'Q': 1000*np.diag([1.0, 1.0]),
        'q': 10.0,
        'R': 10*np.diag([0.1, 1.0, 1.0])
        }

# Load splines for path and track
#track_spline = scipy.io.loadmat(track_spline_filename)
beta_limit = track_spline['breaks'][0,-1]

# Constraintst
constraints = {
        'omega_min': -5,
        'v_min': 0,
        'gamma_min': 0,
        'theta_min': -np.inf,
        'beta_min': 0,
        'omega_max': 5,
        'v_max': 2,
        'gamma_max': 2,
        'theta_max': np.inf,
        'beta_max': beta_limit-5
        }


# Initial states/inputs
init_omega = 0.0
init_gamma = 0.0
init_v = 0.0
init_beta = 0.0

# Set simulation time
start_time = 0.0
end_time = 10.0
times = np.arange(start_time, end_time, Ts)

# Initialise inputs
u_curr = np.vstack(np.array([init_omega, init_v, init_gamma])) # Guess for 
# initial u
u_guess = np.tile(u_curr, [1, N]) # Solved inputs over horizon

# Initialise states
xi_curr = np.vstack(np.array([track_points['path'][0,0],
                              track_points['path'][1,0],
                              math.atan2(track_points['path'][1,1]-track_points['path'][1,0],
                                         track_points['path'][0,1]-track_points['path'][0,0]), 
                              init_beta])) # Initial state

# Initialise history of inputs and states
history_u = np.zeros((N_INPUTS, times.size-1))
history_xi = np.hstack((xi_curr, np.zeros((N_STATES, times.size-1))))

# Setup plotting

#fig = plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
#ax = fig.add_subplot(111)
#plt.ion()
#fig.show()
#fig.canvas.draw()

# Determine initial guesses for inputs
for n in range(2, N+1):
    sol = ltvcmpc.unicycle_raw_solver(weights, constraints, xi_curr, u_guess[:,0:n], n,
                                      Ts, track_spline)

    # Update guesses for u
    u_guess[:,0:n] = sol['u']

    # Plot prediction along horizon
    #ax.clear()
    #ax.plot(track_points['outer'][0,:], track_points['outer'][1,:], 'k')
    #ax.plot(track_points['inner'][0,:], track_points['inner'][1,:], 'k')
    #ax.plot(sol['xi'][0,:], sol['xi'][1,:])
    #fig.canvas.draw()

# Run simulation in loop
for i in range(0, times.size-1):
    for j in range(0, M):
        sol = ltvcmpc.unicycle_raw_solver(weights, constraints, xi_curr, u_guess, N,
                                      Ts, track_spline)
        
    # Apply input to plant
    xi_curr = np.vstack(sol['xi'][:,1]) # take state at second time step for now
    xi_curr = np.array(ltvcmpc.unicycle_raw_taylor_order2_next_step(xi_curr, sol['u'][:,0], Ts))

    # Update history
    history_xi[:,i+1] = sol['xi'][:,1]
    history_u[:,i] = sol['u'][:,0]
    
    # Update guesses for u
    u_guess = np.hstack([sol['u'][:,1:], np.zeros((N_INPUTS,1))])
    
    #ax.clear()
    #ax.plot(track_points['outer'][0,:], track_points['outer'][1,:], 'k')
    #ax.plot(track_points['inner'][0,:], track_points['inner'][1,:], 'k')
    #ax.plot(history_xi[0,0:i+2], history_xi[1,0:i+2], lw=0, marker='o', fillstyle='none')
    #ax.plot(sol['xi'][0,1:], sol['xi'][1,1:])
    #fig.canvas.draw()
    print((float(i)/(times.size-1)))
print("finished")
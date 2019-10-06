# -*- coding: utf-8 -*-
"""
ltv_contouring_mpc
Created on Wed Aug  7 07:53:41 2019

Supporting functions for ltv contouring mpc

@author: khewk
"""
import osqp
# import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import sparse
import scipy as sp
import socket, time
import interpolate as ip
import scipy.io

def generate_track_spline(track_points, cycles):
    """ Load points for center, outer of track and path, and returns
    spline cofficients corresponding to center, outer and path """
    
    # Generate path spline
    path_spline = generate_spline_path(track_points['path'],cycles)
    
    # Generate center spline
    center_spline = generate_spline_2d(track_points['center'],path_spline['breaks'],cycles)
    
    # Generate outer spline
    outer_spline = generate_spline_2d(track_points['outer'],path_spline['breaks'],cycles)
    

    
    
    # Return splines
    track_spline = {
            'breaks': np.reshape(path_spline['breaks'], (1, len(path_spline['breaks']))),
            'break_fin': path_spline['break_fin'],
            'coefs_x_path': path_spline['coefs_x_points'],
            'coefs_y_path': path_spline['coefs_y_points'],
            'coefs_x_center': center_spline['coefs_x_points'],
            'coefs_y_center': center_spline['coefs_y_points'],
            'coefs_x_outer': outer_spline['coefs_x_points'],
            'coefs_y_outer': outer_spline['coefs_y_points']
            }
    
    return track_spline
    
def generate_spline_2d(points, breaks, cycles):
    """ Generate spline for 2d points based on existing breaks """
    # Cycle points to account for horizon
    xpoints_cycled = np.tile(points[0,:], (cycles+2))
    ypoints_cycled = np.tile(points[1,:], (cycles+2))
    
    # Generate cubic spline object
#    csx = sp.interpolate.CubicSpline(breaks, xpoints_cycled)
#    csy = sp.interpolate.CubicSpline(breaks, ypoints_cycled)
    
    # Try out new interpolation function
    xpoints = [ip.Point(breaks[i],xpoints_cycled[i]) for i in range(0,len(xpoints_cycled))]
    ypoints = [ip.Point(breaks[i],ypoints_cycled[i]) for i in range(0,len(ypoints_cycled))]
    cx_points = np.zeros((len(xpoints_cycled) - 1, 4))
    cy_points = np.zeros((len(ypoints_cycled) - 1, 4))
    cx_points[:,0],cx_points[:,1],cx_points[:,2],cx_points[:,3] = ip.Spline(xpoints)
    cy_points[:,0],cy_points[:,1],cy_points[:,2],cy_points[:,3] = ip.Spline(ypoints)
    
    # Obtain cublic spline coefficients
    points_spline = {
            'coefs_x_points': cx_points,
            'coefs_y_points': cy_points,
            }
    
    return points_spline

def generate_spline_path(path, cycles):
    """ Generate spline coefficients and breaks corresponding to path """
    
    # Cycle points to account for horizon
    xpoints_cycled = np.tile(path[0,:], (cycles+2))
    ypoints_cycled = np.tile(path[1,:], (cycles+2))
    
    # Calculate breaks
    distances = np.sqrt((xpoints_cycled[1:] - xpoints_cycled[0:-1])**2 + 
                 (ypoints_cycled[1:] - ypoints_cycled[0:-1])**2)
    breaks = np.zeros(len(xpoints_cycled))
    breaks[1:] = np.cumsum(distances)
    
    # Generate cubic spline object
#    csx = sp.interpolate.CubicSpline(breaks, xpoints_cycled)
#    csy = sp.interpolate.CubicSpline(breaks, ypoints_cycled)
   
    # Try out new interpolation function
    xpoints = [ip.Point(breaks[i],xpoints_cycled[i]) for i in range(0,len(xpoints_cycled))]
    ypoints = [ip.Point(breaks[i],ypoints_cycled[i]) for i in range(0,len(ypoints_cycled))]
    cx_points = np.zeros((len(xpoints_cycled) - 1, 4))
    cy_points = np.zeros((len(ypoints_cycled) - 1, 4))
    cx_points[:,0],cx_points[:,1],cx_points[:,2],cx_points[:,3] = ip.Spline(xpoints)
    cy_points[:,0],cy_points[:,1],cy_points[:,2],cy_points[:,3] = ip.Spline(ypoints)
    
    # Obtain cublic spline coefficients
    path_spline = {
            'breaks': breaks,
            'coefs_x_points': cx_points,
            'coefs_y_points': cy_points,
            'break_fin': breaks[cycles*len(path[0,:])]
            }
    
    return path_spline
    
def get_track():
    serverIp = '10.42.0.239'
    tcpPort = 9998
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Wait till server goes online
    while True:
      try:
          server.connect((serverIp, tcpPort))
          break
      except socket.error:
          time.sleep(2)
    
    # Get track from server
    track = {}
    track_elements = ['inner', 'outer', 'path']
    track_index = 0
    while track_index < 3:
        server.send(b'ready')
        msg = ''
        while True:
            try:
                pkt_length = int(server.recv(3))
            except ValueError:
                print("error")
                server.send(b'resend')
                msg = ''
                break
            print(pkt_length)
            if pkt_length == 0:
                break
            pkt = server.recv(pkt_length)
            pkt = pkt.decode("utf-8")
            msg = msg + pkt
        if msg == '':
            track_index = 0
            track = {}
            # Wait till server reconnects
            server.close()
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            while True:
                try:
                    server.connect((serverIp, tcpPort))
                    break
                except socket.error:
                    time.sleep(2)
            # restart the sending process (loop)
            continue
        msg = msg.split(';')
        data = []
        for dim in msg:
            data.append(list(map(float,dim.split(','))))
        time.sleep(2)
        data = np.array(data)
        track[track_elements[track_index]] = data
        track_index+=1
    track['center'] = 0.5*(track['inner']+track['outer'])
    return track
    
def load_track(filename):
    """ Load positions for inner and outer track boundaries and path, 
    calculate track center, return in dictionary """
    mat = scipy.io.loadmat(filename)
    center = 0.5*(mat['inner'] + mat['outer'])
    track = {
            'inner': 10*mat['inner'],
            'outer': 10*mat['outer'],
            'path': 10*mat['path'],
            'center': 10*center
            }
    return track

def view_track(track):
    """ View the track on a plot """
    plt.plot(track['inner'][0,:], track['inner'][1,:])
    plt.plot(track['outer'][0,:], track['outer'][1,:])
    plt.plot(track['center'][0,:], track['center'][1,:])
    plt.plot(track['path'][0,:], track['path'][1,:])

def unicycle_raw_solver(weights, constraints, xi_curr, u_guess, N, Ts, 
                    track_spline):
    """ Solve for states and inputs over horizon using MPC """
    
    # Constants
    BETA_INDEX = 3
    N_STATES = 4
    N_INPUTS = 3
    
    # Set weighting for virtual reference to 0 at end of track
    if xi_curr[BETA_INDEX,0] > track_spline['break_fin']:
        weights['q'] = 0
 
    # Calculate guesses for states over the horizon  
    xi_guess = np.hstack((xi_curr, np.zeros((N_STATES, N))))
    for n in range(0, N):
        xi_guess[:,n+1] = unicycle_raw_taylor_order2_next_step(xi_guess[:,n],
                u_guess[:,n], Ts)
    
    # Linearise the state space mdoel %%%%%%%%%%%%%
    state_space_mats = unicycle_raw_linearise_state_space_horizon(u_guess, xi_guess, N, Ts)

    # Get Jacobians for tracking error and boundary variable
    track_params = unicycle_raw_linearise_track_horizon(xi_guess, N, track_spline)
    
    # Apply change of variable for linearisation to constraints
    del_constraints = raw_change_of_variable_constraints_over_horizon(xi_guess,
                                                                  u_guess,
                                                                  constraints, 
                                                                  N)
    
    # Convert mpc to qp problem
    #qp_mats = raw_unicycle_ltv_contouring_mpc2qp(state_space_mats, track_params,
    #                                        del_constraints, weights, u_guess, 
    #                                        N)
    qp_mats = raw_unicycle_ltv_contouring_mpc2qp_loop(state_space_mats, track_params,
                                            del_constraints, weights, u_guess, 
                                            N)
    
    
    # Solve qp problem
    prob = osqp.OSQP()
    prob.setup(qp_mats['P'], qp_mats['q'], qp_mats['A'], qp_mats['l'],
               qp_mats['u'], warm_start=True, verbose=False)
    res = prob.solve()
    
    u_delta_flat = res.x[(N+1)*N_STATES:(N+1)*N_STATES+N*N_INPUTS]
    xi_delta_flat = res.x[0:(N+1)*N_STATES]
    
    # Update states and input
    u_delta = np.reshape(u_delta_flat, (N_INPUTS, N), 'F')
    xi_delta = np.reshape(xi_delta_flat, (N_STATES, N+1), 'F')
    
    solution = {
            'u': u_guess + u_delta,
            'xi': xi_guess + xi_delta
            }
        
    return solution

def unicycle_solver(weights, constraints, xi_curr, u_guess, N, Ts, 
                    track_spline):
    """ Solve for states and inputs over horizon using MPC """
    
    # Constants
    BETA_INDEX = 4
    N_STATES = 6
    N_INPUTS = 3
    
    # Set weighting for virtual reference to 0 at end of track
    if xi_curr[BETA_INDEX,0] > track_spline['break_fin'][0,0]:
        weights['q'] = 0
 
    # Calculate guesses for states over the horizon  
    xi_guess = np.hstack((xi_curr, np.zeros((N_STATES, N))))
    for n in range(0, N):
        xi_guess[:,n+1] = unicycle_taylor_order2_next_step(xi_guess[:,n],
                u_guess[:,n], Ts)
    
    # Linearise the state space mdoel
    state_space_mats = unicycle_linearise_state_space_horizon(xi_guess, N, Ts)

    # Get Jacobians for tracking error and boundary variable
    track_params = unicycle_linearise_track_horizon(xi_guess, N, track_spline)
    
    # Apply change of variable for linearisation to constraints
    del_constraints = change_of_variable_constraints_over_horizon(xi_guess,
                                                                  u_guess,
                                                                  constraints, 
                                                                  N)
    
    # Convert mpc to qp problem
    qp_mats = unicycle_ltv_contouring_mpc2qp(state_space_mats, track_params,
                                            del_constraints, weights, u_guess, 
                                            N)
    
    # Solve qp problem
    prob = osqp.OSQP()
    prob.setup(qp_mats['P'], qp_mats['q'], qp_mats['A'], qp_mats['l'],
               qp_mats['u'], warm_start=True, verbose=False)
    res = prob.solve()
    
    u_delta_flat = res.x[(N+1)*N_STATES:(N+1)*N_STATES+N*N_INPUTS]
    xi_delta_flat = res.x[0:(N+1)*N_STATES]
    
    # Update states and input
    u_delta = np.reshape(u_delta_flat, (N_INPUTS, N), 'F')
    xi_delta = np.reshape(xi_delta_flat, (N_STATES, N+1), 'F')
    
    solution = {
            'u': u_guess + u_delta,
            'xi': xi_guess + xi_delta
            }
        
    return solution


def unicycle_raw_taylor_order2_next_step(xi_curr, u_curr, Ts):
    """ Calculate next step using second order Taylor series discretization """
    dotx = u_curr[1]*math.cos(xi_curr[2]);
    ddotx = -u_curr[1]*math.sin(xi_curr[2])*u_curr[0];
    doty = u_curr[1]*math.sin(xi_curr[2]);
    ddoty = u_curr[1]*math.cos(xi_curr[2])*u_curr[0];
    dottheta = u_curr[0];
    dotbeta = u_curr[2];
    xi_next = [0] * len(xi_curr);
    xi_next[0] = xi_curr[0] + Ts*dotx + Ts**2/2.0*ddotx;
    xi_next[1] = xi_curr[1] + Ts*doty + Ts**2/2.0*ddoty;
    xi_next[2] = xi_curr[2] + Ts*dottheta;
    xi_next[3] = xi_curr[3] + Ts*dotbeta;
    return xi_next

def unicycle_taylor_order2_next_step(xi_curr, u_curr, Ts):
    """ Calculate next step using second order Taylor series discretization """
    dotx = xi_curr[2]*math.cos(xi_curr[3]);
    ddotx = -xi_curr[2]*math.sin(xi_curr[3])*u_curr[0] + math.cos(xi_curr[3])*u_curr[1];
    doty = xi_curr[2]*math.sin(xi_curr[3]);
    ddoty = xi_curr[2]*math.cos(xi_curr[3])*u_curr[0] + math.sin(xi_curr[3])*u_curr[1];
    dotv = u_curr[1];
    dottheta = u_curr[0];
    dotbeta = xi_curr[5];
    ddotbeta = u_curr[2];
    xi_next = [0] * len(xi_curr)
    xi_next[0] = xi_curr[0] + Ts*dotx + Ts**2/2.0*ddotx;
    xi_next[1] = xi_curr[1] + Ts*doty + Ts**2/2.0*ddoty;
    xi_next[2] = xi_curr[2] + Ts*dotv;
    xi_next[3] = xi_curr[3] + Ts*dottheta;
    xi_next[4] = xi_curr[4] + Ts*dotbeta + Ts**2/2*ddotbeta;
    xi_next[5] = xi_curr[5] + Ts*ddotbeta;
    return xi_next

def unicycle_raw_linearise_state_space_horizon(u_horizon, xi_horizon, N, Ts):
    """ Generate linearised state and input matrices over the horizon N """
    mats = {
            'Ad': [unicycle_raw_linearise_statemat(u_horizon[:, n], xi_horizon[:, n], Ts) for n in range(0, N)],
            'Bd': [unicycle_raw_linearise_inputmat(u_horizon[:, n], xi_horizon[:, n], Ts) for n in range(0, N)]
            }
    return mats
 
def unicycle_linearise_state_space_horizon(xi_horizon, N, Ts):
    """ Generate linearised state and input matrices over the horizon N """
    mats = {
            'Ad': [unicycle_linearise_statemat(xi_horizon[:, n], Ts) for n in range(0, N)],
            'Bd': [unicycle_linearise_inputmat(xi_horizon[:, n], Ts) for n in range(0, N)]
            }
    return mats

def unicycle_raw_linearise_statemat(u, xi, Ts):
    """ Generate unicycle state matrix linearised over xi with sample 
    period Ts """
    Ad = np.array([[1, 0, -Ts*u[1]*math.sin(xi[2]), 0],
                   [0, 1, Ts*u[1]*math.cos(xi[2]), 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype='f')
    return Ad

def unicycle_linearise_statemat(xi, Ts):
    """ Generate unicycle state matrix linearised over xi with sample 
    period Ts """
    Ad = np.array([[1, 0, Ts*math.cos(xi[3]), -Ts*xi[2]*math.sin(xi[3]), 0, 0],
                   [0, 1, Ts*math.sin(xi[3]), Ts*xi[2]*math.cos(xi[3]), 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, Ts],
                   [0, 0, 0, 0, 0, 1]], dtype='f')
    return Ad

def unicycle_raw_linearise_inputmat(u, xi, Ts):
    """ 
    Generate unicycle input matrix linearised over xi with sample period 
    Ts 
    """
    Bd = np.array([[-(Ts**2*u[1]*math.sin(xi[2]))/2, 
                    Ts*math.cos(xi[2]), 0],
                    [(Ts**2*u[1]*math.cos(xi[2]))/2, 
                     Ts*math.sin(xi[2]), 0],
                    [Ts, 0, 0],
                    [0, 0, Ts]], dtype='f')
    return Bd

def unicycle_linearise_inputmat(xi, Ts):
    """ 
    Generate unicycle input matrix linearised over xi with sample period 
    Ts 
    """
    Bd = np.array([[-(Ts**2*xi[2]*math.sin(xi[3]))/2, 
                    (Ts**2*math.cos(xi[3]))/2, 0],
                    [(Ts**2*xi[2]*math.cos(xi[3]))/2, 
                     (Ts**2*math.sin(xi[3]))/2, 0],
                    [0, Ts, 0],
                    [Ts, 0, 0],
                    [0, 0, Ts**2/2],
                    [0, 0, Ts]], dtype='f')
    return Bd
    
def unicycle_raw_linearise_track_horizon(xi_horizon, N, track_spline):
    """ 
    Generate Jacobian and offsets for linear approximation of tracking 
    error and track boundary variable for unicycle
    """
    N = xi_horizon.shape[1] - 1
    BETA_INDEX = 3

    # Get coefficients for path, track center and track outer boundary
    # corresponding to virtual state 
    beta = xi_horizon[BETA_INDEX,:]
    track_pieces = [get_track_piece_from_virtual_state_bisect(x, track_spline) for 
                    x in beta]
    #track_pieces = [get_track_piece_from_virtual_state_loop(x, track_spline) for 
    #                    x in beta]
    
    # Generate Jacobian for path error term 'eps' and boundary variable 'p'
    #linearised_track_params = [raw_linearise_error_and_boundary(xi_horizon[:,n], track_pieces[n]) for n in range(0, N+1)]
    linearised_track_params = [raw_linearise_error_and_boundary_loop(xi_horizon[:,n], track_pieces[n]) for n in range(0, N+1)]
    params = {
            'J_eps': [linearised_track_params[n]['J_eps'] for n in range(0,N+1)],
            'eps_offset': [linearised_track_params[n]['eps_offset'] for 
                           n in range(0,N+1)],
            'J_p': [linearised_track_params[n]['J_p'] for n in range(0,N+1)],
            'p_offset': [linearised_track_params[n]['p_offset'] for 
                         n in range(0,N+1)]
            }
    
    return params

def unicycle_linearise_track_horizon(xi_horizon, N, track_spline):
    """ 
    Generate Jacobian and offsets for linear approximation of tracking 
    error and track boundary variable for unicycle
    """
    N = xi_horizon.shape[1] - 1
    BETA_INDEX = 4

    # Get coefficients for path, track center and track outer boundary
    # corresponding to virtual state 
    beta = xi_horizon[BETA_INDEX,:]
    track_pieces = [get_track_piece_from_virtual_state(x, track_spline) for 
                    x in beta]
    
    # Generate Jacobian for path error term 'eps' and boundary variable 'p'
    linearised_track_params = [linearise_error_and_boundary(xi_horizon[:,n], track_pieces[n]) for n in range(0, N+1)]
    
    params = {
            'J_eps': [linearised_track_params[n]['J_eps'] for n in range(0,N+1)],
            'eps_offset': [linearised_track_params[n]['eps_offset'] for 
                           n in range(0,N+1)],
            'J_p': [linearised_track_params[n]['J_p'] for n in range(0,N+1)],
            'p_offset': [linearised_track_params[n]['p_offset'] for 
                         n in range(0,N+1)]
            }
    
    return params
    
    

def get_track_piece_from_virtual_state(beta, track_spline):
    """ Take spline information and path parameter (beta) and return corresponding coefficients """
    pieces = track_spline['coefs_x_center'].shape[0]
    for i in range(0, pieces):
        if ((track_spline['breaks'][0,i] <= beta) and (beta <= track_spline['breaks'][0,i+1])):
            break
        elif (i == pieces-1):
            raise ValueError('beta not in domain of spline')

    track_piece = {
            'beta': beta,
            'break': track_spline['breaks'][0,i],
            'coefs_x_center': track_spline['coefs_x_center'][i,:],
            'coefs_y_center': track_spline['coefs_y_center'][i,:],
            'coefs_x_outer': track_spline['coefs_x_outer'][i,:],
            'coefs_y_outer': track_spline['coefs_y_outer'][i,:],
            'coefs_x_path': track_spline['coefs_x_path'][i,:],
            'coefs_y_path': track_spline['coefs_y_path'][i,:]
            }
    
    return track_piece

from bisect import bisect_left
def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

def get_track_piece_from_virtual_state_bisect(beta, track_spline):
    """ Take spline information and path parameter (beta) and return corresponding coefficients """
    pieces = track_spline['coefs_x_center'].shape[0]
    
    if (beta < 0) or (beta > track_spline['breaks'][0,-1]):
        raise ValueError('beta not in domain of spline')
        
    i = bisect_left(track_spline['breaks'][0], beta)
    
    #for i in range(0, pieces):
    #    if ((track_spline['breaks'][0,i] <= beta) and (beta <= track_spline['breaks'][0,i+1])):
    #        break
    #    elif (i == pieces-1):
    #        raise ValueError('beta not in domain of spline')
            
    # Search for track piece using binary search (bisect)
    #a = 4

    track_piece = {
            'beta': beta,
            'break': track_spline['breaks'][0,i],
            'coefs_x_center': track_spline['coefs_x_center'][i,:],
            'coefs_y_center': track_spline['coefs_y_center'][i,:],
            'coefs_x_outer': track_spline['coefs_x_outer'][i,:],
            'coefs_y_outer': track_spline['coefs_y_outer'][i,:],
            'coefs_x_path': track_spline['coefs_x_path'][i,:],
            'coefs_y_path': track_spline['coefs_y_path'][i,:]
            }
    
    return track_piece
    
def raw_linearise_error_and_boundary(xi, track_piece):
    """ Take piece of track spline and state and return the corresponding 
    Jacobian matrices and offsets for linearly approximatting path tracking
    error and boundary variable """
    
    X_INDEX = 0
    Y_INDEX = 1
    BETA_INDEX = 3
    
    # Coefficient vectors for path
    a_path_vec = np.array([[track_piece['coefs_x_path'][0]],
                           [track_piece['coefs_y_path'][0]]])
    b_path_vec = np.array([[track_piece['coefs_x_path'][1]],
                           [track_piece['coefs_y_path'][1]]])
    c_path_vec = np.array([[track_piece['coefs_x_path'][2]],
                           [track_piece['coefs_y_path'][2]]])
    d_path_vec = np.array([[track_piece['coefs_x_path'][3]],
                           [track_piece['coefs_y_path'][3]]])
    
    # Coefficient vectors for track center
    a_center_vec = np.array([[track_piece['coefs_x_center'][0]],
                             [track_piece['coefs_y_center'][0]]])
    b_center_vec = np.array([[track_piece['coefs_x_center'][1]],
                             [track_piece['coefs_y_center'][1]]])
    c_center_vec = np.array([[track_piece['coefs_x_center'][2]],
                             [track_piece['coefs_y_center'][2]]])
    d_center_vec = np.array([[track_piece['coefs_x_center'][3]],
                             [track_piece['coefs_y_center'][3]]])
    
    # Coefficient vectors for track outer boundary
    a_outer_vec = np.array([[track_piece['coefs_x_outer'][0]],
                            [track_piece['coefs_y_outer'][0]]])
    b_outer_vec = np.array([[track_piece['coefs_x_outer'][1]],
                            [track_piece['coefs_y_outer'][1]]])
    c_outer_vec = np.array([[track_piece['coefs_x_outer'][2]],
                            [track_piece['coefs_y_outer'][2]]])
    d_outer_vec = np.array([[track_piece['coefs_x_outer'][3]],
                            [track_piece['coefs_y_outer'][3]]])
    
    # Path linearised around virtual state
    beta_diff = xi[BETA_INDEX] - track_piece['break']
    J_path_beta = (3*a_path_vec*(beta_diff)**2 + 
                  2*b_path_vec*(beta_diff) +
                  c_path_vec)
    path_beta = (a_path_vec*beta_diff**3 + 
                  b_path_vec*beta_diff**2 + 
                  c_path_vec*beta_diff + 
                  d_path_vec)
    
    # Track center linearised around virtual state
    J_center_beta = (3*a_center_vec*(beta_diff)**2 + 
                  2*b_center_vec*(beta_diff) +
                  c_center_vec)
    center_beta = (a_center_vec*beta_diff**3 + 
                  b_center_vec*beta_diff**2 + 
                  c_center_vec*beta_diff + 
                  d_center_vec)
    
    # Boundary variable lienarised around virtual state
    
    # First order derivative wrt x y
    
    bound_out = (a_outer_vec*beta_diff**3 +
                 b_outer_vec*beta_diff**2 + 
                 c_outer_vec*beta_diff + 
                 d_outer_vec)
    bound_center_diff = bound_out - center_beta
    G = (1/(bound_center_diff[0]**2 + bound_center_diff[1]**2) *
         np.array([[bound_center_diff[0,0], bound_center_diff[1,0]], 
                   [-bound_center_diff[1,0], bound_center_diff[0,0]]]))
        
    # First order derivative wrt beta
    pos_diff = np.vstack(xi[[X_INDEX,Y_INDEX]]) - center_beta
    J_bound_out = (3*a_outer_vec*(beta_diff)**2 +
                   2*b_outer_vec*(beta_diff) + 
                   c_outer_vec)
    J_bound_center_diff = J_bound_out - J_center_beta
    J_a_vec = (
            (bound_center_diff[0,0]**2 - bound_center_diff[1,0]**2) 
            * J_bound_center_diff - 
            2*(bound_center_diff[0,0]*J_bound_center_diff[0,0] 
            - bound_center_diff[1,0]*J_bound_center_diff[1,0]) * 
            bound_center_diff)
    c_vec = np.array([[0, 1], [-1, 0]]) @ bound_center_diff
    J_c_vec = np.array([[0, 1], [-1, 0]]) @ J_bound_center_diff               
    J_b_vec = ((bound_center_diff[0,0]**2 - bound_center_diff[1,0]**2) * 
               J_c_vec - 
               2*(bound_center_diff[0,0]*J_bound_center_diff[0,0] - 
                  bound_center_diff[1,0]*J_bound_center_diff[1,0]) * 
                  c_vec)
    J_G = np.vstack((J_a_vec.T, J_b_vec.T))
    J_pos_diff = -J_center_beta
    dpdbeta = G @ J_pos_diff + J_G @ pos_diff
    
    
    # Error between position and path linearised around state
    J_eps_xi = np.hstack((np.eye(2), np.zeros((2,1)), -J_path_beta))
    eps_xi = np.vstack(xi[[X_INDEX,Y_INDEX]]) - path_beta
    
    # Boundary variable linearised around state
    J_p_xi = np.hstack((G, np.zeros((2,1)), dpdbeta))
    p_xi = G @ pos_diff
    
    params = {
            'J_eps': J_eps_xi,
            'eps_offset': eps_xi,
            'J_p': J_p_xi,
            'p_offset': p_xi
            }
    
    return params

def raw_linearise_error_and_boundary_loop(xi, track_piece):
    """ Take piece of track spline and state and return the corresponding 
    Jacobian matrices and offsets for linearly approximatting path tracking
    error and boundary variable """
    
    X_INDEX = 0
    Y_INDEX = 1
    BETA_INDEX = 3
    N_REFS = 2
    N_STATES = 4 
    
    # Coefficient vectors for path
    a_path_vec = np.array([[track_piece['coefs_x_path'][0]],
                           [track_piece['coefs_y_path'][0]]])
    b_path_vec = np.array([[track_piece['coefs_x_path'][1]],
                           [track_piece['coefs_y_path'][1]]])
    c_path_vec = np.array([[track_piece['coefs_x_path'][2]],
                           [track_piece['coefs_y_path'][2]]])
    d_path_vec = np.array([[track_piece['coefs_x_path'][3]],
                           [track_piece['coefs_y_path'][3]]])
    
    # Coefficient vectors for track center
    a_center_vec = np.array([[track_piece['coefs_x_center'][0]],
                             [track_piece['coefs_y_center'][0]]])
    b_center_vec = np.array([[track_piece['coefs_x_center'][1]],
                             [track_piece['coefs_y_center'][1]]])
    c_center_vec = np.array([[track_piece['coefs_x_center'][2]],
                             [track_piece['coefs_y_center'][2]]])
    d_center_vec = np.array([[track_piece['coefs_x_center'][3]],
                             [track_piece['coefs_y_center'][3]]])
    
    # Coefficient vectors for track outer boundary
    a_outer_vec = np.array([[track_piece['coefs_x_outer'][0]],
                            [track_piece['coefs_y_outer'][0]]])
    b_outer_vec = np.array([[track_piece['coefs_x_outer'][1]],
                            [track_piece['coefs_y_outer'][1]]])
    c_outer_vec = np.array([[track_piece['coefs_x_outer'][2]],
                            [track_piece['coefs_y_outer'][2]]])
    d_outer_vec = np.array([[track_piece['coefs_x_outer'][3]],
                            [track_piece['coefs_y_outer'][3]]])
    
    # Path linearised around virtual state
    beta_diff = xi[BETA_INDEX] - track_piece['break']
    J_path_beta = (3*a_path_vec*(beta_diff)**2 + 
                  2*b_path_vec*(beta_diff) +
                  c_path_vec)
    path_beta = (a_path_vec*beta_diff**3 + 
                  b_path_vec*beta_diff**2 + 
                  c_path_vec*beta_diff + 
                  d_path_vec)
    
    # Track center linearised around virtual state
    J_center_beta = (3*a_center_vec*(beta_diff)**2 + 
                  2*b_center_vec*(beta_diff) +
                  c_center_vec)
    center_beta = (a_center_vec*beta_diff**3 + 
                  b_center_vec*beta_diff**2 + 
                  c_center_vec*beta_diff + 
                  d_center_vec)
    
    # Boundary variable lienarised around virtual state
    
    # First order derivative wrt x y
    
    bound_out = (a_outer_vec*beta_diff**3 +
                 b_outer_vec*beta_diff**2 + 
                 c_outer_vec*beta_diff + 
                 d_outer_vec)
    bound_center_diff = bound_out - center_beta
    G = (1/(bound_center_diff[0]**2 + bound_center_diff[1]**2) *
         np.array([[bound_center_diff[0,0], bound_center_diff[1,0]], 
                   [-bound_center_diff[1,0], bound_center_diff[0,0]]]))
        
    # First order derivative wrt beta
    pos_diff = (xi[[X_INDEX,Y_INDEX]] - center_beta.T).T
    J_bound_out = (3*a_outer_vec*(beta_diff)**2 +
                   2*b_outer_vec*(beta_diff) + 
                   c_outer_vec)
    J_bound_center_diff = J_bound_out - J_center_beta
    c_vec = np.array([[0, 1], [-1, 0]]) @ bound_center_diff
    J_c_vec = np.array([[0, 1], [-1, 0]]) @ J_bound_center_diff                
    J_G = np.zeros((N_REFS,N_REFS))
    J_G_var1 = bound_center_diff[0,0]**2 - bound_center_diff[1,0]**2
    J_G_var2 = 2*(bound_center_diff[0,0]*J_bound_center_diff[0,0] - 
                  bound_center_diff[1,0]*J_bound_center_diff[1,0])
    J_G[0,:] = ((
            (J_G_var1) 
            * J_bound_center_diff - 
            J_G_var2 * 
            bound_center_diff).T)
    J_G[1,:] = ((
            (J_G_var1) * 
               J_c_vec - 
               J_G_var2 * 
                  c_vec).T)
               
    J_pos_diff = -J_center_beta
    dpdbeta = G @ J_pos_diff + J_G @ pos_diff
    
    
    # Error between position and path linearised around state
    J_eps_xi = np.zeros((N_REFS,N_STATES))
    J_eps_xi[:,0:N_REFS] = np.eye(2)
    J_eps_xi[:,BETA_INDEX] = -J_path_beta.T
    eps_xi = (xi[[X_INDEX,Y_INDEX]] - path_beta.T).T
    
    # Boundary variable linearised around state
    J_p_xi = np.zeros((N_REFS,N_STATES))
    J_p_xi[:,0:N_REFS] = G
    J_p_xi[:,BETA_INDEX] = dpdbeta.T
    p_xi = G @ pos_diff
    
    params = {
            'J_eps': J_eps_xi,
            'eps_offset': eps_xi,
            'J_p': J_p_xi,
            'p_offset': p_xi
            }
    
    return params

def linearise_error_and_boundary(xi, track_piece):
    """ Take piece of track spline and state and return the corresponding 
    Jacobian matrices and offsets for linearly approximatting path tracking
    error and boundary variable """
    
    X_INDEX = 0
    Y_INDEX = 1
    BETA_INDEX = 4
    
    # Coefficient vectors for path
    a_path_vec = np.array([[track_piece['coefs_x_path'][0]],
                           [track_piece['coefs_y_path'][0]]])
    b_path_vec = np.array([[track_piece['coefs_x_path'][1]],
                           [track_piece['coefs_y_path'][1]]])
    c_path_vec = np.array([[track_piece['coefs_x_path'][2]],
                           [track_piece['coefs_y_path'][2]]])
    d_path_vec = np.array([[track_piece['coefs_x_path'][3]],
                           [track_piece['coefs_y_path'][3]]])
    
    # Coefficient vectors for track center
    a_center_vec = np.array([[track_piece['coefs_x_center'][0]],
                             [track_piece['coefs_y_center'][0]]])
    b_center_vec = np.array([[track_piece['coefs_x_center'][1]],
                             [track_piece['coefs_y_center'][1]]])
    c_center_vec = np.array([[track_piece['coefs_x_center'][2]],
                             [track_piece['coefs_y_center'][2]]])
    d_center_vec = np.array([[track_piece['coefs_x_center'][3]],
                             [track_piece['coefs_y_center'][3]]])
    
    # Coefficient vectors for track outer boundary
    a_outer_vec = np.array([[track_piece['coefs_x_outer'][0]],
                            [track_piece['coefs_y_outer'][0]]])
    b_outer_vec = np.array([[track_piece['coefs_x_outer'][1]],
                            [track_piece['coefs_y_outer'][1]]])
    c_outer_vec = np.array([[track_piece['coefs_x_outer'][2]],
                            [track_piece['coefs_y_outer'][2]]])
    d_outer_vec = np.array([[track_piece['coefs_x_outer'][3]],
                            [track_piece['coefs_y_outer'][3]]])
    
    # Path linearised around virtual state
    beta_diff = xi[BETA_INDEX] - track_piece['break']
    J_path_beta = (3*a_path_vec*(beta_diff)**2 + 
                  2*b_path_vec*(beta_diff) +
                  c_path_vec)
    path_beta = (a_path_vec*beta_diff**3 + 
                  b_path_vec*beta_diff**2 + 
                  c_path_vec*beta_diff + 
                  d_path_vec)
    
    # Track center linearised around virtual state
    J_center_beta = (3*a_center_vec*(beta_diff)**2 + 
                  2*b_center_vec*(beta_diff) +
                  c_center_vec)
    center_beta = (a_center_vec*beta_diff**3 + 
                  b_center_vec*beta_diff**2 + 
                  c_center_vec*beta_diff + 
                  d_center_vec)
    
    # Boundary variable lienarised around virtual state
    
    # First order derivative wrt x y
    
    bound_out = (a_outer_vec*beta_diff**3 +
                 b_outer_vec*beta_diff**2 + 
                 c_outer_vec*beta_diff + 
                 d_outer_vec)
    bound_center_diff = bound_out - center_beta
    G = (1/(bound_center_diff[0]**2 + bound_center_diff[1]**2) *
         np.array([[bound_center_diff[0,0], bound_center_diff[1,0]], 
                   [-bound_center_diff[1,0], bound_center_diff[0,0]]]))
        
    # First order derivative wrt beta
    pos_diff = np.vstack(xi[[X_INDEX,Y_INDEX]]) - center_beta
    J_bound_out = (3*a_outer_vec*(beta_diff)**2 +
                   2*b_outer_vec*(beta_diff) + 
                   c_outer_vec)
    J_bound_center_diff = J_bound_out - J_center_beta
    J_a_vec = (
            (bound_center_diff[0,0]**2 - bound_center_diff[1,0]**2) 
            * J_bound_center_diff - 
            2*(bound_center_diff[0,0]*J_bound_center_diff[0,0] 
            - bound_center_diff[1,0]*J_bound_center_diff[1,0]) * 
            bound_center_diff)
    c_vec = np.array([[0, 1], [-1, 0]]) @ bound_center_diff
    J_c_vec = np.array([[0, 1], [-1, 0]]) @ J_bound_center_diff               
    J_b_vec = ((bound_center_diff[0,0]**2 - bound_center_diff[1,0]**2) * 
               J_c_vec - 
               2*(bound_center_diff[0,0]*J_bound_center_diff[0,0] - 
                  bound_center_diff[1,0]*J_bound_center_diff[1,0]) * 
                  c_vec)
    J_G = np.vstack((J_a_vec.T, J_b_vec.T))
    J_pos_diff = -J_center_beta
    dpdbeta = G @ J_pos_diff + J_G @ pos_diff
    
    
    # Error between position and path linearised around state
    J_eps_xi = np.hstack((np.eye(2), np.zeros((2,2)), -J_path_beta, np.zeros((2,1))))
    eps_xi = np.vstack(xi[[X_INDEX,Y_INDEX]]) - path_beta
    
    # Boundary variable linearised around state
    J_p_xi = np.hstack((G, np.zeros((2,2)), dpdbeta, np.zeros((2,1))))
    p_xi = G @ pos_diff
    
    params = {
            'J_eps': J_eps_xi,
            'eps_offset': eps_xi,
            'J_p': J_p_xi,
            'p_offset': p_xi
            }
    
    return params

def raw_change_of_variable_constraints_over_horizon(xi_guess, u_guess, 
                                                constraints, N):
    """ Obtain delta constraints according to linearisation around 
    trajectory"""
        
    xi_guess_flat = np.vstack(xi_guess.flatten('F'))
    u_guess_flat = np.vstack(u_guess.flatten('F'))
    
    umin = np.tile(np.vstack([constraints['omega_min'], 
            constraints['v_min'], 
            constraints['gamma_min']]), [N, 1])
    
    umax = np.tile(np.vstack([constraints['omega_max'], 
            constraints['v_max'], 
            constraints['gamma_max']]), [N, 1])
    
    ximin = np.tile(np.vstack([-np.inf, 
             -np.inf, 
             constraints['theta_min'], 
             constraints['beta_min']]), [N+1, 1])
    
    ximax = np.tile(np.vstack([np.inf, 
             np.inf, 
             constraints['theta_max'], 
             constraints['beta_max']]), [N+1, 1])
    
    del_constraints = {
            'umin': umin - u_guess_flat,
            'umax': umax - u_guess_flat, 
            'ximin': ximin - xi_guess_flat,
            'ximax': ximax - xi_guess_flat
            }
    return del_constraints

def change_of_variable_constraints_over_horizon(xi_guess, u_guess, 
                                                constraints, N):
    """ Obtain delta constraints according to linearisation around 
    trajectory"""
        
    xi_guess_flat = np.vstack(xi_guess.flatten('F'))
    u_guess_flat = np.vstack(u_guess.flatten('F'))
    
    umin = np.tile(np.vstack([constraints['omega_min'], 
            constraints['a_min'], 
            constraints['gamma_min']]), [N, 1])
    
    umax = np.tile(np.vstack([constraints['omega_max'], 
            constraints['a_max'], 
            constraints['gamma_max']]), [N, 1])
    
    ximin = np.tile(np.vstack([-np.inf, 
             -np.inf, 
             constraints['v_min'], 
             constraints['theta_min'], 
             constraints['beta_min'], 
             constraints['dotbeta_min']]), [N+1, 1])
    
    ximax = np.tile(np.vstack([np.inf, 
             np.inf, 
             constraints['v_max'], 
             constraints['theta_max'], 
             constraints['beta_max'], 
             constraints['dotbeta_max']]), [N+1, 1])
    
    del_constraints = {
            'umin': umin - u_guess_flat,
            'umax': umax - u_guess_flat, 
            'ximin': ximin - xi_guess_flat,
            'ximax': ximax - xi_guess_flat
            }
    return del_constraints

def raw_unicycle_ltv_contouring_mpc2qp(state_space_mats, track_params, 
                                   del_constraints, weights, u_guess, N):
    """ Convert unicycle MPC into matrices for QP problems """
    
    N_STATES = state_space_mats['Bd'][0].shape[0]
    N_INPUTS = state_space_mats['Bd'][0].shape[1]
    
    # Set initial delta-state to be zero column vector
    x0 = np.zeros((N_STATES, 1))
    
    # Construct P matrix
    Pdiagstate = [sparse.csc_matrix(x.T @ weights['Q'] @ x) for 
                  x in track_params['J_eps']]
    Pdiagstate.append(sparse.kron(sparse.identity(N), weights['R']))
    P = sparse.block_diag(Pdiagstate).multiply(2).tocsc()
    
    # Construct q matrix
    qvertstatelist = [2*track_params['J_eps'][n].T @ weights['Q'] @
                      track_params['eps_offset'][n] + 
                      np.vstack([0.0,0.0,0.0,-weights['q']]) for 
                      n in range(0, N+1)]
    qvertstatelist.append(2.0*np.kron(np.eye(N), weights['R']) @
                          np.vstack(u_guess.flatten('F')))
    q = np.vstack(qvertstatelist)

    # Construct Ax (working)
    diagblkAd = sparse.block_diag(state_space_mats['Ad'])
    Ax = ((sparse.kron(sparse.eye(N+1), -sparse.eye(N_STATES)) +
           sparse.vstack([sparse.csc_matrix((N_STATES, (N+1)*N_STATES)), 
                          sparse.hstack((diagblkAd, 
                                         sparse.csc_matrix(
                                                 (N*N_STATES,
                                                  N_STATES))))])).tocsc())
    
    # Construct Bu
    diagblkBd = sparse.block_diag(state_space_mats['Bd'])
    Bu = sparse.vstack([sparse.csc_matrix((N_STATES, N*N_INPUTS)), 
                        diagblkBd]).tocsc()
    
    # Construct Aeq
    Aeq = sparse.hstack([Ax, Bu])
    
    # Construct leq
    leq = np.vstack([-x0, np.zeros((N*N_STATES, 1))])
    
    # Construct ueq
    ueq = leq
    
    # Construct Aineq
    Aineqlist = [sparse.vstack([sparse.csc_matrix(x), 
                                sparse.hstack([sparse.csc_matrix((N_STATES-2,2)), 
                                               sparse.eye(N_STATES-2)])]) for 
    x in track_params['J_p']]
    Aineqlist.append(sparse.eye(N*N_INPUTS))
    Aineq = sparse.block_diag(Aineqlist).tocsc()
    
    # Construct lineq
    lineqlist = [np.vstack([-np.ones((2,1))-track_params['p_offset'][n], 
                            np.vstack(del_constraints['ximin'][(n-1)*
                                      N_STATES+2:n*N_STATES,0])]) for 
    n in range(1, N+1)]
    lineqlist.insert(0, np.full((N_STATES, 1), -np.inf))
    lineqlist.append(del_constraints['umin'])
    lineq = np.vstack(lineqlist)
        
    # Construct uineq
    uineqlist = [np.vstack([np.ones((2,1))-track_params['p_offset'][n], 
                            np.vstack(del_constraints['ximax'][(n-1)* 
                                      N_STATES+2:n*N_STATES,0])]) for 
    n in range(1, N+1)]
    uineqlist.insert(0, np.full((N_STATES, 1), np.inf))
    uineqlist.append(del_constraints['umax'])
    uineq = np.vstack(uineqlist)
    
    #  Construct A, l, u
    A = sparse.vstack([Aeq, Aineq]).tocsc()
    l = np.vstack([leq, lineq])
    u = np.vstack([ueq, uineq])
    
        
    # Print shapes
#    print("SIZES")
#    print(P.shape)
#    print(q.shape)
#    print(A.shape)
#    print(l.shape)
#    print(u.shape)
#    print(Aineq.shape)
#    print(lineq.shape)
#    print(uineq.shape)
#    print(Aeq.shape)
#    print(leq.shape)
#    print(ueq.shape)
    
    qp_mats = {
            'P': P,
            'q': q,
            'A': A,
            'l': l,
            'u': u
                }
    #scipy.io.savemat('qpmat', qp_mats)
    
    return qp_mats

def raw_unicycle_ltv_contouring_mpc2qp_loop(state_space_mats, track_params, 
                                   del_constraints, weights, u_guess, N):
    N_STATES = state_space_mats['Bd'][0].shape[0]
    N_INPUTS = state_space_mats['Bd'][0].shape[1]
    
    # Setup qp matrices
    P = np.zeros(((N+1)*N_STATES + N*N_INPUTS, (N+1)*N_STATES + N*N_INPUTS))
    q = np.zeros(((N+1)*N_STATES + N*N_INPUTS, 1))
    #Aeq = np.zeros(((N+1)*N_STATES, (N+1)*N_STATES + N*N_INPUTS))
    #leq = np.zeros(((N+1)*N_STATES, 1)) # column matrix of zeros
    #ueq = np.zeros(((N+1)*N_STATES, 1)) # column matrix of zeros
    #Aineq = np.zeros(((N+1)*N_STATES + N*N_INPUTS, (N+1)*N_STATES + N*N_INPUTS))
    #lineq = np.zeros(((N+1)*N_STATES + N*N_INPUTS, 1))
    #uineq = np.zeros(((N+1)*N_STATES + N*N_INPUTS, 1))
    A = np.zeros((2*(N+1)*N_STATES + N*N_INPUTS, (N+1)*N_STATES + N*N_INPUTS))
    l = np.zeros((2*(N+1)*N_STATES + N*N_INPUTS, 1))
    u = np.zeros((2*(N+1)*N_STATES + N*N_INPUTS, 1))
    
    # Construct P matrix
    # Set block diagonals to be tracking cost
    for i in range(0, N+1):
        P[i*N_STATES:(i+1)*N_STATES, i*N_STATES:(i+1)*N_STATES] = (
                2.0 * track_params['J_eps'][i].T @ weights['Q'] @ 
                track_params['J_eps'][i])  
    # Set block diagonsl to be input cost
    for i in range(0, N):
        P[(N+1)*N_STATES+i*N_INPUTS:(N+1)*N_STATES+(i+1)*N_INPUTS,
          (N+1)*N_STATES+i*N_INPUTS:(N+1)*N_STATES+(i+1)*N_INPUTS] = 2.0 * weights['R']
    
    for i in range(0, N+1):
        q[i*N_STATES:(i+1)*N_STATES, 0] = ( (2 * track_params['eps_offset'][i])[:,0].T @ weights['Q'] @
         track_params['J_eps'][i] )
        q[(i+1)*N_STATES-1, 0] = q[(i+1)*N_STATES-1, 0] - weights['q']
    
    
#    for i in range(0, N+1):
#        q[i*N_STATES:(i+1)*N_STATES, 0] = ( (2*track_params['J_eps'][i].T @ weights['Q'] @ 
#         track_params['eps_offset'][i])[:,0] )
#        q[(i+1)*N_STATES-1, 0] = q[(i+1)*N_STATES-1, 1] - weights['q']
        
    for i in range(0, N):
        q[(N+1)*N_STATES+i*N_INPUTS:(N+1)*N_STATES+(i+1)*N_INPUTS, 0] = (
                2.0 * weights['R'] @ u_guess[:,i] )
    # Construct Aeq matrix
    # Ax
    for i in range(0, (N+1)*N_STATES):
        A[i,i] = -1
    for i in range(0, N):
        A[(i+1)*N_STATES:(i+2)*N_STATES,i*N_STATES:(i+1)*N_STATES] += (
                state_space_mats['Ad'][i])
    # Bu
    for i in range(0, N):
        A[(i+1)*N_STATES:(i+2)*N_STATES, (N+1)*N_STATES + i*N_INPUTS:(N+1)*N_STATES + (i+1)*N_INPUTS] = (
                state_space_mats['Bd'][i])
        
    # Ignore leq and ueq since just vector of zeros
    
    # Construct Aineq
    
    # State part of Aineq
    for i in range(0, N+1):
        A[(N+1)*N_STATES + i*N_STATES:(N+1)*N_STATES + i*N_STATES+2, i*N_STATES:(i+1)*N_STATES] = (
                track_params['J_p'][i])
        A[(N+1)*N_STATES + i*N_STATES+2:(N+1)*N_STATES + (i+1)*N_STATES,i*N_STATES+2:(i+1)*N_STATES] = (
                np.eye(2))
        
    # Input part of Aineq
    for i in range(0, N*N_INPUTS):
        A[2*(N+1)*N_STATES + i, (N+1)*N_STATES + i] = 1
        
    # Consruct l
    for i in range(0,N+1):
        if i == 0:
            l[(N+1)*N_STATES:(N+1)*N_STATES + N_STATES, 0] = np.full((1, N_STATES), -np.inf)
            
        else:
            # constraint for track boundary
            l[(N+1)*N_STATES + i*N_STATES:(N+1)*N_STATES + i*N_STATES+2, 0] = (
                    -np.ones((1,2)) - track_params['p_offset'][i].T)
        
            # constraint for angle and virtual state
            l[(N+1)*N_STATES + i*N_STATES+2:(N+1)*N_STATES + (i+1)*N_STATES, 0] = (
                    del_constraints['ximin'][(i-1)*N_STATES+2:i*N_STATES,0])
            
    l[2*(N+1)*N_STATES:2*(N+1)*N_STATES + N*N_INPUTS] = del_constraints['umin']
    
    # Consruct u
    for i in range(0,N+1):
        if i == 0:
            u[(N+1)*N_STATES:(N+1)*N_STATES + N_STATES, 0] = np.full((1, N_STATES), np.inf)
            
        else:
            # constraint for track boundary
            u[(N+1)*N_STATES + i*N_STATES:(N+1)*N_STATES + i*N_STATES+2, 0] = (
                    np.ones((1,2)) - track_params['p_offset'][i].T)
        
            # constraint for angle and virtual state
            u[(N+1)*N_STATES + i*N_STATES+2:(N+1)*N_STATES + (i+1)*N_STATES, 0] = (
                    del_constraints['ximax'][(i-1)*N_STATES+2:i*N_STATES,0])
            
    u[2*(N+1)*N_STATES:2*(N+1)*N_STATES + N*N_INPUTS,0] = del_constraints['umax'].T
    qp_mats = {
            'P': sparse.csc_matrix(P),
            'q': q,
            'A': sparse.csc_matrix(A),
            'l': l,
            'u': u
                }
    
    #scipy.io.savemat('qpmats_fast_v2', qp_mats)
    
    return qp_mats
    

def unicycle_ltv_contouring_mpc2qp(state_space_mats, track_params, 
                                   del_constraints, weights, u_guess, N):
    """ Convert unicycle MPC into matrices for QP problems """
    
    N_STATES = state_space_mats['Bd'][0].shape[0]
    N_INPUTS = state_space_mats['Bd'][0].shape[1]
    
    # Set initial delta-state to be zero column vector
    x0 = np.zeros((N_STATES, 1))
    
    # Construct P matrix
    Pdiagstate = [sparse.csc_matrix(x.T @ weights['Q'] @ x) for 
                  x in track_params['J_eps']]
    Pdiagstate.append(sparse.kron(sparse.identity(N), weights['R']))
    P = sparse.block_diag(Pdiagstate).multiply(2).tocsc()
    
    # Construct q matrix
    qvertstatelist = [2*track_params['J_eps'][n].T @ weights['Q'] @
                      track_params['eps_offset'][n] + 
                      np.vstack([0.0,0.0,0.0,0.0,-weights['q'],0.0]) for 
                      n in range(0, N+1)]
    qvertstatelist.append(2.0*np.kron(np.eye(N), weights['R']) @
                          np.vstack(u_guess.flatten('F')))
    q = np.vstack(qvertstatelist)

    # Construct Ax (working)
    diagblkAd = sparse.block_diag(state_space_mats['Ad'])
    Ax = ((sparse.kron(sparse.eye(N+1), -sparse.eye(N_STATES)) +
           sparse.vstack([sparse.csc_matrix((N_STATES, (N+1)*N_STATES)), 
                          sparse.hstack((diagblkAd, 
                                         sparse.csc_matrix(
                                                 (N*N_STATES,
                                                  N_STATES))))])).tocsc())
    
    # Construct Bu
    diagblkBd = sparse.block_diag(state_space_mats['Bd'])
    Bu = sparse.vstack([sparse.csc_matrix((N_STATES, N*N_INPUTS)), 
                        diagblkBd]).tocsc()
    
    # Construct Aeq
    Aeq = sparse.hstack([Ax, Bu])
    
    # Construct leq
    leq = np.vstack([-x0, np.zeros((N*N_STATES, 1))])
    
    # Construct ueq
    ueq = leq
    
    # Construct Aineq
    Aineqlist = [sparse.vstack([sparse.csc_matrix(x), 
                                sparse.hstack([sparse.csc_matrix((4,2)), 
                                               sparse.eye(4)])]) for 
    x in track_params['J_p']]
    Aineqlist.append(sparse.eye(N*N_INPUTS))
    Aineq = sparse.block_diag(Aineqlist).tocsc()
    
    # Construct lineq
    lineqlist = [np.vstack([-np.ones((2,1))-track_params['p_offset'][n], 
                            np.vstack(del_constraints['ximin'][(n-1)*
                                      N_STATES+2:n*N_STATES,0])]) for 
    n in range(1, N+1)]
    lineqlist.insert(0, np.full((N_STATES, 1), -np.inf))
    lineqlist.append(del_constraints['umin'])
    lineq = np.vstack(lineqlist)
        
    # Construct uineq
    uineqlist = [np.vstack([np.ones((2,1))-track_params['p_offset'][n], 
                            np.vstack(del_constraints['ximax'][(n-1)* 
                                      N_STATES+2:n*N_STATES,0])]) for 
    n in range(1, N+1)]
    uineqlist.insert(0, np.full((N_STATES, 1), np.inf))
    uineqlist.append(del_constraints['umax'])
    uineq = np.vstack(uineqlist)
    
    #  Construct A, l, u
    A = sparse.vstack([Aeq, Aineq]).tocsc()
    l = np.vstack([leq, lineq])
    u = np.vstack([ueq, uineq])
    
    
    qp_mats = {
            'P': P,
            'q': q,
            'A': A,
            'l': l,
            'u': u
                }
    #scipy.io.savemat('qpmat', qp_mats)
    
    return qp_mats
    
def unicycle_ltv_contouring_mpc2qp_allsparse(state_space_mats, track_params, 
                                   del_constraints, weights, u_guess, N):
    """ Convert unicycle MPC into matrices for QP problems """
    
    N_STATES = state_space_mats['Bd'][0].shape[0]
    N_INPUTS = state_space_mats['Bd'][0].shape[1]
    
    # Set initial delta-state to be zero column vector
    x0 = sparse.csc_matrix((N_STATES, 1))
    
    # Construct P matrix
    Pdiagstate = [sparse.csc_matrix(x.T @ weights['Q'] @ x) for 
                  x in track_params['J_eps']]
    Pdiagstate.append(sparse.kron(sparse.identity(N), weights['R'], 'csc'))
    P = sparse.block_diag(Pdiagstate, 'csc').multiply(2)
    
    # Construct q matrix
    qvertstatelist = [sparse.csc_matrix(2*track_params['J_eps'][n].T @ weights['Q'] @
                      track_params['eps_offset'][n] + 
                      np.vstack([0,0,0,0,-weights['q'],0])) for 
                      n in range(0, N+1)]
    qvertstatelist.append(2*sparse.kron(sparse.eye(N, format='csc'), weights['R']) *
                          sparse.csc_matrix(np.vstack(u_guess.flatten('F'))))
    q = sparse.vstack(qvertstatelist, 'csc')

    # Construct Ax (working)
    diagblkAd = sparse.block_diag(state_space_mats['Ad'], 'csc')
    Ax = (sparse.kron(sparse.eye(N+1, format = 'csc'), -sparse.eye(N_STATES, format = 'csc')) +
           sparse.vstack([sparse.csc_matrix((N_STATES, (N+1)*N_STATES)), 
                          sparse.hstack((diagblkAd, 
                                         sparse.csc_matrix(
                                                 (N*N_STATES,
                                                  N_STATES))), 'csc')], 'csc'))
    
    # Construct Bu
    diagblkBd = sparse.block_diag(state_space_mats['Bd'], 'csc')
    Bu = sparse.vstack([sparse.csc_matrix((N_STATES, N*N_INPUTS)), 
                        diagblkBd], 'csc')
    
    # Construct Aeq
    Aeq = sparse.hstack([Ax, Bu], 'csc')
    
    # Construct leq
    leq = sparse.vstack([-x0, np.zeros((N*N_STATES, 1))], 'csc')
    
    # Construct ueq
    ueq = leq
    
    # Construct Aineq
    Aineqlist = [sparse.vstack([sparse.csc_matrix(x), 
                                sparse.hstack([sparse.csc_matrix((4,2)), 
                                               sparse.eye(4)], 'csc')], 'csc') for 
    x in track_params['J_p']]
    Aineqlist.append(sparse.eye(N*N_INPUTS, format='csc'))
    Aineq = sparse.block_diag(Aineqlist, 'csc')
    
    # Construct lineq
    lineqlist = [sparse.vstack([sparse.csc_matrix(-np.ones((2,1))-track_params['p_offset'][n]), 
                            sparse.csc_matrix(np.vstack(del_constraints['ximin'][(n-1)*
                                      N_STATES+2:n*N_STATES,0]))], 'csc') for 
    n in range(1, N+1)]
    lineqlist.insert(0, sparse.csc_matrix(np.full((N_STATES, 1), -np.inf)))
    lineqlist.append(sparse.csc_matrix(del_constraints['umin']))
    lineq = sparse.vstack(lineqlist, 'csc')
        
    # Construct uineq
    uineqlist = [sparse.vstack([sparse.csc_matrix(np.ones((2,1))-track_params['p_offset'][n]), 
                            sparse.csc_matrix(np.vstack(del_constraints['ximax'][(n-1)* 
                                      N_STATES+2:n*N_STATES,0]))], 'csc') for 
    n in range(1, N+1)]
    uineqlist.insert(0, sparse.csc_matrix(np.full((N_STATES, 1), np.inf)))
    uineqlist.append(sparse.csc_matrix(del_constraints['umax']))
    uineq = sparse.vstack(uineqlist, 'csc')
    
    #  Construct A, l, u
    A = sparse.vstack([Aeq, Aineq], 'csc')
    l = sparse.vstack([leq, lineq], 'csc')
    u = sparse.vstack([ueq, uineq], 'csc')
    
    qp_mats = {
            'P': P,
            'q': q,
            'A': A,
            'l': l,
            'u': u
                }
    scipy.io.savemat('qpmat', qp_mats)
    
    return qp_mats
    

#def generate_spline_path(path, cycles):
#    """ Generate spline from track path """
#    
#    xpoints_cycled = np.tile(path.x, (1, cycles+2))
#    ypoints_cycled = np.tile(path.y, (1, cycles+2))
#    npoints = len(xpoints_cycled)
#    distances = np.sqrt(np.square(xpoints_cycled[1:npoints] - xpoints_cycled[0:npoints-1])
#        + np.square(ypoints_cycled[1:npoints] - ypoints_cycled[0:npoints-1]))
#    breaks = np.zeros




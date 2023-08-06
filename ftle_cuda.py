import os
import sys
import glob
import gc

from concurrent.futures import ThreadPoolExecutor as Executor

from typing import Callable, Tuple, List

import h5py
import time
import numpy as np
from numba import cuda, uint32, uint64, float32, float64, njit, prange
from numba.cuda.libdevice import fast_log2f, fmaf_rn, fast_fdividef, fsqrt_rn, fminf, fmaxf, fast_logf, floorf, float2uint_rn, int2float_rn, float2int_rn
from math import sqrt, log2, ceil, floor

from scipy.spatial import KDTree

from progressbar import ProgressBar
from pyevtk.hl import pointsToVTK, gridToVTK

from mpi4py import MPI


THREADS_PER_BLOCK = 64
MAX_THREADS_PER_SM = 2048
SM_COUNT = 80
BLOCKS_PER_GRID = (MAX_THREADS_PER_SM // THREADS_PER_BLOCK) * SM_COUNT

THREADS_PER_BLOCK_3D = (2, 2, 32)
BLOCKS_PER_GRID_3D = (16, 16, 7)

REAL_NP = np.float32
REAL_CP = float32
CUDA_INT = uint64

FAST_MATH = True


def master_print(msg):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print(msg)
    sys.stdout.flush()


def get_my_gpu_id():
    ngpus = len(cuda.gpus)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return rank % ngpus

def select_gpu():
    return cuda.select_device(get_my_gpu_id())

select_gpu()

def read_UVW(filename, gen_2d=False):
    if not gen_2d:
        with h5py.File(filename, "r", driver='mpio', comm=MPI.COMM_WORLD) as f:
            data = f["dataset"]
            Nx, Ny, _, Nz = data.shape
            U = cuda.pinned_array((Nx, Ny, Nz), dtype=REAL_NP)
            V = cuda.pinned_array((Nx, Ny, Nz), dtype=REAL_NP)
            W = cuda.pinned_array((Nx, Ny, Nz), dtype=REAL_NP)
            data.read_direct(U, np.s_[:, :, 1, :], np.s_[:, :, :])
            data.read_direct(V, np.s_[:, :, 2, :], np.s_[:, :, :])
            data.read_direct(W, np.s_[:, :, 3, :], np.s_[:, :, :])
    else:
        #with h5py.File(filename, "r", driver='mpio', comm=MPI.COMM_WORLD) as f:
        with h5py.File(filename, "r") as f:
            data = f["dataset"]
            #with data.collective:
            Nx, Ny, _, Nz = data.shape
            U = data.astype(REAL_NP)[:, :, 1, Nz//2].reshape((Nx, Ny, 1))
            V = data.astype(REAL_NP)[:, :, 2, Nz//2].reshape((Nx, Ny, 1))
            W = np.zeros_like(U)
    return U, V, W


@cuda.jit(fastmath=FAST_MATH)
def interp_UVW_flat(U0, V0, W0, U1, V1, W1, Un, Vn, Wn, dt, fraction=0.1, N=0):
    global_id = cuda.grid(1)

    gridDim = cuda.gridsize(1)

    tm = REAL_CP(dt / 2.)
    t_target = REAL_CP(fraction * N * dt)
    dt_inv = REAL_CP(1./dt)

    N = U0.shape[0]

    for ii in range(global_id, N, gridDim):
        Um = REAL_CP(REAL_CP(0.5) * (U0[ii] + U1[ii]))
        Vm = REAL_CP(REAL_CP(0.5) * (V0[ii] + V1[ii]))
        Wm = REAL_CP(REAL_CP(0.5) * (W0[ii] + W1[ii]))

        dUdt = REAL_CP((U1[ii] - U0[ii]) * dt_inv)
        dVdt = REAL_CP((V1[ii] - V0[ii]) * dt_inv)
        dWdt = REAL_CP((W1[ii] - W0[ii]) * dt_inv)

        deltaT = REAL_CP((t_target - tm))

        Un[ii] = REAL_CP(Um + dUdt * deltaT)
        Vn[ii] = REAL_CP(Vm + dVdt * deltaT)
        Wn[ii] = REAL_CP(Wm + dWdt * deltaT)


@cuda.jit(fastmath=FAST_MATH)
def interp_UVW(U0, V0, W0, U1, V1, W1, Un, Vn, Wn, dt, fraction=0.1, N=0):
    global_idx, global_idy, global_idz = cuda.grid(3)

    gridDimX, gridDimY, gridDimZ = cuda.gridsize(3)

    tm = REAL_CP(dt / 2.)
    t_target = REAL_CP(fraction * N * dt)
    dt_inv = REAL_CP(1./dt)

    Nx, Ny, Nz = U0.shape

    for ii in range(global_idx, Nx, gridDimX):
        for jj in range(global_idy, Ny, gridDimY):
            for kk in range(global_idz, Nz, gridDimZ):
                Um = REAL_CP(REAL_CP(0.5) * (U0[ii, jj, kk] + U1[ii, jj, kk]))
                Vm = REAL_CP(REAL_CP(0.5) * (V0[ii, jj, kk] + V1[ii, jj, kk]))
                Wm = REAL_CP(REAL_CP(0.5) * (W0[ii, jj, kk] + W1[ii, jj, kk]))

                dUdt = REAL_CP((U1[ii, jj, kk] - U0[ii, jj, kk]) * dt_inv)
                dVdt = REAL_CP((V1[ii, jj, kk] - V0[ii, jj, kk]) * dt_inv)
                dWdt = REAL_CP((W1[ii, jj, kk] - W0[ii, jj, kk]) * dt_inv)

                deltaT = REAL_CP((t_target - tm))

                Un[ii, jj, kk] = REAL_CP(Um + dUdt * deltaT)
                Vn[ii, jj, kk] = REAL_CP(Vm + dVdt * deltaT)
                Wn[ii, jj, kk] = REAL_CP(Wm + dWdt * deltaT)


@cuda.jit(fastmath=FAST_MATH, device = True, inline = True)
def max_cube(Uc):
    U000 = Uc[0,0,0]
    U100 = Uc[1,0,0]
    U010 = Uc[0,1,0]
    U001 = Uc[0,0,1]
    U110 = Uc[1,1,0]
    U011 = Uc[0,1,1]
    U101 = Uc[1,0,1]
    U111 = Uc[1,1,1]
    return max(U000, U100, U010, U001, U110, U011, U101, U111)


@cuda.jit(fastmath=FAST_MATH, device = True, inline = True)
def min_cube(Uc):
    U000 = Uc[0,0,0]
    U100 = Uc[1,0,0]
    U010 = Uc[0,1,0]
    U001 = Uc[0,0,1]
    U110 = Uc[1,1,0]
    U011 = Uc[0,1,1]
    U101 = Uc[1,0,1]
    U111 = Uc[1,1,1]
    return min(U000, U100, U010, U001, U110, U011, U101, U111)

@cuda.jit(fastmath=FAST_MATH, device = True, inline = True)
def minmax_cube(Uc):
    U000 = Uc[0,0,0]
    U100 = Uc[1,0,0]
    U010 = Uc[0,1,0]
    U001 = Uc[0,0,1]
    U110 = Uc[1,1,0]
    U011 = Uc[0,1,1]
    U101 = Uc[1,0,1]
    U111 = Uc[1,1,1]
    mini = min(U000, U100, U010, U001, U110, U011, U101, U111)
    maxi = max(U000, U100, U010, U001, U110, U011, U101, U111)
    return mini, maxi


@cuda.jit(fastmath=FAST_MATH, device = True, inline = True)
def clip(v, minimum, maximum):
    return max(min(maximum, v), minimum)


@cuda.jit(fastmath=FAST_MATH, device = True, inline = True)
def inner_trilinear_interpolate(Uc, dx, dy, dz):
    dxdy = dx * dy
    dxdz = dx * dz
    dydz = dy * dz
    dxdydz = dx * dy * dz

    mini, maxi = minmax_cube(Uc)

    U000 = Uc[0,0,0]
    U100 = Uc[1,0,0]
    U010 = Uc[0,1,0]
    U001 = Uc[0,0,1]
    U110 = Uc[1,1,0]
    U011 = Uc[0,1,1]
    U101 = Uc[1,0,1]
    U111 = Uc[1,1,1]

    c0u = (U000)
    c1u = (U100 - U000)
    c2u = (U010 - U000)
    c3u = (U001 - U000)
    c4u = (U110 - U010 - U100 + U000)
    c5u = (U011 - U001 - U010 + U000)
    c6u = (U101 - U001 - U100 + U000)
    c7u = (U111 - U011 - U101 - U110 + U100 + U001 + U010 - U000)

    return clip(c0u + c1u * dx + c2u * dy + c3u * dz + c4u * dxdy + c5u * dydz + c6u * dxdz + c7u * dxdydz, mini, maxi) # 80


@cuda.jit(fastmath=FAST_MATH, device = True)
def gpu_nns(x, y, z, xp, yp, zp, coarse_levels):
    Nx, Ny, Nz = x.shape

    COARSE_FACTOR = ((2)**coarse_levels)

    best_x = 0
    best_y = 0
    best_z = 0

    distance = float32(1e6)
    for ii in range((COARSE_FACTOR), (Nx-COARSE_FACTOR), (COARSE_FACTOR//2)):
        for jj in range((COARSE_FACTOR), (Ny-COARSE_FACTOR), (COARSE_FACTOR//2)):
            for kk in range((COARSE_FACTOR), (Nz-COARSE_FACTOR), (COARSE_FACTOR//2)):
                current_distance = fsqrt_rn(
                    (x[ii,jj,kk] - xp)*(x[ii,jj,kk] - xp) + 
                    (y[ii,jj,kk] - yp)*(y[ii,jj,kk] - yp) + 
                    (z[ii,jj,kk] - zp)*(z[ii,jj,kk] - zp)
                )
                if (current_distance <= distance):
                    best_x = (ii)
                    best_y = (jj) 
                    best_z = (kk) 
                    distance = current_distance 
    while COARSE_FACTOR > (1):
        for ii in range((best_x - COARSE_FACTOR), (best_x + COARSE_FACTOR), (COARSE_FACTOR//2)):
            for jj in range((best_y - COARSE_FACTOR), (best_y + COARSE_FACTOR), (COARSE_FACTOR//2)):
                for kk in range((best_z - COARSE_FACTOR), (best_z + COARSE_FACTOR), (COARSE_FACTOR//2)):
                    current_distance = fsqrt_rn(
                        (x[ii,jj,kk] - xp)*(x[ii,jj,kk] - xp) + 
                        (y[ii,jj,kk] - yp)*(y[ii,jj,kk] - yp) + 
                        (z[ii,jj,kk] - zp)*(z[ii,jj,kk] - zp)
                    )
                    if (current_distance <= distance):
                        best_x = (ii)
                        best_y = (jj) 
                        best_z = (kk) 
                        distance = current_distance     
        COARSE_FACTOR = (COARSE_FACTOR // 2)
    return CUDA_INT(best_x), CUDA_INT(best_y), CUDA_INT(best_z)


@cuda.jit(fastmath=FAST_MATH)
def get_particle_velocity_with_integrated_nns(x, y, z, xp, yp, zp, U, V, W, dt):
    local_idx = cuda.threadIdx.x
    global_idx = cuda.grid(1)
    blockDim = cuda.blockDim.x
    gridDim = cuda.gridsize(1) #cuda.gridDim.x * cuda.blockDim.x

    NParticles = xp.size
    Nx, Ny, Nz = x.shape
    SMALLEST_DIM = min(Nx, Ny, Nz)
    P = 1
    while (2**(P+1)) < (SMALLEST_DIM//2):
        P+=1
    for particle in range(global_idx, NParticles, gridDim):
        xp_local = REAL_CP(xp[particle])
        yp_local = REAL_CP(yp[particle])
        zp_local = REAL_CP(zp[particle])

        i, j, k = gpu_nns(x, y, z, xp_local, yp_local, zp_local, P)

        i -= 1 * ((xp_local <= x[i,j,k]))
        j -= 1 * ((yp_local <= y[i,j,k]))
        k -= 1 * ((zp_local <= z[i,j,k]) and (zp_local >= z[i,j,max(0, k-1)]))

        in_domain = (0 <= i) and (i < Nx - 1) and (0 <= j) and (j < Ny - 1) and (0 <= k) and (k < Nz - 1)
        i = clip(i, 0, Nx - 2)
        j = clip(j, 0, Ny - 2)
        k = clip(k, 0, Nz - 2)

        xc = (x[i:i+2, j:j+2, k:k+2])
        yc = (y[i:i+2, j:j+2, k:k+2])
        zc = (z[i:i+2, j:j+2, k:k+2])

        minX, maxX = minmax_cube(xc)

        minY, maxY = minmax_cube(yc)
        
        minZ, maxZ = minmax_cube(zc)

        dx = fast_fdividef(xp_local - minX, maxX - minX)
        dy = fast_fdividef(yp_local - minY, maxY - minY)
        dz = fast_fdividef(zp_local - minZ, maxZ - minZ)

        Uc = (U[i:i+2, j:j+2, k:k+2])
        Vc = (V[i:i+2, j:j+2, k:k+2])
        Wc = (W[i:i+2, j:j+2, k:k+2])

        Up = REAL_CP(inner_trilinear_interpolate(Uc, dx, dy, dz)) * REAL_CP(in_domain) + REAL_CP(not in_domain) * REAL_CP(U[i, j, k])
        Vp = REAL_CP(inner_trilinear_interpolate(Vc, dx, dy, dz)) * REAL_CP(in_domain) + REAL_CP(not in_domain) * REAL_CP(V[i, j, k])
        Wp = REAL_CP(inner_trilinear_interpolate(Wc, dx, dy, dz)) * REAL_CP(in_domain) + REAL_CP(not in_domain) * REAL_CP(W[i, j, k])

        xp[particle] = fmaf_rn(dt, Up, xp_local) #xp_local + REAL_CP(dt * Up)
        yp[particle] = fmaf_rn(dt, Vp, yp_local) #yp_local + REAL_CP(dt * Vp)
        zp[particle] = fmaf_rn(dt, Wp, zp_local) #zp_local + REAL_CP(dt * Wp)


def update_particle_velocity(
    point: Tuple[np.ndarray, np.ndarray, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    is_2d: bool = False,
    stream: cuda.stream = cuda.default_stream(),
    dt: float = None,
):
    xtarget, ytarget, ztarget = point

    get_particle_velocity_with_integrated_nns[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        x, 
        y, 
        z, 
        xtarget, 
        ytarget, 
        ztarget, 
        U, 
        V,
        W, 
        REAL_CP(dt)
    )
    stream.synchronize()
    return


@cuda.jit(fastmath=FAST_MATH)
def step_particles_euler(
    Up, Vp, Wp, xp, yp, zp, dt, maxz=0.0, minz=0.0
):
    global_idx = cuda.grid(1)
    gridDim = cuda.gridsize(1) #cuda.gridDim.x * cuda.blockDim.x

    NParticles = xp.size
    for particle in range(global_idx, NParticles, gridDim):
        xp[particle] = REAL_CP(dt * Up[particle] + xp[particle])
        yp[particle] = REAL_CP(dt * Vp[particle] + yp[particle])
        zp[particle] = REAL_CP(dt * Wp[particle] + zp[particle])


def particle_first_step(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dt: float,
    mask: List[int], 
    step_particles: Callable,
    is_2d: bool,
    newx: np.ndarray, 
    newy: np.ndarray,
    newz: np.ndarray,
    stream: cuda.stream = cuda.default_stream()
):
    with stream.auto_synchronize():
        N = len(mask) - 1

        xp = cuda.to_device(np.ascontiguousarray(newx.reshape((-1,))[mask[0]:mask[-1]]), stream)
        yp = cuda.to_device(np.ascontiguousarray(newy.reshape((-1,))[mask[0]:mask[-1]]), stream)
        zp = cuda.to_device(np.ascontiguousarray(newz.reshape((-1,))[mask[0]:mask[-1]]), stream)

        update_particle_velocity(
            (xp, yp, zp), x, y, z, U, V, W, is_2d, stream, dt
        )
        #step_particles_euler[BLOCKS_PER_GRID, THREADS_PER_BLOCK, stream](Up, Vp, Wp, xp, yp, zp, dt, maxz, minz)
    return xp, yp, zp


def particle_update_velocities(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dt: float,
    xp: np.ndarray,
    yp: np.ndarray,
    zp: np.ndarray,
    step_particles: Callable,
    is_2d: bool = False,
    stream: cuda.stream = cuda.default_stream()
):
    N = xp.size
    with stream.auto_synchronize():
        update_particle_velocity(
            (xp, yp, zp), x, y, z, U, V, W, is_2d, stream, dt
        )
        #step_particles_euler[BLOCKS_PER_GRID, THREADS_PER_BLOCK, stream](Up, Vp, Wp, xp, yp, zp, dt, maxz, minz)
    return xp, yp, zp


def write_particles_to_file(
    h5file: h5py.File,
    xp: cuda.device_array,
    yp: cuda.device_array,
    zp: cuda.device_array,
    timestep: int,
    mask: List[int]
): 
    print(f"Writing timestep: # {timestep}")
    sys.stdout.flush()
    dset_1 = h5file["Particle_xyz"]
    #with dset_1.collective:
    dset_1[timestep, 0, mask[0]:mask[-1]] = np.array(xp)
    dset_1[timestep, 1, mask[0]:mask[-1]] = np.array(yp)
    dset_1[timestep, 2, mask[0]:mask[-1]] = np.array(zp)
    return


@njit(parallel=True, fastmath=False, nogil=True)
def __irregular_mesh_refine(x, y, z, ref_x, ref_y, ref_z):
    Nx, Ny, Nz = x.shape
    
    newx = np.zeros((Nx * ref_x, Ny * ref_y, Nz*ref_z), dtype = x.dtype) + np.min(x)
    newy = np.zeros((Nx * ref_x, Ny * ref_y, Nz*ref_z), dtype = y.dtype) + np.min(y)
    newz = np.zeros((Nx * ref_x, Ny * ref_y, Nz*ref_z), dtype = z.dtype) + np.min(z)
    for ii in prange(1, Nx+1):
        for jj in range(1, Ny+1):
            for kk in range(1, Nz+1):
                for io in range(ref_x):
                    for jo in range(ref_y):
                        for ko in range(ref_z):
                            idx = (ii - 1)*ref_x + io
                            idy = (jj - 1)*ref_y + jo
                            idz = (kk - 1)*ref_z + ko
                            
                            xcube = x[max(0, ii-1):min(Nx, ii+1), max(0, jj-1):min(Ny, jj+1), max(0, kk-1):min(Nz, kk+1)]
                            ycube = y[max(0, ii-1):min(Nx, ii+1), max(0, jj-1):min(Ny, jj+1), max(0, kk-1):min(Nz, kk+1)]
                            zcube = z[max(0, ii-1):min(Nx, ii+1), max(0, jj-1):min(Ny, jj+1), max(0, kk-1):min(Nz, kk+1)]

                            dx = (io) / (ref_x)
                            dy = (jo) / (ref_y)
                            dz = (ko) / (ref_z)

                            Uc = xcube
                            Vc = ycube
                            Wc = zcube
                            
                            c0u = (Uc[0,0,0])
                            c1u = (Uc[-1,0,0] - Uc[0,0,0])
                            c2u = (Uc[0,-1,0] - Uc[0,0,0])
                            c3u = (Uc[0,0,-1] - Uc[0,0,0])
                            c4u = (Uc[-1,-1,0] - Uc[0,-1,0] - Uc[-1,0,0] + Uc[0,0,0])
                            c5u = (Uc[0,-1,-1] - Uc[0,0,-1] - Uc[0,-1,0] + Uc[0,0,0])
                            c6u = (Uc[-1,0,-1] - Uc[0,0,-1] - Uc[-1,0,0] + Uc[0,0,0])
                            c7u = (Uc[-1,-1,-1] - Uc[0,-1,-1] - Uc[-1,0,-1] - Uc[-1,-1,0] + Uc[-1,0,0] + Uc[0,0,-1] + Uc[0,-1,0] - Uc[0,0,0])

                            c0v = (Vc[0,0,0])
                            c1v = (Vc[-1,0,0] - Vc[0,0,0])
                            c2v = (Vc[0,-1,0] - Vc[0,0,0])
                            c3v = (Vc[0,0,-1] - Vc[0,0,0])
                            c4v = (Vc[-1,-1,0] - Vc[0,-1,0] - Vc[-1,0,0] + Vc[0,0,0])
                            c5v = (Vc[0,-1,-1] - Vc[0,0,-1] - Vc[0,-1,0] + Vc[0,0,0])
                            c6v = (Vc[-1,0,-1] - Vc[0,0,-1] - Vc[-1,0,0] + Vc[0,0,0])
                            c7v = (Vc[-1,-1,-1] - Vc[0,-1,-1] - Vc[-1,0,-1] - Vc[-1,-1,0] + Vc[-1,0,0] + Vc[0,0,-1] + Vc[0,-1,0] - Vc[0,0,0])

                            c0w = (Wc[0,0,0])
                            c1w = (Wc[-1,0,0] - Wc[0,0,0])
                            c2w = (Wc[0,-1,0] - Wc[0,0,0])
                            c3w = (Wc[0,0,-1] - Wc[0,0,0])
                            c4w = (Wc[-1,-1,0] - Wc[0,-1,0] - Wc[-1,0,0] + Wc[0,0,0])
                            c5w = (Wc[0,-1,-1] - Wc[0,0,-1] - Wc[0,-1,0] + Wc[0,0,0])
                            c6w = (Wc[-1,0,-1] - Wc[0,0,-1] - Wc[-1,0,0] + Wc[0,0,0])
                            c7w = (Wc[-1,-1,-1] - Wc[0,-1,-1] - Wc[-1,0,-1] - Wc[-1,-1,0] + Wc[-1,0,0] + Wc[0,0,-1] + Wc[0,-1,0] - Wc[0,0,0])

                            newx[idx,idy,idz] = c0u + c1u * dx + c2u * dy + c3u * dz + c4u * dx * dy + c5u * dy * dz + c6u * dz * dx + c7u * dx * dy * dz
                            newy[idx,idy,idz] = c0v + c1v * dx + c2v * dy + c3v * dz + c4v * dx * dy + c5v * dy * dz + c6v * dz * dx + c7v * dx * dy * dz
                            newz[idx,idy,idz] = c0w + c1w * dx + c2w * dy + c3w * dz + c4w * dx * dy + c5w * dy * dz + c6w * dz * dx + c7w * dx * dy * dz
    return newx, newy, newz


def refine_mesh(
    x, y, z, ref_x = 10, ref_y = 10, ref_z = 1, regular_mesh=False
):
    Nx, Ny, Nz = x.shape

    if regular_mesh:
        min_x = np.min(x)
        min_y = np.min(y)
        max_x = np.max(x)
        max_y = np.max(y)
        min_z = np.min(z)
        max_z = np.max(z)
        newx, newy, newz = np.meshgrid(
            np.linspace(min_x, max_x, num=Nx * ref_x, endpoint=True),
            np.linspace(min_y, max_y, num=Ny * ref_y, endpoint=True),
            np.linspace(min_z, max_z, num=Nz * ref_z, endpoint=True),
            indexing='ij'
        )
    else:
        newx, newy, newz = __irregular_mesh_refine(x, y, z, ref_x, ref_y, ref_z)
    return newx.astype(x.dtype), newy.astype(x.dtype), newz.astype(x.dtype)


def particle_simulator(
    filenames: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dt: float,
    output_filename: str,
    upscale: int = 10,
    make_2d_simulation: bool = False,
    refinement_parameters: Tuple[int, int, int] = (1, 1, 1),
    write_all_time_steps: bool = True,
    number_of_time_steps=-1,
    initial_time_step=0,
):

    comm = MPI.COMM_WORLD
    my_rank = comm.rank
    total_ranks = comm.size
    
    stream = cuda.default_stream()

    print("Creating output HDF5 files...")
    h5Out = h5py.File(output_filename, "w", driver='mpio', comm=MPI.COMM_WORLD)

    #filenames = sorted(filenames)
    U, V, W = read_UVW(filenames[0], gen_2d=make_2d_simulation)

    newx, newy, newz = refine_mesh(
        x, y, z, refinement_parameters[0], refinement_parameters[1], refinement_parameters[2]
    )
    Nx, Ny, Nz = newy.shape
    TotalElements = Nx * Ny * Nz

    localN = -(TotalElements // -total_ranks)
    
    MyIndexList = [ii for ii in range(my_rank*localN, min((my_rank+1)*localN, TotalElements))]
    print(f"    * Rank {my_rank} to process {len(MyIndexList)} particles out of {TotalElements} in the range: [{MyIndexList[0]}, {MyIndexList[-1]})")
    print(f"Allocating HDF5 datasets for {TotalElements} particles...")
    sys.stdout.flush()

    h5Out.create_dataset(
        f"refined_x",
        data = newx
    )
    h5Out.create_dataset(
        f"refined_y",
        data = newy
    )
    h5Out.create_dataset(
        f"refined_z",
        data = newz
    )
    filenames = filenames[initial_time_step:(initial_time_step+number_of_time_steps)]
    if write_all_time_steps:
        length_of_files = len(filenames) + 1
    else:
        length_of_files = 2
    h5Out.create_dataset(
        f"Particle_xyz",
        shape=(length_of_files, 3, TotalElements),
        dtype=REAL_NP,
    )
    #h5Out.flush()
    timestep = 0
    write_particles_to_file(
        h5Out,
        newx.ravel()[MyIndexList[0]:MyIndexList[-1]],
        newy.ravel()[MyIndexList[0]:MyIndexList[-1]],
        newz.ravel()[MyIndexList[0]:MyIndexList[-1]],
        timestep,
        MyIndexList
    )
    with stream.auto_synchronize():
        d_U = cuda.to_device(np.ascontiguousarray(U), stream)
        d_V = cuda.to_device(np.ascontiguousarray(V), stream)
        d_W = cuda.to_device(np.ascontiguousarray(W), stream)

        d_x = cuda.to_device(np.ascontiguousarray(x), stream)
        d_y = cuda.to_device(np.ascontiguousarray(y), stream)
        d_z = cuda.to_device(np.ascontiguousarray(z), stream)

        d_U0 = cuda.device_array(U.shape, dtype=REAL_NP, stream=stream)
        d_V0 = cuda.device_array(V.shape, dtype=REAL_NP, stream=stream)
        d_W0 = cuda.device_array(W.shape, dtype=REAL_NP, stream=stream)
        d_Un = cuda.device_array(U.shape, dtype=REAL_NP, stream=stream)
        d_Vn = cuda.device_array(V.shape, dtype=REAL_NP, stream=stream)
        d_Wn = cuda.device_array(W.shape, dtype=REAL_NP, stream=stream)

        print("Initializing the simulation and allocating additional arrays...")
        xp, yp, zp = particle_first_step(
            d_U, 
            d_V, 
            d_W, 
            d_x, 
            d_y, 
            d_z, 
            dt/upscale, 
            MyIndexList, 
            step_particles_euler, 
            make_2d_simulation, 
            newx, 
            newy, 
            newz, 
            stream
        )
    timestep += 1
    if write_all_time_steps:
        write_particles_to_file(h5Out, xp, yp, zp, timestep, MyIndexList)
    reader = lambda filename: read_UVW(filename, gen_2d=make_2d_simulation)
    with ProgressBar(max_value=len(filenames)) as bar, Executor(1) as pool:
        FRACTION = 1./upscale
        TIMER_LEN = len(filenames[1:]) - 1
        for ii, (U, V, W) in enumerate(pool.map(read_UVW, filenames[1:])):
            if (ii % TIMER_LEN) == 0:
                gc.collect()
                bar.update(ii)
            with stream.auto_synchronize():
                d_U0, d_U = d_U, d_U0
                d_V0, d_V = d_V, d_V0
                d_W0, d_W = d_W, d_W0
                d_U.copy_to_device(np.ascontiguousarray(U), stream)
                d_V.copy_to_device(np.ascontiguousarray(V), stream)
                d_W.copy_to_device(np.ascontiguousarray(W), stream)
                for N in range(1, upscale):
                    interp_UVW_flat[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
                        d_U0.reshape(-1), d_V0.reshape(-1), d_W0.reshape(-1), 
                        d_U.reshape(-1), d_V.reshape(-1), d_W.reshape(-1), 
                        d_Un.reshape(-1), d_Vn.reshape(-1), d_Wn.reshape(-1), 
                        dt, FRACTION, N
                    )
                    xp, yp, zp = particle_update_velocities(
                        d_Un, d_Vn, d_Wn, d_x, d_y, d_z, dt/upscale, xp, yp, zp, step_particles_euler, make_2d_simulation, stream
                    )
                    #bar.update(ii)
                xp, yp, zp = particle_update_velocities(
                    d_U, d_V, d_W, d_x, d_y, d_z, dt/upscale, xp, yp, zp, step_particles_euler, make_2d_simulation, stream
                )
            timestep += 1
            if write_all_time_steps:
                write_particles_to_file(h5Out, xp, yp, zp, timestep, MyIndexList)
        if not write_all_time_steps:
            write_particles_to_file(h5Out, xp, yp, zp, 1, MyIndexList)
            bar.update(ii+1)
    h5Out.close()
    return xp, yp, zp


@cuda.jit(fastmath=FAST_MATH, device=True, inline=True)
def power_iteration_3d(A, b_k, b_k1):
    b_k[0] = 0.801784
    b_k[1] = 0.534522
    b_k[2] = -0.267261

    bkp_0 = b_k[0]
    bkp_1 = b_k[1]
    bkp_2 = b_k[2]

    difference = 100

    while difference > 1e-6:
        b_k1[0] = A[0, 0] * b_k[0] + A[0, 1] * b_k[1] + A[0, 2] * b_k[2]
        b_k1[1] = A[1, 0] * b_k[0] + A[1, 1] * b_k[1] + A[1, 2] * b_k[2]
        b_k1[2] = A[2, 0] * b_k[0] + A[2, 1] * b_k[1] + A[2, 2] * b_k[2]

        b_k1_norm_inv = 1./sqrt(b_k1[0] * b_k1[0] + b_k1[1] * b_k1[1] + b_k1[2] * b_k1[2])

        bkp_0 = b_k[0]
        bkp_1 = b_k[1]
        bkp_2 = b_k[2]

        b_k[0] = b_k1[0] * b_k1_norm_inv
        b_k[1] = b_k1[1] * b_k1_norm_inv
        b_k[2] = b_k1[2] * b_k1_norm_inv

        difference = fsqrt_rn((bkp_0 - b_k[0])*(bkp_0 - b_k[0]) + (bkp_1 - b_k[1])*(bkp_1 - b_k[1]) + (bkp_2 - b_k[2])*(bkp_2 - b_k[2]))
    rho = (b_k1[0]* b_k[0] + b_k1[1]* b_k[1] + b_k1[2] * b_k[2]) / (b_k[0]* b_k[0] + b_k[1]* b_k[1] + b_k[2] * b_k[2])
    return rho


@cuda.jit(fastmath=FAST_MATH, device=True, inline=True)
def power_iteration_2d(A, b_k, b_k1):
    # 0.81901029, 0.57377882
    b_k[0] = 0.81901029
    b_k[1] = 0.57377882
    rho_k = 1024.
    for _ in range(100000):
        b_k1[0] = A[0, 0] * b_k[0] + A[0, 1] * b_k[1]
        b_k1[1] = A[1, 0] * b_k[0] + A[1, 1] * b_k[1]
        
        b_k1_norm_inv = 1./sqrt(b_k1[0] * b_k1[0] + b_k1[1] * b_k1[1])

        b_k[0] = b_k1[0] * b_k1_norm_inv
        b_k[1] = b_k1[1] * b_k1_norm_inv
        rho = (b_k1[0]* b_k[0] + b_k1[1]* b_k[1]) / (b_k[0]* b_k[0] + b_k[1]* b_k[1])
        if abs(rho - rho_k) < 1.192093e-07:
            break
        rho_k = rho
    return rho


@cuda.jit(fastmath=FAST_MATH, device=True, inline=True)
def device_transposed_matmul_and_make_symmetric_2d(J, C):
    # C = J.T @ J
    for ii in range(2):
        for jj in range(2):
            C[ii,jj] = REAL_CP(0.)
            for kk in range(2):
                C[ii,jj] += J[kk,ii] * J[kk,jj]
    # C = (C + C.T)/2.
    for ii in range(2):
        for jj in range(ii, 2):
            C[jj,ii] = REAL_CP(0.5) * (C[ii,jj] + C[jj,ii])
            C[ii,jj] = C[jj,ii]
    return C


@cuda.jit(fastmath=FAST_MATH, device=True, inline=True)
def device_transposed_matmul_and_make_symmetric_3d(J, C):
    # C = J.T @ J
    for ii in range(3):
        for jj in range(3):
            C[ii,jj] = REAL_CP(0.)
            for kk in range(3):
                C[ii,jj] += J[kk,ii] * J[kk,jj]
    # C = (C + C.T)/2.
    for ii in range(3):
        for jj in range(ii, 3):
            C[jj,ii] = REAL_CP(0.5) * (C[ii,jj] + C[jj,ii])
            C[ii,jj] = C[jj,ii]
    return C


@cuda.jit(fastmath=FAST_MATH)
def calculate_particle_ftle_and_fsle(xp0, yp0, zp0, xp1, yp1, zp1, ftle, fsle, T):
    EPS = REAL_CP(1.192093e-07)
    Nx, Ny, Nz = xp0.shape
    dz = abs(zp0[0,0,1] - zp0[0,0,0])

    sp0 = xp0 #np.zeros_like(xp0)
    np0 = yp0 #np.zeros_like(yp0)

    T_inv = REAL_CP(REAL_CP(1.)/REAL_CP(T))

    global_idx, global_idy, global_idz = cuda.grid(3)

    gridDimX, gridDimY, gridDimZ = cuda.gridsize(3)

    distance_mean_factor = REAL_CP(0.038461538461538)

    J = cuda.local.array((3, 3), REAL_CP)
    _C = cuda.local.array((3, 3), REAL_CP)
    b_k = cuda.local.array((3,), REAL_CP)
    b_k1 = cuda.local.array((3,), REAL_CP)
    for ii in range(global_idx, Nx, gridDimX):
        for jj in range(global_idy, Ny, gridDimY):
            for kk in range(global_idz, Nz, gridDimZ):
                # X Derivatives
                if ii == 0:
                    J[0, 0] = (xp1[ii+1,jj,kk] - xp1[ii,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii,jj,kk] + EPS)
                    J[1, 0] = (yp1[ii+1,jj,kk] - yp1[ii,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii,jj,kk] + EPS)
                    J[2, 0] = (zp1[ii+1,jj,kk] - zp1[ii,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii,jj,kk] + EPS)
                elif ii == Nx - 1:
                    J[0, 0] = (xp1[ii,jj,kk] - xp1[ii-1,jj,kk]) / (sp0[ii,jj,kk] - sp0[ii-1,jj,kk] + EPS)
                    J[1, 0] = (yp1[ii,jj,kk] - yp1[ii-1,jj,kk]) / (sp0[ii,jj,kk] - sp0[ii-1,jj,kk] + EPS)
                    J[2, 0] = (zp1[ii,jj,kk] - zp1[ii-1,jj,kk]) / (sp0[ii,jj,kk] - sp0[ii-1,jj,kk] + EPS)
                else:
                    J[0, 0] = (xp1[ii+1,jj,kk] - xp1[ii-1,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii-1,jj,kk] + EPS)
                    J[1, 0] = (yp1[ii+1,jj,kk] - yp1[ii-1,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii-1,jj,kk] + EPS)
                    J[2, 0] = (zp1[ii+1,jj,kk] - zp1[ii-1,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii-1,jj,kk] + EPS)

                # Y Derivatives
                if jj == 0:
                    J[0, 1] = (xp1[ii,jj+1,kk] - xp1[ii,jj,kk]) / (np0[ii,jj+1,kk] - np0[ii,jj,kk] + EPS)
                    J[1, 1] = (yp1[ii,jj+1,kk] - yp1[ii,jj,kk]) / (np0[ii,jj+1,kk] - np0[ii,jj,kk] + EPS)
                    J[2, 1] = (zp1[ii,jj+1,kk] - zp1[ii,jj,kk]) / (np0[ii,jj+1,kk] - np0[ii,jj,kk] + EPS)
                elif jj == Ny - 1:
                    J[0, 1] = (xp1[ii,jj,kk] - xp1[ii,jj-1,kk]) / (np0[ii,jj,kk] - np0[ii,jj-1,kk] + EPS)
                    J[1, 1] = (yp1[ii,jj,kk] - yp1[ii,jj-1,kk]) / (np0[ii,jj,kk] - np0[ii,jj-1,kk] + EPS)
                    J[2, 1] = (zp1[ii,jj,kk] - zp1[ii,jj-1,kk]) / (np0[ii,jj,kk] - np0[ii,jj-1,kk] + EPS)
                else:
                    h1 = (np0[ii,jj+1,kk] - np0[ii,jj,kk])
                    h2 = (np0[ii,jj,kk] - np0[ii,jj-1,kk])
                    alpha = h1 / h2
                    dy = h1 * (1. + alpha) + EPS
                    J[0, 1] = (xp1[ii,jj+1,kk] - alpha**2 * xp1[ii,jj-1,kk] - (1 - alpha**2) * xp1[ii,jj,kk]) / dy
                    J[1, 1] = (yp1[ii,jj+1,kk] - alpha**2 * yp1[ii,jj-1,kk] - (1 - alpha**2) * yp1[ii,jj,kk]) / dy
                    J[2, 1] = (zp1[ii,jj+1,kk] - alpha**2 * zp1[ii,jj-1,kk] - (1 - alpha**2) * zp1[ii,jj,kk]) / dy
                # Z Derivatives
                if kk == 0:
                    J[0, 2] = (xp1[ii,jj,1] - xp1[ii,jj,Nz - 1]) / (2 * dz + EPS)
                    J[1, 2] = (yp1[ii,jj,1] - yp1[ii,jj,Nz - 1]) / (2 * dz + EPS)
                    J[2, 2] = (zp1[ii,jj,kk+1] - zp1[ii,jj,kk]) / (dz + EPS)
                elif kk == Nz - 1:
                    J[0, 2] = (xp1[ii,jj,0] - xp1[ii,jj,kk-1]) / (2 * dz + EPS)
                    J[1, 2] = (yp1[ii,jj,0] - yp1[ii,jj,kk-1]) / (2 * dz + EPS)
                    J[2, 2] = (zp1[ii,jj,kk] - zp1[ii,jj,kk-1]) / (dz + EPS)
                else:
                    J[0, 2] = (xp1[ii,jj,kk+1] - xp1[ii,jj,kk-1]) / (2 * dz + EPS)
                    J[1, 2] = (yp1[ii,jj,kk+1] - yp1[ii,jj,kk-1]) / (2 * dz + EPS)
                    J[2, 2] = (zp1[ii,jj,kk+1] - zp1[ii,jj,kk-1]) / (2 * dz + EPS)

                if (0 < ii) and (ii < Nx - 1) and (0 < jj) and (jj < Ny - 1) and (0 < kk) and (kk < Nz - 1):
                    for _i in range(-1, 2):
                        for _j in range(-1, 2):
                            for _k in range(-1, 2):
                                if (_i != 0) or (_j != 0) or (_k != 0):
                                    d0 = fsqrt_rn(
                                        (xp0[ii+_i,jj+_j,kk+_k] - xp0[ii,jj,kk])*(xp0[ii+_i,jj+_j,kk+_k] - xp0[ii,jj,kk]) +
                                        (yp0[ii+_i,jj+_j,kk+_k] - yp0[ii,jj,kk])*(yp0[ii+_i,jj+_j,kk+_k] - yp0[ii,jj,kk]) +
                                        (zp0[ii+_i,jj+_j,kk+_k] - zp0[ii,jj,kk])*(zp0[ii+_i,jj+_j,kk+_k] - zp0[ii,jj,kk])
                                    )
                                    d1 = fsqrt_rn(
                                        (xp1[ii+_i,jj+_j,kk+_k] - xp1[ii,jj,kk])*(xp1[ii+_i,jj+_j,kk+_k] - xp1[ii,jj,kk]) +
                                        (yp1[ii+_i,jj+_j,kk+_k] - yp1[ii,jj,kk])*(yp1[ii+_i,jj+_j,kk+_k] - yp1[ii,jj,kk]) +
                                        (zp1[ii+_i,jj+_j,kk+_k] - zp1[ii,jj,kk])*(zp1[ii+_i,jj+_j,kk+_k] - zp1[ii,jj,kk])
                                    )
                                    fsle[ii, jj, kk] += distance_mean_factor * T_inv * fast_logf(d1 / d0)
                _C = device_transposed_matmul_and_make_symmetric_3d(J, _C)
                ftle[ii, jj, kk] = fast_logf( fsqrt_rn( power_iteration_3d(_C, b_k, b_k1) )) * T_inv


@cuda.jit(fastmath=FAST_MATH)
def calculate_particle_ftle_2d(xp0, yp0, zp0, xp1, yp1, zp1, C, N, dt):
    EPS = REAL_CP(1e-6)
    Nx, Ny, Nz = xp0.shape

    sp0 = xp0 #np.zeros_like(xp0)
    np0 = yp0 #np.zeros_like(yp0)

    global_idx, global_idy = cuda.grid(2)

    gridDimX, gridDimY = cuda.gridsize(2) #cuda.gridDim.x * cuda.blockDim.x
    #gridDimY = #cuda.gridDim.y * cuda.blockDim.y

    T = (N * dt)

    kk = 0
    J = cuda.local.array((2, 2), REAL_CP)
    _C = cuda.local.array((2, 2), REAL_CP)
    for ii in range(global_idx, Nx, gridDimX):
        for jj in range(global_idy, Ny, gridDimY):
            # X Derivatives
            if ii == 0:
                J[0, 0] = (xp1[ii+1,jj,kk] - xp1[ii,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii,jj,kk] + EPS)
                J[1, 0] = (yp1[ii+1,jj,kk] - yp1[ii,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii,jj,kk] + EPS)
            elif ii == Nx - 1:
                J[0, 0] = (xp1[ii,jj,kk] - xp1[ii-1,jj,kk]) / (sp0[ii,jj,kk] - sp0[ii-1,jj,kk] + EPS)
                J[1, 0] = (yp1[ii,jj,kk] - yp1[ii-1,jj,kk]) / (sp0[ii,jj,kk] - sp0[ii-1,jj,kk] + EPS)
            else:
                J[0, 0] = (xp1[ii+1,jj,kk] - xp1[ii-1,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii-1,jj,kk] + EPS)
                J[1, 0] = (yp1[ii+1,jj,kk] - yp1[ii-1,jj,kk]) / (sp0[ii+1,jj,kk] - sp0[ii-1,jj,kk] + EPS)

            # Y Derivatives
            if jj == 0:
                J[0, 1] = (xp1[ii,jj+1,kk] - xp1[ii,jj,kk]) / (np0[ii,jj+1,kk] - np0[ii,jj,kk] + EPS)
                J[1, 1] = (yp1[ii,jj+1,kk] - yp1[ii,jj,kk]) / (np0[ii,jj+1,kk] - np0[ii,jj,kk] + EPS)
            elif jj == Ny - 1:
                J[0, 1] = (xp1[ii,jj,kk] - xp1[ii,jj-1,kk]) / (np0[ii,jj,kk] - np0[ii,jj-1,kk] + EPS)
                J[1, 1] = (yp1[ii,jj,kk] - yp1[ii,jj-1,kk]) / (np0[ii,jj,kk] - np0[ii,jj-1,kk] + EPS)
            else:
                h1 = (np0[ii,jj+1,kk] - np0[ii,jj,kk])
                h2 = (np0[ii,jj,kk] - np0[ii,jj-1,kk])
                alpha = h1 / h2
                dy = h1 * (1. + alpha) + EPS
                J[0, 1] = (xp1[ii,jj+1,kk] - alpha**2 * xp1[ii,jj-1,kk] - (1 - alpha**2) * xp1[ii,jj,kk]) / dy
                J[1, 1] = (yp1[ii,jj+1,kk] - alpha**2 * yp1[ii,jj-1,kk] - (1 - alpha**2) * yp1[ii,jj,kk]) / dy
            _C = device_transposed_matmul_and_make_symmetric_2d(J, _C)
            for i in range(2):
                for j in range(2):
                    C[ii, jj, 0, i, j] = _C[i,j] 


def write_viz_files(xp0, yp0, zp0, ftle, fsle, N, filename):
    filename_ftle = filename + f".ftle.{N}"
    gridToVTK(filename_ftle, xp0, yp0, zp0, pointData={'FTLE': ftle, 'FSLE':fsle})
    return


@njit(parallel=True, fastmath=False, nogil=True)
def correct_ftle(U, ftle):
    Nx, Ny, Nz = U.shape
    for ii in prange(1, Nx-1):
        for jj in range(1, Ny - 1):
            for kk in range(Nz):
                if np.abs(ftle[ii,jj,kk]) > 0.5:
                    ftle[ii-1:ii+1, jj-1:jj+2, kk] = 0.0
    return ftle


def process_final_particle_info(filename, dt, output_file, viz_filename, x, y, z, make_2d_simulation = False, NFlowFields = None):
    comm = MPI.COMM_WORLD
    my_rank = comm.rank
    total_ranks = comm.size
    with h5py.File(filename, 'r', driver='mpio', comm=comm) as f, \
         h5py.File(output_file, 'w', driver='mpio', comm=comm) as h5Out:
        xyz_particle = f["Particle_xyz"]
        x = f["refined_x"][...]
        y = f["refined_y"][...]
        z = f["refined_z"][...]
        Nx, Ny, Nz = x.shape

        NTimeSteps = xyz_particle.shape[0]

        xp0 = xyz_particle[0, 0, :].reshape((Nx, Ny, Nz))
        yp0 = xyz_particle[0, 1, :].reshape((Nx, Ny, Nz))
        zp0 = xyz_particle[0, 2, :].reshape((Nx, Ny, Nz))
        
        h5Out.create_dataset(
            f"FTLE",
            shape=(NTimeSteps, Nx, Ny, Nz),
            dtype=REAL_NP,
        )
        FTLE = h5Out['FTLE']

        with ProgressBar(max_value=len(list(range(my_rank, NTimeSteps, total_ranks))) + 1) as bar:
            iteration = 0
            for N in range(my_rank, NTimeSteps, total_ranks):
                #bar.update(iteration)
                #sys.stdout.flush()

                if make_2d_simulation:
                    C = cuda.device_array((Nx, Ny, 1, 2, 2), dtype=REAL_NP)
                    calculate_ftle = calculate_particle_ftle_2d
                else:
                    C = cuda.device_array((Nx, Ny, Nz, 3, 3), dtype=REAL_NP)
                    calculate_ftle = calculate_particle_ftle
                calculate_ftle[BLOCKS_PER_GRID_3D, THREADS_PER_BLOCK_3D](xp0, yp0, zp0, C, N, dt)
                LambdaMax = np.max(np.linalg.eigvalsh(np.array(C)), axis=-1)
                ftle = (1. / (abs(N * NFlowFields * dt) + 1.19e-07)) * np.log(np.sqrt(LambdaMax))
                ftle = np.nan_to_num(ftle, nan=0.0, posinf=0.0, neginf=0.0)
                ftle -= np.min(ftle)
                ftle /= (np.max(ftle) * np.sign(dt) + 1.19e-07)
                ftle[:, -3:, :] = ftle[:, -3, np.newaxis, :]
                ftle[:, :, -3:] = ftle[:, :, np.newaxis, -3]
                ftle[:, :, :3] = ftle[:, :, np.newaxis, 3]
                FTLE[N,...] = ftle 
                write_viz_files(x, y, z, ftle, int(N*NFlowFields), viz_filename)
                iteration += 1
                #bar.update(iteration)
                #sys.stdout.flush()
    return


def process_multiple_parallel_final_particle_info(filename, dt, output_file, viz_filename, x, y, z, make_2d_simulation = False, NFlowFields = None):
    with h5py.File(filename, 'r') as f, \
         h5py.File(output_file, 'w') as h5Out:
        xyz_particle = f["Particle_xyz"]
        x = f["refined_x"][...]
        y = f["refined_y"][...]
        z = f["refined_z"][...]
        Nx, Ny, Nz = x.shape

        NTimeSteps = xyz_particle.shape[0]

        xp0 = xyz_particle[0, 0, :].reshape((Nx, Ny, Nz))
        yp0 = xyz_particle[0, 1, :].reshape((Nx, Ny, Nz))
        zp0 = xyz_particle[0, 2, :].reshape((Nx, Ny, Nz))

        h5Out.create_dataset(
            f"FTLE",
            shape=(NTimeSteps-1, Nx, Ny, Nz),
            dtype=REAL_NP,
        )
        h5Out.create_dataset(
            f"FSLE",
            shape=(NTimeSteps-1, Nx, Ny, Nz),
            dtype=REAL_NP,
        )
        FTLE = h5Out['FTLE']
        FSLE = h5Out['FSLE']

        with ProgressBar(max_value=len(list(range(1, NTimeSteps))) + 1) as bar:
            iteration = 0
            for N in range(1,NTimeSteps):
                bar.update(iteration)
                sys.stdout.flush()
                xp1 = xyz_particle[N, 0, :].reshape((Nx, Ny, Nz))
                yp1 = xyz_particle[N, 1, :].reshape((Nx, Ny, Nz))
                zp1 = xyz_particle[N, 2, :].reshape((Nx, Ny, Nz))

                if make_2d_simulation:
                    C = cuda.device_array((Nx, Ny, 1, 2, 2), dtype=REAL_NP)
                    calculate_ftle = calculate_particle_ftle_2d
                else:
                    ftle = cuda.device_array((Nx, Ny, Nz), dtype=REAL_NP)
                    fsle = cuda.device_array((Nx, Ny, Nz), dtype=REAL_NP)
                    calculate_ftle_and_fsle = calculate_particle_ftle_and_fsle
                T = REAL_NP(1.0 / (N * NFlowFields * dt))
                calculate_ftle_and_fsle[BLOCKS_PER_GRID_3D, THREADS_PER_BLOCK_3D](xp0, yp0, zp0, xp1, yp1, zp1, ftle, fsle, T)

                ftle = np.array(ftle)
                fsle = np.array(fsle)

                ftle = np.nan_to_num(ftle, nan=0.0, posinf=0.0, neginf=0.0)
                fsle = np.nan_to_num(fsle, nan=0.0, posinf=0.0, neginf=0.0)
                if T > 0:
                    ftle -= np.min(ftle)
                    ftle /= np.max(ftle)
                    fsle -= np.min(fsle)
                    fsle /= np.max(fsle)
                else:
                    ftle -= np.max(ftle)
                    ftle /= abs(np.min(ftle))
                    fsle -= np.max(fsle)
                    fsle /= abs(np.min(fsle))

                ftle[:, -3:, :] = ftle[:, -3, np.newaxis, :]
                ftle[:, :, -3:] = ftle[:, :, np.newaxis, -3]
                ftle[:, :, :3] = ftle[:, :, np.newaxis, 3]

                fsle[:, -3:, :] = fsle[:, -3, np.newaxis, :]
                fsle[:, :, -3:] = fsle[:, :, np.newaxis, -3]
                fsle[:, :, :3] = fsle[:, :, np.newaxis, 3]

                FTLE[N-1,...] = ftle
                FSLE[N-1,...] = fsle
                write_viz_files(x, y, z, ftle, fsle, int(N*NFlowFields), viz_filename)
                iteration += 1
                bar.update(iteration)
                sys.stdout.flush()
    return


if __name__ == "__main__":
    select_gpu()
    RUNSIM = True
    CALCULATE_FTLE = True

    NFIELDS = 14 # Number of Flow Fields to reach ~t^+ = 40
    UPSCALE = 40 #20 # 8 additional flow fields between every real field.
                  # Set this value to 1 to avoid upscaling.
    FLOWFIELDS = 1 # For a dynamic FTLE, set this number to the desired number of FTLEs
    SKIP = 1 # If you require skipping underlying flow fields, set this number to the desired value.
    MAKE_2D_SIMULATION = False # Particle advection can neglect spanwise valocity.
    WRITE_ALL_TIME_STEPS = False # Write intermediate steps for particle advection visualization.
    # CONFIGS Allows for a list of refinement values along x, y, z directions
    CONFIGS = [
        (8, 8, 3), # 415.8M particles
    ]
    FCENTERS = list(range(0, FLOWFIELDS, SKIP))
    DIRECTIONS = [-1,1] # Backward and Forw

    for X_UP, Y_UP, Z_UP in CONFIGS:
        master_print("*"*100)
        master_print("*"*100)
        master_print("-"*100)
        master_print("-"*100)
        master_print("Configuration:")
        master_print(f"    + ({X_UP}, {Y_UP}, {Z_UP})")

        # Configure Base Directory
        BASE_DIR = "./M08/Z_hdf5_c_260_300K"
        CASE_NAME = "ZPG_LOW_RE"
        MACH_MOD = "M08"
        WALL_CONDITION = "Adiabatic"

        coord_path = f"{BASE_DIR}/coord_1_440.txt.h5"
        filenames_full = sorted(
            glob.glob(f"{BASE_DIR}/PUVWT*.h5")
            )
        Ntime = len(filenames_full)
        t1 = time.time()
        with h5py.File(coord_path, "r", driver='mpio', comm=MPI.COMM_WORLD) as hf:
            dataset = hf["dataset"]
            Nx, Ny, _, Nz = dataset.shape
            x = dataset.astype(REAL_NP)[:,:,0,:].reshape((Nx,Ny,Nz))
            y = dataset.astype(REAL_NP)[:,:,1,:].reshape((Nx,Ny,Nz))
            z = dataset.astype(REAL_NP)[:,:,2,:].reshape((Nx,Ny,Nz))
        master_print(f"(Nx, Ny, Nz) = ({Nx}, {Ny}, {Nz})")
        PARTICLE_COUNT = X_UP * Y_UP * Z_UP * Nx * Ny * Nz
        master_print(f"Particle Count = {PARTICLE_COUNT}")
        t2 = time.time()
        master_print(f"Time Elapsed Reading Coordinates: {t2 - t1} s")
        sys.stdout.flush()

        for DIRECTION in DIRECTIONS:
            for FLOW_CENTER in FCENTERS:
                master_print(f"Global Step: {FLOW_CENTER}")
                gc.collect()
                dt = DIRECTION * 4.0e-4 * 10
                FIELDS = Ntime // 2 - 2
                filenames = filenames_full[Ntime//2 : (Ntime//2) * (DIRECTION * FIELDS) : DIRECTION]
                CURRENT_ID = os.path.split(filenames[FLOW_CENTER])[1][:-3]
                if DIRECTION == 1:
                    output_filename = f"./Temporal/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.forward.gpu.skip.step.{CURRENT_ID}.h5"
                    ftle_filename = f"./Temporal/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.forward.ftle.gpu.skip.step.{CURRENT_ID}.h5"
                    viz_filename = f"./Temporal/PV_VIZ/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.forward.gpu.skip.step.{CURRENT_ID}"
                elif DIRECTION == -1:
                    output_filename = f"./Temporal/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.backward.gpu.skip.step.{CURRENT_ID}.h5"
                    ftle_filename = f"./Temporal/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.backward.ftle.gpu.skip.step.{CURRENT_ID}.h5"
                    viz_filename = f"./Temporal/PV_VIZ/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.backward.gpu.skip.step.{CURRENT_ID}"

                if RUNSIM:
                    t3 = time.time()
                    particle_simulator(
                        filenames,
                        x,
                        y,
                        z,
                        dt,
                        output_filename,
                        UPSCALE,
                        MAKE_2D_SIMULATION,
                        (X_UP, Y_UP, Z_UP),
                        WRITE_ALL_TIME_STEPS,
                        NFIELDS,
                        FLOW_CENTER
                    )
                    t4 = time.time()
                    print(f"Time Elapsed Running Particle Simulation {t4 - t3} s")
                    sys.stdout.flush()
                gc.collect()
        master_print("-"*100)
        master_print("*"*100)
    FUSED_LOOP_COUNTER = [
        ((X_UP, Y_UP, Z_UP), FLOW_CENTER, DIRECTION)
        for X_UP, Y_UP, Z_UP in CONFIGS
        for FLOW_CENTER in FCENTERS
        for DIRECTION in DIRECTIONS
    ]
    comm = MPI.COMM_WORLD
    my_rank = comm.rank
    total_ranks = comm.size
    for (X_UP, Y_UP, Z_UP), FLOW_CENTER, DIRECTION in FUSED_LOOP_COUNTER[my_rank::total_ranks]:
        master_print("*"*100)
        master_print("*"*100)
        master_print("-"*100)
        master_print("-"*100)
        master_print("Configuration:")
        master_print(f"    + ({X_UP}, {Y_UP}, {Z_UP})")
        PARTICLE_COUNT = X_UP * Y_UP * Z_UP * Nx * Ny * Nz
        if CALCULATE_FTLE:
            gc.collect()
            dt = DIRECTION * 4.0e-4 * 10
            FIELDS = Ntime // 2 - 2
            filenames = filenames_full[Ntime//2 : (Ntime//2) * (DIRECTION * FIELDS) : DIRECTION]
            CURRENT_ID = os.path.split(filenames[FLOW_CENTER])[1][:-3]
            if DIRECTION == 1:
                output_filename = f"./Temporal/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.forward.gpu.skip.step.{CURRENT_ID}.h5"
                ftle_filename = f"./Temporal/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.forward.ftle.gpu.skip.step.{CURRENT_ID}.h5"
                viz_filename = f"./Temporal/PV_VIZ/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.forward.gpu.skip.step.{CURRENT_ID}"
            elif DIRECTION == -1:
                output_filename = f"./Temporal/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.backward.gpu.skip.step.{CURRENT_ID}.h5"
                ftle_filename = f"./Temporal/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.backward.ftle.gpu.skip.step.{CURRENT_ID}.h5"
                viz_filename = f"./Temporal/PV_VIZ/ParticleTracking_{CASE_NAME}_{X_UP}x_{Y_UP}x_{Z_UP}x_{PARTICLE_COUNT}P_{NFIELDS}Fields_TEMPORAL_UPS{UPSCALE}_{MACH_MOD}_{WALL_CONDITION}.backward.gpu.skip.step.{CURRENT_ID}"
            process_multiple_parallel_final_particle_info(output_filename, dt, ftle_filename, viz_filename, x, y, z, MAKE_2D_SIMULATION, NFIELDS)
            gc.collect()
    cuda.close()


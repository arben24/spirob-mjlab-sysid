import itertools
import json
import math
import time
from collections.abc import Callable
from typing import Any

import mujoco as mj
import numpy as np
import polars as pl

from .segment_estimator import SimpleSegmentEstimator

# Controller interface (callback signature).
# A controller takes mj.MjModel, mj.MjData, the current time (float) and the
# step index (int).
ControllerFunc = Callable[[mj.MjModel, mj.MjData, float, int], None]

# --- Helper function for body contact forces ---

def extract_body_contact_forces(model: mj.MjModel, data: mj.MjData) -> dict[int, np.ndarray]:
    """
    Extracts the total contact forces for each body in world frame.
    
    Returns a dict body_id -> np.array([Fx, Fy, Fz]) in world coordinates.
    
    Frame Convention:
    - mj_contactForce returns force[:3] in the contact frame, where force is the force applied to geom1 by geom2.
    - contact.frame is the 3x3 rotation matrix from contact frame to world frame.
    - Thus, F_world = contact.frame @ force[:3] gives the force on geom1 in world coordinates.
    - The force on geom2 is -F_world.
    - body_forces[body1] accumulates +F_world (force on body1), body_forces[body2] accumulates -F_world.
    """
    body_forces = {i: np.zeros(3, dtype=np.float64) for i in range(model.nbody)}
    
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        
        # Get force in contact frame (6D: Fx, Fy, Fz, Tx, Ty, Tz)
        force = np.zeros(6, dtype=np.float64)
        mj.mj_contactForce(model, data, contact_id, force)
        
        # Transform force from contact frame to world frame
        # contact.frame is rotation matrix: contact -> world
        if len(contact.frame) != 9:
            raise ValueError(f"contact.frame has unexpected length: {len(contact.frame)}")
        rotation_matrix = contact.frame.reshape(3, 3)
        force_world = rotation_matrix @ force[:3]  # F_world = R_contact_to_world @ F_contact
        #print(f"Contact ID {contact_id}: force_contact={force[:3]}, force_world={force_world}")
        #print(rotation_matrix)

        # Get body IDs
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        
        # Apply forces: body1 gets +force_world, body2 gets -force_world
        body_forces[body1] += force_world
        body_forces[body2] -= force_world
    
    return body_forces

# --- 1. Helper functions (data handling) ---

def get_sliced_dict(data_dict: dict[str, np.ndarray], final_length: int) -> dict[str, np.ndarray]:
    """Trim every array in the dict to the actual length (keeps them in sync)."""
    return {name: arr[:final_length] for name, arr in data_dict.items()}

# --- 2. Core simulation function ---

def initialize_data_structures(model: mj.MjModel, sim_time: float) -> tuple[dict, dict, dict, list, np.ndarray, int]:
    """Allocate every array and collect the metadata before the simulation."""
    
    num_steps = int(sim_time / model.opt.timestep) + 1

    # Dictionaries holding the sensor time series
    acc_over_time, gyro_over_time, tendon_frc_over_time, tendon_pos_over_time, \
    tendon_vel_over_time, joint_pos_over_time, joint_vel_over_time = {}, {}, {}, {}, {}, {}, {}
    positions_over_time = {}  # for geoms
    body_contact_force_over_time = {}  # for per-body contact forces
    
    SENSOR_CONFIG = {
        mj.mjtSensor.mjSENS_ACCELEROMETER:    ('acc',    acc_over_time),
        mj.mjtSensor.mjSENS_GYRO:             ('gyro',   gyro_over_time),
        mj.mjtSensor.mjSENS_TENDONACTFRC:     ('tendon_frc', tendon_frc_over_time),
        mj.mjtSensor.mjSENS_TENDONPOS:        ('tendon_pos', tendon_pos_over_time),
        mj.mjtSensor.mjSENS_TENDONVEL:        ('tendon_vel', tendon_vel_over_time),
        mj.mjtSensor.mjSENS_JOINTPOS:         ('joint_pos',  joint_pos_over_time),
        mj.mjtSensor.mjSENS_JOINTVEL:         ('joint_vel',  joint_vel_over_time),
    }

    sensor_metadata = []
    
    # 1. Sensoren initialisieren
    for i in range(model.nsensor):
        sensor_type = model.sensor_type[i]
        
        if sensor_type in SENSOR_CONFIG:
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
            _, data_dict = SENSOR_CONFIG[sensor_type]
            dim = model.sensor_dim[i] 
            
            time_series_array = np.zeros((num_steps, dim), dtype=np.float64)
            data_dict[name] = time_series_array 
            
            sensor_metadata.append({
                'name': name,
                'array': time_series_array, 
                'index': i                  
            })

    # 1b. Initialise the per-body contact forces
    body_metadata = []
    for i in range(1, model.nbody):  # Skip worldbody (0)
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        force_array = np.zeros((num_steps, 3), dtype=np.float64)  # Fx, Fy, Fz
        body_contact_force_over_time[body_name] = force_array
        body_metadata.append({
            'name': body_name,
            'array': force_array,
            'id': i
        })

    # 2. Geoms initialisieren
    geom_metadata = []
    i = 0
    while True:
        name = f"g_{i}"
        geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            break
        
        pos_array = np.zeros((num_steps, 3), dtype=np.float64)
        positions_over_time[name] = pos_array
        
        geom_metadata.append({
            'name': name,
            'pos_array': pos_array,
            'id': geom_id
        })
        i += 1
        
    time_array = np.zeros(num_steps, dtype=np.float64)

    sensor_dicts = {
        "acc": acc_over_time, "gyro": gyro_over_time, "tendon_frc": tendon_frc_over_time, 
        "tendon_pos": tendon_pos_over_time, "tendon_vel": tendon_vel_over_time, 
        "joint_pos": joint_pos_over_time, "joint_vel": joint_vel_over_time, 
        "geom_pos": positions_over_time, "bodycontactfrc": body_contact_force_over_time
    }

    return sensor_dicts, sensor_metadata, geom_metadata, body_metadata, time_array, num_steps

def create_single_polars_dataframe(
    sensor_groups_config: list[tuple[str, dict[str, np.ndarray]]], 
    time_series_data: np.ndarray, 
    final_length: int
) -> pl.DataFrame:
    """
    Creates a single Polars DataFrame from sensor groups config.
    """
    df_dict = {"time_s": time_series_data[:final_length]}
    
    for group_name, sensor_dict in sensor_groups_config:
        for sensor_name, array in sensor_dict.items():
            array = array[:final_length]
            if group_name == "bodycontactfrc":
                df_dict[f"{sensor_name}_contact_force_X"] = array[:, 0]
                df_dict[f"{sensor_name}_contact_force_Y"] = array[:, 1]
                df_dict[f"{sensor_name}_contact_force_Z"] = array[:, 2]
            elif group_name in ["geom_pos", "geom_quat"]:
                # Special handling for geom
                geom_id = sensor_name.split('_')[-1]
                if group_name == "geom_pos":
                    base = f"geom_pos_{geom_id}"
                    df_dict[f"{base}_x"] = array[:, 0]
                    df_dict[f"{base}_y"] = array[:, 1]
                    df_dict[f"{base}_z"] = array[:, 2]
                elif group_name == "geom_quat":
                    base = f"geom_quat_{geom_id}"
                    df_dict[f"{base}_w"] = array[:, 0]
                    df_dict[f"{base}_x"] = array[:, 1]
                    df_dict[f"{base}_y"] = array[:, 2]
                    df_dict[f"{base}_z"] = array[:, 3]
            elif array.ndim == 1:
                # 1D sensor
                df_dict[sensor_name] = array
            elif array.ndim == 2:
                if array.shape[1] == 3:
                    # 3D vector
                    df_dict[f"{sensor_name}_X"] = array[:, 0]
                    df_dict[f"{sensor_name}_Y"] = array[:, 1]
                    df_dict[f"{sensor_name}_Z"] = array[:, 2]
                elif array.shape[1] == 4:
                    # Quaternion
                    df_dict[f"{sensor_name}_w"] = array[:, 0]
                    df_dict[f"{sensor_name}_x"] = array[:, 1]
                    df_dict[f"{sensor_name}_y"] = array[:, 2]
                    df_dict[f"{sensor_name}_z"] = array[:, 3]
                else:
                    # Other dimensions
                    for i in range(array.shape[1]):
                        df_dict[f"{sensor_name}_{i}"] = array[:, i]
    
    return pl.DataFrame(df_dict)

def run_simulation_and_get_dataframe(
    model: mj.MjModel, 
    data: mj.MjData, 
    sim_time: float, 
    controller: ControllerFunc,
    enable_viewer: bool,
    boost_viewer: float,
    include_geom_pos: bool = False,
    record_video: bool = False,
    video_fps: int = 30,
    video_resolution: tuple[int, int] = (640, 480),
    video_path: str = None,
    video_flip_vertical: bool = True,
    enable_position_estimation: bool = False,
    position_estimator_segments: list[int] = None
) -> pl.DataFrame:
    """
    Run the simulation, collect the data and convert it into a Polars DataFrame.
    """
    
    # Allocate every storage structure
    sensor_dicts, sensor_metadata, geom_metadata, body_metadata, time_array, num_steps = \
        initialize_data_structures(model, sim_time)
        
    # --- Position-estimation setup ---
    estimator = None
    pos_estimate_arrays = {}
    quat_estimate_arrays = {}
    vel_estimate_arrays = {}
    if enable_position_estimation and position_estimator_segments:
        # Read the initial values from MuJoCo
        initial_positions = {}
        initial_orientations = {}
        for seg_id in position_estimator_segments:
            geom_name = f'geom_{seg_id}'
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                initial_positions[seg_id] = data.geom_xpos[geom_id].tolist()
                initial_orientations[seg_id] = data.geom_xquat[geom_id].tolist()
        
        estimator = SimpleSegmentEstimator(
            segment_ids=position_estimator_segments,
            initial_positions=initial_positions,
            initial_orientations=initial_orientations
        )
        
        # Initialise the arrays for the estimates
        for seg_id in position_estimator_segments:
            pos_estimate_arrays[seg_id] = np.zeros((num_steps, 3))
            quat_estimate_arrays[seg_id] = np.zeros((num_steps, 4))
            vel_estimate_arrays[seg_id] = np.zeros((num_steps, 4))  # x,y,z,norm
        
    # --- Video Rendering Initialisierung ---
    frames = []
    width, height = video_resolution if record_video else (640, 480)
    mjv_scene = None
    mjv_camera = None
    mjv_option = None
    mjr_context = None
    rgb_buffer = None
    render_every = 1
    render_counter = 0
    if record_video:
        try:
            # Select the OpenGL platform for headless rendering
            import os
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            
            # Initialise the EGL context for headless rendering
            import mujoco.egl
            width, height = video_resolution
            egl_context = mujoco.egl.GLContext(width, height)
            egl_context.make_current()
            
            # Offscreen Rendering Setup
            mjv_scene = mj.MjvScene(model, maxgeom=1000)
            mjv_camera = mj.MjvCamera()
            mjv_option = mj.MjvOption()
            
            # Kamera einstellen: Feste globale Kamera
            mjv_camera.type = mj.mjtCamera.mjCAMERA_FREE
            mjv_camera.lookat = np.array([0.0, 0.0, 0.1])  # Blickpunkt
            mjv_camera.distance = 1.0
            mjv_camera.azimuth = 90.0
            mjv_camera.elevation = -20.0
            
            # Adjust fovy to maintain consistent horizontal FOV across different aspect ratios
            aspect = width / height
            aspect_ref = 4/3  # Reference aspect ratio (640x480)
            fovy_ref = 45.0   # Reference fovy for reference aspect
            hfov_ref_rad = 2 * math.atan(math.tan(math.radians(fovy_ref)/2) * aspect_ref)
            fovy_rad = 2 * math.atan(math.tan(hfov_ref_rad/2) / aspect)
            try:
                mjv_camera.fovy = math.degrees(fovy_rad)
            except AttributeError:
                print("Warning: camera FOV adjustment unsupported (no fovy), using the default FOV")
            
            # Context for offscreen rendering
            mjr_context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
            
            # Set the buffer for offscreen rendering
            mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, mjr_context)
            
            # Resize offscreen buffer to match video resolution
            mj.mjr_resizeOffscreen(width, height, mjr_context)
            
            # Verify offscreen buffer size
            off_width = mjr_context.offWidth
            off_height = mjr_context.offHeight
            print(f"  Offscreen buffer size: {off_width}x{off_height}")
            if off_width < width or off_height < height:
                print(f"Warning: offscreen buffer ({off_width}x{off_height}) smaller than the video resolution ({width}x{height})")
            
            # RGB framebuffer
            rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Render interval that yields the requested FPS
            # render_every = max(1, round(1.0 / (video_fps * dt)))
            # so that video length ~= simulated time
            dt = model.opt.timestep
            render_every = max(1, int(round(1.0 / (video_fps * dt))))
            
            # Debug logging
            print(f"Video-Aufzeichnung initialisiert: {width}x{height} @ {video_fps} FPS (render every {render_every} steps)")
            print(f"  aspect: {aspect:.3f}")
            try:
                print(f"  fovy: {mjv_camera.fovy:.1f}°")
            except AttributeError:
                print("  fovy: unavailable (using the default)")
        except ImportError:
            print("Warning: mujoco.egl unavailable. Headless video rendering is not supported.")
            record_video = False
        except Exception as e:
            print(f"Warning: video rendering could not be initialised: {e}")
            record_video = False
    
    step_index = 0
    
    # Zustand t=0 speichern
    time_array[step_index] = data.time
    for meta in sensor_metadata:
        meta['array'][step_index] = data.sensor(meta['index']).data
    if include_geom_pos:
        for meta in geom_metadata:
            meta['pos_array'][step_index] = data.geom_xpos[meta['id']]
        
    steps_to_run = num_steps - 1

    if enable_viewer:
        with mj.viewer.launch_passive(model, data) as viewer:
            # data.time drives the stop condition,
            # so that exactly sim_time seconds of physical time are simulated
            while viewer.is_running() and data.time < sim_time:
                step_start = time.time()

                step_index += 1
                
                # --- CALL THE EXTERNAL CONTROLLER ---
                controller(model, data, data.time, step_index) 
                
                # --- Simulationsschritt ---
                mj.mj_step(model, data)
                
                # --- Position Estimation ---
                if estimator:
                    dt = model.opt.timestep
                    sensor_data = {}
                    for seg_id in position_estimator_segments:
                        acc_cols = [f'acc_{seg_id}_X', f'acc_{seg_id}_Y', f'acc_{seg_id}_Z']
                        gyro_cols = [f'gyro_{seg_id}_X', f'gyro_{seg_id}_Y', f'gyro_{seg_id}_Z']
                        acc_data = []
                        gyro_data = []
                        for col in acc_cols:
                            sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, col)
                            if sensor_id >= 0:
                                acc_data.append(data.sensor(sensor_id).data[0])
                        for col in gyro_cols:
                            sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, col)
                            if sensor_id >= 0:
                                gyro_data.append(data.sensor(sensor_id).data[0])
                        if acc_data and gyro_data:
                            sensor_data[seg_id] = {'acc': acc_data, 'gyro': gyro_data}
                    if sensor_data:
                        estimator.update_batch(sensor_data, dt)
                        states = estimator.get_all_states()
                        for seg_id in position_estimator_segments:
                            if seg_id in states:
                                state = states[seg_id]
                                pos_estimate_arrays[seg_id][step_index] = state.position
                                quat_estimate_arrays[seg_id][step_index] = state.orientation
                                vel_estimate_arrays[seg_id][step_index] = [state.velocity[0], state.velocity[1], state.velocity[2], np.linalg.norm(state.velocity)]
                
                # --- Datenspeicherung ---
                # Guard against writing past the end of the arrays
                if step_index < len(time_array):
                    time_array[step_index] = data.time
                    for meta in sensor_metadata:
                        meta['array'][step_index] = data.sensor(meta['index']).data
                    if include_geom_pos:
                        for meta in geom_metadata:
                            meta['pos_array'][step_index] = data.geom_xpos[meta['id']]
                    
                    # Collect the per-body contact forces
                    body_forces = extract_body_contact_forces(model, data)
                    for meta in body_metadata:
                        meta['array'][step_index] = body_forces[meta['id']]

                # GUI aktualisieren
                viewer.sync()

                # --- Video Frame aufzeichnen ---
                if record_video:
                    render_counter += 1
                    if render_counter % render_every == 0:
                        try:
                            mj.mjv_updateScene(model, data, mjv_option, None, mjv_camera, mj.mjtCatBit.mjCAT_ALL, mjv_scene)
                            mj.mjr_render(mj.MjrRect(0, 0, width, height), mjv_scene, mjr_context)
                            mj.mjr_readPixels(rgb_buffer, None, mj.MjrRect(0, 0, width, height), mjr_context)
                            frame = rgb_buffer.copy()
                            if video_flip_vertical:
                                frame = np.flipud(frame)  # MuJoCo rendert bottom-up, Videos brauchen top-down
                            frames.append(frame)
                        except Exception as e:
                            print(f"Warning: frame capture failed: {e}")

                # --- Pacing, with a boost factor ---
                # Divide the physical time step by the boost factor
                target_step_duration = model.opt.timestep / boost_viewer
                elapsed_time = time.time() - step_start
                
                time_until_next_step = target_step_duration - elapsed_time
                
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
        for _ in range(steps_to_run):
                
            step_index += 1
            
            # --- CALL THE EXTERNAL CONTROLLER ---
            controller(model, data, data.time, step_index) 
            
            # --- Simulationsschritt ---
            mj.mj_step(model, data)
            
            # --- Position Estimation ---
            if estimator:
                dt = model.opt.timestep
                sensor_data = {}
                for seg_id in position_estimator_segments:
                    acc_cols = [f'acc_{seg_id}_X', f'acc_{seg_id}_Y', f'acc_{seg_id}_Z']
                    gyro_cols = [f'gyro_{seg_id}_X', f'gyro_{seg_id}_Y', f'gyro_{seg_id}_Z']
                    acc_data = []
                    gyro_data = []
                    for col in acc_cols:
                        sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, col)
                        if sensor_id >= 0:
                            acc_data.append(data.sensor(sensor_id).data[0])
                    for col in gyro_cols:
                        sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, col)
                        if sensor_id >= 0:
                            gyro_data.append(data.sensor(sensor_id).data[0])
                    if acc_data and gyro_data:
                        sensor_data[seg_id] = {'acc': acc_data, 'gyro': gyro_data}
                if sensor_data:
                    estimator.update_batch(sensor_data, dt)
                    states = estimator.get_all_states()
                    for seg_id in position_estimator_segments:
                        if seg_id in states:
                            state = states[seg_id]
                            pos_estimate_arrays[seg_id][step_index] = state.position
                            quat_estimate_arrays[seg_id][step_index] = state.orientation
                            vel_estimate_arrays[seg_id][step_index] = [state.velocity[0], state.velocity[1], state.velocity[2], np.linalg.norm(state.velocity)]
            
            # --- Datenspeicherung ---
            time_array[step_index] = data.time
            for meta in sensor_metadata:
                meta['array'][step_index] = data.sensor(meta['index']).data
            if include_geom_pos:
                for meta in geom_metadata:
                    meta['pos_array'][step_index] = data.geom_xpos[meta['id']]
            
            # Collect the per-body contact forces
            body_forces = extract_body_contact_forces(model, data)
            for meta in body_metadata:
                meta['array'][step_index] = body_forces[meta['id']]

            # --- Video Frame aufzeichnen ---
            if record_video:
                render_counter += 1
                if render_counter % render_every == 0:
                    try:
                        mj.mjv_updateScene(model, data, mjv_option, None, mjv_camera, mj.mjtCatBit.mjCAT_ALL, mjv_scene)
                        mj.mjr_render(mj.MjrRect(0, 0, width, height), mjv_scene, mjr_context)
                        mj.mjr_readPixels(rgb_buffer, None, mj.MjrRect(0, 0, width, height), mjr_context)
                        frame = rgb_buffer.copy()
                        if video_flip_vertical:
                            frame = np.flipud(frame)  # MuJoCo rendert bottom-up, Videos brauchen top-down
                        frames.append(frame)
                    except Exception as e:
                        print(f"Warning: frame capture failed: {e}")

    final_length = step_index + 1
    time_series_data = time_array[:final_length] 
    
    # --- 2. Polars Konvertierung ---
    
    SENSOR_GROUPS_CONFIG: list[tuple[str, dict[str, np.ndarray]]] = [
        ("acc", sensor_dicts["acc"]),
        ("gyro", sensor_dicts["gyro"]),
        ("tendon_frc", sensor_dicts["tendon_frc"]),
        ("tendon_pos", sensor_dicts["tendon_pos"]),
        ("tendon_vel", sensor_dicts["tendon_vel"]),
        ("joint_pos", sensor_dicts["joint_pos"]),
        ("joint_vel", sensor_dicts["joint_vel"]),
        ("bodycontactfrc", sensor_dicts["bodycontactfrc"]),
    ]
    
    if include_geom_pos:
         SENSOR_GROUPS_CONFIG.append(("geom_pos", sensor_dicts["geom_pos"]))
    
    # Add position estimates
    if enable_position_estimation and position_estimator_segments:
        pos_estimate_dict = {}
        quat_estimate_dict = {}
        vel_estimate_dict = {}
        for seg_id in position_estimator_segments:
            pos_estimate_dict[f'pos_estimate_{seg_id}'] = pos_estimate_arrays[seg_id][:final_length]
            quat_estimate_dict[f'quat_estimate_{seg_id}'] = quat_estimate_arrays[seg_id][:final_length]
            vel_estimate_dict[f'vel_estimate_{seg_id}'] = vel_estimate_arrays[seg_id][:final_length]
        SENSOR_GROUPS_CONFIG.append(("pos_estimate", pos_estimate_dict))
        SENSOR_GROUPS_CONFIG.append(("quat_estimate", quat_estimate_dict))
        SENSOR_GROUPS_CONFIG.append(("vel_estimate", vel_estimate_dict))

    final_wide_df = create_single_polars_dataframe(
        SENSOR_GROUPS_CONFIG, 
        time_series_data, 
        final_length
    )
    
    # --- Video speichern ---
    if record_video and frames and video_path:
        try:
            # Make sure the directory exists
            import os
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            print(f"Writing video to {video_path} ...")
            # Imported here, not at module scope: video export belongs to the
            # optional 'vision' extra, and the core install must stay importable
            # without it.
            import imageio

            with imageio.get_writer(video_path, fps=video_fps, macro_block_size=None) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Video saved: {video_path}")
        except Exception as e:
            print(f"Error writing the video: {e}")
    
    return final_wide_df

# --- 3. Controller-Templates (Beispiele) ---

def static_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Apply a constant tendon force (0.2) to the first actuator."""
    # Example: drive only the first actuator

    #data.ctrl[0] = 0.2
    data.ctrl[1] = 0.3

def ramped_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Ramp the first actuator linearly from 0.0 to 1.0 over 10 seconds."""
    max_time = 2.0
    max_force = -10.0
    force = (current_time / max_time) * max_force
    force = min(force, max_force)  # Begrenze auf max_force
    data.ctrl[0] = force

def sine_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Drive the first actuator with a sine (amplitude 0.5, frequency 0.5 Hz)."""

    amplitude = 0.5
    frequency = 2.0 * math.pi * 0.5 
    data.ctrl[0] = amplitude * np.sin(frequency * current_time)

# These functions are called LATER, inside the loop
def setup_cylinder(worldbody, pos, size, euler, **kwargs):
    body = worldbody.add_body(name="cylinder_obj", pos=pos)
    body.add_geom(
        name="cyl_geom",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=size,  # Erwartet [radius, half_length, unused]
        euler=euler,
        rgba=[0.2, 0.8, 0.5, 1],
        density=1000
    )

def setup_box(worldbody, pos, size, euler, **kwargs):
    body = worldbody.add_body(name="box_obj", pos=pos)
    body.add_geom(
        name="box_geom",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=size,  # Erwartet [x_half, y_half, z_half]
        euler=euler,
        rgba=[0.8, 0.2, 0.2, 1],
        density=1000
    )

def generate_grid_configs(variable_params: dict[str, Any], fixed_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of configuration dictionaries."""
    
    # Split keys and values so they can be handed to itertools.product
    keys = list(variable_params.keys())
    values = [
        # If the value is a dict (as for 'controller'), use its values
        list(v.values()) if isinstance(v, dict) else v
        for v in variable_params.values()
    ]
    

    SIM_CONFIGS = []
    run_counter = 1
    
    # itertools.product yields every combination of the values
    for combination in itertools.product(*values):
        
        # Build a dict from the current combination
        config = dict(zip(keys, combination))
        
        # Add the fixed parameters
        config.update(fixed_params)
        
        # If 'controller' is in the grid, find its name for the ID
        ctrl_name = "Custom"
        if "controller" in keys:
            # Find the controller's name from its function/object value
            # (slightly awkward: the value has to be mapped back to its key)
            # Simplified by assuming 'controller' is the last variable:
            if isinstance(variable_params["controller"], dict):
                ctrl_value = config["controller"]
                
                # Find the name belonging to this value
                ctrl_name = next((name for name, func in variable_params["controller"].items() if func == ctrl_value), "Unknown")
            
        # Build the unique ID
        id_parts = [
            ctrl_name,
            f"L{config['L_target']:.2f}",
            f"T{config['sim_time']:.1f}",
            f"d{config['base_d']:.3f}",
            # ... add further relevant parameters here
        ]
        config["id"] = f"Run_{run_counter:03d}_{'_'.join(id_parts)}"
        
        SIM_CONFIGS.append(config)
        run_counter += 1
        
    return SIM_CONFIGS

def generate_hybrid_grid_configs(common_params, geom_scenarios, fixed_params):
    configs = []
    run_counter = 1

    # Step 1: build the grid over the shared parameters (L_target, controller, ...)
    common_keys = list(common_params.keys())
    # Controllers need special handling: we want the values, but the names for the ID
    common_values = []
    for k, v in common_params.items():
        if k == "controller" and isinstance(v, dict):
            common_values.append(list(v.items())) # Speichert (Name, Funktion) Tupel
        else:
            common_values.append(v)

    # Iterate over the base parameters
    for common_prod in itertools.product(*common_values):
        
        # Basis-Config Dictionary bauen
        base_config = {}
        ctrl_name_id = ""
        
        for i, key in enumerate(common_keys):
            val = common_prod[i]
            if key == "controller":
                # Unpack the tuple: (name, function)
                ctrl_name_id = val[0]
                base_config[key] = val[1]
            else:
                base_config[key] = val

        # Step 2: for EVERY base config, iterate through the geometry scenarios
        for scenario in geom_scenarios:
            geom_func = scenario["setup_func"]
            geom_name = scenario["obj_name"]
            
            # Fetch this scenario's parameters (size, pos, euler)
            scen_params = scenario["params"]
            scen_keys = list(scen_params.keys())
            scen_values = list(scen_params.values())
            
            # Mini-grid for this scenario
            for geom_prod in itertools.product(*scen_values):
                
                # Copy the base config so we do not overwrite it
                final_config = base_config.copy()
                final_config.update(fixed_params)
                
                # Add the geometry data
                final_config["geom_func"] = geom_func
                
                # Put the geometry parameters into the config dict individually AND into 'geom_kwargs'
                geom_kwargs = {}
                geom_id_parts = [geom_name]
                
                for i, key in enumerate(scen_keys):
                    val = geom_prod[i]
                    geom_kwargs[key] = val  # for the later function call
                    
                    # ID Teil generieren (z.B. Size -> S0.1)
                    if key == "size":
                        s_str = "-".join([f"{x:.2f}" for x in val])
                        geom_id_parts.append(f"Sz{s_str}")
                    elif key == "pos":
                        # Optional, if pos matters for the ID
                        pass 

                final_config["geom_kwargs"] = geom_kwargs
                
                # ID erstellen
                id_parts = [
                    ctrl_name_id,
                    "_".join(geom_id_parts),
                    f"L{base_config['L_target']:.2f}",
                    f"T{base_config['sim_time']:.1f}"
                ]
                final_config["id"] = f"Run_{run_counter:03d}_{'_'.join(id_parts)}"
                
                configs.append(final_config)
                run_counter += 1
                
    return configs



def format_value_for_print(value: Any) -> str:
    """Render lists/arrays as a compact, readable string."""
    if isinstance(value, (list, tuple, np.ndarray)):
        # Round floats and stringify: [0.10, 0.20]
        return "[" + ", ".join([f"{x:.2f}" for x in value]) + "]"
    
    if isinstance(value, float):
        return f"{value:.3f}"
        
    # If it is the function's name as a string (e.g. 'setup_cylinder')
    if isinstance(value, str):
        return value.replace('setup_', '')
        
    return str(value)


def print_configs_formatted(config_list: list[dict[str, Any]], preview_limit: int = 5, print_all: bool = False):
    """
    Print a formatted preview of the configuration list to the console.
    Includes the details of the variable geometry.
    """
    
    print("\n" + "="*80)
    print(f"CONFIGURATION PREVIEW ({len(config_list)} runs)")
    print("="*80)

    if print_all:
        preview_limit = len(config_list)
    
    for i, config in enumerate(config_list):
        if i >= preview_limit:
            print(f"  ... and {len(config_list) - i} further configurations.")
            break
            
        # --- 1. Controller Name ---
        ctrl = config.get('controller')
        ctrl_name = ctrl.__name__ if hasattr(ctrl, '__name__') else str(ctrl)

        # --- 2. Geometrie-Informationen ---
        
        # Geometry setup function
        geom_func = config.get('geom_func')
        geom_type_name = geom_func.__name__.replace('setup_', '') if hasattr(geom_func, '__name__') else "Unbekannt"
        
        # Geometrie-Parameter (pos, size, euler)
        geom_kwargs = config.get('geom_kwargs', {})
        
        geom_pos_str = format_value_for_print(geom_kwargs.get('pos', 'N/A'))
        geom_size_str = format_value_for_print(geom_kwargs.get('size', 'N/A'))
        geom_euler_str = format_value_for_print(geom_kwargs.get('euler', 'N/A'))
        
        # --- Output ---
        print(f"[{i+1}/{len(config_list)}] ID: {config['id']}")
        
        # Allgemeine Parameter
        print(f"  > Modell: L_target={config.get('L_target', 'N/A'):.2f}, base_d={config.get('base_d', 'N/A'):.3f}")
        print(f"  > Kontext: Time={config.get('sim_time', 'N/A'):.1f}, Controller={ctrl_name}")
        
        # Geometrie-Details
        print(f"  > OBJEKT ({geom_type_name.upper()}):")
        print(f"      pos: {geom_pos_str}, size: {geom_size_str}, euler: {geom_euler_str}")
        
    print("\n" + "="*80)

def save_configs_to_json(
    config_list: list[dict[str, Any]], 
    filename: str = "simulation_configs.json", 
    indent: int = 4
):
    """
    Export the configuration list to a JSON file.
    Non-serialisable objects (functions, NumPy arrays) are converted to strings 
    or into standard Python types.
    
    Args:
        config_list: the list of configuration dictionaries (SIM_CONFIGS).
        filename: name of the export file.
        indent: number of spaces used for the JSON indentation.
    """
    
    exportable_list = []
    
    for config in config_list:
        # Copy the dict so the original is left untouched
        export_config = config.copy()
        
        # --- 1. Handle the controller (function/object) ---
        if 'controller' in export_config:
            ctrl = export_config['controller']
            # Store the function/class name as a string
            ctrl_str = ctrl.__name__ if hasattr(ctrl, '__name__') else str(ctrl)
            export_config['controller_info_str'] = ctrl_str
            # Drop the non-serialisable object
            del export_config['controller']

        # --- 2. Handle the geometry function ---
        if 'geom_func' in export_config:
            geom_func = export_config['geom_func']
            # Store the function name as a string
            geom_func_str = geom_func.__name__ if hasattr(geom_func, '__name__') else "unknown function"
            export_config['geom_func_info_str'] = geom_func_str
            # Drop the non-serialisable object
            del export_config['geom_func']

        # --- 3. Handle the geometry arguments (geom_kwargs) and other lists ---
        # This is the critical step that turns NumPy arrays into plain lists.
        
        # Recursive helper that converts NumPy types
        def convert_to_serializable(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                # Recurse into lists/arrays
                return [convert_to_serializable(x) for x in item]
            elif isinstance(item, dict):
                # Recurse into dictionaries
                return {k: convert_to_serializable(v) for k, v in item.items()}
            elif isinstance(item, (np.float32, np.float64, np.generic)):
                # Konvertiere NumPy-Floats zu nativem Python-Float
                return float(item)
            elif isinstance(item, (np.int32, np.int64)):
                # Konvertiere NumPy-Integers zu nativem Python-Integer
                return int(item)
            else:
                return item

        # Apply the conversion to geom_kwargs, if present
        if 'geom_kwargs' in export_config:
            export_config['geom_kwargs'] = convert_to_serializable(export_config['geom_kwargs'])
            
        # Apply the conversion to other top-level values that may be lists/arrays
        # (e.g. L_target values that came from an np.array)
        for key, value in export_config.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                export_config[key] = convert_to_serializable(value)
        
        
        # --- 4. Append to the export list ---
        exportable_list.append(export_config)

    # --- JSON-Export ---
    try:
        with open(filename, 'w') as f:
            json.dump(exportable_list, f, indent=indent)
        print(f"\nConfigurations exported to: {filename}")
    except Exception as e:
        print(f"\nError exporting to JSON ({filename}): {e}")
        print("Make sure no non-serialisable types (e.g. complex objects) are left.")

def get_video_path(run_id: str, base_dir: str = "build/experiments") -> str:
    """Build the video path from the run_id."""
    return f"{base_dir}/{run_id}/video.mp4"

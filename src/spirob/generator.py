from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import bisect

from . import spiral as ms

# ===================================================================
# 1) High-level API (public functions)
# ===================================================================

def generate_xml_string(
    L_target: float,
    base_d: float,
    tip_d: float,
    Delta_theta_deg: float = 30.0,
    model_name: str = "spiral_chain",
    auto_format: bool = False,
):
    """
    Generate an MJCF XML string representing a discretized logarithmic spiral chain.

    This function solves the spiral parameters, computes the segment geometry,
    discretizes the chain, and constructs a complete MJCF model using ``XMLBuilder``.

    Parameters
    ----------
    L_target : float
        Desired total centerline length of the spiral.
    base_d : float
        Diameter at the base of the spiral (start).
    tip_d : float
        Diameter at the tip of the spiral (end).
    Delta_theta_deg : float, optional
        Target angular resolution per segment in degrees.
        The final angular resolution may differ slightly due to rounding.
    model_name : str, optional
        Name of the root model in the generated MJCF XML.
    auto_format : bool, optional
        If True, reformat the generated XML using MuJoCo’s ``MjSpec`` for
        consistent indentation and structure.

    Returns
    -------
    str
        The generated MJCF XML model as a string.
    """

    calc = SpiralCalculator(
        L_target=L_target,
        base_d=base_d,
        tip_d=tip_d,
        Delta_theta_deg=Delta_theta_deg,
    )

    geometry = calc.compute_geometry()

    SensorRegistry.register("acc", "accelerometer")
    SensorRegistry.register("gyro", "gyro")
    SensorRegistry.register("angle", "jointpos")
    SensorRegistry.register("joint_vel", "jointvel")
    SensorRegistry.register("tendon_frc", "tendonactuatorfrc")
    SensorRegistry.register("tendon_pos", "tendonpos")
    SensorRegistry.register("tendon_vel", "tendonvel")
    #SensorRegistry.register("frc", "force")    # DO NOT WORK THIS WAY YET

    xml = XMLBuilder(
        geometry.seg_lengths,
        geometry.seg_halfwidths,
        geometry.phi_taper,
        geometry.new_Delta_theta,
        geometry.beta,
    ).build(model_name=model_name)

    if auto_format:
        spec = mj.MjSpec.from_string(xml)
        xml = spec.to_xml()

    return xml


def generate_and_save_xml(
    filepath: str | Path,
    **kwargs,
):
    """
    Generate an MJCF XML file and save it to disk.

    This is a thin wrapper around :func:`generate_xml_string`.
    All keyword arguments are passed directly to that function.

    Parameters
    ----------
    filepath : str or Path
        Destination path where the XML file will be written.
    **kwargs
        Additional keyword arguments forwarded to :func:`generate_xml_string`.

    Returns
    -------
    pathlib.Path
        The final path of the saved XML file.
    """
    xml = generate_xml_string(**kwargs)
    filepath = Path(filepath)
    filepath.write_text(xml, encoding="utf-8")
    return filepath


# ===================================================================
# 2) Computation (formerly top-level code)
# ===================================================================

class SpiralGeometry:
    """
    Container class for all geometric quantities of the discretized spiral.

    Parameters
    ----------
    seg_lengths : array_like
        Length of each segment along the chain.
    seg_halfwidths : array_like
        Half-width of each segment (radius), controlling the box dimensions.
    phi_taper : float
        Tapering angle of the spiral body.
    new_Delta_theta : float
        Actual angular resolution used after discretization.
    beta : float
        Radial growth factor per segment.
    """

    def __init__(self, seg_lengths, seg_halfwidths, phi_taper, new_Delta_theta, beta,
                 b=None, a=None, theta0=None, L_check=None):
        self.seg_lengths = seg_lengths
        self.seg_halfwidths = seg_halfwidths
        self.phi_taper = phi_taper
        self.new_Delta_theta = new_Delta_theta
        self.beta = beta
        self.b = b
        self.a = a
        self.theta0 = theta0
        self.L_check = L_check

    @property
    def n_segments(self):
        """Number of discretized segments."""
        return len(self.seg_lengths)

    def summary(self) -> str:
        """Return a human-readable summary of the computed spiral parameters."""
        lines = [
            "--- Spiral parameters ---",
            f"  growth parameter b     : {self.b:.6f}",
            f"  scale ratio beta       : {self.beta:.6f}   (rho_(i+1)/rho_i = exp(b*dTheta))",
            f"  scale factor a         : {self.a:.6f} m",
            f"  total angle theta0     : {self.theta0:.6f} rad "
            f"({np.rad2deg(self.theta0):.3f} deg)",
            f"  taper angle phi        : {self.phi_taper:.6f} rad "
            f"({np.rad2deg(self.phi_taper):.3f} deg)",
            f"  centreline length L    : {self.L_check:.6f} m",
            "--- Discretisation ---",
            f"  segments N             : {self.n_segments}",
            f"  dTheta (effective)     : {self.new_Delta_theta:.6f} rad "
            f"({np.rad2deg(self.new_Delta_theta):.3f} deg)",
            f"  segment lengths        : {self.seg_lengths[0]:.6f} m (base) ... "
            f"{self.seg_lengths[-1]:.6f} m (tip), sum {float(np.sum(self.seg_lengths)):.6f} m",
            f"  half widths            : {self.seg_halfwidths[0]:.6f} m (base) ... "
            f"{self.seg_halfwidths[-1]:.6f} m (tip)",
        ]
        return "\n".join(lines)


class SpiralCalculator:
    """
    Compute and discretize a logarithmic spiral for the soft robotic chain.

    This class encapsulates the complete mathematical handling:
    solving the spiral parameters, computing radii, angles, and
    discretizing the curve into mechanical segments.

    Parameters
    ----------
    L_target : float
        Desired total chain length.
    base_d : float
        Diameter at the base.
    tip_d : float
        Diameter at the tip.
    Delta_theta_deg : float
        Angular resolution in degrees for discretization.
    """

    def __init__(self, L_target, base_d, tip_d, Delta_theta_deg):
        self.L_target = L_target
        self.base_d = base_d
        self.tip_d = tip_d
        self.Delta_theta = np.deg2rad(Delta_theta_deg)

        self.params = ms.SpiralParams(
            base_d=self.base_d,
            tip_d=self.tip_d,
            L_target=self.L_target
        )

    def compute_geometry(self) -> SpiralGeometry:
        """
        Compute all geometric quantities of the logarithmic spiral.

        This includes solving the nonlinear equation for the spiral growth
        parameter ``b``, computing the spiral radius function, taper angle,
        centerline arc length, and discretizing the curve into segments.

        Returns
        -------
        SpiralGeometry
            A container holding all geometric values such as segment lengths,
            radii, taper angles, and discretization parameters.
        """

        f = ms.make_f_of_b(self.params)
        b_sol = bisect(f, 1e-4, 1.0, xtol=1e-12, rtol=1e-12, maxiter=200)

        theta0 = ms.theta0_from_ratio(
            b_sol,
            base_d=self.base_d,
            tip_d=self.tip_d
        )
        a = ms.a_from_tip(b_sol, tip_d=self.tip_d)
        L_check = ms.length_central(a, b_sol, theta0)
        phi_taper = ms.taper_angle_phi(b_sol)

        # Discretisation
        N_cont = (1.0 / (b_sol * self.Delta_theta)) * np.log(self.base_d / self.tip_d)
        N = max(1, int(np.round(N_cont)))
        new_Delta_theta = theta0 / N
        thetas = np.linspace(0.0, theta0, N + 1)

        delta_vals = ms.delta_width(thetas, a=a, b=b_sol)
        factor = np.sqrt(b_sol**2 + 1.0) / b_sol

        s_vals = factor * (ms.rho_c(thetas, a=a, b=b_sol) - ms.rho_c(0.0, a=a, b=b_sol))
        Y_vals = L_check - s_vals
        w_vals = 0.5 * delta_vals

        seg_lengths = (Y_vals[:-1] - Y_vals[1:])
        seg_halfwidths = w_vals[:-1]

        beta = np.exp(b_sol * new_Delta_theta)

        return SpiralGeometry(
            seg_lengths=seg_lengths,
            seg_halfwidths=seg_halfwidths,
            phi_taper=phi_taper,
            new_Delta_theta=new_Delta_theta,
            beta=beta,
            b=b_sol,
            a=a,
            theta0=theta0,
            L_check=L_check,
        )


class SensorRegistry:
    _registry: dict[str, str] = {}

    @classmethod
    def register(cls, key: str, tag: str):
        cls._registry[key] = tag

    @classmethod
    def get_xml_tag(cls, key: str) -> str:
        if key not in cls._registry:
            raise ValueError(f"Unknown sensor key: {key}")
        return cls._registry[key]
    
    @classmethod
    def exists(cls, key: str) -> bool:
        return key in cls._registry

    @classmethod
    def allowed(cls):
        return list(cls._registry.keys())


# ===================================================================
# 3) XML emitters (formerly body_block, tendons_xml, ...)
# ===================================================================

# can also be build with Mujoco Spec API. Here, we keep it simple with string building.

class XMLBuilder:
    """
    Build a complete MJCF string for a discretized spiral soft robot.

    Parameters
    ----------
    seg_lengths : array_like
        Length of each segment.
    seg_halfwidths : array_like
        Half-width (radius) for each segment.
    phi_taper : float
        Taper angle of the spiral.
    Delta_theta : float
        Angular step between segments.
    beta : float
        Radial growth factor per segment.
    """
    NUM_CABLES = 2
    SITE_SIZE = 0.001

    def __init__(self, seg_lengths, seg_halfwidths, phi_taper, Delta_theta, beta):
        self.seg_lengths = seg_lengths
        self.seg_halfwidths = seg_halfwidths
        self.phi_taper = phi_taper
        self.Delta_theta = Delta_theta
        self.beta = beta
    # ---------- Helper blocks ----------

    def mjcf_header(self, model_name):
        """
        Create the MJCF header including worldbody and base.

        Parameters
        ----------
        model_name : str
            Name of the root MJCF model.

        Returns
        -------
        str
            XML snippet.
        """
        rng = np.rad2deg(self.Delta_theta)
        return f'''<mujoco model="{model_name}">
  <compiler/>
  <option timestep="0.005" gravity="0 0 -9.81" impratio="10" iterations="50"/>
  <default>
    <joint damping="0.2" stiffness="0.01" limited="true" range="{-rng+5} {rng-5}" solimplimit="0.9 0.95 0.001" solreflimit="0.02 0.5" armature="0.01"/>
    <geom contype="1" conaffinity="1"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="plane" size="2 2 0.1" rgba="0 0 1 0.6" contype="1" conaffinity="1"/>
      <!-- The actual chain root is created here as a child -->
'''

    def mjcf_footer(self):
        """Return the closing MJCF tag."""
        return "</mujoco>\n"

    def worldbody_footer(self):
        """Return closing tags for worldbody."""
        return "    </body>\n  </worldbody>\n"

    # ---------- Body segment ----------


    def body_block(self, i, seg_len, half_width, add_color=False):
        """
        Generate the MJCF XML block for a single chain segment.

        Parameters
        ----------
        i : int
            Segment index.
        seg_len : float
            Length of the segment.
        half_width : float
            Half-width (radius) of the segment.
        add_color : bool, optional
            Whether to use an alternating color scheme.

        Returns
        -------
        str
            XML snippet representing the segment.
        """
        half_seg_len = seg_len / 2.0
        hx, hy, hz = float(half_width), float(half_width), float(half_seg_len)
        
        a = -np.tan(np.pi/2 - (self.phi_taper/2))
        b = -1
        c = np.tan(np.pi/2 - (self.phi_taper/2)) * ((half_width * self.beta) - 0.008)

        solv0 = ms.solve_for_points(a=a, b=b, c=c, theta_deg=np.rad2deg(self.Delta_theta))
        #xC0, yC0 = solv0["C"]
        xD0, yD0 = solv0["D"]
        x_in, y_in, z_in = xD0, 0, yD0

        c = np.tan(np.pi/2 - (self.phi_taper/2)) * ((half_width) - 0.008)
        solv1 = ms.solve_for_points(a=a, b=b, c=c, theta_deg=np.rad2deg(self.Delta_theta))
        xC1, yC1 = solv1["C"]
        #xD1, yD1 = solv1["D"]
        x_out, y_out, z_out = xC1, 0, yC1 + seg_len

        rgba = '0.6 0.75 0.95 0.3' if add_color else '0.2 0.7 0.2 0.3'

        N = len(self.seg_lengths)

        if i==N-1:
            if SensorRegistry.exists("acc") or SensorRegistry.exists("gyro"):
                return f'''      <body name="seg_{i}" pos="0 0 {seg_len * self.beta:.6g}">
                <geom name="g_{i}" type="box" size="{hx:.6g} {hy:.6g} {hz:.6g}" pos="0 0 {hz:.6g}" 
                    rgba="{rgba}" contype="1" conaffinity="1" density="1100"/>
                <site name="site_imu_{i}"  pos="0 0 {hz:.6g}" size="{self.SITE_SIZE}" rgba="1 0 1 1"/>
                <site name="site_in_{i}_0"  pos="{x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
                <site name="site_out_{i}_0" pos="{x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
                <site name="site_in_{i}_1"  pos="{-x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
                <site name="site_out_{i}_1" pos="{-x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
'''
            return f'''      <body name="seg_{i}" pos="0 0 0">
            <geom name="g_{i}" type="box" size="{hx:.6g} {hy:.6g} {hz:.6g}" pos="0 0 {hz:.6g}" 
                  rgba="{rgba}" contype="1" conaffinity="1" density="1100"/>
            <site name="site_in_{i}_0"  pos="{x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_out_{i}_0" pos="{x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_in_{i}_1"  pos="{-x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_out_{i}_1" pos="{-x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
'''
        if SensorRegistry.exists("acc") or SensorRegistry.exists("gyro"):
            return f'''      <body name="seg_{i}" pos="0 0 {seg_len * self.beta:.6g}">
            <joint name="j_{i}" type="hinge" axis="0 1 0" pos="0 0 0" stiffness="0.05" damping="0.05"
                   limited="true" range="{-np.rad2deg(self.Delta_theta)+0.02} {np.rad2deg(self.Delta_theta)-0.02}"
                   solimplimit="0.9 0.95 0.001" solreflimit="0.01 0.5"/>
            <geom name="g_{i}" type="box" size="{hx:.6g} {hy:.6g} {hz:.6g}" pos="0 0 {hz:.6g}" 
                  rgba="{rgba}" contype="1" conaffinity="1" density="1100"/>
            <site name="site_imu_{i}"  pos="0 0 {hz:.6g}" size="{self.SITE_SIZE}" rgba="1 0 1 1"/>
            <site name="site_in_{i}_0"  pos="{x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_out_{i}_0" pos="{x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_in_{i}_1"  pos="{-x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_out_{i}_1" pos="{-x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
'''

        return f'''      <body name="seg_{i}" pos="0 0 {seg_len * self.beta:.6g}">
            <joint name="j_{i}" type="hinge" axis="0 1 0" pos="0 0 0" stiffness="0.05" damping="0.05"
                   limited="true" range="{-np.rad2deg(self.Delta_theta)+0.02} {np.rad2deg(self.Delta_theta)-0.02}"
                   solimplimit="0.9 0.95 0.001" solreflimit="0.01 0.5"/>
            <geom name="g_{i}" type="box" size="{hx:.6g} {hy:.6g} {hz:.6g}" pos="0 0 {hz:.6g}" 
                  rgba="{rgba}" contype="1" conaffinity="1" density="1100"/>
            <site name="site_in_{i}_0"  pos="{x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_out_{i}_0" pos="{x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_in_{i}_1"  pos="{-x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_out_{i}_1" pos="{-x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
'''


    def close_body_block(self):
        """
        Emit the closing MJCF block of a body segment.

        This block closes both the inner helper body element
        both the inner end element (``seg_i_end``) and the outer segment (``seg_i``).

        Returns
        -------
        str
            XML fragment closing the open ``<body>`` tags.
        """
        return "        </body>\n"

    # ---------- Contact Exclusions ----------

    def exclude_contacts_xml(self, N):
        """
        Generate XML fragment to exclude self-collisions between two segments.

        The first two segments are considered static relative to the plane, so the collision
        detection should ignore collisions between `seg_{N-1}` and `seg_{N-2}`.

        Args:
            N (int): Index of the last segment (e.g., number of segments minus 1).

        Returns:
            str: An XML string containing a `<contact>` element that excludes collision
            between `seg_{N-1}` and `seg_{N-2}`.
        """
        return f'''  <contact>
            <exclude body1="seg_{N - 1}" body2="seg_{N - 2}"/>
            </contact> 
            '''


    # ---------- Tendons / Actuators ----------

    def tendons_xml(self, N):
        """
        Emit the MJCF tendon definitions for every cable.

        One ``<spatial>`` tendon per cable (`NUM_CABLES`), routed through the
        generated ``site_in`` and ``site_out`` sites of every segment.

        Parameters
        ----------
        N : int
            Number of segments in the chain.

        Returns
        -------
        str
            XML block with every tendon definition.
        """
        out = ["  <tendon>"]
        for k in range(self.NUM_CABLES):
            out.append(f'    <spatial name="tendon_{k}" width="0.001" rgba="1 0 0 1" frictionloss="0.1" stiffness="50">')
            for i in reversed(range(N)):
                out.append(f'      <site site="site_in_{i}_{k}"/>')
                out.append(f'      <site site="site_out_{i}_{k}"/>')
            out.append("    </spatial>")
        out.append("  </tendon>\n")
        return "\n".join(out)

    def actuators_xml(self):
        """
        Emit the MJCF actuator definitions for the tendons.

        One force actuator per cable, acting on that tendon.

        Returns
        -------
        str
            XML block with every actuator definition.
        """
        out = ["  <actuator>"]
        for k in range(self.NUM_CABLES):
            out.append(
                f'    <motor name="tendon_act_{k}" tendon="tendon_{k}" '
                f'gear="1" forcerange="-200 200" ctrlrange="-50 0"/>'
            )
        out.append("  </actuator>\n")
        return "\n".join(out)

    def sensors_xml(self):
        """
        Emit the MJCF sensor definitions for the tendons.

        Position and velocity sensors are emitted for every cable
        of the corresponding tendon.

        Returns
        -------
        str
            XML block with every sensor definition.
        """
        N = len(self.seg_lengths)
        out = ["  <sensor>"]
        for k in range(self.NUM_CABLES):
            if SensorRegistry.exists("tendon_frc"):
                out.append(f'<{SensorRegistry.get_xml_tag("tendon_frc")} name="tendonfrc_{k}" tendon="tendon_{k}"/>')
            if SensorRegistry.exists("tendon_pos"):
                out.append(f'<{SensorRegistry.get_xml_tag("tendon_pos")} name="tendonpos_{k}" tendon="tendon_{k}"/>')
            if SensorRegistry.exists("tendon_vel"):
                out.append(f'<{SensorRegistry.get_xml_tag("tendon_vel")} name="tendonvel_{k}" tendon="tendon_{k}"/>')
        for i in range(N-1):
            if SensorRegistry.exists("angle"):
                out.append(f'<{SensorRegistry.get_xml_tag("angle")} name="jointpos_{i}" joint="j_{i}"/>')
            if SensorRegistry.exists("joint_vel"):
                out.append(f'<{SensorRegistry.get_xml_tag("joint_vel")} name="jointvel_{i}" joint="j_{i}"/>')
        for i in range(N):
            if SensorRegistry.exists("gyro"):
                out.append(f'<{SensorRegistry.get_xml_tag("gyro")} name="gyro_{i}" site="site_imu_{i}"/>')
            if SensorRegistry.exists("acc"):
                out.append(f'<{SensorRegistry.get_xml_tag("acc")} name="acc_{i}" site="site_imu_{i}"/>')
            if SensorRegistry.exists("frc"):
                out.append(f'<{SensorRegistry.get_xml_tag("frc")} name="frc_{i}" site="site_imu_{i}"/>')
        out.append("  </sensor>\n")
        return "\n".join(out)

    # ---------- Gesamt-XML ----------

    def build(self, model_name="spiral_chain"):
        """
        Assemble the complete MJCF XML model.

        The builder joins all parts (body segments, tendons, actuators,
        sensors and the header/footer blocks) into one MuJoCo XML model.

        Parameters
        ----------
        model_name : str, optional
            Name of the MJCF model (default: ``"spiral_chain"``).

        Returns
        -------
        str
            The complete MJCF XML as a string.
        """

        N = len(self.seg_lengths)
        xml = [self.mjcf_header(model_name)]

        # Generate bodies
        for i in reversed(range(N)):
            xml.append(
                self.body_block(
                    i,
                    self.seg_lengths[i],
                    self.seg_halfwidths[i],
                    add_color=(i % 2 == 0)
                )
            )

        # Close bodies
        for _ in reversed(range(N)):
            xml.append(self.close_body_block())

        xml.append(self.worldbody_footer())
        xml.append(self.exclude_contacts_xml(N))
        xml.append(self.tendons_xml(N))
        xml.append(self.actuators_xml())
        xml.append(self.sensors_xml())
        xml.append(self.mjcf_footer())

        return "".join(xml)

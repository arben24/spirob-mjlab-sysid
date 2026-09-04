"""SpiRob — logarithmic-spiral continuum robot: model generation and simulation.

Public API. Import from here, not from the submodules::

    import spirob

    xml = spirob.generate_xml_string(L_target=0.44, base_d=0.10, tip_d=0.03,
                                     Delta_theta_deg=30.0)

Layers
------
``spirob.spiral``            pure logarithmic-spiral math (no MuJoCo)
``spirob.generator``         geometry solver + MuJoCo XML builder
``spirob.simulate``          rollout loop, tendon controllers, contact forces
``spirob.data_schema``       Pydantic records for experiment metadata
``spirob.exporter``          atomic ``data.parquet`` + ``meta.json`` per run
``spirob.analyzer``          lazy loading of stored runs
``spirob.plotstyle``         shared Matplotlib style for publication figures
"""

from .analyzer import load_experiment  # noqa: F401
from .exporter import generate_sensor_meta, save_experiment  # noqa: F401
from .generator import (  # noqa: F401
    SensorRegistry,
    SpiralCalculator,
    SpiralGeometry,
    XMLBuilder,
    generate_and_save_xml,
    generate_xml_string,
)
from .simulate import *  # noqa: F401,F403
from .spiral import *  # noqa: F401,F403

__version__ = "0.1.0"

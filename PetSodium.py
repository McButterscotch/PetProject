import opengate as gate
from pathlib import Path
import opengate.contrib.pet.philipsvereos as pet_vereos
from scripts.pet_helpers import add_vereos_digitizer_v1
from opengate.geometry.utility import get_circular_repetition

if __name__ == "__main__":
    sim = gate.Simulation()

    # ---- options
    simple = True                       # 2 modules vs 18
    sim.visu = True                     # visualization (see note below)
    sim.visu_type = "vrml"              # or "vrml_file_only" to avoid GUI
    sim.visu_filename = "scene.wrl"     # Gate 10: required to save VRML/GDML
    sim.random_seed = "auto"
    sim.number_of_threads = 1

    # ---- units
    m   = gate.g4_units.m
    mm  = gate.g4_units.mm
    cm  = gate.g4_units.cm
    sec = gate.g4_units.s
    Bq  = gate.g4_units.Bq

    # ---- folders
    output_path = Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- world
    world = sim.world
    world.size = [2 * m, 2 * m, 2 * m]
    world.material = "G4_AIR"

    # ---- PET geometry (Philips Vereos contrib)
    pet = pet_vereos.add_pet(sim, "pet")

    # ---- ring repetitions: either 2 modules or 18 modules
    module = sim.volume_manager.get_volume("pet_module")
    
    if simple:
        translations_ring, rotations_ring = get_circular_repetition(
            2, [391.5 * mm, 0, 0], start_angle_deg=190, axis=[0, 0, 1]
        )
    else:
        translations_ring, rotations_ring = get_circular_repetition(
            18, [391.5 * mm, 0, 0], start_angle_deg=190, axis=[0, 0, 1]
        )
    module.translation = translations_ring
    module.rotation = rotations_ring

    # ---- Sources (two Na-22 positron sources)
    # Gate 10 helper for β+ yield fraction:
    # total_yield = gate.sources.generic.get_rad_yield("Na22")

    source1 = sim.add_source("GenericSource", "hot_sphere_source_1")
    source1.attached_to = "world"
    source1.particle = "e+"
    source1.energy.type = "Na22"          # built-in β+ spectrum
    source1.activity = 100000 * Bq
    source1.half_life = 2.6 * 365.25 * 24 * 3600 * sec
    source1.position.translation = [-5 * cm, 0, 0]

    source2 = sim.add_source("GenericSource", "hot_sphere_source_2")
    source2.attached_to = "world"
    source2.particle = "e+"
    source2.energy.type = "Na22"
    source2.activity = 100000 * Bq
    source2.half_life = 2.6 * 365.25 * 24 * 3600 * sec
    source2.position.translation = [5 * cm, 0, 0]

    # ---- physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut("world", "all", 1 * m)

    # ---- PET digitizer (your helper)
    output_root = output_path / "output_vereos.root"
    add_vereos_digitizer_v1(sim, pet, output_root)

    # ---- stats actor
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.track_types_flag = True
    # IMPORTANT: use output_filename (Path or str). Do NOT use user_output.
    stats.output_filename = output_path / "stats_vereos.txt"

    # ---- timing
    sim.run_timing_intervals = [[0, 2.0 * sec]]

    # ---- go
    sim.run()

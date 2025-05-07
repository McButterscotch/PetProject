import opengate as gate
from pathlib import Path
import opengate.contrib.pet.philipsvereos as pet_vereos
from pet_helpers import add_vereos_digitizer_v1
from opengate.geometry.utility import get_grid_repetition, get_circular_repetition


if __name__ == "__main__":
    sim = gate.Simulation()

    # options
    # warning the visualisation is slow !
    simple = True 
    sim.visu = True
    sim.visu_type = "vrml"
    sim.random_seed = "auto"
    sim.number_of_threads = 1

    # units
    m = gate.g4_units.m
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    sec = gate.g4_units.s
    ps = gate.g4_units.ps
    keV = gate.g4_units.keV
    Bq = gate.g4_units.Bq
    gcm3 = gate.g4_units.g_cm3

    # folders
    data_path = Path("data")
    output_path = Path("output")
    

    # world
    world = sim.world
    world.size = [2 * m, 2 * m, 2 * m]
    world.material = "G4_AIR"

    pet = pet_vereos.add_pet(sim, "pet")
    
    # We want two systems: One with 18 modules and one with 2 modules
    if simple:
        module = sim.volume_manager.get_volume("pet_module")
        # only 2 repetition instead of 18   
        translations_ring, rotations_ring = get_circular_repetition(
            2, [391.5 * mm, 0, 0], start_angle_deg=190, axis=[0, 0, 1]
        )
        module.translation = translations_ring
        module.rotation = rotations_ring
    else:
        module = sim.volume_manager.get_volume("pet_module")
        # 18 repetitions
        translations_ring, rotations_ring = get_circular_repetition(
            18, [391.5 * mm, 0, 0], start_angle_deg=190, axis=[0, 0, 1]
        )
        module.translation = translations_ring
        module.rotation = rotations_ring

    # source is attached to world. Without translation, its in the center.
    source1 = sim.add_source("GenericSource", "hot_sphere_source 1")
    source1.attached_to = "world"
    total_yield1 = gate.sources.base.get_rad_yield("Na22")
    source1.particle = "e+"
    source1.energy.type = "Na22"
    source1.activity = 20 * Bq * total_yield1 
    source1.half_life = 2.6 * 365.25 * 1440 * sec
    source1.position.translation = [-5 * cm, 0 * cm, 0 * cm]
    # source1.position.translation = [-5 * cm, -2 * cm, 0 * cm]
    
    source2 = sim.add_source("GenericSource", "hot_sphere_source 2")
    total_yield2 = gate.sources.base.get_rad_yield("Na22")
    source2.attached_to = "world"
    source2.particle = "e+"
    source2.energy.type = "Na22"
    source2.activity = 20 * Bq * total_yield2
    source2.half_life = 2.6 * 365.25 * 1440 * sec
    source2.position.translation = [5 * cm, 0 * cm, 0 * cm]   
    # source2.position.translation = [5 * cm, 2 * cm, 0 * cm]
    

    # physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut("world", "all", 1 * m)

    # add the PET digitizer.
    output = output_path / f"output_vereos.root"
    add_vereos_digitizer_v1(sim, pet, output)

    # add stat actor - Apparently this crashes the simulation. Do know why.
    # stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    # stats.track_types_flag = True
    # stats.user_output = Path("output") / "stats_vereos.txt"

    # timing
    sim.run_timing_intervals = [[0, 2.0 * sec]]
    
    # go
    sim.run()
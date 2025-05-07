import opengate as gate
from pathlib import Path
import opengate.contrib.pet.philipsvereos as pet_vereos
from pet_helpers import add_vereos_digitizer_v1
from opengate.geometry.utility import get_grid_repetition, get_circular_repetition


if __name__ == "__main__":
    sim = gate.Simulation()

    # options
    # warning the visualisation is slow !
    sim.visu = False
    simple = True  
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

    # add the Philips Vereos PET
    pet = pet_vereos.add_pet(sim, "pet")

    # If visu is enabled, we simplified the PET system, otherwise it is too slow
    if sim.visu:
        module = sim.volume_manager.get_volume("pet_module")
        # only 2 repetition instead of 18
        translations_ring, rotations_ring = get_circular_repetition(
        2, [391.5 * mm, 0, 0], start_angle_deg=190, axis=[0, 0, 1]
        )
        module.translation = translations_ring
        module.rotation = rotations_ring
        
      
    
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

    # add table
    bed = pet_vereos.add_table(sim, "pet")

    # add a simple waterbox with a hot sphere inside
    waterbox = sim.add_volume("Box", "waterbox")
    waterbox.size = [10 * cm, 10 * cm, 10 * cm]
    waterbox.translation = [0 * cm, -10 * cm, 0 * cm]
    waterbox.material = "G4_WATER"
    waterbox.color = [0, 0, 1, 1]

    hot_sphere = sim.add_volume("Sphere", "hot_sphere")
    hot_sphere.mother = waterbox.name
    hot_sphere.radius = 5 * cm
    hot_sphere.material = "G4_WATER"
    hot_sphere.color = [1, 0, 0, 1]

    # source for tests
    source = sim.add_source("GenericSource", "hot_sphere_source")
    total_yield = gate.sources.base.get_rad_yield("F18")
    print("Yield for F18 (nb of e+ per decay) : ", total_yield)
    source.particle = "e+"
    source.energy.type = "F18"
    source.activity = 1e3 * Bq * total_yield 
    if sim.visu:
        source.activity = 1e2 * Bq * total_yield
    source.half_life = 6586.26 * sec

    # physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut("world", "all", 1 * m)
    sim.physics_manager.set_production_cut("waterbox", "all", 1 * mm)

    # add the PET digitizer.
    output = output_path / f"output_vereos.root"
    add_vereos_digitizer_v1(sim, pet, output)

    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.authorize_repeated_volumes = True
    stats.track_types_flag = True
    stats.output = Path("output") / "stats_vereos.txt"

    # timing
    sim.run_timing_intervals = [[0, 2.0 * sec]]

    # go
    sim.run()

    # end
    """print(f"Output statistics are in {stats.output}")
    print(f"Output edep map is in {dose.output}")
    print(f"vv {ct.image} --fusion {dose.output}")
    stats = sim.output.get_actor("Stats")
    print(stats)"""

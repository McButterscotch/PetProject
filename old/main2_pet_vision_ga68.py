#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from pathlib import Path
import opengate.contrib.pet.philipsvereos as pet_vereos
import opengate.contrib.pet.siemensbiograph as pet_vision
import opengate.contrib.phantoms.nemaiec as gate_iec
from opengate.geometry.utility import get_grid_repetition

if __name__ == "__main__":
    sim = gate.Simulation()

    # options
    # warning the visualisation is slow !
    sim.visu = False
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
    cm3 = gate.g4_units.cm3
    gcm3 = gate.g4_units.g_cm3
    BqmL = Bq / cm3

    # folders
    data_path = Path("data")
    output_path = Path("output")

    # world
    world = sim.world
    world.size = [2 * m, 2 * m, 2 * m]
    world.material = "G4_AIR"

    # add the Philips Vereos PET
    pet = pet_vision.add_pet(sim, "pet")

    # If visu is enabled, we simplified the PET system, otherwise it is too slow
    if sim.visu:
        crystal = sim.volume_manager.get_volume("pet_crystal")
        # 3x3 crystal instead of 13x13
        crystal.translation = get_grid_repetition([1, 3, 3], [0, 4 * mm, 4 * mm])

    # add table
    bed = pet_vereos.add_table(sim, "pet")

    # add IEC phantom
    iec_phantom = gate_iec.add_iec_phantom(sim, "iec")
    iec_phantom.translation = [0 * cm, 0 * cm, 0 * cm]

    # source for tests
    total_yield = gate.sources.generic.get_rad_yield("Ga68")
    print("Yield for Ga68 (nb of e+ per decay) : ", total_yield)
    a = 1e2 * BqmL * total_yield
    if sim.visu:
        a = 1e-1 * BqmL * total_yield
    activities = [a]*6
    sources = gate_iec.add_spheres_sources(sim, "iec", "iec_source", "all", activities)
    for source in sources:
        source.particle = "e+"
        source.energy.type = "Ga68"
        source.half_life = 67.71 * 60 * sec

    # physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut("world", "all", 1 * m)
    sim.physics_manager.set_production_cut("iec", "all", 1 * mm)

    # add the PET digitizer.
    output = output_path / f"output_vision.root"
    pet_vision.add_digitizer(sim, pet.name, output)

    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.track_types_flag = True
    stats.output = Path("output") / "stats_vision.txt"

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

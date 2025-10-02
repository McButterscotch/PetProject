import opengate as gate
import scipy
import numpy as np
import re

from pathlib import Path
import opengate.contrib.pet.philipsvereos as pet_vereos
import opengate.contrib.phantoms.necr as phantom_necr


def add_vereos_digitizer_v1(sim, pet, output, energy_resolution=0.112, time_resolution=220.0):
    """
    add a  PET digitizer.

    It is composed of a chain of several modules.
    - Module1: HitsCollection
    - Module2: Readout
    - Module3: Efficiency
    - Module4: energy resolution
    - Module5: time resolution
    - Module6: energy selection

    This is a simplified digitizer : no noise, no piles-up, no dead-time
    """

    # units
    keV = gate.g4_units.keV
    ps = gate.g4_units.ps
    ns = gate.g4_units.ns

    # Hits
    crystal = sim.volume_manager.get_volume(f"{pet.name}_crystal")
    hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
    hc.authorize_repeated_volumes = True
    hc.attached_to = crystal
    hc.output_filename = output
    hc.attributes = [
        "PostPosition",
        "TotalEnergyDeposit",
        "PreStepUniqueVolumeID",
        "GlobalTime",
        "LocalTime",
    ]

    # Readout
    module = sim.volume_manager.get_volume(f"{pet.name}_module")
    sc = sim.add_actor("DigitizerReadoutActor", "Singles1")
    sc.output_filename = output
    sc.input_digi_collection = "Hits"
    sc.group_volume = module.name
    sc.discretize_volume = crystal.name
    sc.policy = "EnergyWeightedCentroidPosition"
    # policies : EnergyWeightedCentroidPosition or EnergyWinnerPosition

    # Detection Efficiency
    ea = sim.add_actor("DigitizerEfficiencyActor", "Singles2")
    ea.authorize_repeated_volumes = True
    ea.output_filename = output
    ea.input_digi_collection = "Singles1"
    ea.efficiency = 0.86481

    # energy blurring
    eb = sim.add_actor("DigitizerBlurringActor", "Singles3")
    eb.authorize_repeated_volumes = True
    eb.output_filename = output
    eb.input_digi_collection = "Singles2"
    eb.blur_attribute = "TotalEnergyDeposit"
    eb.blur_method = "InverseSquare"
    eb.blur_resolution = energy_resolution
    eb.blur_reference_value = 511 * keV

    # time blurring
    tb = sim.add_actor("DigitizerBlurringActor", "Singles4")
    tb.authorize_repeated_volumes = True
    tb.output_filename = output
    tb.input_digi_collection = "Singles3"
    tb.blur_attribute = "GlobalTime"
    tb.blur_method = "Gaussian"
    tb.blur_fwhm = time_resolution * ps
    #tb.blur_fwhm = 220.0 * ns

    # EnergyWindows
    ew = sim.add_actor("DigitizerEnergyWindowsActor", "Singles5")
    ew.authorize_repeated_volumes = True
    ew.output_filename = output
    ew.input_digi_collection = tb.name
    ew.channels = [{"name": ew.name, "min": 449.68 * keV, "max": 613.20 * keV}]




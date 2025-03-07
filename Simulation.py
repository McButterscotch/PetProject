import opengate as gate

if __name__ == "__main__":
    sim = gate.Simulation()
    sim.output_dir = "/data"

    wb = sim.add_volume("Box", name="waterbox")
    # configure the volume ...
    cm = gate.g4_units.cm
    wb.size = [10 * cm, 5 * cm, 10 * cm]
    # ...

    source = sim.add_source("GenericSource", name="Default")
    MeV = gate.g4_units.MeV
    Bq = gate.g4_units.Bq
    source.particle = "proton"
    source.activity = 100 * Bq
    source.energy.mono = 240 * gate.g4_units.MeV
    # ...

    stats = sim.add_actor("SimulationStatisticsActor", "Stats")

    spectrum = gate.sources.utility.get_spectrum("F18")

    sim.run()
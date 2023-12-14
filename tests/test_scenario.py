from pathlib import Path

from numpy.random import default_rng

from environment.scenario import Scenario


def test_random_create():
    Scenario.create_random(400, default_rng())


def test_calculate_specials():
    scenario = Scenario.create_random(400, default_rng())
    scenario.calculate_specials()


def test_create_visualisation():
    scenario = Scenario.create_random(400, default_rng())
    scenario.calculate_specials()
    scenario.generate_visualisation()


def test_save_and_load():
    scenario = Scenario.create_random(400, default_rng())
    scenario.calculate_specials()
    scenario.generate_visualisation()
    scenario.to_directory(Path("tests/test_scenario"))
    scenario = Scenario.from_directory(Path("tests/test_scenario"))

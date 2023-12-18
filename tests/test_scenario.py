from pathlib import Path
import shutil

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
    bid = next(scenario.iter_bids())
    utility_before = scenario.get_utilities(bid)
    scenario.to_directory(Path("tests/test_scenarios/utility"))
    scenario.calculate_specials()
    scenario.to_directory(Path("tests/test_scenarios/specials"))
    scenario.generate_visualisation()
    scenario.to_directory(Path("tests/test_scenarios/full"))
    scenario = Scenario.from_directory(Path("tests/test_scenarios/full"))
    utility_after = scenario.get_utilities(bid)
    assert utility_before == utility_after
    shutil.copyfile(
        Path("tests/test_scenarios/full/objectives.json"),
        Path("tests/test_scenarios/objectives/objectives.json"),
    )
    scenario = Scenario.from_directory(Path("tests/test_scenarios/objectives"))
    utility_after = scenario.get_utilities(bid)
    assert utility_before != utility_after

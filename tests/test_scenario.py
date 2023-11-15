from environment.scenario import Scenario


def test_random_create():
    Scenario.create_random("test", 400)


def test_calculate_specials():
    scenario = Scenario.create_random("test", 400)
    scenario.calculate_specials()

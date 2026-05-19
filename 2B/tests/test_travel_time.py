import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from travel_time import flow_to_speed, travel_time, SPEED_LIMIT, CAPACITY_SPEED


def test_zero_flow_at_speed_limit():
    assert flow_to_speed(0) == SPEED_LIMIT


def test_low_flow_capped_at_limit():
    assert flow_to_speed(200) == SPEED_LIMIT


def test_capacity_flow_near_capacity_speed():
    s = flow_to_speed(1500)
    assert 30 <= s <= 35


def test_over_capacity_slows_down():
    assert flow_to_speed(2000) < flow_to_speed(500)


def test_travel_time_includes_intersection_delay():
    t = travel_time(0, 1.0)
    assert t >= 30


def test_longer_distance_more_time():
    assert travel_time(500, 2.0) > travel_time(500, 1.0)

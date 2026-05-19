from math import sqrt

SPEED_LIMIT = 60
CAPACITY_FLOW = 1500
CAPACITY_SPEED = 32
INTERSECTION_DELAY = 30

# flow (veh/hr) = A_COEFF * speed^2 + B_COEFF * speed  (fundamental diagram)
A_COEFF = -1.4648375
B_COEFF = 93.75


def flow_to_speed(flow_per_hour):
    """
    Convert hourly traffic flow to speed (km/h) using the assignment diagram.
    Uses the higher-speed root when under capacity, lower-speed when over capacity.
    """
    # Rearrange: |A_COEFF|*speed^2 - B_COEFF*speed + flow = 0
    a = abs(A_COEFF)
    b = B_COEFF
    discriminant = b * b - 4 * a * flow_per_hour

    if discriminant < 0:
        return 1.0

    sqrt_d = sqrt(discriminant)

    if flow_per_hour <= CAPACITY_FLOW:
        # Green branch — higher speed
        speed = (b + sqrt_d) / (2 * a)
    else:
        # Red branch — lower speed (congested)
        speed = (b - sqrt_d) / (2 * a)

    return max(1.0, min(speed, SPEED_LIMIT))


def travel_time(flow_per_hour, distance_km):
    """Travel time in seconds for one link plus intersection delay."""
    speed = flow_to_speed(flow_per_hour)
    time_sec = (distance_km / speed) * 3600
    return time_sec + INTERSECTION_DELAY


if __name__ == "__main__":
    print(f"Flow=0:    speed={flow_to_speed(0):.1f} km/h")
    print(f"Flow=351:  speed={flow_to_speed(351):.1f} km/h")
    print(f"Flow=1500: speed={flow_to_speed(1500):.1f} km/h")
    print(f"Flow=2000: speed={flow_to_speed(2000):.1f} km/h")
    print(f"\n1km at flow=500:  {travel_time(500, 1):.1f} seconds")
    print(f"1km at flow=1500: {travel_time(1500, 1):.1f} seconds")

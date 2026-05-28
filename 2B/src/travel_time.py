from math import sqrt

# ─────────────────────────────────────────────
# Constants  (from Traffic Flow to Travel Time v1.0 PDF)
# ─────────────────────────────────────────────
SPEED_LIMIT        = 60      # km/h — speed cap when flow is low
CAPACITY_FLOW      = 1500    # veh/hr — flow at capacity (turning point)
CAPACITY_SPEED     = 32      # km/h  — speed at capacity
INTERSECTION_DELAY = 30      # seconds added per intersection

# Quadratic coefficients for: flow = A·speed² + B·speed
A = -CAPACITY_FLOW / (CAPACITY_SPEED ** 2)   # ≈ -1.46484375
B = -2 * CAPACITY_SPEED * A                  # =  93.75


# ─────────────────────────────────────────────
# Flow → Speed conversion
# ─────────────────────────────────────────────
def flow_to_speed(flow_per_hour: float) -> float:
    """
    Convert predicted traffic flow (veh/hr) to estimated speed (km/h)
    using the simplified fundamental diagram parabola.

    The parabola  flow = A·s² + B·s  has two branches:
      Green (under capacity) : higher-speed root — road is not yet congested
      Red   (over capacity)  : lower-speed root  — traffic breakdown

    Solving A·s² + B·s − flow = 0 via the quadratic formula:
        discriminant = B² − 4·A·(−flow) = B² + 4·A·flow
        s = (−B ± √discriminant) / (2·A)

    Because A < 0, the higher-speed root uses the '−' sign in the numerator:
        s_high = (−B − √discriminant) / (2·A)   ← green branch
        s_low  = (−B + √discriminant) / (2·A)   ← red branch

    Edge cases
    ----------
    • flow ≤ ~351 veh/hr : green branch gives speed > 60 → capped at SPEED_LIMIT
    • flow = 1500 veh/hr : discriminant = 0 → both roots = 32 km/h (capacity)
    • flow > 1500 veh/hr : discriminant < 0 (model breaks down) → return CAPACITY_SPEED

    Parameters
    ----------
    flow_per_hour : float — predicted flow in vehicles per hour

    Returns
    -------
    float — estimated speed in km/h, bounded to [1.0, SPEED_LIMIT]
    """
    # Discriminant of A·s² + B·s − flow = 0
    discriminant = B ** 2 + 4 * A * flow_per_hour

    if discriminant < 0:
        # Flow exceeds the model's capacity — return capacity speed
        return float(CAPACITY_SPEED)

    sqrt_disc = sqrt(discriminant)

    if flow_per_hour <= CAPACITY_FLOW:
        # Under/at capacity — green branch (higher speed root)
        speed = (-B - sqrt_disc) / (2 * A)
    else:
        # Over capacity — red branch (lower speed root)
        speed = (-B + sqrt_disc) / (2 * A)

    return max(1.0, min(speed, SPEED_LIMIT))


# ─────────────────────────────────────────────
# Travel time for one road segment
# ─────────────────────────────────────────────
def travel_time(flow_per_hour: float, distance_km: float) -> float:
    """
    Estimate travel time (seconds) for one road segment.

    Parameters
    ----------
    flow_per_hour : float — traffic flow in vehicles per hour
                            (convert from veh/15min by multiplying × 4)
    distance_km   : float — segment length in kilometres

    Returns
    -------
    float — travel time in seconds (driving time + intersection delay)
    """
    speed    = flow_to_speed(flow_per_hour)
    time_hrs = distance_km / speed
    time_sec = time_hrs * 3600
    return time_sec + INTERSECTION_DELAY


# ─────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Flow → Speed verification")
    print(f"  Flow=   0 veh/hr : {flow_to_speed(0):.2f} km/h  (expect 60.00)")
    print(f"  Flow= 351 veh/hr : {flow_to_speed(351):.2f} km/h  (expect ~60.00)")
    print(f"  Flow= 800 veh/hr : {flow_to_speed(800):.2f} km/h  (expect ~53.86)")
    print(f"  Flow=1500 veh/hr : {flow_to_speed(1500):.2f} km/h  (expect 32.00)")
    print(f"  Flow=2000 veh/hr : {flow_to_speed(2000):.2f} km/h  (expect 32.00 — over capacity)")

    print("\nTravel time for 1 km segment")
    print(f"  flow=   0 : {travel_time(0, 1):.1f}s  (expect ~90.0s)")
    print(f"  flow= 500 : {travel_time(500, 1):.1f}s  (expect ~92.0s)")
    print(f"  flow=1500 : {travel_time(1500, 1):.1f}s  (expect ~142.5s)")

    print("\n⚠  Note: flow_per_hour = predicted_flow_15min × 4")
    print(f"  Example: 200 veh/15min → {200*4} veh/hr → {flow_to_speed(200*4):.1f} km/h")

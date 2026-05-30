from math import sqrt
from config import cfg

# ─────────────────────────────────────────────
# Constants  (from Traffic Flow to Travel Time v1.0 PDF)
# ─────────────────────────────────────────────
SPEED_LIMIT        = cfg["travel_time"]["speed_limit_kmh"]
CAPACITY_FLOW      = cfg["travel_time"]["capacity_flow_vph"]
CAPACITY_SPEED     = cfg["travel_time"]["capacity_speed_kmh"]
INTERSECTION_DELAY = cfg["travel_time"]["intersection_delay_sec"]

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
def travel_time(flow_15min: float, distance_km: float) -> float:
    """
    Estimate travel time (minutes) for one road segment.

    Parameters
    ----------
    flow_15min : float — predicted traffic flow in vehicles per 15 minutes
    distance_km : float — segment length in kilometres

    Returns
    -------
    float — travel time in minutes (driving time + intersection delay)
    """
    flow_hr = flow_15min * 4.0
    speed = flow_to_speed(flow_hr)
    return (distance_km / speed) * 60.0 + (INTERSECTION_DELAY / 60.0)


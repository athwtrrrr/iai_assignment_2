from math import sqrt

SPEED_LIMIT     = 60     
CAPACITY_FLOW   = 1500   
CAPACITY_SPEED  = 32    
INTERSECTION_DELAY = 30  

A = -CAPACITY_FLOW / (CAPACITY_SPEED ** 2)   
B = -2 * CAPACITY_SPEED * A                  

def flow_to_speed(flow_per_hour):

    # Formula to calculate delta
    delta = B**2 - 4 * A * flow_per_hour

    if delta < 0:
        # No valid value floor at 1 to avoid division by zero
        return 1.0

    sqrt_delta = sqrt(delta)

    if flow_per_hour <= CAPACITY_FLOW:
        # Under capacity — green line — higher speed root
        speed = (-B + sqrt_delta) / (2 * A)
    else:
        # Over capacity — red line — lower speed root
        speed = (-B - sqrt_delta) / (2 * A)

    # Cap at speed limit, floor at 1 to avoid division by zero
    return max(1.0, min(speed, SPEED_LIMIT))

def travel_time(flow_per_hour, distance_km):
    
    speed    = flow_to_speed(flow_per_hour)
    time_hrs = distance_km / speed
    time_sec = time_hrs * 3600
    return time_sec + INTERSECTION_DELAY

# ── Quick sanity check ──
if __name__ == "__main__":
    # At 0 flow → should be speed limit (60 km/h)
    print(f"Flow=0:    speed={flow_to_speed(0):.1f} km/h")
    # At 351 flow → should be ~60 km/h (blue dashed line from PDF)
    print(f"Flow=351:  speed={flow_to_speed(351):.1f} km/h")
    # At 1500 flow → should be 32 km/h (capacity point)
    print(f"Flow=1500: speed={flow_to_speed(1500):.1f} km/h")
    # At 2000 flow → should be slow (over capacity)
    print(f"Flow=2000: speed={flow_to_speed(2000):.1f} km/h")

    # Travel time for 1km at different flows
    print(f"\n1km at flow=500:  {travel_time(500, 1):.1f} seconds")
    print(f"1km at flow=1500: {travel_time(1500, 1):.1f} seconds")
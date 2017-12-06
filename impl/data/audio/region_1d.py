
def remove_regions(allowed_regions, blocked_regions):
    """
    allowed_regions contains a list of "allowed" regions, e.g.:
    [(0, 6), (6.2, 7)]

    blocked_regions contains a list of regions which are bad / blocked, e.g.:
    [(3, 4), (6.1, 6.3)]

    Now we remove the blocked_regions from the allowed, then we get:
    [(0, 3), (4, 6), (6.3, 7)]

    :param allowed_regions:
    :param blocked_regions:
    :return:
    """

    # Get now all "unblocked" ranges in the input range
    for blocked_region in blocked_regions:
        allowed_regions_new = []
        for allowed_region in allowed_regions:

            # Is there an overlap? If yes: Nothing to do
            if allowed_region[1] < blocked_region[0] or allowed_region[0] > blocked_region[1]:
                allowed_regions_new.append(allowed_region)
                continue

            # Does the blocked range contain the good range? If yes: Then the good range may be removed
            if allowed_region[0] >= blocked_region[0] and allowed_region[1] <= blocked_region[1]:
                continue

            # Does the good range contain the blocked range? If yes: Then we have to split it
            if allowed_region[0] <= blocked_region[0] and allowed_region[1] >= blocked_region[1]:
                allowed_regions_new.append((allowed_region[0], blocked_region[0]))
                allowed_regions_new.append((blocked_region[1], allowed_region[1]))
                continue

            # Is there a left side overlap of the good range?
            if allowed_region[0] <= blocked_region[1] <= allowed_region[1]:
                allowed_regions_new.append((blocked_region[1], allowed_region[1]))
                continue

            # Is there a right side overlap of the good range?
            if allowed_region[0] <= blocked_region[0] <= allowed_region[1]:
                allowed_regions_new.append((allowed_region[0], blocked_region[0]))
                continue

            raise Exception()

        # Remove all empty ranges
        allowed_regions_new = list(filter(lambda r: r[1] - r[0] > 0, allowed_regions_new))
        allowed_regions = allowed_regions_new
    return allowed_regions

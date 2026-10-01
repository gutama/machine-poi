"""Small geometry fixtures for the containment research design (stdlib only).

Cl(3,0), basis e1,e2,e3, orientation e123, ab=a.b+a^b, reversal ~A,
commutator A x B=(AB-BA)/2. This verifies algebra/transport fixtures, not
model behavior or the existing workspace_diagnostics runtime implementation.
Run: python experiments/geometry_sanity.py
"""

import json
import math
import random


def gp(a, b):
    """Independent bitmask geometric product in Euclidean Cl(3,0)."""
    out = [0.0] * 8
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            inversions = sum((j & ((1 << k) - 1)).bit_count()
                             for k in range(3) if i & (1 << k))
            out[i ^ j] += (-1 if inversions % 2 else 1) * ai * bj
    return out


def reverse(a):
    return [v * (-1 if (i.bit_count() * (i.bit_count() - 1) // 2) % 2 else 1)
            for i, v in enumerate(a)]


def mv(v):
    out = [0.0] * 8
    for i, x in enumerate(v):
        out[1 << i] = x
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(dot(v, v))


def unit(v):
    length = norm(v)
    return [x / length for x in v]


def rotor_step(z, s, theta_max, eps=1e-10):
    """Research fixture with explicit no-op degeneracies; not a runtime hook."""
    if len(z) != 3 or len(s) != 3:
        raise ValueError("Expected Cl(3,0) vectors")
    if not all(math.isfinite(x) for x in [*z, *s, theta_max]):
        raise ValueError("Non-finite input")
    if not 0 <= theta_max <= math.pi:
        raise ValueError("Invalid angle cap")
    if norm(z) < eps or norm(s) < eps:
        return list(z), list(z), "zero", 0.0, 0.0
    u, target = unit(z), unit(s)
    cosine = max(-1.0, min(1.0, dot(u, target)))
    tangent = [x - cosine * y for x, y in zip(target, u)]
    if norm(tangent) < eps:
        return list(z), list(z), "parallel" if cosine >= 0 else "antipodal", 0.0, 0.0
    t = unit(tangent)
    theta = min(theta_max, math.acos(cosine))
    biv = gp(mv(u), mv(t))
    # Orthogonal u,t imply geometric product equals their wedge.
    r = [-math.sin(theta / 2) * x for x in biv]
    r[0] += math.cos(theta / 2)
    sandwich = gp(gp(r, mv(z)), reverse(r))
    ga = [sandwich[1 << i] for i in range(3)]
    vector = [norm(z) * (math.cos(theta) * x + math.sin(theta) * y)
              for x, y in zip(u, t)]
    square = gp(biv, biv)
    square_error = max(abs(v - (-1.0 if i == 0 else 0.0))
                       for i, v in enumerate(square))
    rotor_norm = gp(r, reverse(r))
    rotor_error = max(abs(v - (1.0 if i == 0 else 0.0))
                      for i, v in enumerate(rotor_norm))
    return ga, vector, "rotated", theta, max(square_error, rotor_error)


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matvec(a, v):
    return [dot(row, v) for row in a]


def wedge_matrix(u, v):
    return [[u[i] * v[j] - v[i] * u[j] for j in range(3)]
            for i in range(3)]


def difference(a, b):
    return math.sqrt(sum((a[i][j] - b[i][j]) ** 2
                         for i in range(3) for j in range(3)))


def rotation(axis, angle):
    """Independent Rodrigues rotation with unit positive orientation."""
    x, y, z = unit(axis)
    k = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
    kk = mm(k, k)
    return [[float(i == j) + math.sin(angle) * k[i][j]
             + (1 - math.cos(angle)) * kk[i][j] for j in range(3)]
            for i in range(3)]


def main():
    rng = random.Random(730)
    identity = [[float(i == j) for j in range(3)] for i in range(3)]
    worst = {"ga_vs_vector": 0.0, "norm": 0.0,
             "displacement_identity": 0.0, "bivector_and_rotor": 0.0}
    for _ in range(100):
        z = [rng.uniform(-2, 2) for _ in range(3)]
        s = [rng.uniform(-2, 2) for _ in range(3)]
        ga, vector, reason, theta, algebra_error = rotor_step(z, s, 0.2)
        assert reason == "rotated"
        delta = norm([a - b for a, b in zip(ga, z)]) / norm(z)
        worst["ga_vs_vector"] = max(worst["ga_vs_vector"], norm([a - b for a, b in zip(ga, vector)]))
        worst["norm"] = max(worst["norm"], abs(norm(ga) - norm(z)))
        worst["displacement_identity"] = max(worst["displacement_identity"], abs(delta - 2 * math.sin(theta / 2)))
        worst["bivector_and_rotor"] = max(worst["bivector_and_rotor"], algebra_error)
    assert max(worst.values()) < 1e-10

    # Sign and cap: e1 moves toward e2; no overshoot of a nearby target.
    a, _, _, _, _ = rotor_step([1, 0, 0], [0, 1, 0], 0.2)
    assert norm([a[0] - math.cos(0.2), a[1] - math.sin(0.2), a[2]]) < 1e-12
    target = [math.cos(0.01), math.sin(0.01), 0]
    a, _, _, theta, _ = rotor_step([1, 0, 0], target, 0.2)
    assert norm([x - y for x, y in zip(a, target)]) < 1e-10
    assert abs(theta - 0.01) < 1e-10

    degenerate = []
    for z, s, expected in [([0, 0, 0], [1, 0, 0], "zero"),
                           ([1, 0, 0], [0, 0, 0], "zero"),
                           ([1, 0, 0], [1, 0, 0], "parallel"),
                           ([1, 0, 0], [-1, 0, 0], "antipodal")]:
        ga, _, reason, _, _ = rotor_step(z, s, 0.2)
        assert ga == z and reason == expected
        degenerate.append(reason)
    try:
        rotor_step([math.nan, 0, 0], [1, 0, 0], 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite input accepted")

    # Same expression as legacy diagnostics: three vertex rotations, no closure.
    t = rotation([0, 0, 1], 0.1)
    legacy = mm(t, mm(t, t))
    legacy_angle = math.acos(max(-1, min(1, (sum(legacy[i][i] for i in range(3)) - 1) / 2)))
    assert abs(legacy_angle - 0.3) < 1e-12

    # Distinct same-axis rotations commute and compose to the angle sum.
    same_axis = rotation([0, 0, 1], 0.23)
    forward = mm(t, same_axis)
    backward = mm(same_axis, t)
    commuting_error = difference(forward, backward)
    assert difference(t, same_axis) > 1e-3
    assert commuting_error < 1e-12
    assert difference(forward, rotation([0, 0, 1], 0.33)) < 1e-12

    # A flat closed path with forward angles .1,.1 and return angle -.2.
    closed = mm(rotation([0, 0, 1], -0.2), mm(t, t))
    flat_error = difference(closed, identity)
    assert flat_error < 1e-12

    # Varying local frames: U_ij=F_j F_i^T remains flat around every loop.
    frames = [rotation([1, 0, 0], 0.3), rotation([0, 1, 0], -0.2),
              rotation([0, 0, 1], 0.4)]
    links = {(i, j): mm(frames[j], transpose(frames[i]))
             for i in range(3) for j in range(3)}
    pure_gauge = mm(links[2, 0], mm(links[1, 2], links[0, 1]))
    pure_gauge_error = difference(pure_gauge, identity)
    inverse_error = max(difference(mm(links[j, i], links[i, j]), identity)
                        for i in range(3) for j in range(3))
    assert max(pure_gauge_error, inverse_error) < 1e-12

    # Noncommuting loop has nonzero rotation; frame change conjugates it.
    u, v = rotation([1, 0, 0], 0.2), rotation([0, 1, 0], 0.3)
    hol = mm(transpose(v), mm(transpose(u), mm(v, u)))
    signal = difference(hol, identity)
    assert signal > 1e-3
    loop_links = [u, v, mm(transpose(v), transpose(u))]
    gauged = [mm(frames[1], mm(loop_links[0], transpose(frames[0]))),
              mm(frames[2], mm(loop_links[1], transpose(frames[1]))),
              mm(frames[0], mm(loop_links[2], transpose(frames[2])))]
    gauged_hol = mm(gauged[2], mm(gauged[1], gauged[0]))
    conjugated = mm(frames[0], mm(hol, transpose(frames[0])))
    gauge_error = difference(gauged_hol, conjugated)
    assert gauge_error < 1e-12
    assert abs(difference(gauged_hol, identity) - signal) < 1e-12

    # Independent value-coordinate change preserves attention output after
    # output-projection compensation, but changes the raw query/value wedge.
    # Two opposite values have zero prefix mean, matching legacy centering.
    q = [1.0, 0.0, 0.0]
    values = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
    weights = [0.2, 0.8]
    n = rotation([0, 0, 1], 0.5)
    weighted = [sum(w * v[i] for w, v in zip(weights, values)) for i in range(3)]
    changed_values = [matvec(n, v) for v in values]
    changed = [sum(w * v[i] for w, v in zip(weights, changed_values)) for i in range(3)]
    recovered_output = matvec(transpose(n), changed)
    output_error = norm([a - b for a, b in zip(recovered_output, weighted)])
    generator_change = difference(wedge_matrix(q, weighted), wedge_matrix(q, changed))
    assert output_error < 1e-12 and generator_change > 1e-3

    print(json.dumps({"scope": "algebra and synthetic transport fixtures; no model or host evaluated",
                      "random_rotor_cases": 100, "maximum_errors": worst,
                      "degenerate_noops": degenerate,
                      "legacy_constant_generator_angle_rad": legacy_angle,
                      "same_axis_commutation_error": commuting_error,
                      "flat_closed_loop_error": flat_error,
                      "pure_gauge_loop_error": pure_gauge_error,
                      "inverse_edge_error": inverse_error,
                      "noncommuting_loop_signal": signal,
                      "frame_covariance_error": gauge_error,
                      "value_reparameterization_output_error": output_error,
                      "value_reparameterization_generator_change": generator_change}, indent=2))


if __name__ == "__main__":
    main()

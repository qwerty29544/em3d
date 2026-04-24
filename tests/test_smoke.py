def test_package_imports():
    import em3d

    # public surface sanity
    for sym in [
        "Backend",
        "Precision",
        "Grid",
        "Problem",
        "Operator",
        "cylinder_refraction",
        "flat_wave_vec",
        "SIM",
        "BiCGStab",
        "TwoStep",
        "SolverConfig",
    ]:
        assert hasattr(em3d, sym), f"em3d is missing public symbol {sym}"

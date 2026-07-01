import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

import em3d
from experiments import chapter06_em as c6


def _fresh_output_root(name: str) -> Path:
    return Path("experiments") / "outputs" / f"test-{name}-{uuid.uuid4().hex}"


def test_n_series_for_mode_quick_and_full():
    assert c6.n_series_for_mode("quick") == [8, 16, 24]
    assert c6.n_series_for_mode("full") == [8, 16, 24, 32, 40, 48, 56, 64]


def test_n_series_for_mode_rejects_unknown():
    with pytest.raises(ValueError, match="mode"):
        c6.n_series_for_mode("interactive")


def test_ensure_output_dirs_creates_expected_tree():
    root = _fresh_output_root("chapter06")
    try:
        paths = c6.ensure_output_dirs(root)
        assert paths["root"] == root
        assert paths["raw"].is_dir()
        assert paths["tables"].is_dir()
        assert paths["figures"].is_dir()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_make_sphere_case_uses_cubic_grid_and_transverse_wave():
    case = c6.make_sphere_case(N=16, eps_r=2.0 + 0.0j, k0a=1.0, a=0.3)
    assert case.name == "sphere_eps2_k0a1_N16"
    assert case.N == (16, 16, 16)
    assert case.L == (1.0, 1.0, 1.0)
    assert case.k0 == pytest.approx(1.0 / 0.3)
    assert case.geometry == "sphere"
    assert case.radius == (0.3, 0.3, 0.3)
    assert case.wave_orient == (0.0, 0.0, 1.0)
    assert case.wave_amplitude == (1.0, 0.0, 0.0)


def test_make_anisotropic_ellipsoid_case_preserves_tensor():
    eps_real = np.array([[2.0, 0.1, 0.0], [0.1, 1.6, 0.0], [0.0, 0.0, 1.3]])
    eps_imag = np.zeros((3, 3))
    case = c6.make_anisotropic_ellipsoid_case(N=8, eps_real=eps_real, eps_imag=eps_imag, k0=2.5)
    assert case.name == "anisotropic_ellipsoid_N8"
    assert case.N == (8, 8, 8)
    assert case.geometry == "ellipsoid"
    np.testing.assert_allclose(case.eps_real, eps_real)
    np.testing.assert_allclose(case.eps_imag, eps_imag)


def test_solver_run_to_row_serializes_grid_shape():
    run = c6.SolverRun(
        case_name="case",
        solver_name="BiCGStab",
        N=(8, 8, 8),
        dof=1536,
        converged=True,
        iterations=4,
        final_residual=1e-6,
        elapsed_sec=0.1,
        residual_history=[1.0, 1e-6],
    )
    assert run.to_row() == {
        "case_name": "case",
        "solver_name": "BiCGStab",
        "N": "8x8x8",
        "dof": 1536,
        "converged": True,
        "iterations": 4,
        "final_residual": 1e-6,
        "elapsed_sec": 0.1,
    }


def test_build_problem_sphere_shapes():
    case = c6.make_sphere_case(N=8, eps_r=2.0 + 0.0j, k0a=1.0, a=0.3)
    problem, operator = c6.build_problem(case)
    assert isinstance(problem, em3d.Problem)
    assert isinstance(operator, em3d.Operator)
    assert problem.grid.N == (8, 8, 8)
    assert problem.eps_tensor.shape == (3, 3, 8, 8, 8)
    assert problem.wave.shape == (3, 8, 8, 8)
    assert problem.k0 == pytest.approx(case.k0)


def test_build_problem_rejects_unknown_geometry():
    case = c6.ExperimentCase(
        name="bad",
        N=(8, 8, 8),
        L=(1.0, 1.0, 1.0),
        k0=1.0,
        geometry="cube",
        eps_real=2.0,
        eps_imag=0.0,
        center=(0.0, 0.0, 0.0),
        radius=(0.3, 0.3, 0.3),
        wave_orient=(0.0, 0.0, 1.0),
        wave_amplitude=(1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="geometry"):
        c6.build_problem(case)


def test_estimate_gamma0_returns_analysis():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, _ = c6.build_problem(case)
    analysis = c6.estimate_gamma0(problem, coarse_N=(2, 2, 2))
    assert analysis.coarse_N == (3, 3, 3)
    assert analysis.matrix_shape == (81, 81)
    assert analysis.radius > 0.0


def test_run_solver_bicgstab_smoke():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, operator = c6.build_problem(case)
    run = c6.run_solver(problem, operator, case, "BiCGStab", max_iter=50, rtol=1e-5)
    assert run.case_name == case.name
    assert run.solver_name == "BiCGStab"
    assert run.N == (8, 8, 8)
    assert run.dof == 3 * 8 * 8 * 8
    assert run.iterations >= 0
    assert run.final_residual >= 0.0
    assert run.elapsed_sec >= 0.0
    assert len(run.residual_history) >= 1


def test_run_solver_suite_returns_requested_solvers():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, operator = c6.build_problem(case)
    analysis = c6.estimate_gamma0(problem, coarse_N=(2, 2, 2))
    runs = c6.run_solver_suite(
        problem,
        operator,
        case,
        ["SIM", "BiCGStab", "TwoStep"],
        gamma0_analysis=analysis,
        max_iter=50,
        rtol=1e-5,
    )
    assert [run.solver_name for run in runs] == ["SIM", "BiCGStab", "TwoStep"]


def test_save_runs_csv_writes_solver_rows():
    root = _fresh_output_root("csv")
    try:
        run = c6.SolverRun(
            case_name="case",
            solver_name="BiCGStab",
            N=(8, 8, 8),
            dof=1536,
            converged=True,
            iterations=4,
            final_residual=1e-6,
            elapsed_sec=0.1,
            residual_history=[1.0, 1e-6],
        )
        path = root / "runs.csv"
        c6.save_runs_csv([run], path)
        text = path.read_text(encoding="utf-8")
        assert "case_name,solver_name,N,dof,converged,iterations,final_residual,elapsed_sec" in text
        assert "case,BiCGStab,8x8x8,1536,True,4,1e-06,0.1" in text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_save_json_roundtrip():
    root = _fresh_output_root("json")
    try:
        path = root / "data.json"
        c6.save_json({"value": 1, "items": [1, 2]}, path)
        assert path.read_text(encoding="utf-8").startswith("{")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_benchmark_matvec_returns_timings():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, operator = c6.build_problem(case)
    row = c6.benchmark_matvec(problem, operator, case, repeats=2)
    assert row["case_name"] == case.name
    assert row["N"] == "8x8x8"
    assert row["dof"] == case.dof
    assert row["matvec_avg_sec"] >= 0.0
    assert row["operator_build_sec"] == 0.0


def test_compute_mie_rcs_diagnostics_smoke():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, operator = c6.build_problem(case)
    run_result = em3d.BiCGStab(em3d.SolverConfig(max_iter=50, rtol=1e-5)).solve(operator, problem.wave)
    diagnostics = c6.compute_mie_rcs_diagnostics(
        run_result.u,
        problem,
        a=0.25,
        eps_r=1.2 + 0.0j,
        n_phi=36,
    )
    assert diagnostics["phi"].shape == (36,)
    assert diagnostics["sigma_num_norm"].shape == (36,)
    assert diagnostics["shape_err"] >= 0.0
    assert diagnostics["scale_ratio"] >= 0.0


def test_run_quick_experiment_writes_artifacts():
    root = _fresh_output_root("quick")
    try:
        summary = c6.run_quick_experiment(
            output_root=root,
            n_values=[8],
            solver_names=["BiCGStab"],
            max_iter=50,
            rtol=1e-5,
            rcs_n_phi=24,
        )
        assert summary["mode"] == "quick"
        assert summary["n_values"] == [8]
        assert summary["num_solver_runs"] == 1
        assert (root / "tables" / "solver_runs.csv").is_file()
        assert (root / "raw" / "summary.json").is_file()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_main_invokes_requested_mode(monkeypatch):
    calls = []

    def fake_run_quick_experiment(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(c6, "run_quick_experiment", fake_run_quick_experiment)
    assert c6.main(["--mode", "quick", "--output-root", "out", "--max-iter", "7", "--rtol", "1e-4"]) == 0
    assert calls[0]["output_root"] == "out"
    assert calls[0]["n_values"] == [8, 16, 24]
    assert calls[0]["max_iter"] == 7
    assert calls[0]["rtol"] == pytest.approx(1e-4)


def test_chapter06_notebook_exists_and_has_sections():
    import json

    path = Path("notebooks") / "chapter-06-em.ipynb"
    assert path.is_file()
    nb = json.loads(path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "markdown"
    )
    for section in ["6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9", "6.10"]:
        assert section in markdown


def test_material_spec_isotropic_to_eps():
    material = c6.MaterialSpec.isotropic(2.5)
    eps_real, eps_imag = c6.material_eps(material, k0=3.0)
    assert eps_real == pytest.approx(2.5)
    assert eps_imag == pytest.approx(0.0)


def test_material_spec_anisotropic_lossy_to_eps():
    eps_real_in = np.diag([2.0, 1.5, 1.2])
    eps_imag_in = np.diag([0.1, 0.05, 0.02])
    material = c6.MaterialSpec.anisotropic_lossy(eps_real_in, eps_imag_in)
    eps_real, eps_imag = c6.material_eps(material, k0=3.0)
    np.testing.assert_allclose(eps_real, eps_real_in)
    np.testing.assert_allclose(eps_imag, eps_imag_in)


def test_material_spec_drude_matches_formula():
    material = c6.MaterialSpec.plasma_drude(eps_inf=1.0, omega_p=2.0, gamma=0.1)
    eps_real, eps_imag = c6.material_eps(material, k0=2.0)
    expected = 1.0 - 2.0**2 / (2.0**2 + 1j * 0.1 * 2.0)
    assert eps_real == pytest.approx(expected.real)
    assert eps_imag == pytest.approx(expected.imag)


def test_rotate_tensor_preserves_eigenvalues():
    eps = np.diag([2.0, 1.5, 1.2])
    theta = np.pi / 4.0
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = c6.rotate_tensor(eps, R)
    np.testing.assert_allclose(np.linalg.eigvalsh(rotated), np.linalg.eigvalsh(eps), atol=1e-12)


def test_make_uniaxial_crystal_ellipsoid_case_builds_tensor():
    case = c6.make_uniaxial_crystal_ellipsoid_case(N=8, eps_o=2.2, eps_e=1.4, k0=3.0)
    assert case.geometry == "ellipsoid"
    assert case.material.kind == "anisotropic"
    eps_real, eps_imag = c6.material_eps(case.material, k0=case.k0)
    np.testing.assert_allclose(eps_real, np.diag([2.2, 2.2, 1.4]))
    np.testing.assert_allclose(eps_imag, np.zeros((3, 3)))


def test_make_layered_box_case_builds_three_layers():
    layers = [
        c6.LayerSpec(
            z_min=-0.5,
            z_max=-1.0 / 6.0,
            material=c6.MaterialSpec.anisotropic_lossy(
                np.diag([1.5, 1.4, 1.3]),
                np.diag([0.01, 0.01, 0.01]),
            ),
        ),
        c6.LayerSpec(
            z_min=-1.0 / 6.0,
            z_max=1.0 / 6.0,
            material=c6.MaterialSpec.anisotropic_lossy(
                np.diag([2.0, 1.8, 1.6]),
                np.diag([0.02, 0.02, 0.02]),
            ),
        ),
        c6.LayerSpec(
            z_min=1.0 / 6.0,
            z_max=0.5,
            material=c6.MaterialSpec.anisotropic_lossy(
                np.diag([2.5, 2.2, 1.9]),
                np.diag([0.03, 0.03, 0.03]),
            ),
        ),
    ]
    case = c6.make_layered_box_case(N=6, k0=4.0, layers=layers)
    problem, _ = c6.build_problem(case)
    assert case.geometry == "layered_box"
    assert problem.eps_tensor.shape == (3, 3, 6, 6, 6)
    assert np.count_nonzero(np.asarray(problem.eps_tensor[0, 0])) > 0


def test_scan_gamma0_returns_rows_for_small_subset():
    def factory(coarse_N, k0):
        return c6.make_anisotropic_ellipsoid_case(
            N=8,
            eps_real=np.diag([2.0, 1.6, 1.3]),
            eps_imag=np.zeros((3, 3)),
            k0=k0,
        )

    rows = c6.scan_gamma0(factory, coarse_values=[2], k_values=[1, 2], scenario="test-aniso")
    assert len(rows) == 2
    assert rows[0]["scenario"] == "test-aniso"
    assert rows[0]["coarse_N"] >= 2
    assert rows[0]["k0"] in (1.0, 2.0)
    assert "rho" in rows[0]


def test_anisotropic_ellipsoid_gamma0_factory_uses_k0():
    case = c6.make_anisotropic_gamma0_case(coarse_N=4, k0=7.0)
    assert case.k0 == pytest.approx(7.0)
    assert case.geometry == "ellipsoid"


def test_scan_sim_convergence_by_gamma0_smoke():
    case = c6.make_anisotropic_ellipsoid_case(
        N=8,
        eps_real=np.diag([1.3, 1.2, 1.1]),
        eps_imag=np.zeros((3, 3)),
        k0=1.0,
    )
    rows = c6.scan_sim_convergence_by_gamma0(case, coarse_values=[2], max_iter=10, rtol=1e-4)
    assert len(rows) == 1
    assert rows[0]["solver_name"] == "SIM"
    assert rows[0]["case_name"] == case.name


def test_run_solver_comparison_returns_reference_solution():
    case = c6.make_anisotropic_ellipsoid_case(
        N=8,
        eps_real=np.diag([1.3, 1.2, 1.1]),
        eps_imag=np.zeros((3, 3)),
        k0=1.0,
    )
    result = c6.run_solver_comparison(case, sim_coarse_N=2, max_iter=10, rtol=1e-4)
    assert [run.solver_name for run in result["runs"]] == ["SIM", "BiCGStab", "TwoStep"]
    assert result["reference_u"].shape == (3, 8, 8, 8)


def test_benchmark_fft_vs_dense_matches_dense_on_small_grid():
    def factory(N):
        return c6.make_anisotropic_ellipsoid_case(
            N=N,
            eps_real=np.diag([1.2, 1.1, 1.05]),
            eps_imag=np.zeros((3, 3)),
            k0=1.0,
        )

    rows = c6.benchmark_fft_vs_dense(factory, n_values=[2], repeats=1)
    assert len(rows) == 1
    assert rows[0]["N"] == "2x2x2"
    assert rows[0]["relative_error"] < 1e-10
    assert rows[0]["fft_avg_sec"] >= 0.0
    assert rows[0]["dense_avg_sec"] >= 0.0


def test_scan_mie_rcs_by_k0a_smoke():
    rows = c6.scan_mie_rcs_by_k0a(
        N=8,
        a=0.25,
        eps_r=1.2,
        k0a_values=[0.25],
        n_phi=24,
        max_iter=50,
        rtol=1e-5,
    )
    assert len(rows) == 1
    assert rows[0]["k0a"] == pytest.approx(0.25)
    assert rows[0]["shape_err"] >= 0.0
    assert rows[0]["scale_ratio"] >= 0.0


def test_plot_three_field_slices_calls_scalar_slice(monkeypatch):
    calls = []

    def fake_plot(u, grid, **kwargs):
        calls.append(kwargs)
        return object(), object()

    monkeypatch.setattr(em3d.vis, "plot_field_scalar_slice", fake_plot)
    case = c6.make_sphere_case(N=4, eps_r=1.2, k0a=0.5, a=0.25)
    problem, _ = c6.build_problem(case)
    figs = c6.plot_three_field_slices(problem.wave, problem.grid, part="abs", component=None)
    assert len(figs) == 3
    assert [call["plane"] for call in calls] == ["xy", "xz", "yz"]


def test_chapter06_notebook_mentions_extended_experiments():
    import json

    path = Path("notebooks") / "chapter-06-em.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    for token in [
        "MaterialSpec.plasma_drude",
        "make_uniaxial_crystal_ellipsoid_case",
        "scan_gamma0",
        "scan_sim_convergence_by_gamma0",
        "run_solver_comparison",
        "benchmark_fft_vs_dense",
        "plot_three_field_slices",
        "scan_mie_rcs_by_k0a",
    ]:
        assert token in source


def test_experiment_logger_writes_jsonl_and_text():
    root = _fresh_output_root("logger")
    try:
        logger = c6.ExperimentLogger(root, "smoke")
        logger.event("start", case_name="case", value=np.float64(1.5))
        logger.event("finish", ok=True)

        jsonl_path = root / "raw" / "smoke.jsonl"
        log_path = root / "raw" / "smoke.log"
        assert jsonl_path.is_file()
        assert log_path.is_file()

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert '"event": "start"' in lines[0]
        assert '"case_name": "case"' in lines[0]
        assert "start" in log_path.read_text(encoding="utf-8")
        assert "finish" in log_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chapter06_default_scan_constants():
    assert c6.FFT_DENSE_N_VALUES == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert c6.RCS_K0A_VALUES == [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert c6.RCS_DEFAULT_RADIUS == pytest.approx(0.5)


def test_benchmark_fft_vs_dense_logs_events():
    root = _fresh_output_root("fft-log")
    try:
        logger = c6.ExperimentLogger(root, "fft")

        def factory(N):
            return c6.make_anisotropic_ellipsoid_case(
                N=N,
                eps_real=np.diag([1.2, 1.1, 1.05]),
                eps_imag=np.zeros((3, 3)),
                k0=1.0,
            )

        rows = c6.benchmark_fft_vs_dense(factory, n_values=[2], repeats=1, logger=logger)
        assert len(rows) == 1
        text = (root / "raw" / "fft.jsonl").read_text(encoding="utf-8")
        assert '"event": "start"' in text
        assert '"event": "benchmark_done"' in text
        assert '"event": "finish"' in text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_plot_fft_vs_dense_timing_uses_matplotlib(monkeypatch):
    calls = []

    class FakeFig:
        def savefig(self, filename, dpi=None, bbox_inches=None):
            calls.append(("savefig", filename, dpi, bbox_inches))

    class FakeAx:
        def plot(self, *args, **kwargs):
            calls.append(("plot", args, kwargs))

        def set_xlabel(self, value):
            calls.append(("xlabel", value))

        def set_ylabel(self, value):
            calls.append(("ylabel", value))

        def set_title(self, value):
            calls.append(("title", value))

        def grid(self, *args, **kwargs):
            calls.append(("grid", args, kwargs))

        def legend(self):
            calls.append(("legend",))

    class FakePlt:
        def subplots(self, **kwargs):
            calls.append(("subplots", kwargs))
            return FakeFig(), FakeAx()

    monkeypatch.setattr(em3d.vis, "_require_matplotlib", lambda: FakePlt())
    fig, ax = c6.plot_fft_vs_dense_timing(
        [{"N": "2x2x2", "fft_avg_sec": 0.1, "dense_avg_sec": 0.3}],
        output_dir=_fresh_output_root("fft-plot") / "figures",
    )
    assert fig is not None
    assert ax is not None
    assert any(call[0] == "plot" for call in calls)
    assert any(call[0] == "savefig" for call in calls)


def test_plot_rcs_scan_calls_cartesian_and_polar(monkeypatch):
    calls = []

    def fake_cartesian(phi, sigma_num, sigma_mie, **kwargs):
        calls.append(("cartesian", kwargs))
        return object(), object()

    def fake_polar(phi, sigma_num, sigma_mie, **kwargs):
        calls.append(("polar", kwargs))
        return object(), object()

    monkeypatch.setattr(em3d.vis, "plot_rcs_comparison", fake_cartesian)
    monkeypatch.setattr(em3d.vis, "plot_rcs_comparison_polar", fake_polar)
    rows = [
        {
            "k0a": 0.25,
            "phi": np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
            "sigma_num_norm": np.ones(8),
            "sigma_mie_norm": np.ones(8),
            "shape_err": 0.1,
            "scale_ratio": 0.9,
        }
    ]
    figs = c6.plot_rcs_scan(rows)
    assert len(figs["directional"]) == 2
    assert calls[0][0] == "cartesian"
    assert calls[1][0] == "polar"


def test_chapter06_notebook_mentions_logging_and_crash_test():
    import json

    path = Path("notebooks") / "chapter-06-em.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    for token in [
        "ExperimentLogger",
        "plot_fft_vs_dense_timing",
        "plot_rcs_scan",
        "RUN_CRASH_TEST",
        "N=128",
        "TwoStep",
        "plot_field_vector_slice",
    ]:
        assert token in source


def test_make_structured_lattice_case_defaults_to_large_grid_and_coarse_gamma0():
    case = c6.make_structured_lattice_case()
    assert case.name == "structured_lattice_5x5x5_N100"
    assert case.N == (100, 100, 100)
    assert case.coarse_N == (9, 9, 9)
    assert case.lattice_shape == (5, 5, 5)
    assert [inclusion.material.kind for inclusion in case.inclusions[:3]] == [
        "isotropic",
        "isotropic",
        "isotropic",
    ]
    assert case.solver_names == ("SIM", "BiCGStab", "TwoStep")


def test_structured_lattice_problem_marks_multiple_inclusions():
    material = c6.MaterialSpec.isotropic(2.0 + 0.1j)
    case = c6.make_structured_lattice_case(
        N=8,
        coarse_N=3,
        lattice_shape=(2, 1, 1),
        inclusion_radius=(0.12, 0.12, 0.12),
        material=material,
        k0=2.0,
    )
    problem, operator = c6.build_structured_lattice_problem(case)
    assert isinstance(problem, em3d.Problem)
    assert isinstance(operator, em3d.Operator)
    assert problem.grid.N == (8, 8, 8)
    assert problem.eps_tensor.shape == (3, 3, 8, 8, 8)

    eta_xx = np.asarray(problem.eps_tensor[0, 0])
    assert np.count_nonzero(np.abs(eta_xx) > 0.0) > 0
    assert np.max(eta_xx.real) == pytest.approx(1.0)
    assert np.max(eta_xx.imag) == pytest.approx(0.1)


def test_run_structured_lattice_experiment_writes_metrics_and_logs(monkeypatch):
    root = _fresh_output_root("structured-lattice")
    try:
        monkeypatch.setattr(c6, "plot_three_field_slices", lambda *args, **kwargs: [])
        monkeypatch.setattr(c6, "plot_residual_histories", lambda *args, **kwargs: (object(), object()))
        monkeypatch.setattr(em3d.vis, "plot_rcs", lambda *args, **kwargs: (object(), object()))
        monkeypatch.setattr(em3d.vis, "plot_rcs_polar", lambda *args, **kwargs: (object(), object()))

        case = c6.make_structured_lattice_case(
            N=6,
            coarse_N=3,
            lattice_shape=(1, 1, 1),
            inclusion_radius=(0.18, 0.18, 0.18),
            k0=1.0,
            solver_names=("BiCGStab",),
        )
        summary = c6.run_structured_lattice_experiment(
            case=case,
            output_root=root,
            max_iter=5,
            rtol=1e-4,
            rcs_n_phi=12,
        )

        assert summary["case_name"] == case.name
        assert summary["N"] == "6x6x6"
        assert summary["coarse_N"] == "3x3x3"
        assert summary["solver_names"] == ["BiCGStab"]
        assert summary["num_solver_runs"] == 1
        assert (root / "tables" / "structured_lattice_solver_runs.csv").is_file()
        assert (root / "raw" / "structured_lattice_summary.json").is_file()
        assert (root / "raw" / "structured_lattice_residual_histories.json").is_file()
        log_text = (root / "raw" / "structured_lattice.jsonl").read_text(encoding="utf-8")
        assert '"event": "start"' in log_text
        assert '"event": "solver_finished"' in log_text
        assert '"event": "finish"' in log_text
    finally:
        shutil.rmtree(root, ignore_errors=True)

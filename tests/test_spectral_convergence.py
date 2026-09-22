import numpy as np

from em3d.spectral import ConvergenceClass, analyze_residual_history


def test_convergence_classification_distinguishes_unresolved_modes():
    converged = analyze_residual_history(0.5 ** np.arange(30), rtol=1e-6)
    assert converged.classification is ConvergenceClass.CONVERGED

    contracting = analyze_residual_history(0.99 ** np.arange(30), rtol=1e-12)
    assert contracting.classification is ConvergenceClass.CONTRACTING_UNRESOLVED

    unstable = analyze_residual_history(1.05 ** np.arange(30), rtol=1e-12)
    assert unstable.classification is ConvergenceClass.UNSTABLE

    unresolved = analyze_residual_history(np.ones(30), rtol=1e-12)
    assert unresolved.classification is ConvergenceClass.UNRESOLVED

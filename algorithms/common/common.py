from qutip import qutrit_basis, qutrit_ops, QobjEvo, sesolve, basis, tensor, sigmap, sigmam, identity, ket2dm, sigmaz, sigmax
import numpy as np
import random

def tsoa_coupled_spins_quantum_simulation(input_params, initial_quantum_state, final_time,
                                          nb_harmonics=3, max_steps=100, ξ=1):
    rabi_freq_params = input_params[0:2 * nb_harmonics + 1]
    detuning_params = input_params[2 * nb_harmonics + 1: len(input_params)]

    # construct trigonometric series for Ω
    def omega(t, args):
        Ω = rabi_freq_params[0]

        for harmonic in range(1, nb_harmonics + 1):
            Ω += rabi_freq_params[2 * harmonic - 1] * np.cos(harmonic * t) + rabi_freq_params[2 * harmonic] * np.sin(harmonic * t)

        return Ω

    # construct trigonometric series for Δ
    def detuning(t, args):
        Δ = detuning_params[0]

        for harmonic in range(1, nb_harmonics + 1):
            Δ += detuning_params[2 * harmonic - 1] * np.cos(harmonic * t) + detuning_params[2 * harmonic] * np.sin(harmonic * t)

        return Δ

    [proj1, proj2, proj3, trans12, trans23, _] = qutrit_ops()
    trans21 = trans12.dag()
    trans32 = trans23.dag()

    constant = lambda t, args: 4 * ξ
    
    H = QobjEvo([[(1 / np.sqrt(2)) * (trans12 + trans21 + trans23 + trans32), omega], [proj3, constant], [-proj3, detuning], [proj1, detuning]])
    
    time = np.linspace(start = 0, stop = final_time, num = max_steps)

    omegas = [omega(t, args = None) for t in time]
    detunings = [detuning(t, args = None) for t in time]

    result = sesolve(H, initial_quantum_state, time, [])

    return (result.states, omegas, detunings)

def tsoa_coupled_spins_4lvl_quantum_simulation(input_params, initial_quantum_state, final_time,
                                          nb_harmonics=3, max_steps=100, ξ=1):
    
    rabi_freq_params = input_params[0:2 * nb_harmonics + 1]
    detuning_params = input_params[2 * nb_harmonics + 1: len(input_params)]

    # construct trigonometric series for Ω
    def omega(t, args):
        Ω = rabi_freq_params[0]

        for harmonic in range(1, nb_harmonics + 1):
            Ω += rabi_freq_params[2 * harmonic - 1] * np.cos(harmonic * t) + rabi_freq_params[2 * harmonic] * np.sin(harmonic * t)

        return Ω

    # construct trigonometric series for Δ
    def detuning(t, args):
        Δ = detuning_params[0]

        for harmonic in range(1, nb_harmonics + 1):
            Δ += detuning_params[2 * harmonic - 1] * np.cos(harmonic * t) + detuning_params[2 * harmonic] * np.sin(harmonic * t)

        return Δ

    """ [proj1, proj2, proj3, trans12, trans23, _] = qutrit_ops()
    trans21 = trans12.dag()
    trans32 = trans23.dag() """

    P1 = ket2dm(basis(2, 0))
    P2 = ket2dm(basis(2, 1))
    proj1 = tensor(P1, P1)
    proj2 = tensor(P1, P2)
    proj3 = tensor(P2, P1)
    proj4 = tensor(P2, P2)
    trans12 = tensor(identity(2), sigmap())
    trans21 = trans12.dag()
    trans23 = tensor(sigmap(), sigmam())
    trans32 = trans23.dag()

    constant = lambda t, args: 4 * ξ
    constant4 = lambda t, args: -ξ
    
    H = QobjEvo([[(1 / np.sqrt(2)) * (trans12 + trans21 + trans23 + trans32), omega], [proj3, constant], [-proj3, detuning], [proj1, detuning], [proj4, constant4]])
    
    time = np.linspace(start = 0, stop = final_time, num = max_steps)

    omegas = [omega(t, args = None) for t in time]
    detunings = [detuning(t, args = None) for t in time]

    result = sesolve(H, initial_quantum_state, time, [])

    return (result.states, omegas, detunings)

def tsoa_tensor_product_qubits_quantum_simulation(input_params, initial_quantum_state, final_time,
                                          nb_harmonics=3, max_steps=100, ξ=1):
    
    rabi_freq_params = input_params[0:2 * nb_harmonics + 1]
    detuning_params = input_params[2 * nb_harmonics + 1: len(input_params)]

    # construct trigonometric series for Ω
    def omega(t, args):
        Ω = rabi_freq_params[0]

        for harmonic in range(1, nb_harmonics + 1):
            Ω += rabi_freq_params[2 * harmonic - 1] * np.cos(harmonic * t) + rabi_freq_params[2 * harmonic] * np.sin(harmonic * t)

        return Ω

    # construct trigonometric series for Δ
    def detuning(t, args):
        Δ = detuning_params[0]

        for harmonic in range(1, nb_harmonics + 1):
            Δ += detuning_params[2 * harmonic - 1] * np.cos(harmonic * t) + detuning_params[2 * harmonic] * np.sin(harmonic * t)

        return Δ

    P1 = ket2dm(basis(2, 0))
    P2 = ket2dm(basis(2, 1))
    proj1 = tensor(P1, P1)
    proj2 = tensor(P1, P2)
    proj3 = tensor(P2, P1)
    proj4 = tensor(P2, P2)
    trans12 = tensor(identity(2), sigmap())
    trans21 = trans12.dag()
    trans23 = tensor(sigmap(), sigmam())
    trans32 = trans23.dag()

    H_qubit = QobjEvo([[(1 / np.sqrt(2)) * sigmaz(), detuning], [0.5 * sigmax(), omega]])
    
    H_composite = tensor(H_qubit, H_qubit)
    
    time = np.linspace(start = 0, stop = final_time, num = max_steps)

    omegas = [omega(t, args = None) for t in time]
    detunings = [detuning(t, args = None) for t in time]

    result = sesolve(H_composite, initial_quantum_state, time, [])

    return (result.states, omegas, detunings)

def tsoa_coupled_spins_quantum_simulation_with_uncertainty(input_params, initial_quantum_state, final_time,
                                          nb_harmonics=3, max_steps=100, ξ=1,
                                          error=0.05):
    rabi_freq_params = input_params[0:2 * nb_harmonics + 1]
    detuning_params = input_params[2 * nb_harmonics + 1: len(input_params)]

    # construct trigonometric series for Ω
    def omega(t, args):
        Ω = rabi_freq_params[0]

        for harmonic in range(1, nb_harmonics + 1):
            Ω += rabi_freq_params[2 * harmonic - 1] * np.cos(harmonic * t) + rabi_freq_params[2 * harmonic] * np.sin(harmonic * t)

        # add random perturbation / uncertainty on omega value
        return Ω + error

    # construct trigonometric series for Δ
    def detuning(t, args):
        Δ = detuning_params[0]

        for harmonic in range(1, nb_harmonics + 1):
            Δ += detuning_params[2 * harmonic - 1] * np.cos(harmonic * t) + detuning_params[2 * harmonic] * np.sin(harmonic * t)

        # add random perturbation / uncertainty on delta value
        return Δ + error

    [proj1, proj2, proj3, trans12, trans23, _] = qutrit_ops()
    trans21 = trans12.dag()
    trans32 = trans23.dag()

    constant = lambda t, args: 4 * ξ
    
    H = QobjEvo([[(1 / np.sqrt(2)) * (trans12 + trans21 + trans23 + trans32), omega], [proj3, constant], [-proj3, detuning], [proj1, detuning]])
    
    time = np.linspace(start = 0, stop = final_time, num = max_steps)

    omegas = [omega(t, args = None) for t in time]
    detunings = [detuning(t, args = None) for t in time]

    result = sesolve(H, initial_quantum_state, time, [])

    return (result.states, omegas, detunings)
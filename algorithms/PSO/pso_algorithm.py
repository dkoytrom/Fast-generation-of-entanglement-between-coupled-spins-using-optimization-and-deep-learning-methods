import numpy as np
from tqdm.auto import trange
import random

"""
A candidate solution in the parameters landscape
"""
class Solution():
    def __init__(self, nb_params) -> None:
        self.position = np.zeros(nb_params)
        self.objective = np.infty

"""
A particle contains a current position and velocity which is its exploration status
It also contains a Best solution, as the position with the best objective function value that 
has been visited so far
"""
class Particle:
    def __init__(self, nb_params, lower_bound, upper_bound) -> None:
        self.nb_params = nb_params
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        # Init position randomly between the parameter bounds
        self.position = np.array([random.uniform(lower_bound[i], upper_bound[i]) for i in range(nb_params)])
        
        # set initial velocity to zero vector
        self.velocity = np.zeros(nb_params)

        # personal best starts with zero position and infinite value objective
        self.personal_best = Solution(nb_params)
        self.objective = np.infty

    """
    Calculate the objective in current position
    """
    def calculate_objective(self, objective_fn):
        self.objective = objective_fn(self.position)

    """
    Update the personal best solution if needed based on current state
    """
    def find_update_best(self):
        if self.objective < self.personal_best.objective:
            self.personal_best.objective = self.objective
            self.personal_best.position = self.position

    """
    Calculate the new velocity and position based on the update rule
    """
    def calculate_velocity_and_position(self, inertia_w, c1, c2, global_best, max_velocity, min_velocity):
        velocity  = inertia_w * self.velocity 
        velocity += c1 * np.array([random.random() for _ in range(self.nb_params)]) * (self.personal_best.position - self.position) 
        velocity += c2 * np.array([random.random() for _ in range(self.nb_params)]) * (global_best.position - self.position)

        # restrict velocity between bounds if available
        if max_velocity is not None and min_velocity is not None:
            for i, vel in enumerate(velocity):
                if vel > max_velocity[i]:
                    self.velocity[i] = max_velocity[i]
                elif vel < min_velocity[i]:
                    self.velocity[i] = min_velocity[i]
                else:
                    self.velocity[i] = vel
        else:
            self.velocity = velocity

        self.position = self.position + self.velocity

        # check positions are inside bounds
        for i, pos in enumerate(self.position):
            if pos > self.upper_bound[i]:
                self.position[i] = self.upper_bound[i]
            elif pos < self.lower_bound[i]:
                self.position[i] = self.lower_bound[i]

"""
Algorithm class, it contains particles and keeps track of the global best solution found by all the individual particles
"""
class PSO:
    def __init__(self, nb_particles, nb_params, lower_bound, upper_bound, 
                 objective_fn, w_max, w_min, c1, c2,
                 max_velocity = None, min_velocity = None,
                 objective_threshold = 1e-4) -> None:
        
        self.nb_particles = nb_particles
        self.objective_fn = objective_fn
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self._global_objectives = []
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self._objective_threshold = objective_threshold

        # initialize particles randomly
        self.particles = [Particle(nb_params, lower_bound, upper_bound) for _ in range(nb_particles)]

        # define the initial global best
        self.global_best = Solution(nb_params)

    """
    Update the global best from the current position of the given particle
    """
    def update_global_best(self, particle):
        if particle.objective < self.global_best.objective:
            self.global_best.objective = particle.objective
            self.global_best.position = particle.position

    """
    getter function of the global objectives as it evolves
    """
    def get_global_objectives(self):
        return self._global_objectives

    """
    function which triggers the optimization process, inits the particles, handles the global objectives evolution
    and updates the particles positions and velocities after each iteration
    """
    def optimize(self, max_iter):
        iter = 0
        with trange(max_iter, dynamic_ncols = False) as tbar:
            for _ in tbar:
                iter += 1
        # for iter in range(max_iter):

                """
                For all particles:
                1. find the current objective value
                2. update personal best if needed
                3. update the global best if needed
                """
                for particle in self.particles:
                    # calculate the objective
                    particle.calculate_objective(self.objective_fn)

                    # update the personal best
                    particle.find_update_best()

                    # update the global best
                    self.update_global_best(particle)

                tbar.set_postfix({"objective": self.global_best.objective})
                # print(f"step: {iter}, best objective: {self.global_best.objective}, best position: {self.global_best.position}")
                self._global_objectives.append(self.global_best.objective)

                if self.global_best.objective < self._objective_threshold:
                    break

                # now we need to update the positions and velocity of the particles
                # Velocity = w*old_velocity + c1 + c2
                # Position = old_position + velocity

                # inertia weight - decrises over time
                inertia_w = self.w_max - iter * (self.w_max - self.w_min) / max_iter

                for particle in self.particles:
                    particle.calculate_velocity_and_position(inertia_w, self.c1, self.c2, self.global_best, self.max_velocity, self.min_velocity)

                    
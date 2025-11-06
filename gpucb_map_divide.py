import functools
import time
import pickle
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import astar
import math
import wandb

import warnings

warnings.filterwarnings('ignore')

time_taken = {
    "'argmax_ucb'": [],
    "'points_between_two_points'": [],
    "'learn'": []
}

def timer(func):
    """A decorator to measure the execution time of methods."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        time_dict = getattr(args[0], 'time_taken', time_taken)
        time_dict.setdefault(repr(func.__name__), []).append(round(run_time, 3))
        return value
    return wrapper

def create_partitions(mask, num_agents):
    """Divides a boolean mask into a grid of partitions for multiple agents."""
    if num_agents == 1:
        return [mask]

    n_rows = 1
    for i in range(int(math.sqrt(num_agents)), 0, -1):
        if num_agents % i == 0:
            n_rows = i
            break
    n_cols = num_agents // n_rows
    print(f"Partitioning map into a {n_rows}x{n_cols} grid for {num_agents} agents.")
    row_splits = np.array_split(np.arange(mask.shape[0]), n_rows)
    col_splits = np.array_split(np.arange(mask.shape[1]), n_cols)

    partition_masks = []
    for r_split in row_splits:
        for c_split in col_splits:
            partition_mask = np.zeros_like(mask, dtype=bool)
            min_row, max_row = r_split[0], r_split[-1] + 1
            min_col, max_col = c_split[0], c_split[-1] + 1
            partition_mask[min_row:max_row, min_col:max_col] = mask[min_row:max_row, min_col:max_col]
            partition_masks.append(partition_mask)

    return partition_masks


class GPUCB(object):
    """Base class for Gaussian Process Upper Confidence Bound algorithm."""
    def __init__(self, img, lawn_mover_distance=0, beta=1e6, debug=True):
        self.points = {}
        self.lawn_mover_distance = lawn_mover_distance
        self.debug = debug
        self.iteration = 0
        self.img = img
        h, w = self.img.shape
        x = np.arange(0, w, 1)
        y = np.arange(0, h, 1)
        self.meshgrid = np.array(np.meshgrid(x, y))
        self.beta = beta

        self.grid = self.meshgrid.reshape(2, -1).T
        self.mu = np.zeros(self.grid.shape[0])
        self.sigma = np.full(self.grid.shape[0], 0.5)
        self.distance = 0
        self.visited_points = []
        self.visited_depth = []
        self.ucb_points = []
        self.parameter = 1
        self.sp = 0
        self.MU = []
        self.SIGMA = []
        self.DIST = []

        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        self.time_taken = {"'argmax_ucb'": [], "'learn'": [], "'points_between_two_points'": []}

    @timer
    def argmax_ucb(self):
        """Finds the index of the grid point with the highest UCB value."""
        if self.sp == 0:
            return 0

        ucb_values = self.mu + self.sigma * np.sqrt(self.beta)
        max_ucb_value = np.amax(ucb_values)
        candidate_indices = np.where(ucb_values == max_ucb_value)[0]

        if len(candidate_indices) == 1:
            return candidate_indices[0]

        last_point = self.visited_points[-1]
        candidate_points = self.grid[candidate_indices]
        distances = np.linalg.norm(candidate_points - last_point, axis=1)

        min_dist_idx = np.argmin(distances)
        return candidate_indices[min_dist_idx]

    @timer
    def learn(self):
        """Performs one iteration of the learning process."""
        dist = 0
        self.idx = self.argmax_ucb()

        if self.sp > 0:
            P = self.visited_points[-1]
            Q = self.grid[self.idx]
            points, dist = self.points_between_two_points(P, Q)

            self.distance += dist
            self.DIST.append(self.distance)

            self.gp.fit(np.array(self.visited_points), self.visited_depth)
            self.mu, self.sigma = self.gp.predict(self.grid, return_std=True)

            self.points[self.sp] = points
            self.mu[self.mu < 0] = 0

            if self.debug:
                self.MU.append(self.mu)
                self.SIGMA.append(self.sigma)
        else:
            self.points[0] = [(0, 0)]
            self.DIST.append(self.distance)
            self.sample(list(self.grid[self.idx]))

            self.gp.fit(np.array(self.visited_points), self.visited_depth)
            self.mu, self.sigma = self.gp.predict(self.grid, return_std=True)
            self.mu[self.mu < 0] = 0

            if self.debug:
                self.MU.append(self.mu)
                self.SIGMA.append(self.sigma)

        self.sp += 1
        return self.mu, dist

    @timer
    def points_between_two_points(self, point_a, point_b):
        """Generates points along a line and samples them."""
        points = []
        dist = np.linalg.norm(np.array(point_a) - np.array(point_b))
        if dist == 0:
            return points, dist

        num_points = int(dist)
        t_values = np.linspace(0, 1, num_points + 1)

        for t in t_values:
            x = (1 - t) * point_a[0] + t * point_b[0]
            y = (1 - t) * point_a[1] + t * point_b[1]
            self.sample([x, y])
            points.append((x, y))

        return points, dist


    def sample(self, x):
        """Samples a point and adds it to the visited set."""
        point_tuple = tuple(x)
        if point_tuple not in [tuple(p) for p in self.visited_points]:
            self.ucb_points.append(self.grid[self.idx])
            self.visited_points.append(x)
            row = np.clip(int(np.rint(x[0])), 0, self.img.shape[0] - 1)
            col = np.clip(int(np.rint(x[1])), 0, self.img.shape[1] - 1)
            self.visited_depth.append(self.img[row, col])


    def error(self):
        """Calculates the reconstruction error."""
        if not self.MU:
            return 100.0
        # For the base class, the predicted mean sum is compared against the entire image sum
        # return abs(1 - np.sum(self.MU[-1]) / np.sum(self.img)) * 100
        valid_mask = ~np.isnan(self.img)
        return abs(1 - np.sum(self.MU[-1][valid_mask]) / np.sum(self.img[valid_mask])) * 100

    def calculate_r_squared(self):
        """
        Calculates the R-squared (coefficient of determination) score.
        """
        if len(self.visited_points) < 2:
            return 0.0

        predicted_values = self.gp.predict(np.array(self.visited_points))
        actual_values = np.array(self.visited_depth)
        mean_actual = np.mean(actual_values)
        sst = np.sum((actual_values - mean_actual)**2)

        if sst == 0.0:
            return 1.0 if np.allclose(actual_values, predicted_values) else 0.0

        ssr = np.sum((actual_values - predicted_values)**2)
        r_squared = 1 - (ssr / sst)
        return r_squared

    def run(self):
        """Runs the exploration until the distance budget is met."""
        print(f'Target Distance: {self.lawn_mover_distance}')
        error = 100
        # while self.distance < self.lawn_mover_distance:
        while error > 10:
            self.learn()
            error = self.error()
            r_squared = self.calculate_r_squared()
            print(f'Iteration: {self.iteration}; Distance: {self.distance:.2f}; Error: {error:.2f}%; R-squared: {r_squared:.3f}', end='\r')
            self.iteration += 1

        print(f'\nTotal Iterations: {self.iteration}')

        if self.MU:
            final_mean_1d = self.MU[-1]
            final_mean_map = final_mean_1d.reshape(self.img.shape)
            final_mean_map[~self.mask] = np.nan
            output_filename = 'predicted_mean_map_divide.npy'
            np.save(output_filename, final_mean_map)
            print(f"Final predicted mean map saved as '{output_filename}'")
        else:
            print("No data was generated to save. Exploration might have stopped prematurely.")


class GPUCB_NEW(GPUCB):
    """
    An enhanced GPUCB that operates on a masked area and uses python-astar for pathfinding.
    """
    def __init__(self, img, mask, step=(1, 1), lawn_mover_distance=0, beta=1e6, debug=True):
        super().__init__(img, lawn_mover_distance, beta, debug)
        self.step = step
        self.mask = mask
        self.valid_indices = np.argwhere(self.mask)
        self.grid = self.valid_indices

        if self.grid.shape[0] == 0:
            self.mu = np.array([])
            self.sigma = np.array([])
        else:
            self.mu = np.zeros(self.grid.shape[0])
            self.sigma = np.full(self.grid.shape[0], 0.5)

        self.ERROR = []
        self.R2_SCORE = []


    def get_distance(self, point_a, point_b):
        """Calculates distance considering the step size."""
        pa = np.array(point_a) * self.step
        pb = np.array(point_b) * self.step
        return np.linalg.norm(pa - pb)

    def get_neighbors(self, node):
        """Yields valid, walkable neighbors for a given node."""
        r, c = node
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = r + dr, c + dc

                if 0 <= nr < self.mask.shape[0] and 0 <= nc < self.mask.shape[1] and self.mask[nr, nc]:
                    yield (nr, nc)

    def distance_between(self, n1, n2):
        """Calculates Euclidean distance between two nodes."""
        return np.linalg.norm(np.array(n1) - np.array(n2))

    def get_mu(self, MU):
        """Projects the mean prediction back onto the full image shape."""
        mu_full = np.zeros_like(self.img, dtype=float)
        if self.grid.size > 0:
            rows, cols = self.grid.T
            mu_full[rows, cols] = MU
        return mu_full

    def learn(self):
        """Performs one iteration of learning with pathfinding."""
        if self.grid.size == 0 or len(self.visited_points) == self.grid.shape[0]:
            return None, 0
            
        dist = 0
        self.idx = self.argmax_ucb()

        if self.sp > 0:
            P = self.visited_points[-1]
            Q = self.grid[self.idx]

            points, dist = self.get_path(P, Q)
            self.distance += dist
            self.DIST.append(self.distance)

            self.sample(list(Q))

            self.gp.fit(np.array(self.visited_points), self.visited_depth)
            self.mu, self.sigma = self.gp.predict(self.grid, return_std=True)
            self.points[self.sp] = points
            self.mu[self.mu < 0] = 0

            if self.debug:
                self.MU.append(self.get_mu(self.mu))
                self.SIGMA.append(self.get_mu(self.sigma))
        else:
            Q = self.grid[self.idx]
            self.points[0] = [tuple(Q)]
            self.DIST.append(self.distance)
            self.sample(list(Q))

            self.gp.fit(np.array(self.visited_points), self.visited_depth)
            self.mu, self.sigma = self.gp.predict(self.grid, return_std=True)
            self.mu[self.mu < 0] = 0
            if self.debug:
                self.MU.append(self.get_mu(self.mu))
                self.SIGMA.append(self.get_mu(self.sigma))

        self.sp += 1
        return self.mu, dist

    def get_path(self, point_a, point_b):
        """Finds a pruned A* path and samples points along it."""
        dist = 0
        all_points = []

        start_node = tuple(int(np.rint(c)) for c in point_a)
        end_node = tuple(int(np.rint(c)) for c in point_b)

        astar_path_generator = astar.find_path(
            start=start_node,
            goal=end_node,
            neighbors_fnct=self.get_neighbors,
            distance_between_fnct=self.get_distance,
        )

        if not astar_path_generator:
            pruned_path = [start_node, end_node]
        else:
            astar_path = list(astar_path_generator)
            pruned_path = self.prune_path(astar_path)

        for i in range(len(pruned_path) - 1):
            p1 = pruned_path[i]
            p2 = pruned_path[i+1]
            d = self.get_distance(p1, p2)
            sub_points, _ = self.points_between_two_points(p1, p2)
            dist += d
            all_points.extend(sub_points)

        return all_points, dist

    def line_of_sight(self, p1, p2):
        """Checks for a clear line of sight between two points on the mask."""
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        if dist == 0:
            return True

        num_steps = int(dist * 2)
        x = np.linspace(p1[0], p2[0], num_steps)
        y = np.linspace(p1[1], p2[1], num_steps)

        rows = np.clip(np.rint(x).astype(int), 0, self.mask.shape[0] - 1)
        cols = np.clip(np.rint(y).astype(int), 0, self.mask.shape[1] - 1)

        return np.all(self.mask[rows, cols])

    def prune_path(self, path):
        """Prunes an A* path using line-of-sight checks."""
        if not path:
            return []
        pruned_path = [path[0]]
        last_los_node = path[0]
        for current_node in path[1:]:
            if not self.line_of_sight(last_los_node, current_node):
                pruned_path.append(path[path.index(current_node)-1])
                last_los_node = path[path.index(current_node)-1]

        if pruned_path[-1] != path[-1]:
             pruned_path.append(path[-1])
        return pruned_path
    
    def error(self):
        """Calculates the reconstruction error within the agent's partition."""
        if not self.MU:
            return 100.0
 
        true_sum = np.sum(self.img[self.mask])
        predicted_sum = np.sum(self.MU[-1][self.mask])

        if true_sum == 0:
            return 0.0
            
        return abs(1 - predicted_sum / true_sum) * 100


class GPUCB_ADAPTIVE_RADIUS(GPUCB_NEW):
    """A GPUCB variant with an adaptive search radius."""
    def __init__(self, img, mask, radius=20, **kwargs):
        super().__init__(img, mask, **kwargs)
        self.radius = radius
        self.RADIUS = []

    def argmax_ucb(self):
        """Selects the best UCB point within the current radius."""
        if self.sp == 0:
            ucb_values = self.mu + self.sigma * np.sqrt(self.beta)
            return np.argmax(ucb_values)

        ucb_values = self.mu + self.sigma * np.sqrt(self.beta)
        sorted_indices = np.argsort(ucb_values)[::-1]

        last_point = self.visited_points[-1]

        for idx in sorted_indices:
            candidate_point = self.grid[idx]
            distance = self.get_distance(candidate_point, last_point)
            if distance < self.radius:
                return idx

        return sorted_indices[0]

    def update_radius(self, points):
        """Dynamically adjusts the search radius based on prediction error."""
        if not points:
            return

        rows = np.clip(np.rint([p[0] for p in points]).astype(int), 0, self.img.shape[0] - 1)
        cols = np.clip(np.rint([p[1] for p in points]).astype(int), 0, self.img.shape[1] - 1)

        true_values = self.img[rows, cols]
        predicted_values = self.MU[-1][rows, cols]

        error_std = np.std(np.abs(true_values - predicted_values))

        if error_std > 0.2 and self.radius > 8:
            self.radius -= 1
        elif error_std < 0.1 and self.radius < 32:
            self.radius += 1

        self.RADIUS.append(self.radius)

    def learn(self):
        """Overrides the learn method to include radius updates."""
        mu, dist = super().learn()
        if mu is not None and self.debug:
            points_from_last_step = self.points.get(self.sp - 1, [])
            self.update_radius(points_from_last_step)
        return mu, dist


if __name__ == '__main__':

    wandb.init(
        project="thompson_sampling",
        name="GPUCB_NEW_3_agents_divide_map"
    )

    try:
        with open('/home/moonlab/multi_agent_thompson/src/maps/real_maps/windmill_lake_1.pkl', 'rb') as f:
            cost_map = pickle.load(f, encoding='latin1')
        print("✅ Successfully loaded 'windmill_lake_1.pkl'")
        print(f"🗺️ Map dimensions: {cost_map.shape}, Data type: {cost_map.dtype}")
    except FileNotFoundError:
        print("❌ Error: 'cost_map_1.pkl' not found. Creating a dummy map for demonstration.")
        cost_map = np.random.rand(100, 100) * 50


    NUM_AGENTS = 3
    AGENT_CLASS = GPUCB_NEW
    LAWN_MOWER_DISTANCE = 899
   
    if AGENT_CLASS == GPUCB:
        print("\n Error: The base GPUCB class cannot be used for partitioning because it does not accept a 'mask'.")
        print("Please choose either 'GPUCB_NEW' or 'GPUCB_ADAPTIVE_RADIUS'.")
        exit()

    # overall_mask = cost_map != 0
    overall_mask = ~np.isnan(cost_map)
    partition_masks = create_partitions(overall_mask, NUM_AGENTS)

    agents = []
    print(f"--- Initializing {NUM_AGENTS} agents of type {AGENT_CLASS.__name__} ---")

    if AGENT_CLASS == GPUCB_ADAPTIVE_RADIUS:
        base_params = {'img': cost_map, 'beta': 1e6, 'radius': 25, 'debug': True}
        agents = [GPUCB_ADAPTIVE_RADIUS(mask=p_mask, **base_params) for p_mask in partition_masks]
    
    elif AGENT_CLASS == GPUCB_NEW:
        base_params = {'img': cost_map, 'beta': 1e6, 'debug': True}
        agents = [GPUCB_NEW(mask=p_mask, **base_params) for p_mask in partition_masks]


    iteration = 0
    max_iterations = 270
    while iteration < max_iterations:
        total_distance = 0
        distance_limit_reached = False

        for i, agent in enumerate(agents):
            agent.learn()
            total_distance += agent.distance
            # if agent.distance > LAWN_MOWER_DISTANCE:
            #     print(f"\n Exploration stopped: Agent {i} exceeded distance limit of {LAWN_MOWER_DISTANCE}.")
            #     distance_limit_reached = True
            #     break
        
        combined_map_iteration = np.zeros_like(cost_map, dtype=float)
        active_agents = 0
        for agent in agents:
            if agent.MU: 
                latest_prediction = agent.MU[-1]
                combined_map_iteration[agent.mask] = latest_prediction[agent.mask]
                active_agents +=1

        if active_agents > 0:
            total_true_sum = np.sum(cost_map[overall_mask])
            if total_true_sum == 0:
                combined_error = 0.0
            else:
                total_predicted_sum = np.sum(combined_map_iteration[overall_mask])
                combined_error = abs(1 - total_predicted_sum / total_true_sum) * 100
        else:
            combined_error = 100.0


        print(f'Iteration: {iteration+1}; Combined Error: {combined_error:.2f}%; Total Distance: {total_distance:.2f}', end='\r')
        
        # Check all termination conditions
        # if distance_limit_reached:
        #     break
            
        if combined_error < 1.0:
            print(f"\n Exploration stopped: Combined error is below 1%.")
            break
        
        iteration += 1
        wandb.log({
            "Combined Error (%)": combined_error,
            "Total Distance": total_distance,
            "Iteration": iteration
        })

    if iteration == max_iterations:
        print(f"\n Exploration stopped: Reached max iterations ({max_iterations}).")

    print("\n--- 💾 Saving Results ---")
    combined_map = np.zeros_like(cost_map, dtype=float)
    for i, agent in enumerate(agents):
        if agent.MU:
            final_map = agent.MU[-1]
            filename = f'predicted_map_agent_{i}.npy'
            np.save(filename, final_map)
            print(f"Saved agent {i}'s map to '{filename}'")
            
            combined_map[agent.mask] = final_map[agent.mask]
        else:
            print(f"Agent {i} did not generate data to save.")

    np.save('oasgpucb_combined_predicted_map_1.npy', combined_map)
    print("Final combined map saved as 'oasgpucb_combined_predicted_map.npy'")

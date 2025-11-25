import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class SpatialVotingSimulation:
    def __init__(self, voter_groups, candidates, learning_rate=0.1):
        """
        voter_groups: List of dictionaries, e.g.,
                      [{'center': [-3, 0], 'size': 0.5, 'n': 50}, ...]
        candidates: List of starting coordinates, e.g., [[0,1], [0,-1]]
        """
        self.lr = learning_rate
        self.history = []

        # 1. Generate Voters
        self.voters = []
        self.group_labels = []
        for i, g in enumerate(voter_groups):
            # Generate random points around the center
            cluster = np.random.normal(
                loc=g["center"], scale=g["size"], size=(g["n"], 2)
            )
            self.voters.append(cluster)
            self.group_labels.extend([i] * g["n"])

        self.voters = np.vstack(self.voters)
        self.group_labels = np.array(self.group_labels)
        self.candidates = np.array(candidates, dtype=np.float64)

        # Save initial state
        self.history.append(self.candidates.copy())

    def _get_gradient(self, candidate_idx):
        """
        Calculates the gradient for a specific candidate based on
        softmax probability (multinomial logit).
        """
        # Calculate Utilities: u = -distance^2
        # Shape: (n_voters, n_candidates)
        diffs = self.voters[:, np.newaxis, :] - self.candidates[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        utilities = -dists_sq

        # Softmax probabilities
        # Subtract max for numerical stability
        max_u = np.max(utilities, axis=1, keepdims=True)
        exp_u = np.exp(utilities - max_u)
        sum_exp_u = np.sum(exp_u, axis=1, keepdims=True)
        probs = exp_u / sum_exp_u

        # Gradient component for candidate_idx
        # grad = sum( P_i * (1 - P_i) * 2 * (voter_pos - cand_pos) )
        p_i = probs[:, candidate_idx]
        weights = p_i * (1 - p_i)

        direction_vecs = self.voters - self.candidates[candidate_idx]
        weighted_vecs = weights[:, np.newaxis] * direction_vecs

        return 2 * np.sum(weighted_vecs, axis=0)

    def run(self, steps=100):
        print(f"Running simulation for {steps} steps...")
        for _ in range(steps):
            # Sequential updates (Turn-based)
            for i in range(len(self.candidates)):
                grad = self._get_gradient(i)
                self.candidates[i] += self.lr * grad

            self.history.append(self.candidates.copy())
        self.history = np.array(self.history)
        print("Simulation complete.")

    def plot(self, title="Spatial Voting Simulation"):
        plt.figure(figsize=(10, 8))

        # Plot Voters
        unique_groups = np.unique(self.group_labels)
        colors = cm.get_cmap("tab10", len(unique_groups))

        for g_id in unique_groups:
            mask = self.group_labels == g_id
            plt.scatter(
                self.voters[mask, 0],
                self.voters[mask, 1],
                alpha=0.3,
                color=colors(g_id),
                label=f"Voter Group {g_id+1}",
            )

        # Plot Candidates
        cand_colors = ["black", "red", "gold", "purple", "cyan"]
        styles = ["-", "--", "-.", ":"]

        for i in range(len(self.candidates)):
            # Fallback color if more than 5 candidates
            c = cand_colors[i % len(cand_colors)]
            s = styles[i % len(styles)]

            path = self.history[:, i, :]

            # Trajectory
            plt.plot(
                path[:, 0], path[:, 1], linestyle=s, color=c, linewidth=2, alpha=0.8
            )
            # Start
            plt.plot(path[0, 0], path[0, 1], "o", color=c, markersize=8)
            # End
            plt.plot(
                path[-1, 0],
                path[-1, 1],
                "X",
                color=c,
                markersize=12,
                markeredgecolor="white",
                label=f"Cand {i+1}",
            )

        plt.title(title)
        plt.legend()
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
        plt.close()


# ==========================================
# SCENARIO CONFIGURATION
# ==========================================

# Example 1: The "Cannibalization" Effect
# Two candidates fight over Group 1, leaving Group 2 open for Candidate 3.
groups = [
    {"center": [-4, 0], "size": 0.8, "n": 900},  # Left Base
    {"center": [4, 0], "size": 0.8, "n": 50},  # Right Base
]

start_positions = [
    [0.0, 0.0],  # Cand 1: Incumbent on Left
    [2.0, 0.0],  # Cand 2: Challenger on Left
]

# Initialize and Run
sim = SpatialVotingSimulation(groups, start_positions, learning_rate=0.01)
sim.run(steps=600)
sim.plot(title="Scenario: The Primary Challenge vs. Open Field")

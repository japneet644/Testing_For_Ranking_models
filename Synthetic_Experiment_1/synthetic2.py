import numpy as np
from scipy.special import erf
import json


class Experiment:
    def __init__(self, n, k, topology, activation):
        self.n = n
        self.k = k
        self.E = np.zeros([n, n], dtype=bool)
        self.topology = topology
        self.activation = activation
        self.w = 0 * np.random.uniform(-1.0, 1.0, self.n)
        self.thurstone_weights = 0 * np.random.uniform(-1.0, 1.0, self.n)
        self.test_statistics = 0

    def generate_weight_vector(self):
        # Draw a weight vector w from uniformly from [-2,2] length n
        # Make it zero mean
        if self.activation == "sigmoid":
            self.w = np.random.uniform(-1.47, 1.47, self.n)
        elif self.activation == "qfunc":
            self.w = np.random.uniform(-0.8, 0.8, self.n)
        self.w -= self.w.mean()

        # print('True weights', self.w)

    def generate_graph(self):
        # Generate a matrix E with topologies complete graph on n nodes, single cycle on n nodes, Barbell graph (n/2---n/2 nodes)
        if self.topology == "complete":
            Etemp = np.ones((self.n, self.n)) - np.eye(self.n)

        elif self.topology == "cycle":
            Etemp = np.eye(self.n, k=1) + np.eye(self.n, k=-1)
            Etemp[0, -1] = Etemp[-1, 0] = 1
            np.fill_diagonal(Etemp, 0)  # Add this line to set diagonal elements to 0

        elif self.topology == "barbell":
            if self.n % 2 != 0:
                print("Skipping barbell topology for odd n")
                return
            Etemp = np.block(
                [
                    [
                        np.ones((self.n // 2, self.n // 2)),
                        np.zeros((self.n // 2, self.n // 2)),
                    ],
                    [
                        np.zeros((self.n // 2, self.n // 2)),
                        np.ones((self.n // 2, self.n // 2)),
                    ],
                ]
            )
            Etemp[self.n // 2 - 1, self.n // 2] = 1
            Etemp[self.n // 2, self.n // 2 - 1] = 1

        elif self.topology == "2Dgrid":  # sqrt{n}xsqrt{n} square grid
            if self.n**0.5 != int(self.n**0.5):
                print("Skipping 2D grid topology for non-square n")
                return
            m1 = int(np.sqrt(self.n))
            m2 = self.n // m1
            Etemp = np.zeros((self.n, self.n), dtype=int)
            for i in range(self.n):
                if (i + 1) % m2 != 0:  # Connect horizontally within rows
                    Etemp[i, i + 1] = 1
                    Etemp[i + 1, i] = 1
                else:  # Wrap around horizontally
                    Etemp[i, i + 1 - m2] = 1
                    Etemp[i + 1 - m2, i] = 1
                if i + m2 < self.n:  # Connect vertically
                    Etemp[i, i + m2] = 1
                    Etemp[i + m2, i] = 1
                else:  # Wrap around vertically
                    Etemp[i, i + m2 - self.n] = 1
                    Etemp[i + m2 - self.n, i] = 1
        elif self.topology == "Erdos":
            p = 2 * np.log(self.n) ** 2 / self.n
            Etemp = np.zeros((self.n, self.n), dtype=int)
            for i in range(self.n):
                for j in range(i):
                    Etemp[i, j] = np.random.binomial(1, p)
                    Etemp[j, i] = Etemp[i, j]
        self.E = Etemp > 0.5
        #         print(self.E)

    def compute_P(self):
        # Compute P[i,j] = qfunc(w_i-w_j) or sigmoid(w_i - w_j)
        if self.activation == "qfunc":
            self.P = np.zeros((self.n, self.n))
            indices = np.argwhere(self.E)
            for i, j in indices:
                self.P[i, j] = qfunc(self.w[i] - self.w[j])
        elif self.activation == "sigmoid":
            self.P = np.zeros((self.n, self.n))
            indices = np.argwhere(self.E)
            for i, j in indices:
                self.P[i, j] = F(self.w[i] - self.w[j])
        np.fill_diagonal(self.P, 0)

    def modify_P(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.E[i, j]:
                    if i > j:
                        self.P[i, j] += 0.2
                    elif i < j:
                        self.P[i, j] -= 0.2
        return self.P

    def generate_Z(self):
        # Generate a matrix Z of size nxnxk such that for (i,j) in E fill Z[i,j,k] is sampled from Ber(p_{ij})
        self.Z = np.zeros((self.n, self.n, self.k), dtype=bool)

        for i in range(self.n):
            for j in range(self.n):
                if self.E[i, j]:  # Check if (i, j) is in E
                    self.Z[i, j] = np.random.binomial(1, self.P[i, j], self.k).astype(
                        bool
                    )
        return self.Z

    def compute_P_empirical(self):
        # Compute P_empirical P[i,j] = sum_{m=1}^k (Z[i,j][m]) for (i,j) in E
        self.P_empirical = self.Z.sum(axis=2) / self.k
        return self.P_empirical

    def compute_thurstone_weights(self):
        # Compute Thurstone weights
        if self.activation == "qfunc":
            self.thurstone_weights = compute_weights_qfunc(self.P_empirical, self.E)
        elif self.activation == "sigmoid":
            self.thurstone_weights = compute_weights_sigmoid(self.P_empirical, self.E)
        # print(self.thurstone_weights)

    def compute_test_statistic(self):
        # Compute test statistic
        Z2 = self.Z.sum(axis=2)
        K2 = self.k * self.E
        self.test_statistics = Compute_mytest_statistic(
            self.thurstone_weights, Z2, K2, self.activation
        )
        return self.test_statistics


def qfunc(x):
    return 1 - 0.5 * (1 - erf(x / np.sqrt(2)))


def F(x):
    return 1 / (1 + np.exp(-x))


def compute_weights_qfunc(P, E):
    learning_rate = 0.01
    max_iterations = 3000
    n = P.shape[0]
    w = np.zeros(n)
    for iter in range(max_iterations):
        gradient = np.zeros_like(w)
        for i in range(n):
            for j in range(n):
                if E[i, j]:
                    gradient[i] += (P[i, j] + 1 - P[j, i]) * np.exp(
                        -((w[j] - w[i]) ** 2) / 2
                    ) / qfunc(w[i] - w[j]) - (1 - P[i, j] + P[j, i]) * np.exp(
                        -((w[j] - w[i]) ** 2) / 2
                    ) / qfunc(
                        w[j] - w[i]
                    )
        if np.linalg.norm(gradient) < 1e-5:
            break
        w = w + learning_rate * gradient
        # w = np.clip(w, -1.0, 1.0)
        # w -= np.mean(w)
        if iter == max_iterations - 1:
            print("max iterations reached")
    return w


def compute_weights_sigmoid(P, E):
    learning_rate = 0.01
    max_iterations = 3000
    n = P.shape[0]
    w = np.zeros(n)
    for iter in range(max_iterations):
        gradient = np.zeros_like(w)
        for i in range(n):
            for j in range(n):
                if E[i, j]:
                    gradient[i] += 0.5 * (P[i, j] + 1 - P[j, i]) - F(w[i] - w[j])
        if np.linalg.norm(gradient) < 1e-5:
            break
        w = w + learning_rate * gradient
        # w = np.clip(w, -1.6, 1.6)
        # w -= np.mean(w)
        if iter == max_iterations - 1:
            print("max iterations reached")
    return w


def Compute_mytest_statistic(what, Z2, K2, activation):
    n = len(what)
    T = 0
    for i in range(n):
        for j in range(n):
            if activation == "qfunc":
                Fijhat = qfunc(what[i] - what[j])
            elif activation == "sigmoid":
                Fijhat = F(what[i] - what[j])
            if K2[i, j] > 1 and i != j:
                T += (
                    (Z2[i, j] * (Z2[i, j] - 1)) / (K2[i, j] * (K2[i, j] - 1))
                    + Fijhat**2
                    - 2 * Fijhat * (Z2[i, j] / K2[i, j])
                )
    return T


def compute_graph_properties(E):
    n = E.shape[0]
    D = np.diag(E.sum(axis=1))
    L = D - E
    eigenvalues = np.linalg.eigvalsh(L)

    lambda_2 = sorted(eigenvalues)[1]
    print(sorted(eigenvalues))
    return lambda_2


n_values = [15, 25, 35, 45, 55]
k_values = [12, 20]


test_statistics = {}
for model in ["qfunc", "sigmoid"]:
    for topology in [
        "2Dgrid",
        "Erdos",
        "complete",
    ]:  #
        for n in n_values:
            if topology == "2Dgrid":
                n = int(np.ceil(np.sqrt(n)) ** 2)
            for k in k_values:
                exp = Experiment(n=n, k=k, topology=topology, activation=model)
                stats = []
                for iter in range(400):
                    exp.generate_weight_vector()
                    exp.generate_graph()
                    exp.compute_P()
                    exp.generate_Z()
                    exp.compute_P_empirical()
                    exp.compute_thurstone_weights()
                    exp.generate_Z()
                    test_statistic = exp.compute_test_statistic()
                    dmax = max(np.sum(exp.E, axis=0))
                    lambda2 = compute_graph_properties(exp.E)
                    normalized_test_statistic = (
                        k * test_statistic * lambda2 / (n * dmax)
                    )
                    stats.append(normalized_test_statistic)
                test_statistics[f"{model}_{topology}_{n}_{k}"] = stats
                with open(f"{model}_{topology}_{n}_{k}.json", "w") as f:
                    json.dump(stats, f)

                # Calculate and print the 95th percentile
                percentile_95 = np.percentile(stats, 95)
                print(
                    f"Model: {model}, Topology: {topology}, n: {n}, k: {k}, dmax: {dmax},lambda2: {lambda2}, 95th Percentile: {percentile_95}"
                )

np.save("test_statisticsErdos.npy", test_statistics)

#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <mutex>
#include <omp.h>

struct GridWorld
{
    int rows = 0;
    int cols = 0;
    int n_states = 0;
    int n_actions = 4; // 0=up, 1=down, 2=left, 3=right
    int start_state = 0;
    int goal_state = 0;
    int max_steps = 0;

    std::vector<bool> is_wall;

    GridWorld(int rows, int cols, double wall_fraction, unsigned seed) {
        this->rows = rows;
        this->cols = cols;
        n_states = rows * cols;
        max_steps = 4 * rows * cols;
        is_wall.assign(n_states, false);

        start_state = 0;
        goal_state = n_states - 1;
        
        // Random number generator and uniform distribution for epsilon-greedy action selection
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        // Regenerate walls until the start and goal are connected.
        bool solvable = false;
        while (!solvable) {
            std::fill(is_wall.begin(), is_wall.end(), false);
            for (int s = 0; s < n_states; s++) {
                if (s == start_state || s == goal_state) continue;
                if (uni(rng) < wall_fraction) is_wall[s] = true;
            }
            solvable = is_reachable();
        }
    }

    int rc_to_state(int r, int c) const { return r * cols + c; }
    bool in_bounds(int r, int c) const { return r >= 0 && r < rows && c >= 0 && c < cols; }

    int step_state(int s, int action) const {
        int r = s / cols, c = s % cols;
        int nr = r, nc = c;

        if (action == 0) nr--;
        else if (action == 1) nr++;
        else if (action == 2) nc--;
        else if (action == 3) nc++;

        if (!in_bounds(nr, nc) || is_wall[rc_to_state(nr, nc)]) return s;
        return rc_to_state(nr, nc);
    }

    bool is_reachable() const {
        std::vector<bool> visited(n_states, false);
        std::queue<int> q;
        q.push(start_state);
        visited[start_state] = true;

        while (!q.empty()) {
            int s = q.front();
            q.pop();

            if (s == goal_state) return true;

            for (int a = 0; a < 4; a++) {
                int ns = step_state(s, a);
                if (ns != s && !visited[ns]) {
                    visited[ns] = true;
                    q.push(ns);
                }
            }
        }
        return false;
    }
};

struct QLearningAgent
{
    const GridWorld& env;
    std::vector<std::vector<double>> Q;

    double alpha = 0.1;
    double gamma = 0.95;

    QLearningAgent(const GridWorld& env_) : env(env_) {
        Q.assign(env.n_states, std::vector<double>(env.n_actions, 0.0));
    }

    int greedy_action(int state) const {
        int best_a = 0;
        double best_q = Q[state][0];

        for (int a = 1; a < env.n_actions; a++) {
            if (Q[state][a] > best_q) {
                best_q = Q[state][a];
                best_a = a;
            }
        }
        return best_a;
    }

    void QLearningAlgorithmParallel(int n_episodes, int num_threads, unsigned seed) {
        double epsilon = 1.0;
        const double epsilon_min = 0.05;
        const double epsilon_decay = 0.9995;
        const int episodes_per_thread = n_episodes / num_threads;

        // One mutex protects each row of the shared Q-table.
        std::vector<std::mutex> locks(env.n_states);

        std::cout << "Threads: " << num_threads << "\n";
        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::mt19937 rng(seed + 1000u * static_cast<unsigned>(tid) + 1u);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            double eps = epsilon;

            for (int ep = 0; ep < episodes_per_thread; ep++) {
                int state = env.start_state;

                for (int t = 0; t < env.max_steps; t++) {
                    int action;
                    if (uni(rng) < eps) {
                        std::uniform_int_distribution<int> act(0, env.n_actions - 1);
                        action = act(rng);
                    } else {
                        std::lock_guard<std::mutex> lock(locks[state]);
                        action = greedy_action(state);
                    }

                    int next_state = env.step_state(state, action);
                    double reward = -1.0;
                    bool done = false;

                    if (next_state == env.goal_state) {
                        reward = 50.0;
                        done = true;
                    } else if (t + 1 >= env.max_steps) {
                        done = true;
                    }

                    double max_next = 0.0;
                    if (!done) {
                        std::lock_guard<std::mutex> lock(locks[next_state]);
                        max_next = Q[next_state][0];
                        for (int a = 1; a < env.n_actions; a++) {
                            max_next = std::max(max_next, Q[next_state][a]);
                        }
                    }

                    double target = reward + (done ? 0.0 : gamma * max_next);
                    {
                        std::lock_guard<std::mutex> lock(locks[state]);
                        Q[state][action] += alpha * (target - Q[state][action]);
                    }

                    state = next_state;
                    if (done) break;
                }

                eps = std::max(epsilon_min, eps * epsilon_decay);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Training time: " << elapsed.count() << " seconds\n";
    }

    void Evaluate(int n_eval_episodes) const {
        int successes = 0;
        double total_steps = 0.0;

        for (int ep = 0; ep < n_eval_episodes; ep++) {
            int state = env.start_state;
            bool reached = false;
            int steps = 0;

            for (; steps < env.max_steps; steps++) {
                state = env.step_state(state, greedy_action(state));
                if (state == env.goal_state) {
                    reached = true;
                    steps++;
                    break;
                }
            }

            if (reached) {
                successes++;
                total_steps += steps;
            }
        }

        std::cout << "Greedy evaluation: " << successes << "/" << n_eval_episodes
                  << " episodes reached the goal";
        if (successes > 0) {
            std::cout << " (avg steps = " << std::fixed << std::setprecision(1)
                      << (total_steps / successes) << ")";
        }
        std::cout << "\n";
    }
};

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Run: ./qlearning_openmp <rows> <cols> <episodes> <threads> [seed]\n";
        return 1;
    }

    int rows = std::atoi(argv[1]);
    int cols = std::atoi(argv[2]);
    int n_episodes = std::atoi(argv[3]);
    int num_threads = std::atoi(argv[4]);
    unsigned seed = (argc > 5) ? static_cast<unsigned>(std::atoi(argv[5])) : 42u;

    if (rows <= 0 || cols <= 0 || n_episodes <= 0 || num_threads <= 0) {
        std::cerr << "Error: rows, cols, episodes and threads must be positive.\n";
        return 1;
    }

    const double wall_fraction = 0.2;

    GridWorld env(rows, cols, wall_fraction, seed);
    QLearningAgent agent(env);

    std::cout << "Grid: " << rows << "x" << cols
              << ", Walls: " << std::count(env.is_wall.begin(), env.is_wall.end(), true)
              << ", Episodes: " << n_episodes << "\n";

    agent.QLearningAlgorithmParallel(n_episodes, num_threads, seed);
    agent.Evaluate(200);

    return 0;
}

// Compile: g++ -fopenmp -O3 -std=c++17 qlearning_shared.cpp -o qlearning_shared
// Run: ./qlearning_shared <rows> <cols> <episodes> <threads> <seed>
//      ./qlearning_shared 60 60 2000000 4 42

# Parallel Systems: Parallel Q-Learning on a GridWorld

This project implements a Q-Learning agent that learns to navigate a randomly generated GridWorld environment from the top-left state to the bottom-right goal state.

Files:
- qlearning_serial.cpp – Sequential Q-Learning implementation
- qlearning_shared.cpp – OpenMP implementation using a shared Q-table protected with per-state mutexes
- qlearning_localsync.cpp – OpenMP implementation using private Q-tables with periodic synchronization
- Makefile – Build rules for all implementations

## Dependencies

Before compiling the project, the following software must be installed:

- make
- g++
- OpenMP support
- Standard C++ libraries

## Compilation

Compile all implementations using:

```bash
make
```

To remove the compiled executables:

```bash
make clean
```

## Usage

### Serial Version

```bash
./qlearning_serial <rows> <cols> <episodes> <seed>
```

Example:

```bash
./qlearning_serial 60 60 2000000 42
```

### Shared Q-Table Version

```bash
./qlearning_shared <rows> <cols> <episodes> <threads> <seed>
```

Example:

```bash
./qlearning_shared 60 60 2000000 4 42
```

### Local-Sync Version

```bash
./qlearning_localsync <rows> <cols> <episodes> <threads> <sync_interval> <seed>
```

Example:

```bash
./qlearning_localsync 60 60 2000000 4 5000 42
```

## Parameters

- rows – Number of GridWorld rows
- cols – Number of GridWorld columns
- episodes – Total number of training episodes
- threads – Number of OpenMP threads
- sync_interval – Number of episodes per thread between Q-table synchronizations
- seed – Random seed used for reproducible GridWorld generation

If no seed is provided, the default value is 42

For the Local-Sync implementation, if no synchronization interval is provided, the default value is 5000

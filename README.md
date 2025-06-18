# NYC Bike Share Simulator

A discrete event simulation of New York City's bike sharing service using real Citi Bike data from June 2022. This project implements a sophisticated simulation model to analyze bike availability, rider behavior, and system performance under various operational scenarios.

## Description

This simulator models the complex dynamics of a bike sharing system where riders arrive at stations, pick up bikes, ride to destinations, and return bikes. The simulation uses real-world data to accurately represent:

- **Rider arrival patterns** using exponential distributions
- **Station selection probabilities** based on actual Citi Bike usage data
- **Trip duration patterns** using log-normal distributions
- **Queue management** for both bike pickups and returns
- **Station capacity constraints** and bike availability

The project is designed for urban planners, transportation researchers, and bike share operators who need to understand system performance, optimize bike distribution, and predict service quality under different operational parameters.

## Features

- **Discrete Event Simulation**: High-performance event-driven simulation engine
- **Real Data Integration**: Uses actual Citi Bike station probabilities and trip statistics
- **Queue Management**: Handles both arrival and return queues at stations
- **Statistical Analysis**: Calculates success rates, wait times, and system metrics
- **Configurable Parameters**: Adjustable arrival rates, trip durations, and station capacities
- **Multiple Simulation Modes**: Baseline and idealized scenarios for different analysis needs
- **Confidence Interval Estimation**: Statistical validation of simulation results

## Tech Stack

- **Python 3.x**: Core programming language
- **NumPy**: Numerical computations and random number generation
- **SciPy**: Statistical functions (Student's t-distribution)
- **CSV**: Data import from Citi Bike datasets
- **Heapq**: Priority queue for event management


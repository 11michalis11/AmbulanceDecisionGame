import csv
import itertools
import os
import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np
import pandas as pd

import ambulance_game as abg

# from nashpy.algorithms.lemke_howson_lex import lemke_howson_lex


def get_index_of_values(problem_parameters, data, atol=1e-08, rtol=1e-05):
    """
    Get the indices of the rows that match the given parameters' values
    """
    index = data.index
    condition = (
        np.isclose(
            data["lambda_2"], problem_parameters["lambda_2"], atol=atol, rtol=rtol
        )
        & np.isclose(
            data["lambda_1_1"], problem_parameters["lambda_1_1"], atol=atol, rtol=rtol
        )
        & np.isclose(
            data["lambda_1_2"], problem_parameters["lambda_1_2"], atol=atol, rtol=rtol
        )
        & np.isclose(data["mu_1"], problem_parameters["mu_1"], atol=atol, rtol=rtol)
        & np.isclose(data["mu_2"], problem_parameters["mu_2"], atol=atol, rtol=rtol)
        & np.isclose(data["num_of_servers_1"], problem_parameters["num_of_servers_1"])
        & np.isclose(data["num_of_servers_2"], problem_parameters["num_of_servers_2"])
        & np.isclose(data["system_capacity_1"], problem_parameters["system_capacity_1"])
        & np.isclose(data["system_capacity_2"], problem_parameters["system_capacity_2"])
        & np.isclose(data["buffer_capacity_1"], problem_parameters["buffer_capacity_1"])
        & np.isclose(data["buffer_capacity_2"], problem_parameters["buffer_capacity_2"])
        & np.isclose(data["alpha"], problem_parameters["alpha"])
        & np.isclose(data["target"], problem_parameters["target"], atol=atol, rtol=rtol)
    )
    indices = index[condition]
    return indices


def create_directory_with_notebook(data, index_value):
    """
    Create the directory with main.ipynb that contains a brief analysis for that scenario
    """
    hash_value = data[index_value : index_value + 1]["hash_value"]
    dirname = hash_value.to_string(index=False)[1:]
    new_dir = pathlib.Path() / dirname
    new_dir.mkdir(exist_ok=True)
    command = "copy _main\main.ipynb " + dirname + "\\"
    os.system(command)
    return dirname


def looks_degenerate(A, B):
    """
    Check if the game looks degenerate i.e. if on a given column of the payoff
    matrix of A or a given row of the payoff matrix of B there are any duplicate
    maximum values
    """
    for col in A.transpose():
        max_value = np.max(col)
        max_duplicates = np.sum(col == max_value)
        if max_duplicates > 1:
            return True
    for row in B:
        max_value = np.max(row)
        max_duplicates = np.sum(row == max_value)
        if max_duplicates > 1:
            return True
    return False


def get_path_of_experiment():
    """Get the name of the directory for the current experiment"""
    path = str(pathlib.Path.cwd())
    parent_dir_index = path.rfind("\\")
    dirname = path[parent_dir_index + 1 :]
    target_path = pathlib.Path("../../data") / dirname
    return target_path


def get_parameters(target_path=None):
    """
    Get the values of the parameters for this experiment
    """
    keys = [
        "alpha",
        "buffer_capacity_1",
        "buffer_capacity_2",
        "lambda_1_1",
        "lambda_1_2",
        "lambda_2",
        "mu_1",
        "mu_2",
        "num_of_servers_1",
        "num_of_servers_2",
        "system_capacity_1",
        "system_capacity_2",
        "target",
    ]
    if target_path is None:
        target_path = get_path_of_experiment()
    else:
        target_path = pathlib.Path("../data") / target_path
    path = pathlib.Path(target_path) / "main.csv"
    with open(path, "r") as file:
        reader = csv.reader(file)
        values = tuple(reader)[0]

    parameters = {}
    for key, value in zip(keys, values):
        parameters[key] = float(value) if "." in value else int(value)
    return parameters


def get_matrices(target_path=None):
    """
    Get the generated matrices for this experiment
    """
    if target_path is None:
        target_path = get_path_of_experiment()
    else:
        target_path = pathlib.Path("../data") / target_path
    matrices = np.load(target_path / "main.npz")
    R = matrices["routing_matrix"]
    A = matrices["payoff_matrix_A"]
    B = matrices["payoff_matrix_B"]
    return R, A, B


def remove_duplicate_entries(tuple_of_arrays):
    """
    Given a tuple of arrays remove the duplicate ones
    """
    strat_1_count = len(tuple_of_arrays[0][0])
    unique_vectors = np.unique(
        np.array([np.concatenate((i, j)) for i, j in tuple_of_arrays]), axis=0
    )
    unique_bi_vectors = tuple(
        tuple((vec[:strat_1_count], vec[strat_1_count:])) for vec in unique_vectors
    )
    return unique_bi_vectors


def get_evolutionary_stable_strategies(
    A, B, method=None, learning_repetitions=10, **kwargs
):
    """
    Get all unique ESS using one of the following functions:
    nash.Game.lemke_howson_enumeration(),
    nash.Game.fictitious_play(),
    nash.Game.stochastic_fictitious_play(),
    nash.Game.asymmetric_replicator_dynamics(),

    Parameters
    ----------
    A : numpy array
    B : numpy array
    method : function, optional
    learning_repetitions : int, optional

    Additional parameters (kwargs)
    ------------------------------
    iterations : int
        required parameter for fictitious_play() and stochastic_fictitious_play()
    play_counts : bi-tuple
        optional parameter for fictitious_play() and stochastic_fictitious_play()
    etha : float
        optional parameter for stochastic_fictitious_play()
    epsilon : float
        optional parameter for stochastic_fictitious_play()
    timepoints : numpy array
        optional parameter for asymmetric_replicator_dynamics()
    x0 : numpy array
        optional parameter for asymmetric_replicator_dynamics()
    y0 : numpy array
        optional parameter for asymmetric_replicator_dynamics()

    Returns
    -------
    tuple of numpy arrays
        all ESS strategies from the specified method
    """
    game = nash.Game(A, B)
    if method is None or method == nash.Game.lemke_howson_enumeration:
        all_equilibs = tuple(nash.Game.lemke_howson_enumeration(game))
        unique_equilibs = remove_duplicate_entries(all_equilibs)
    elif method == nash.Game.asymmetric_replicator_dynamics:
        xs, ys = method(game, **kwargs)
        unique_equilibs = (tuple((np.round(xs[-1], 2), np.round(ys[-1], 2))),)
    else:
        all_equilibs = ()
        for _ in range(learning_repetitions):
            *_, last_iteration = method(game, **kwargs)
            unnormalised_iteration = (
                last_iteration[0] if type(last_iteration) == tuple else last_iteration
            )
            normalised_iteration = [
                np.round(arr / np.sum(unnormalised_iteration[0]), 2)
                for arr in unnormalised_iteration
            ]
            all_equilibs += (normalised_iteration,)
        unique_equilibs = remove_duplicate_entries(all_equilibs)
    return unique_equilibs


def get_performance_measure_for_given_strategies(
    strategy_A, strategy_B, routing, parameters, performance_measure_function
):
    """
    For a given set of strategies get the sum of a given performance measure of the two players
    """
    prop_1 = routing[strategy_A, strategy_B]
    lambda_2_1 = parameters["lambda_2"] * prop_1
    lambda_2_2 = parameters["lambda_2"] * (1 - prop_1)

    performance_measure_1 = performance_measure_function(
        lambda_2=lambda_2_1,
        lambda_1=parameters["lambda_1_1"],
        mu=parameters["mu_1"],
        num_of_servers=int(parameters["num_of_servers_1"]),
        threshold=strategy_A + 1,
        system_capacity=int(parameters["system_capacity_1"]),
        buffer_capacity=int(parameters["buffer_capacity_1"]),
    )

    performance_measure_2 = performance_measure_function(
        lambda_2=lambda_2_2,
        lambda_1=parameters["lambda_1_2"],
        mu=parameters["mu_2"],
        num_of_servers=int(parameters["num_of_servers_2"]),
        threshold=strategy_B + 1,
        system_capacity=int(parameters["system_capacity_2"]),
        buffer_capacity=int(parameters["buffer_capacity_2"]),
    )

    if (
        performance_measure_function
        == abg.markov.get_accepting_proportion_of_class_2_individuals
    ):
        performance_measure_1 = 1 - performance_measure_1
        performance_measure_2 = 1 - performance_measure_2
    return performance_measure_1, performance_measure_2


def build_performance_values_array(routing, parameters, performance_measure_function):
    """
    Get all the values for the current investigated performance measure
    """
    all_performance_values_A = np.zeros(routing.shape)
    all_performance_values_B = np.zeros(routing.shape)
    for strategy_A, strategy_B in itertools.product(
        range(routing.shape[0]), range(routing.shape[1])
    ):
        measure_A, measure_B = get_performance_measure_for_given_strategies(
            strategy_A=strategy_A,
            strategy_B=strategy_B,
            routing=routing,
            parameters=parameters,
            performance_measure_function=performance_measure_function,
        )
        all_performance_values_A[strategy_A, strategy_B] = measure_A
        all_performance_values_B[strategy_A, strategy_B] = measure_B
    return all_performance_values_A, all_performance_values_B


def find_worst_nash_equilibrium_measure(
    all_nash_equilibrias,
    performance_values_array,
):
    """
    Get the maximum value of the performance measure out of all possible
    equilibria
    """
    max_performance_measure = None
    for row_strategies, col_strategies in all_nash_equilibrias:
        current_performance_measure = (
            row_strategies @ performance_values_array @ col_strategies
        )
        if (
            max_performance_measure is None
            or current_performance_measure > max_performance_measure
        ):
            max_performance_measure = current_performance_measure
            max_row = row_strategies
            max_col = col_strategies
    return max_performance_measure, max_row, max_col


def get_price_of_anarchy(
    performance_measure_function,
    equilib_method=nash.Game.lemke_howson_enumeration,
    target_path=None,
    scalar=1,
    **kwargs,
):
    """
    Get the price of anarchy for the performance measure function given. Possible
    performance_measure_functions:
        - Mean Blocking Time
        - Mean Waiting Time
        - Proportion of lost class 2 individuals
    """
    parameters = get_parameters(target_path=target_path)
    routing, A, B = get_matrices(target_path=target_path)
    A, B = A * scalar, B * scalar
    #     if looks_degenerate(A, B):
    #         equilibria = get_fictitious_play_outcome()
    #     else:
    #         equilibria = get_lemke_howson_outcome()
    try:
        equilibria = get_evolutionary_stable_strategies(
            A=A,
            B=B,
            method=equilib_method,
            **kwargs,
        )
        (
            performance_values_array_A,
            performance_values_array_B,
        ) = build_performance_values_array(
            routing=routing,
            parameters=parameters,
            performance_measure_function=performance_measure_function,
        )
        performance_values_array = (
            performance_values_array_A + performance_values_array_B
        )
        minimum_value = np.min(performance_values_array)
        worst_equilib_value, max_row, max_col = find_worst_nash_equilibrium_measure(
            all_nash_equilibrias=equilibria,
            performance_values_array=performance_values_array,
        )
        price_of_anarchy = worst_equilib_value / minimum_value
        return price_of_anarchy, max_row, max_col
    except:
        return np.nan, np.nan, np.nan


def get_x_range(problem_parameters):
    """
    Get the range of values for lambda_2 that is generated by the code
    """
    stop = 2 * (
        problem_parameters["mu_1"] * problem_parameters["num_of_servers_1"]
        + problem_parameters["mu_2"] * problem_parameters["num_of_servers_2"]
    )
    x_range = np.linspace(start=0.1, stop=stop, num=10)
    return x_range


def get_poa_list(
    data,
    problem_parameters,
    x_range,
    key_name,
    equilib_method=nash.Game.lemke_howson_enumeration,
    **kwargs,
):
    """
    Get a list of lists that each contain the value of the price of anarchy for
    the mean waiting time, the mean blocking time and the proportion of
    individuals lost for different values of key_name.

    Also add the equilibrium value for the particular game. Note that only
    the worst set of strategies is recorded even if there are more
    """
    price_of_anarchy_list = []
    for x_value in x_range:
        problem_parameters[key_name] = x_value
        index = get_index_of_values(problem_parameters, atol=1e-1, data=data)
        dirname = data.iloc[index[0]]["hash_value"]
        poa_waiting, equilibs_1, equilibs_2 = get_price_of_anarchy(
            performance_measure_function=abg.markov.get_mean_waiting_time_using_markov_state_probabilities,
            equilib_method=equilib_method,
            target_path=dirname,
            **kwargs,
        )
        poa_blocking, _, _ = get_price_of_anarchy(
            performance_measure_function=abg.markov.get_mean_blocking_time_using_markov_state_probabilities,
            equilib_method=equilib_method,
            target_path=dirname,
            **kwargs,
        )
        poa_lost, _, _ = get_price_of_anarchy(
            performance_measure_function=abg.markov.get_accepting_proportion_of_class_2_individuals,
            equilib_method=equilib_method,
            target_path=dirname,
            **kwargs,
        )
        price_of_anarchy_list += [
            [poa_waiting, poa_blocking, poa_lost, equilibs_1, equilibs_2]
        ]
    return price_of_anarchy_list


def get_poa_plot(
    data,
    problem_parameters,
    x_range,
    key_name,
    equilib_method=None,
    poa_list=None,
    y_min=None,
    y_max=None,
    show_strats=True,
    annotate=False,
    bar_sep=None,
    **kwargs,
):
    """
    Plot the price of anarchy list generated in get_poa_list()
    """
    if poa_list is None:
        poa_list = get_poa_list(
            data=data,
            problem_parameters=problem_parameters,
            x_range=x_range,
            key_name=key_name,
            equilib_method=equilib_method,
            **kwargs,
        )
    poa_array = np.asarray(poa_list, dtype=object)
    waiting_poa_array = poa_array[:, 0]
    blocking_poa_array = poa_array[:, 1]
    lost_poa_array = poa_array[:, 2]
    equils = np.array(
        [[np.argmax(index[3]) + 1, np.argmax(index[4]) + 1] for index in poa_array]
    )

    plt.figure(figsize=(10, 5))
    plt.plot(x_range, waiting_poa_array)
    plt.plot(x_range, blocking_poa_array)
    plt.plot(x_range, lost_poa_array)
    plt.title("Prices of Anarchy")
    plt.xlabel(key_name)
    plt.ylim(bottom=y_min, top=y_max)
    plt.legend(("Waiting", "Blocking", "Lost"))
    if annotate:
        for index, x in enumerate(x_range):
            eq_cords = tuple(equils[index])
            plt.annotate(eq_cords, (x_range[index], lost_poa_array[index]))
    if show_strats:
        plt.figure(figsize=(10, 5))
        num_of_strats = tuple((len(poa_list[0][3]), len(poa_list[0][4])))
        if bar_sep is None:
            bar_sep = np.max(x_range) / (6 * len(x_range))
        for lambda_2, poa_values in zip(x_range, poa_list):
            np.random.seed(10)
            try:
                chromata = np.random.choice(
                    list(mcolors.TABLEAU_COLORS), np.max(num_of_strats), replace=False
                )
            except ValueError:
                chromata = np.random.choice(
                    list(mcolors.cnames), len(poa_values[3]), replace=False
                )
            for lambda_2, poa_values in zip(x_range, poa_list):
                player_2_more_strats = num_of_strats[0] < num_of_strats[1]
                player_order = np.linspace(
                    player_2_more_strats, not player_2_more_strats, 2, dtype=int
                )
                for player in player_order:
                    bottom_bar = 0
                    for index, strat in enumerate(poa_values[3 + player]):
                        plt.bar(
                            lambda_2 - bar_sep + (2 * player * bar_sep),
                            strat,
                            bottom=bottom_bar,
                            edgecolor="black",
                            color=chromata[index],
                        )
                        bottom_bar += strat
        plt.legend(labels=[f"$S_{i + 1}$" for i in range(max(num_of_strats))], loc=3)
        plt.xticks(x_range, np.round(x_range, 1))
    plt.show()
    return poa_list


def get_poa_values_for_given_strategies(
    all_xs, all_ys, poa_span, routing, problem_parameters, performance_measure_function
):
    """Get the price of anarchy values of all_xs and all_ys"""
    (
        performance_values_array_A,
        performance_values_array_B,
    ) = build_performance_values_array(
        routing=routing,
        parameters=problem_parameters,
        performance_measure_function=performance_measure_function,
    )

    minimum_value_A = np.min(performance_values_array_A)
    minimum_value_B = np.min(performance_values_array_B)

    performace_measure_poa_list_A = [
        find_worst_nash_equilibrium_measure(
            all_nash_equilibrias=(tuple((all_xs[i], all_ys[i])),),
            performance_values_array=performance_values_array_A,
        )[0]
        / minimum_value_A
        for i in poa_span
    ]
    performace_measure_poa_list_B = [
        find_worst_nash_equilibrium_measure(
            all_nash_equilibrias=(tuple((all_xs[i], all_ys[i])),),
            performance_values_array=performance_values_array_B,
        )[0]
        / minimum_value_B
        for i in poa_span
    ]
    return performace_measure_poa_list_A, performace_measure_poa_list_B


def run_replicator_dynamics_with_penalty(
    A, B, timepoints, penalty=None, x_init=None, y_init=None
):
    player_A = A.copy()
    player_B = B.copy()
    game = nash.Game(player_A, player_B)
    break_point = int(len(timepoints) / 2)
    if penalty is None:
        break_point = int(len(timepoints))
    all_xs, all_ys = game.asymmetric_replicator_dynamics(
        timepoints=timepoints[:break_point], x0=x_init, y0=y_init
    )
    if penalty is not None:
        player_A[np.argmax(all_xs[-1]), :] *= penalty
        player_B[:, np.argmax(all_ys[-1])] *= penalty
        penalised_game = nash.Game(player_A, player_B)
        new_xs, new_ys = penalised_game.asymmetric_replicator_dynamics(
            timepoints=timepoints[break_point:], x0=all_xs[-1], y0=all_ys[-1]
        )
        all_xs = np.concatenate((all_xs, new_xs))
        all_ys = np.concatenate((all_ys, new_ys))
    return all_xs, all_ys


def plot_asymmetric_replicator_dynamics_with_penalty(
    R,
    A,
    B,
    problem_parameters,
    penalty=None,
    x_init=None,
    y_init=None,
    timepoints=None,
    poa_plot_max=None,
    poa_plot_min=None,
    performance_measure_function=abg.markov.get_mean_blocking_time_using_markov_state_probabilities,
):
    all_xs, all_ys = run_replicator_dynamics_with_penalty(
        A=A,
        B=B,
        penalty=penalty,
        timepoints=timepoints,
        x_init=x_init,
        y_init=y_init,
    )
    poa_span = np.linspace(0, len(all_xs) - 1, 10, dtype=int)
    (
        performance_value_poa_A,
        performance_value_poa_B,
    ) = get_poa_values_for_given_strategies(
        all_xs=all_xs,
        all_ys=all_ys,
        poa_span=poa_span,
        routing=R,
        problem_parameters=problem_parameters,
        performance_measure_function=performance_measure_function,
    )

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(all_xs)
    plt.xlabel("Timepoints")
    plt.ylabel("Probability")
    plt.title("Row player")
    plt.legend([f"$s_{i + 1}$" for i in range(len(all_xs[0]))])
    if penalty is not None:
        plt.plot(
            [len(all_xs) / 2, len(all_xs) / 2],
            [-0.1, 1.1],
            linewidth=1,
            linestyle=":",
            color="green",
        )
        plt.annotate("Penalty", (len(all_xs) / 2, 1.1), ha="center")

    plt.subplot(2, 2, 2)
    plt.plot(all_ys)
    plt.xlabel("Timepoints")
    plt.ylabel("Probability")
    plt.title("Column player")
    plt.legend([f"$s_{i + 1}$" for i in range(len(all_ys[0]))])
    if penalty is not None:
        plt.plot(
            [len(all_ys) / 2, len(all_ys) / 2],
            [-0.1, 1.1],
            linewidth=1,
            linestyle=":",
            color="green",
        )
        plt.annotate("Penalty", (len(all_ys) / 2, 1.1), ha="center")

    plt.subplot(2, 2, 3)
    plt.title("Row player - PoA")
    plt.plot(poa_span, performance_value_poa_A, color="black", linewidth=6)
    plt.ylim(top=poa_plot_max, bottom=poa_plot_min)

    plt.subplot(2, 2, 4)
    plt.title("Column player - PoA")
    plt.plot(poa_span, performance_value_poa_B, color="black", linewidth=6)
    plt.ylim(top=poa_plot_max, bottom=poa_plot_min)


def run_replicator_dynamics_with_dual_parameters(
    A1, B1, A2, B2, timepoints, divide=2, x_init=None, y_init=None
):
    player_A = A1.copy()
    player_B = B1.copy()
    game = nash.Game(player_A, player_B)
    break_point = int(len(timepoints) / divide)
    xs_1, ys_1 = game.asymmetric_replicator_dynamics(
        timepoints=timepoints[:break_point], x0=x_init, y0=y_init
    )

    if A1.shape != A2.shape:
        added_strats_1 = A2.shape[0] - A1.shape[0]
        added_strats_2 = A2.shape[1] - A1.shape[1]

        xs_1 = [
            np.hstack((entry, np.array([0 for _ in range(added_strats_1)])))
            for entry in xs_1
        ]
        ys_1 = [
            np.hstack((entry, np.array([0 for _ in range(added_strats_2)])))
            for entry in ys_1
        ]
    player_A = A2.copy()
    player_B = B2.copy()
    new_game = nash.Game(player_A, player_B)
    xs_2, ys_2 = new_game.asymmetric_replicator_dynamics(
        timepoints=timepoints[break_point:],
        x0=np.array(xs_1[-1]),
        y0=np.array(ys_1[-1]),
    )
    return xs_1, xs_2, ys_1, ys_2


def get_poa_values_for_given_strategies_dual_parameters(
    all_xs,
    all_ys,
    poa_span_1,
    poa_span_2,
    routing_1,
    routing_2,
    problem_parameters_1,
    problem_parameters_2,
    performance_measure_function,
):
    (
        performance_values_array_A_1,
        performance_values_array_B_1,
    ) = build_performance_values_array(
        routing=routing_1,
        parameters=problem_parameters_1,
        performance_measure_function=performance_measure_function,
    )
    (
        performance_values_array_A_2,
        performance_values_array_B_2,
    ) = build_performance_values_array(
        routing=routing_2,
        parameters=problem_parameters_2,
        performance_measure_function=performance_measure_function,
    )

    minimum_value_A_1 = np.min(performance_values_array_A_1)
    minimum_value_B_1 = np.min(performance_values_array_B_1)
    minimum_value_A_2 = np.min(performance_values_array_A_2)
    minimum_value_B_2 = np.min(performance_values_array_B_2)

    performace_measure_poa_list_A_1 = [
        find_worst_nash_equilibrium_measure(
            all_nash_equilibrias=(
                tuple(
                    (all_xs[i][: routing_1.shape[0]], all_ys[i][: routing_1.shape[1]])
                ),
            ),
            performance_values_array=performance_values_array_A_1,
        )[0]
        / minimum_value_A_1
        for i in poa_span_1
    ]

    performace_measure_poa_list_A_2 = [
        find_worst_nash_equilibrium_measure(
            all_nash_equilibrias=(tuple((all_xs[i], all_ys[i])),),
            performance_values_array=performance_values_array_A_2,
        )[0]
        / minimum_value_A_2
        for i in poa_span_2
    ]

    performace_measure_poa_list_B_1 = [
        find_worst_nash_equilibrium_measure(
            all_nash_equilibrias=(
                tuple(
                    (all_xs[i][: routing_1.shape[0]], all_ys[i][: routing_1.shape[1]])
                ),
            ),
            performance_values_array=performance_values_array_B_1,
        )[0]
        / minimum_value_B_1
        for i in poa_span_1
    ]
    performace_measure_poa_list_B_2 = [
        find_worst_nash_equilibrium_measure(
            all_nash_equilibrias=(tuple((all_xs[i], all_ys[i])),),
            performance_values_array=performance_values_array_B_2,
        )[0]
        / minimum_value_B_2
        for i in poa_span_2
    ]

    return np.concatenate(
        (performace_measure_poa_list_A_1, performace_measure_poa_list_A_2)
    ), np.concatenate(
        (performace_measure_poa_list_B_1, performace_measure_poa_list_B_2)
    )


def plot_asymmetric_replicator_dynamics_with_dual_parameters(
    R1,
    A1,
    B1,
    R2,
    A2,
    B2,
    problem_parameters_1,
    problem_parameters_2,
    timepoints,
    divide=2,
    x_init=None,
    y_init=None,
    poa_plot_max=None,
    poa_plot_min=None,
    performance_measure_function=abg.markov.get_mean_blocking_time_using_markov_state_probabilities,
):
    xs_1, xs_2, ys_1, ys_2 = run_replicator_dynamics_with_dual_parameters(
        A1=A1,
        B1=B1,
        A2=A2,
        B2=B2,
        timepoints=timepoints,
        divide=divide,
        x_init=x_init,
        y_init=y_init,
    )
    all_xs = np.concatenate((xs_1, xs_2))
    all_ys = np.concatenate((ys_1, ys_2))

    break_point = int(10 / divide)
    poa_span_1 = np.linspace(0, len(all_xs) - 1, 10, dtype=int)[:break_point]
    poa_span_2 = np.linspace(0, len(all_xs) - 1, 10, dtype=int)[break_point:]

    (
        performace_measures_A,
        performace_measures_B,
    ) = get_poa_values_for_given_strategies_dual_parameters(
        all_xs=all_xs,
        all_ys=all_ys,
        poa_span_1=poa_span_1,
        poa_span_2=poa_span_2,
        routing_1=R1,
        routing_2=R2,
        problem_parameters_1=problem_parameters_1,
        problem_parameters_2=problem_parameters_2,
        performance_measure_function=performance_measure_function,
    )

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(all_xs)
    plt.xlabel("Timepoints")
    plt.ylabel("Probability")
    plt.title("Row player")
    plt.legend([f"$s_{{{i + 1}}}$" for i in range(len(all_xs[0]))])
    plt.plot(
        [len(all_xs) / divide, len(all_xs) / divide],
        [-0.1, 1.1],
        linewidth=1,
        linestyle=":",
        color="green",
    )
    plt.annotate("Parameter increase", (len(all_xs) / divide, 1.1), ha="center")

    plt.subplot(2, 2, 2)
    plt.plot(all_ys)
    plt.xlabel("Timepoints")
    plt.ylabel("Probability")
    plt.title("Column player")
    plt.legend([f"$s_{{{i + 1}}}$" for i in range(len(all_ys[0]))])
    plt.plot(
        [len(all_ys) / divide, len(all_ys) / divide],
        [-0.1, 1.1],
        linewidth=1,
        linestyle=":",
        color="green",
    )
    plt.annotate("Parameter increase", (len(all_ys) / divide, 1.1), ha="center")

    poa_span = np.hstack((poa_span_1, poa_span_2))
    plt.subplot(2, 2, 3)
    plt.title("Row player - PoA")
    plt.plot(poa_span, performace_measures_A, color="black", linewidth=6)
    plt.ylim(top=poa_plot_max, bottom=poa_plot_min)

    plt.subplot(2, 2, 4)
    plt.title("Column player - PoA")
    plt.plot(poa_span, performace_measures_B, color="black", linewidth=6)
    plt.ylim(top=poa_plot_max, bottom=poa_plot_min)

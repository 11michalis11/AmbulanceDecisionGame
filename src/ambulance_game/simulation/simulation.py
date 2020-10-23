import numpy as np
import random
import ciw
import collections
import scipy.optimize


def build_model(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    system_capacity=float("inf"),
    parking_capacity=float("inf"),
):
    """Builds a ciw object that represents a model of a queuing network with two
    service centres; the hospital and the parking space. Patients arrive at the
    hospital and at the parking space with rates that follow the exponential
    distribution of λ_ο and λ_α respectively. The service distribution follows
    a constant distribution of 0 for the parking space and an exponential
    distribution with a rate of μ for the hospital. The variables "num_of_servers"
    and "parking_capacity" indicate the capacities of the two centres. Finally,
    the queue capacity is set to the difference between the number of servers and
    the system capacity for the hospital centre and for the parking space it is
    set to zero, as there should not occur any waiting there, just blockage.

    Parameters
    ----------
    lambda_a : float
        Arrival rate of ambulance patients
    lambda_o : float
        Arrival rate of other patients
    mu : float
        Service rate of hospital
    num_of_servers : integer
        The num_of_servers of the hospital
    """

    model = ciw.create_network(
        arrival_distributions=[
            ciw.dists.Exponential(lambda_a),
            ciw.dists.Exponential(lambda_o),
        ],
        service_distributions=[ciw.dists.Deterministic(0), ciw.dists.Exponential(mu)],
        routing=[[0.0, 1.0], [0.0, 0.0]],
        number_of_servers=[parking_capacity, num_of_servers],
        queue_capacities=[0, system_capacity - num_of_servers],
    )
    return model


def build_custom_node(threshold=float("inf")):
    """Build a custom node to replace the default ciw.Node. Inherits from the original
    ciw.Node class and replaces methods release_blocked_individual and finish_service.
    The methods are modified in such a way such that all individuals that are in
    the parking space node (node 1) remain blocked as long as the number of individuals
    in the hospital node (node 2) exceeds the threshold.

    Parameters
    ----------
    threshold : int, optional
        The capacity threshold to be used by the method

    Returns
    -------
    class
        A custom node class that inherits from ciw.Node
    """

    class CustomNode(ciw.Node):
        def release_blocked_individual(self):
            """
            Releases an individual who becomes unblocked when
            another individual is released:
              - check if individual in node 2 and should remain blocked
                i.e. if the number of individuals in that node > threshold
              - check if anyone is blocked by this node
              - find the individual who has been blocked the longest
              - remove that individual from blocked queue
              - check if that individual had their service interrupted
              - release that individual from their node
            """
            continue_blockage = (
                self.number_of_individuals >= threshold and self.id_number == 2
            )
            if (
                self.len_blocked_queue > 0
                and self.number_of_individuals < self.node_capacity
                and not continue_blockage
            ):
                node_to_receive_from = self.simulation.nodes[self.blocked_queue[0][0]]
                individual_to_receive_index = [
                    ind.id_number for ind in node_to_receive_from.all_individuals
                ].index(self.blocked_queue[0][1])
                individual_to_receive = node_to_receive_from.all_individuals[
                    individual_to_receive_index
                ]
                self.blocked_queue.pop(0)
                self.len_blocked_queue -= 1
                if individual_to_receive.interrupted:
                    individual_to_receive.interrupted = False
                    node_to_receive_from.interrupted_individuals.remove(
                        individual_to_receive
                    )
                    node_to_receive_from.number_interrupted_individuals -= 1
                node_to_receive_from.release(individual_to_receive_index, self)

        def finish_service(self):
            """
            The next individual finishes service:
              - finds the individual to finish service
              - check if they need to change class
              - find their next node
              - release the individual if there is capacity at destination,
                otherwise cause blockage
              - note that blockage also occurs when we are at node 1 and the
                number of individuals on node 2 are more than the 'threshold'
            """
            next_individual, next_individual_index = self.find_next_individual()
            self.change_customer_class(next_individual)
            next_node = self.next_node(next_individual)
            next_individual.destination = next_node.id_number
            if not np.isinf(self.c):
                next_individual.server.next_end_service_date = float("Inf")
            blockage = (
                next_node.number_of_individuals >= threshold and self.id_number == 1
            )
            if (
                next_node.number_of_individuals < next_node.node_capacity
            ) and not blockage:
                self.release(next_individual_index, next_node)
            else:
                self.block_individual(next_individual, next_node)

    return CustomNode


def simulate_model(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    seed_num=None,
    runtime=1440,
    system_capacity=float("inf"),
    parking_capacity=float("inf"),
    tracker=ciw.trackers.NodePopulation(),
):
    """Simulate the model by using the custom node and returning the simulation object.

    It is important to note that when the threshold is greater than the system capacity
    the parking capacity is forced to be 1 because otherwise, when the hospital gets
    full, ambulance patients will flood the parking spaces which is not what should
    happen in this particular scenario.
    TODO: use different approach to handle this scenario

    Additionally, the parking capacity should always be greater or equal to 1

    Parameters
    ----------
    seed_num : [float], optional
        A seed number in order to be able to replicate results, by default random.random()

    Returns
    -------
    object
        An object that contains all simulation records
    """

    if parking_capacity < 1:
        raise ValueError(
            "Simulation only implemented for parking_capacity >= 1"
        )  # TODO Add an option to ciw model to all for no parking capacity.

    if threshold > system_capacity:
        parking_capacity = 1
        # TODO: Different approach to handle this situation

    if seed_num == None:
        seed_num = random.random()
    model = build_model(
        lambda_a, lambda_o, mu, num_of_servers, system_capacity, parking_capacity
    )
    node = build_custom_node(threshold)
    ciw.seed(seed_num)
    simulation = ciw.Simulation(model, node_class=node, tracker=tracker)
    simulation.simulate_until_max_time(runtime)
    return simulation


def get_simulated_state_probabilities(
    simulation_object, output=np.ndarray, system_capacity=None, parking_capacity=None
):
    """Calculates the vector pi in a dictionary format or an array format. For the
    dictionary format the keys are the states (i,j) and the values are the probabilities
    that the system is in a current state. For the 2-dimensional array format the
    probability of being in state (i,j) is placed in the equivalent (i,j) position
    in the numpy array.

    Parameters
    ----------
    simulation_object : object
        The simulation object from ciw to extract pi from
    type : type, optional
        The type of format to be returned, by default np.ndarray

    Returns
    -------
    dictionary OR np.ndarray
        - A dictionary with the Markov states as keys and the equivalent probabilities as values
        - A numpy.ndarray Π where: Π(i,j) = probability of being in state (i,j)
    """
    state_probabilities_dictionary = (
        simulation_object.statetracker.state_probabilities()
    )
    if output == dict:
        return state_probabilities_dictionary
    elif output == np.ndarray:
        if parking_capacity == None:
            parking_capacity = max(
                [state[0] for state in state_probabilities_dictionary.keys()]
            )
        if system_capacity == None:
            system_capacity = max(
                [state[1] for state in state_probabilities_dictionary.keys()]
            )
        state_probabilities_array = np.full(
            (parking_capacity + 1, system_capacity + 1), np.NaN
        )
        for key, value in state_probabilities_dictionary.items():
            if value > 0:
                state_probabilities_array[key] = value
        return state_probabilities_array


def get_average_simulated_state_probabilities(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    parking_capacity,
    seed_num=None,
    runtime=1440,
    num_of_trials=10,
    output=np.ndarray,
):
    """
    This function runs the get_simulated_state_probabilities() for multiple iterations
    to eliminate any stochasticity from the results

    Parameters
    ----------
    output : type, optional
        The format of the output state probabilities, by default np.ndarray
    """
    if seed_num == None:
        seed_num = random.random()

    if output == dict:
        average_state_probabilities = {}
    else:
        average_state_probabilities = np.full(
            (parking_capacity + 1, system_capacity + 1), np.NaN
        )
    for trial in range(num_of_trials):
        simulation_object = simulate_model(
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            seed_num + trial,
            runtime,
            system_capacity,
            parking_capacity,
        )
        state_probabilities = get_simulated_state_probabilities(
            simulation_object, output, system_capacity, parking_capacity
        )
        if output == dict:
            if len(average_state_probabilities) == 0:
                average_state_probabilities = state_probabilities
            else:
                for key in average_state_probabilities.keys():
                    average_state_probabilities[key] += state_probabilities[key]
        else:
            for row in range(parking_capacity + 1):
                for col in range(system_capacity + 1):
                    updated_entry = np.nansum(
                        [
                            average_state_probabilities[row, col],
                            state_probabilities[row, col],
                        ]
                    )
                    average_state_probabilities[row, col] = (
                        updated_entry if updated_entry != 0 else np.NaN
                    )
    if output == dict:
        for key, value in average_state_probabilities.items():
            average_state_probabilities[key] = value / num_of_trials
    else:
        average_state_probabilities /= num_of_trials

    return average_state_probabilities


def extract_times_from_records(simulation_records, warm_up_time):
    """Get the required times (waiting, service, blocking) out of ciw's records
    where all individuals are treated the same way. This function can't distinguish
    between ambulance patients and other patients. It returns the aggregated waiting
    time, service times BUT only blocking times of ambulance patients.

    Parameters
    ----------
    simulation_records : list
        A list of all simulated records
    warm_up_time : int
        The time we start collecting results

    Returns
    -------
    list, list, list
        Three lists that contain waiting, service and blocking times
    """
    waiting_times = [
        r.waiting_time
        for r in simulation_records
        if r.arrival_date > warm_up_time and r.node == 2
    ]
    serving_times = [
        r.service_time
        for r in simulation_records
        if r.arrival_date > warm_up_time and r.node == 2
    ]
    blocking_times = [
        r.time_blocked
        for r in simulation_records
        if r.arrival_date > warm_up_time and r.node == 1
    ]
    return waiting_times, serving_times, blocking_times


def extract_times_from_individuals(
    individuals, warm_up_time, first_node_to_visit, total_node_visits
):
    """
    Extract waiting times and service times for all individuals and proceed to extract
    blocking times for just ambulance patients. The function uses individual's records
    and determines the type of patient that each individual is, based on the number
    of nodes visited.
    """
    waiting_times = []
    serving_times = []
    blocking_times = []

    for ind in individuals:
        if (
            ind.data_records[0].node == first_node_to_visit
            and len(ind.data_records) == total_node_visits
            and ind.data_records[total_node_visits - 1].arrival_date > warm_up_time
        ):
            waiting_times.append(ind.data_records[total_node_visits - 1].waiting_time)
            serving_times.append(ind.data_records[total_node_visits - 1].service_time)
        if (
            first_node_to_visit == ind.data_records[0].node == 1
            and ind.data_records[0].arrival_date > warm_up_time
        ):
            blocking_times.append(ind.data_records[0].time_blocked)

    return waiting_times, serving_times, blocking_times


def get_list_of_results(results):
    """Modify the outputs so that they are returned in a list format where it is
    sometimes easier to be used by other functions.

    Parameters
    ----------
    results : list
        A list of named tuples for each iteration

    Returns
    -------
    list, list, list
        Three lists that include all waits, services and blocks of all runs of all individuals
    """
    all_waits = [w.waiting_times for w in results]
    all_services = [s.service_times for s in results]
    all_blocks = [b.blocking_times for b in results]
    return all_waits, all_services, all_blocks


def get_multiple_runs_results(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    seed_num=None,
    warm_up_time=100,
    num_of_trials=10,
    runtime=1440,
    output_type="tuple",
    system_capacity=float("inf"),
    parking_capacity=float("inf"),
    patient_type="both",
):
    """Get the waiting times, service times and blocking times for multiple runs
    of the simulation. The function may return the times for ambulance patients,
    other patients or the aggregated total of the two.

    Parameters
    ----------
    warm_up_time : int, optional
        Time to start collecting results, by default 100
    num_of_trials : int, optional
        Number of trials to run the model, by default 10
    output_type : str, optional
        The results' output type (either tuple or list)], by default "tuple"
    patients_type : str, optional
        A string to identify what type of patients to get the times for

    Returns
    -------
    list
        A list of records where each record consists of the waiting, service and
        blocking times of one trial. Alternatively if the output_type = "list" then
        returns three lists with all waiting, service and blocking times

    """
    if seed_num == None:
        seed_num = random.random()
    records = collections.namedtuple(
        "records", "waiting_times service_times blocking_times"
    )
    results = []
    for trial in range(num_of_trials):
        simulation = simulate_model(
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            seed_num + trial,
            runtime,
            system_capacity,
            parking_capacity,
        )

        if patient_type == "both":
            sim_results = simulation.get_all_records()
            waiting_times, serving_times, blocking_times = extract_times_from_records(
                sim_results, warm_up_time
            )
            results.append(records(waiting_times, serving_times, blocking_times))

        if patient_type == "ambulance":
            individuals = simulation.get_all_individuals()
            (
                waiting_times,
                serving_times,
                blocking_times,
            ) = extract_times_from_individuals(
                individuals, warm_up_time, first_node_to_visit=1, total_node_visits=2
            )
            results.append(records(waiting_times, serving_times, blocking_times))

        if patient_type == "others":
            individuals = simulation.get_all_individuals()
            (
                waiting_times,
                serving_times,
                blocking_times,
            ) = extract_times_from_individuals(
                individuals, warm_up_time, first_node_to_visit=2, total_node_visits=1
            )
            results.append(records(waiting_times, serving_times, blocking_times))

    if output_type == "list":
        all_waits, all_services, all_blocks = get_list_of_results(results)
        return [all_waits, all_services, all_blocks]

    return results


def get_mean_blocking_difference_between_two_hospitals(
    prop_1,
    lambda_a,
    lambda_o_1,
    lambda_o_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    threshold_1,
    threshold_2,
    seed_num_1,
    seed_num_2,
    num_of_trials,
    warm_up_time,
    runtime,
):
    """Given a predefined proportion of the ambulance's arrival rate calculate the
    mean difference between blocking times of two hospitals with given set of parameters.
    Note that all parameters that end in "_1" correspond to the first hospital while "_2"
    to the second.

    Parameters
    ----------
    prop_1 : float
        Proportion of ambulance's arrival rate that will be distributed to hospital 1
    lambda_a : float
        Total ambulance arrival rate

    Returns
    -------
    float
        The difference between the mean blocking time of the two hospitals
    """
    lambda_a_1 = prop_1 * lambda_a
    lambda_a_2 = (1 - prop_1) * lambda_a

    res_1 = get_multiple_runs_results(
        lambda_a=lambda_a_1,
        lambda_o=lambda_o_1,
        mu=mu_1,
        num_of_servers=num_of_servers_1,
        threshold=threshold_1,
        seed_num=seed_num_1,
        warm_up_time=warm_up_time,
        num_of_trials=num_of_trials,
        output_type="tuple",
        runtime=runtime,
    )
    res_2 = get_multiple_runs_results(
        lambda_a=lambda_a_2,
        lambda_o=lambda_o_2,
        mu=mu_2,
        num_of_servers=num_of_servers_2,
        threshold=threshold_2,
        seed_num=seed_num_2,
        warm_up_time=warm_up_time,
        num_of_trials=num_of_trials,
        output_type="tuple",
        runtime=runtime,
    )

    hospital_1_blockages = [
        np.nanmean(b.blocking_times) if len(b.blocking_times) != 0 else 0 for b in res_1
    ]
    hospital_2_blockages = [
        np.nanmean(b.blocking_times) if len(b.blocking_times) != 0 else 0 for b in res_2
    ]
    diff = np.mean(hospital_1_blockages) - np.mean(hospital_2_blockages)

    return diff


def calculate_ambulance_best_response(
    lambda_a,
    lambda_o_1,
    lambda_o_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    threshold_1,
    threshold_2,
    seed_num_1,
    seed_num_2,
    num_of_trials,
    warm_up_time,
    runtime,
):
    """Obtains the optimal distribution of ambulances such that the blocking times
    of the ambulances in the two hospitals are identical and thus optimal(minimised).

    The brentq function is used which is an algorithm created to find the root of
    a function that combines root bracketing, bisection, and inverse quadratic
    interpolation. In this specific example the root to be found is the difference
    between the blocking times of two hospitals. In essence the brentq algorithm
    attempts to find the value of "prop_1" where the "diff" is zero
    (see function: get_mean_blocking_difference_between_two_hospitals).

    Returns
    -------
    float
        The optimal proportion where the hospitals have identical blocking times
    """
    optimal_prop = scipy.optimize.brentq(
        get_mean_blocking_difference_between_two_hospitals,
        a=0.01,
        b=0.99,
        args=(
            lambda_a,
            lambda_o_1,
            lambda_o_2,
            mu_1,
            mu_2,
            num_of_servers_1,
            num_of_servers_2,
            threshold_1,
            threshold_2,
            seed_num_1,
            seed_num_2,
            num_of_trials,
            warm_up_time,
            runtime,
        ),
    )
    return optimal_prop

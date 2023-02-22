import os
import cplex
from cplex.exceptions import CplexError
import networkx as nx
import random
import numpy as np
import argparse
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import LP_RCPSP_CPLEX


def disable_output_cpx(instance_cpx):
    instance_cpx.set_log_stream(None)
    # instance_cpx.set_error_stream(None)
    instance_cpx.set_warning_stream(None)
    instance_cpx.set_results_stream(None)


def createIP(rcpsp_model: RCPSPModel, ipfilename:str):
    solver = LP_RCPSP_CPLEX(rcpsp_model=rcpsp_model)
    solver.init_model()
    solver.model.export_as_lp(ipfilename+".lp")
    ip = cplex.Cplex(ipfilename+".lp")
    variable_names = [f"x[{key}]" for key in solver.x]
    return ip, variable_names


def extractVCG(g, E2, ip, set_biases, spo, gap=None, bestobj=None):
    num_solutions = 0
    if set_biases:
        num_solutions = ip.solution.pool.get_num()
        bias_arr = np.zeros(len(ip.solution.pool.get_values(0)))

        for sol_idx in range(ip.solution.pool.get_num()):
            # if ip.solution.pool.get_objective_value(sol_idx) <= 0:
            #     num_solutions -= 1
            bias_arr += ip.solution.pool.get_values(sol_idx)
        bias_arr /= num_solutions

        bias_dict = {}
        for index, name in enumerate(ip.variables.get_names()):
            bias_dict[name] = bias_arr[index]

        print("num_solutions = %d" % num_solutions)

    vcg = nx.Graph(num_solutions=num_solutions, gap=gap, bestobj=bestobj)

    vcg.add_nodes_from([("x" + str(node), {'objcoeff':-node_data['revenue']}) for node, node_data in g.nodes(data=True)], bipartite=0)
    vcg.add_nodes_from(["y" + str(edge[0]) + "_" + str(edge[1]) for edge in E2], bipartite=0)

    for node, node_data in g.nodes(data=True):
        node_name = "x" + str(node)
        bias = bias_dict[node_name] if set_biases else 0
        vcg.add_node(node_name, bias=bias, objcoeff=-1*node_data['revenue'], bipartite=0)

        if spo:
            vcg.nodes[node_name]['features'] = node_data['features']
            vcg.nodes[node_name]['model_indicator'] = node_data['model_indicator']

    for edge in E2:
        node_name = "y" + str(edge[0]) + "_" + str(edge[1])
        bias = bias_dict[node_name] if set_biases else 0
        vcg.add_node(node_name, bias=bias, objcoeff=g[edge[0]][edge[1]]['cost'], bipartite=0)

        if spo:
            vcg.nodes[node_name]['features'] = g[edge[0]][edge[1]]['features']
            vcg.nodes[node_name]['model_indicator'] = g[edge[0]][edge[1]]['model_indicator']
    
    constraint_counter = 0        
    for node1, node2, edge in g.edges(data=True):
        node_name = "c" + str(constraint_counter)
        vcg.add_node(node_name, rhs=1.0, bipartite=1)
        if (node1,node2) in E2:
            edge_varname = "y" + str(node1) + "_" + str(node2)
            vcg.add_edge(edge_varname, node_name, coeff=-1)
        vcg.add_edge("x" + str(node1), node_name, coeff=1)
        vcg.add_edge("x" + str(node2), node_name, coeff=1)
        constraint_counter += 1

    return vcg


def solveIP(ip, pool_bool, timelimit, mipgap, relgap_pool, maxsols, threads, memlimit, treememlimit, cpx_tmp):
    ip.parameters.emphasis.mip.set(1)
    ip.parameters.threads.set(threads)
    ip.parameters.workmem.set(memlimit)
    ip.parameters.timelimit.set(timelimit)
    # ip.parameters.mip.limits.treememory.set(treememlimit)
    ip.parameters.mip.strategy.file.set(2)
    ip.parameters.workdir.set(cpx_tmp)
    
    ip.solve()

    phase1_gap = 1e9
    if ip.solution.is_primal_feasible():
        phase1_gap = ip.solution.MIP.get_mip_relative_gap()
    phase1_status = ip.solution.get_status_string()
    phase2_bestobj = ip.solution.get_objective_value()

    phase2_status, phase2_gap = -1, -1
    if pool_bool:
        print("Finished Phase I.")
        ip.parameters.mip.tolerances.mipgap.set(max([phase1_gap, mipgap]))
        # er_200_SET2_1k was with 0.1
        # 2 = Moderate: generate a larger number of solutions
        ip.parameters.mip.pool.intensity.set(2)
        # Replace the solution which has the worst objective
        ip.parameters.mip.pool.replace.set(1)
        ip.parameters.timelimit.set(timelimit)
        # Maximum number of solutions generated for the solution pool by populate
        ip.parameters.mip.limits.populate.set(maxsols)
        # Relative gap for the solution pool
        ip.parameters.mip.pool.relgap.set(relgap_pool) #er_200_SET2_1k was with 0.2

        try:
            ip.populate_solution_pool()
            if ip.solution.is_primal_feasible():
                phase2_gap = ip.solution.MIP.get_mip_relative_gap()
                phase2_bestobj = ip.solution.get_objective_value()
                phase2_status = ip.solution.get_status_string()
            print("Finished Phase II.")

        except CplexError as exc:
            print(exc)
            return

    return phase1_status, phase1_gap, phase2_status, phase2_gap, phase2_bestobj


def script_creation_data():
    folders_kobe = ["../kobe-rcpsp/data/rcpsp/j30.sm/",
                    "../kobe-rcpsp/data/rcpsp/j60.sm/",
                    "../kobe-rcpsp/data/rcpsp/j90.sm/",
                    "../kobe-rcpsp/data/rcpsp/j120.sm/"]

    def explore_folder_sm(folder):
        return [os.path.join(folder, f)
                for f in os.listdir(folder)
                if ".sm" in f]
    files_per_folder = [explore_folder_sm(f) for f in folders_kobe]
    for i in range(3):
        for file in files_per_folder[i]:
            script_one_file(file)


def script_one_file(file: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir", type=str, default="rcpsp/")
    parser.add_argument("-instance", type=str, default='')
    parser.add_argument("-min_n", type=int)
    parser.add_argument("-max_n", type=int)
    parser.add_argument("-er_prob", type=float, default=0.1)
    parser.add_argument("-whichSet", type=str, default='SET2')
    parser.add_argument("-setParam", type=float, default=100.0)
    parser.add_argument("-alphaE2", type=float, default=0.75)
    parser.add_argument("-timelimit", type=float, default=120.0)
    parser.add_argument("-solve", type=int, default=1)
    parser.add_argument("-threads", type=int, default=4)
    parser.add_argument("-memlimit", type=int, default=2000)
    parser.add_argument("-treememlimit", type=int, default=20000)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-mipgap", type=float, default=0.1)
    parser.add_argument("-relgap_pool", type=float, default=0.1)
    parser.add_argument("-maxsols", type=int, default=1000)
    parser.add_argument("-overwrite_data", type=int, default=0)
    parser.add_argument("-cpx_output", type=int, default=0)
    parser.add_argument("-cpx_tmp", type=str, default="./tmp/")

    parser.add_argument("-spo", type=int, default=0)
    parser.add_argument("-spo_halfwidth", type=float, default=0.5)
    parser.add_argument("-spo_polydeg", type=int, default=2)
    parser.add_argument("-spo_bias_nodes", type=float, default=-100)
    parser.add_argument("-spo_bias_edges", type=float, default=10)

    args = parser.parse_args()
    print(args)

    # assert(args.max_n >= args.min_n)

    lp_dir = "LP/" + args.exp_dir
    try:
        os.makedirs(lp_dir)
    except OSError:
        if not os.path.exists(lp_dir):
            raise

    sol_dir = "SOL/" + args.exp_dir
    try:
        os.makedirs(sol_dir)
    except OSError:
        if not os.path.exists(sol_dir):
            raise

    data_dir = "DATA/" + args.exp_dir
    try:
        os.makedirs(data_dir)
    except OSError:
        if not os.path.exists(data_dir):
            raise

    if args.spo:
        spodata_dir = "SPO_DATA/" + args.exp_dir
        try:
            os.makedirs(spodata_dir)
        except OSError:
            if not os.path.exists(spodata_dir):
                raise

    # Seed generator
    random.seed(args.seed)
    np.random.seed(args.seed)
    from discrete_optimization.rcpsp.rcpsp_parser import parse_file
    rcpsp_model = parse_file(file)
    lpname = os.path.basename(file) + "_lp"
    ip, variable_names = createIP(rcpsp_model=rcpsp_model,
                                  ipfilename=lp_dir + "/" + lpname)
    print("Created MIP instance.")

    # disable all cplex output
    if not args.cpx_output:
        disable_output_cpx(ip)

    num_solutions = 0
    phase1_gap = None
    phase2_bestobj = None

    pool_bool = (args.solve == 1)
    if args.solve > 0:
        start_time = ip.get_time()
        phase1_status, phase1_gap, phase2_status, phase2_gap, phase2_bestobj = solveIP(
            ip,
            pool_bool,
            args.timelimit,
            args.mipgap,
            args.relgap_pool,
            args.maxsols,
            args.threads,
            args.memlimit,
            args.treememlimit,
            args.cpx_tmp)
        end_time = ip.get_time()
        total_time = end_time - start_time

        num_solutions = ip.solution.pool.get_num()
        results_str = ("%s,%s,%g,%s,%g,%g,%d,%g\n" % (
            lpname,
            phase1_status,
            phase1_gap,
            phase2_status,
            phase2_gap,
            phase2_bestobj,
            num_solutions,
            total_time))
        print(results_str)

        with open(sol_dir + "/" + lpname + ".sol", "w+") as sol_file:
            sol_file.write(results_str)

        if pool_bool and num_solutions >= 1:
            # Collect solutions from pool
            solutions_matrix = np.zeros((num_solutions, len(ip.solution.pool.get_values(0))))
            objval_arr = np.zeros(num_solutions)
            for sol_idx in range(num_solutions):
                sol_objval = ip.solution.pool.get_objective_value(sol_idx)
                objval_arr[sol_idx] = sol_objval
                # if sol_objval > 0:
                solutions_matrix[sol_idx] = ip.solution.pool.get_values(sol_idx)
                print(solutions_matrix[sol_idx])
            solutions_obj_matrix = np.concatenate((np.expand_dims(objval_arr, axis=0).T, solutions_matrix), axis=1)

            with open(sol_dir + "/" + lpname + ".npz", 'wb') as f:
                np.savez_compressed(f, solutions=solutions_obj_matrix)
            print("Wrote npz file.")


if __name__ == "__main__":
    script_creation_data()

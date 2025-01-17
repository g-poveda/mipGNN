import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx
import argparse
import io
import heapq
from pathlib import Path
import time
import math

import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
from gnn_models.EdgeConv.mip_bipartite_simple_class import SimpleNet

import cplex

import callbacks_cplex
import utils
import predict

import pickle

def rename_variables(var_names):
    for i in range(len(var_names)):
        name = var_names[i]
        name = name.replace('(','[')
        name = name.replace(')',']')
        name = name.replace('_',',')
        var_names[i] = name
    return var_names

# direction=1: branch on most integer first
def set_cplex_priorities(instance_cpx, prediction, direction=1):
    # score variables based on bias prediction
    scores = np.max(((1-prediction), prediction), axis=0)
    priorities = np.argsort(direction * scores)

    # set priorities
    # reference: https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/refpythoncplex/html/cplex._internal._subinterfaces.OrderInterface-class.html
    order_tuples = []
    var_names = instance_cpx.variables.get_names()

    cur_priority = 0
    for priority, var_cpxid in enumerate(priorities):
        var_name = var_names[var_cpxid]
        # print(scores[var_cpxid], scores[priorities[priority-1]])
        # if priority > 0 and scores[var_cpxid] > scores[priorities[priority-1]] + 1e-3:
        cur_priority += 1
            # print(cur_priority)
        order_tuples += [(var_name, cur_priority, instance_cpx.order.branch_direction.up)]

    # print(cur_priority)
    # z=1/0
    instance_cpx.order.set(order_tuples)

def mipeval(
    method, 
    instance, 
    graph='',
    instance_params='',
    model='', 
    logfile='sys.stdout',
    barebones=0,
    cpx_emphasis=1,
    cpx_threads=1,
    cpx_tmp='./cpx_tmp',
    timelimit=60,
    memlimit=1024,
    freq_best=100,
    lb_threshold=5,
    num_mipstarts=10,
    mipstart_strategy='repair',
    branching_direction=1,
    zero_damping=1.0
    ):
    
    print(locals())

    assert (len(method) >= 1)
    assert (cpx_emphasis >= 0 and cpx_emphasis <= 4)
    assert (timelimit > 0)

    """ Create CPLEX instance """
    instance_cpx = cplex.Cplex(instance)
    sense_str = instance_cpx.objective.sense[instance_cpx.objective.get_sense()]
    num_variables = instance_cpx.variables.get_num()
    num_constraints = instance_cpx.linear_constraints.get_num()    
    start_time = instance_cpx.get_time()

    """ CPLEX output management """
    logstring = sys.stdout
    summary_string = sys.stdout
    if logfile != 'sys.stdout':
        logstring = io.StringIO()
        summary_string = io.StringIO()
        instance_cpx.set_log_stream(logstring)
        instance_cpx.set_results_stream(logstring)
        instance_cpx.set_warning_stream(logstring)
        # instance_cpx.set_error_stream(logstring)
        instance_cpx.set_error_stream(open(os.devnull, 'w'))

    """ Set CPLEX parameters, if any """
    instance_cpx.parameters.timelimit.set(timelimit)
    instance_cpx.parameters.emphasis.mip.set(cpx_emphasis)
    instance_cpx.parameters.mip.display.set(3)
    instance_cpx.parameters.threads.set(cpx_threads)
    instance_cpx.parameters.workmem.set(memlimit)
    instance_cpx.parameters.mip.limits.treememory.set(20000)
    instance_cpx.parameters.mip.strategy.file.set(2)
    instance_cpx.parameters.workdir.set(cpx_tmp)
    if barebones:
        instance_cpx.parameters.mip.limits.cutpasses.set(-1)
        instance_cpx.parameters.mip.strategy.heuristicfreq.set(-1)
        instance_cpx.parameters.preprocessing.presolve.set(0)

        # DFS = 0, BEST-BOUND = 1 (default), BEST-EST = 2, BEST-EST-ALT = 3
        # instance_cpx.parameters.mip.strategy.nodeselect.set(3)

    time_rem_cplex = timelimit
    time_vcg = time.time()
    time_vcg_reading = 0
    time_pred = 0

    is_primal_mipstart = False
    """ Solve CPLEX instance with user-selected method """
    if 'default' not in method[0]:
        assert (len(graph) > 0 or len(instance_params) > 0) and len(model) > 0
        """ Read in the pickled graph and the trained model """
        time_vcg_reading = time.time()
        print("Reading VCG...")
        graph = nx.read_gpickle(graph)
        print("\t took %g secs." % (time.time()-time_vcg))
        time_vcg_reading = time.time() - time_vcg_reading

        print("Predicting...")
        timestamp_pred = time.time()
        prediction, node_to_varnode = predict.get_prediction(model_name=model, graph=graph)
        dict_varname_seqid = predict.get_variable_cpxid(graph, node_to_varnode, prediction)
        print("\t took %g secs." % (time.time()-timestamp_pred))
        time_pred = time.time() - timestamp_pred
        # print(prediction)
        # todo check dimensions of p

        time_rem_cplex = timelimit - time_pred
        print("time_rem_cplex = %g" % time_rem_cplex)
        instance_cpx.parameters.timelimit.set(time_rem_cplex)

        var_names = rename_variables(instance_cpx.variables.get_names())
        prediction_reord = [dict_varname_seqid[var_name][1] for var_name in var_names]
        prediction = np.array(prediction_reord)

        if len(method) == 1 and ('local_branching' in method[0]):
            pred_one_coeff = (prediction >= 0.9) * (-1)
            pred_zero_coeff = (prediction <= 0.1)
            num_ones = -np.sum(pred_one_coeff)
            coeffs = pred_one_coeff + pred_zero_coeff

            local_branching_coeffs = [list(range(len(prediction))), coeffs.tolist()]

            if method[0] == 'local_branching_approx':
                instance_cpx.linear_constraints.add(
                    lin_expr=[local_branching_coeffs],
                    senses=['L'],
                    rhs=[float(lb_threshold - num_ones)],
                    names=['local_branching'])
            
            elif method[0] == 'local_branching_exact':
                branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_local_exact)

                branch_cb.coeffs = local_branching_coeffs
                branch_cb.threshold = lb_threshold - num_ones
                branch_cb.is_root = True

        if 'branching_priorities' in method:
            set_cplex_priorities(instance_cpx, prediction, branching_direction)

        if 'node_selection' in method:
            # score variables based on bias prediction
            scores = np.max(((1-prediction), prediction), axis=0)
            rounding = np.round(prediction)
            
            print(np.mean(scores), np.mean(rounding))
            print(np.argsort(prediction), np.sort(prediction)[:10], np.sort(prediction)[-10:])

            branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_attach_data2)
            node_cb = instance_cpx.register_callback(callbacks_cplex.node_selection3)

            branch_cb.scoring_function = 'sum' #'estimate'
            branch_cb.scores = scores
            branch_cb.rounding = rounding
            branch_cb.zero_damping = zero_damping

            node_cb.last_best = 0
            node_cb.freq_best = freq_best

            node_priority = []
            branch_cb.node_priority = node_priority
            node_cb.node_priority = node_priority            

            branch_cb.time = 0
            node_cb.time = 0

        if ('primal_mipstart' in method) or ('primal_mipstart_only' in method):
            is_primal_mipstart = True
            if not barebones or 'primal_mipstart_only' in method:
                instance_cpx.parameters.mip.limits.cutpasses.set(-1)
                instance_cpx.parameters.mip.strategy.heuristicfreq.set(-1)
                instance_cpx.parameters.preprocessing.presolve.set(0)

            mipstart_string = sys.stdout if logfile == "sys.stdout" else io.StringIO()

            #frac_variables = [0.001*(1.5**i) for i in range(18)] #[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            #frac_variables = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            #frac_variables = np.flip(np.linspace(0, 1, num=num_mipstarts+1))[:-1]
            #print(frac_variables)
            #threshold_set = np.minimum(prediction, 1-prediction)
            #threshold_set = np.sort(threshold_set)#[:mipstart_numthresholds]
            
            #threshold_set = [threshold_set[max([0, int(math.ceil(frac_variables[i]*num_variables)) - 1])] for i in range(len(frac_variables))]
            
            threshold_set = [0.01*(2**i) for i in range(6)]
            
            threshold_set.reverse()
            threshold_set = np.clip(threshold_set, a_min=0, a_max=0.5)
            print("threshold_set = ", threshold_set)
            
            if mipstart_strategy == 'repair':
                mipstart_strategy_int = instance_cpx.MIP_starts.effort_level.repair
            elif mipstart_strategy == 'solve_MIP':
                mipstart_strategy_int = instance_cpx.MIP_starts.effort_level.solve_MIP
            else:
                print("invalid mipstart_strategy %s" % mipstart_strategy)
                exit()

            best_objval_mipstart = -math.inf if sense_str == 'maximize' else math.inf
            for idx, threshold in enumerate(threshold_set):
                time_rem_cplex = timelimit - time_pred #(time.time() - time_vcg)
                if time_rem_cplex <= 0:
                    break

                indices_integer = np.where((prediction >= 1-threshold) | (prediction <= threshold))[0]
                print(idx, threshold, len(indices_integer), len(prediction))

                if len(indices_integer) == 0:
                    continue

                instance_cpx.parameters.mip.display.set(0)
                instance_cpx.parameters.mip.limits.nodes.set(0)
                print("time_rem_cplex = %g" % time_rem_cplex)
                instance_cpx.parameters.timelimit.set(time_rem_cplex)

                instance_cpx.MIP_starts.add(
                    cplex.SparsePair(
                        ind=indices_integer.tolist(),
                        val=np.round(prediction[indices_integer]).tolist()),
                    mipstart_strategy_int)

                instance_cpx.solve()
                instance_cpx.MIP_starts.delete()
                
                if instance_cpx.solution.is_primal_feasible(): #and instance_cpx.solution.get_objective_value() > best_objval_mipstart:
                    is_sol_better = (instance_cpx.solution.get_objective_value() > best_objval_mipstart) if sense_str == 'maximize' else (instance_cpx.solution.get_objective_value() < best_objval_mipstart)
                    if not is_sol_better:
                        continue
                    best_objval_mipstart = instance_cpx.solution.get_objective_value()
                    best_time = time.time() - time_vcg
                    incb_str_cur = ("Found incumbent of value %g after %g sec. mipstart %d %g %g\n" % (best_objval_mipstart, best_time, len(indices_integer), threshold, len(indices_integer)/num_variables))
                    print(incb_str_cur)
                    mipstart_string.write(incb_str_cur)#"Found incumbent of value %g after %g sec. mipstart %d %g %g\n" % (best_objval_mipstart, best_time, len(indices_integer), threshold))

            instance_cpx.parameters.mip.display.set(3)
            if not barebones and not 'primal_mipstart_only' in method:
                instance_cpx.parameters.mip.limits.cutpasses.set(0)
                instance_cpx.parameters.mip.strategy.heuristicfreq.set(0)
                instance_cpx.parameters.preprocessing.presolve.set(1)

            if 'primal_mipstart_only' not in method:
                instance_cpx.parameters.mip.limits.nodes.set(1e9)

    elif method[0] == 'default_emptycb':
        branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_empty)

    time_rem_cplex = timelimit - time_pred #(time.time() - time_vcg)
    print("time_rem_cplex = %g" % time_rem_cplex)
    
    if time_rem_cplex > 0:
        instance_cpx.parameters.timelimit.set(time_rem_cplex)

        # todo: consider runseeds 
        #  https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/refpythoncplex/html/cplex.Cplex-class.html?view=kc#runseeds
        instance_cpx.solve()
    end_time = instance_cpx.get_time()

    """ Get solving performance statistics """
    incumbent_str = ''
    cplex_status = instance_cpx.solution.get_status_string()
    best_bound = instance_cpx.solution.MIP.get_best_objective()
    num_nodes = instance_cpx.solution.progress.get_num_nodes_processed()
    total_time = end_time - start_time

    instance_name = os.path.splitext(os.path.basename(instance))[0]
    
    best_objval, gap = math.inf, math.inf
    if instance_cpx.solution.is_primal_feasible():
        best_objval = instance_cpx.solution.get_objective_value()
        gap = instance_cpx.solution.MIP.get_mip_relative_gap()

    summary_string.write('solving stats,%s,%g,%g,%g,%g,%i,%g,%s,%i,%i,%g,%g\n' % (
            cplex_status, 
            best_objval,
            best_bound,
            gap,
            total_time,
            num_nodes,
            timelimit - time_rem_cplex,
            instance_name,
            num_variables,
            num_constraints,
            time_vcg_reading,
            time_pred))
    if logfile != 'sys.stdout':
        if instance_cpx.solution.is_primal_feasible():
            incumbent_str = ''
            if is_primal_mipstart:
                incumbent_str += utils.parse_cplex_log(mipstart_string.getvalue(), time_offset=time_pred)
            incumbent_str += utils.parse_cplex_log(logstring.getvalue(), time_offset=time_pred)
            print(incumbent_str)
            summary_string.write(incumbent_str)
        summary_string = summary_string.getvalue()
        with open(logfile, 'w') as logfile:
            logfile.write(summary_string)

if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    # parser.add_argument("-method", type=str, default='default')
    parser.add_argument('-method', nargs='+', type=str, required=True)
    parser.add_argument("-instance", type=str)
    parser.add_argument("-graph", type=str, default='')
    parser.add_argument("-instance_params", type=str, default='')
    parser.add_argument("-model", type=str, default="../gnn_models/EdgeConv/trained_p_hat300-2")
    parser.add_argument("-cpx_emphasis", type=int, default=1)
    parser.add_argument("-cpx_threads", type=int, default=1)
    parser.add_argument("-cpx_tmp", type=str, default='./cpx_tmp/')
    parser.add_argument("-barebones", type=int, default=0)
    parser.add_argument("-timelimit", type=float, default=60)
    parser.add_argument("-memlimit", type=float, default=1024)
    parser.add_argument("-logfile", type=str, default='sys.stdout')

    # Parameters for node selection
    parser.add_argument("-freq_best", type=int, default=100)
    parser.add_argument("-zero_damping", type=float, default=1.0)

    # Parameters for exact local branching
    parser.add_argument("-lb_threshold", type=int, default=5)

    # Parameters for primal heuristic mip start
    parser.add_argument("-num_mipstarts", type=int, default=6)
    parser.add_argument("-mipstart_strategy", type=str, default="repair")

    # Parameters for branching priorities
    parser.add_argument("-branching_direction", type=int, default=1)

    args = parser.parse_args()
    print(args)

    mipeval(**vars(args))

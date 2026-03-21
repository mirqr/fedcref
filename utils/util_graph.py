

# dd is a dictionary. key is class/cluster --> [tuples of (other dev, class/cluster)] -- LIST!
def append_graph_from_dic(dev_name, dd, filename_graph): # 
    #clear_graph_file(filename_graph)
    append_edge_line(filename_graph, dev_name, dev_name, -1) # add self loop to put the node in the graph
    for key in dd.keys():
        for dev_other, other_key in dd[key]: # unpack tuple

        
            #print('dev_name: ', dev_name, 'other: ', dev_other, 'key: ', key, 'other_key: ', other_key)
            #Dev.append_edge_line(filename_graph, dev_name, dev_other, str(key)+'_'+str(other_key))
            append_edge_line(filename_graph, dev_name, dev_other, str(key))


def clear_graph_file(filename_graph):
    with open(filename_graph, 'w') as f:
        f.close()


def append_edge_line(filename_graph, node1, node2, weight):
    line = node1 + ' ' + node2 + ' ' + str(weight) + '\n'
    with open(filename_graph, 'a') as f:
        f.write(line)
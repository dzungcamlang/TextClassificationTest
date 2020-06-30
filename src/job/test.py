def find_min_path_length(graph, node1, node2):
    nodes2check = graph.get(node1)
    checked_node_set = set()
    if nodes2check == None: return -1#如果返回-1，表示不存在路径
    distance = 0
    while len(nodes2check ) > 0:
        distance += 1
        if node2 in  nodes2check:
            return distance 
        else:
            checked_node_set = checked_node_set | nodes2check 
            temp_nodes2check  = set()  # 如果已经置空，怎么for
            for a_node in nodes2check:
                temp_nodes2check  = temp_nodes2check  | graph.get(a_node , set())
            nodes2check = temp_nodes2check  - checked_node_set 
    return -1

#1, 2, 3, 4

link = {1: set([2, 3]), 3: set([4, 1])}
node_pair = [1, 4]
min_distance = find_min_path_length(link, node_pair[0], node_pair[1])
print(min_distance)
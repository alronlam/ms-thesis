class Node:

    def __init__(self, data):
        self.data = data
        self.children = []

    def __str__(self):
        return "({},[{}])".format(self.data, self.get_children_list_str())

    def get_children_list_str(self):
        children_str_list = (child.__str__() for child in self.children)
        return ",".join(children_str_list)

    def add_child(self, child_node):
        self.children.append(child_node)

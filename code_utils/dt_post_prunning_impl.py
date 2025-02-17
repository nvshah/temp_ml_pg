from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def get_subtree_nodes(tree, node_id):
    """Get all nodes in the subtree rooted at node_id."""
    left = tree.tree_.children_left[node_id]
    right = tree.tree_.children_right[node_id]
    
    if left == -1:  # Leaf node
        return [node_id]
    
    return [node_id] + get_subtree_nodes(tree, left) + get_subtree_nodes(tree, right)

def prune_subtree(tree, node_id):
    """
    Temporarily prune the subtree rooted at node_id by making it a leaf node.
    Returns the original children_left and children_right arrays for restoration.
    """
    left = tree.tree_.children_left.copy()
    right = tree.tree_.children_right.copy()
    
    # Get all nodes in this subtree
    subtree_nodes = get_subtree_nodes(tree, node_id)
    
    # Temporarily make them all leaf nodes
    tree.tree_.children_left[subtree_nodes] = -1
    tree.tree_.children_right[subtree_nodes] = -1
    
    return left, right

def restore_subtree(tree, node_id, left, right):
    """Restore the original subtree structure."""
    tree.tree_.children_left = left
    tree.tree_.children_right = right

def post_prune_tree(tree, X_val, y_val, node_id=0):
    """
    Recursively post-prune the decision tree using reduced error pruning.
    
    Parameters:
    tree: DecisionTreeClassifier
        The trained decision tree to prune
    X_val: array-like
        Validation data features
    y_val: array-like
        Validation data labels
    node_id: int
        Current node being considered for pruning
        
    Returns:
    float
        Best accuracy achieved for this subtree
    """
    # If this is a leaf node, return current accuracy
    if tree.tree_.children_left[node_id] == -1:
        return accuracy_score(y_val, tree.predict(X_val))
    
    # Get accuracy before pruning
    base_acc = accuracy_score(y_val, tree.predict(X_val))
    
    # Try pruning this node
    left, right = prune_subtree(tree, node_id)
    pruned_acc = accuracy_score(y_val, tree.predict(X_val))
    
    # If pruning improves accuracy, keep it pruned
    if pruned_acc >= base_acc:
        return pruned_acc
    
    # Otherwise, restore the node and try pruning its children
    restore_subtree(tree, node_id, left, right)
    left_child = tree.tree_.children_left[node_id]
    right_child = tree.tree_.children_right[node_id]
    
    _ = post_prune_tree(tree, X_val, y_val, left_child)
    _ = post_prune_tree(tree, X_val, y_val, right_child)
    
    return accuracy_score(y_val, tree.predict(X_val))

# Example usage
def train_and_prune_tree(X_train, X_val, y_train, y_val, max_depth=None):
    """
    Train a decision tree and prune it using validation data.
    
    Parameters:
    X_train, y_train: Training data
    X_val, y_val: Validation data
    max_depth: Maximum depth of the initial tree
    
    Returns:
    tree: Pruned DecisionTreeClassifier
    """
    # Train initial tree
    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_train, y_train)
    
    # Print initial accuracy
    initial_acc = accuracy_score(y_val, tree.predict(X_val))
    print(f"Initial validation accuracy: {initial_acc:.4f}")
    
    # Prune tree
    final_acc = post_prune_tree(tree, X_val, y_val)
    print(f"Final validation accuracy: {final_acc:.4f}")
    
    return tree
Based on your default model observations, I can help suggest reasonable hyperparameter ranges for pre-pruning. Let's analyze your current results and set ranges accordingly:

Current observations:
- Default model achieves high accuracy (0.99) with depth=15 and 125 nodes
- Balanced model also has 0.99 accuracy but uses depth=18 and 175 nodes
- Both models show signs of possible overfitting (very high accuracy)

Suggested hyperparameter ranges:

1. max_depth:
- Start lower than current depths (15 and 18)
- Suggested range: [5, 8, 10, 12, 15]
- Rationale: Since default model performs well at depth 15, we probably don't need to go deeper

2. min_samples_split:
- Current = 2 (very permissive)
- Suggested range: [5, 10, 15, 20]
- Rationale: Higher values might reduce overfitting while maintaining performance

3. min_samples_leaf:
- Current = 1 (very permissive)
- Suggested range: [2, 4, 6, 8]
- Rationale: Slightly higher values will ensure more robust leaf nodes

4. max_leaf_nodes:
- Current models have 125-175 nodes
- Suggested range: [50, 75, 100, 125]
- Rationale: Start with fewer nodes to see if we can maintain performance with simpler tree

Additional considerations:
1. Since your current models show high accuracy, focus on reducing complexity while maintaining performance
2. Watch for changes in recall and precision, not just accuracy
3. Consider adding class_weight='balanced' to the best performing pre-pruned model
4. These ranges should give you a good balance between model complexity and performance

Would you like me to adjust any of these ranges based on other characteristics of your dataset?
import numpy as np
from data_poisoning import data_poisoning

"""
## Experiment 1
### Knobs:
  - non-targeted, binary
  - number of test points: 1 x 21
  - number of training points: [1, 2, 4, 8, 16]
  - Datasets: ["Dog/Fish", "Eagle/Mushroom"]

### Metrics:
  - Success rate
  - Angle change
  - Test accuracy change
"""

dog_fish = {
    "name": "Dog_Fish",
    "classes":["dog", "fish"],
    "num_train_ex_per_class":900,
    "num_test_ex_per_class":300
}
eagle_mushroom = {
    "name": "Eagle_Mushroom",
    "classes":["Eagle", "Mushroom"],
    "num_train_ex_per_class":500,
    "num_test_ex_per_class":50
}

for dataset_classes in [dog_fish, eagle_mushroom]:
    sampled_test_points = np.random.choice(np.arange(2*dataset_classes["num_test_ex_per_class"]), 7, replace=False)
    for num_train_points in [1]:
        for test_idx in sampled_test_points:
            imgs_indices_to_poison, poisoned_images = data_poisoning(data_selected = dataset_classes["classes"], 
                                                                     num_train_ex_per_class=dataset_classes["num_train_ex_per_class"], 
                                                                     num_test_ex_per_class=dataset_classes["num_test_ex_per_class"],
                                                                     use_IF = True,
                                                                     target_test_idx = [test_idx],
                                                                     num_to_perterb = num_train_points,
                                                                     target_labels = None)
            np.save("Experiment_results/Experiment_1/Experiment_1_indices_IF_{}_num_train_{}__{}".format(dataset_classes["name"], num_train_points, test_idx), imgs_indices_to_poison)
            np.save("Experiment_results/Experiment_1/Experiment_1_poisons_IF_{}_num_train_{}__{}".format(dataset_classes["name"], num_train_points, test_idx), poisoned_images)
            

# for dataset_classes in [dog_fish, eagle_mushroom]:
#     sampled_test_points = np.random.choice(np.arange(2*dataset_classes["num_test_ex_per_class"]), 21, replace=False)
#     for num_train_points in [1,2,4,8,16]:
#         for test_idx in sampled_test_points:
#             imgs_indices_to_poison, poisoned_images = data_poisoning(data_selected = dataset_classes["classes"], 
#                                                                      num_train_ex_per_class=dataset_classes["num_train_ex_per_class"], 
#                                                                      num_test_ex_per_class=dataset_classes["num_test_ex_per_class"],
#                                                                      use_IF = False,
#                                                                      target_test_idx = [test_idx],
#                                                                      num_to_perterb = num_train_points,
#                                                                      target_labels = None)
#             np.save("Experiment_results/Experiment_1/Experiment_1_indices_FC_{}_num_train_{}__{}".format(dataset_classes["name"], num_train_points, test_idx), imgs_indices_to_poison)
#             np.save("Experiment_results/Experiment_1/Experiment_1_poisons_FC_{}_num_train_{}__{}".format(dataset_classes["name"], num_train_points, test_idx), poisoned_images)
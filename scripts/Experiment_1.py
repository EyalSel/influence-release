import numpy as np
from data_poisoning import data_poisoning
from utils import dataset_metadatas, experiment_result_metadata_to_FN, FN_to_experiment_result_metadata, get_dataset, get_full_model_graph

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


def run_exp_1_IF():
    for dataset_classes in [dataset_metadatas["Eagle-Mushroom"]]:
        cache = {}
        cache['data_sets'] = get_dataset(dataset_classes)
        cache['full_model_and_graph'] = get_full_model_graph(dataset_classes, cache['data_sets'])
        sampled_test_points = np.random.choice(np.arange(2*dataset_classes["num_test_ex_per_class"]), 7, replace=False)
        for num_train_points in [4,8,16]:
            for test_idx in sampled_test_points:
                imgs_indices_to_poison, poisoned_images = data_poisoning(dataset_metadata = dataset_classes, 
                                                                         use_IF = True,
                                                                         target_test_idx = [test_idx],
                                                                         num_to_perterb = num_train_points,
                                                                         target_labels = None,
																		 cache = cache)
                path_to_save = "Experiment_results/Experiment_1/"
                experiment_result_metadata = {
                    "Experiment_number":1,
                    "contents_type":"indices",
                    "method":"IF",
                    "dataset_name":dataset_classes["name"],
                    "num_poisoned_training_points":num_train_points,
                    "test_idx":test_idx
                }
                FN = experiment_result_metadata_to_FN(experiment_result_metadata)
                np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), imgs_indices_to_poison)
                experiment_result_metadata["contents_type"]="poisons"
                np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), poisoned_images)

def run_exp_1_FC():
    for dataset_classes in [dataset_metadatas["Dog-Fish"], dataset_metadatas["Eagle-Mushroom"]]:
        cache = {}
        cache['data_sets'] = get_dataset(dataset_classes)
        cache['full_model_and_graph'] = get_full_model_graph(dataset_classes, cache['data_sets'])
        sampled_test_points = np.random.choice(np.arange(2*dataset_classes["num_test_ex_per_class"]), 7, replace=False)
        for num_train_points in [1,2,4,8,16]:
            for test_idx in sampled_test_points:
                imgs_indices_to_poison, poisoned_images = data_poisoning(dataset_metadata = dataset_classes, 
                                                                         use_IF = False,
                                                                         target_test_idx = [test_idx],
                                                                         num_to_perterb = num_train_points,
                                                                         target_labels = None,
																		 cache = cache)
                path_to_save = "Experiment_results/Experiment_1/"
                experiment_result_metadata = {
                    "Experiment_number":1,
                    "contents_type":"indices",
                    "method":"FC",
                    "dataset_name":dataset_classes["name"],
                    "num_poisoned_training_points":num_train_points,
                    "test_idx":test_idx
                }
                FN = experiment_result_metadata_to_FN(experiment_result_metadata)
                np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), imgs_indices_to_poison)
                experiment_result_metadata["contents_type"]="poisons"
                np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), poisoned_images)

if __name__ == "__main__":
    run_exp_1_IF()